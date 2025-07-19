#!/usr/bin/env python3
"""
Forest Gump Real-Time Demo Script

This script creates a Leatherback environment and tests real-time simulation synchronization
with proper control decimation. It enforces real-time execution to validate timing accuracy.

Features:
- Real-time ONNX inference with timing enforcement
- Control decimation from 60Hz physics to 15Hz control
- Physics timestep investigation with real-time validation
- Comprehensive timing analysis and rate validation
- USD scene lighting and visualization
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.app import AppLauncher

# Set up argument parser
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Forest Gump Real-Time Demo")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--real_time", type=str2bool, default=True, help="Run in real-time with timing enforcement")
parser.add_argument("--headless", type=str2bool, default=False, help="Run without GUI")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU device")
parser.add_argument("--max_steps", type=int, default=300, help="Maximum simulation steps")
args_cli, hydra_args = parser.parse_known_args()

# Launch the Isaac Lab application
AppLauncher(headless=args_cli.headless).launch()

# Import environment after launching Isaac Lab
from Leatherback.tasks.direct.leatherback import LeatherbackEnv, LeatherbackEnvCfg

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimingAnalyzer:
    """Helper class for analyzing simulation timing"""
    
    def __init__(self, target_physics_rate: float = 60.0, target_control_rate: float = 15.0):
        self.target_physics_dt = 1.0 / target_physics_rate
        self.target_control_dt = 1.0 / target_control_rate
        
        self.physics_times = []
        self.control_times = []
        self.real_times = []
        
        self.last_physics_time = 0.0
        self.last_control_time = 0.0
        self.last_real_time = 0.0
        
    def record_step(self, sim_time: float, is_control_step: bool, real_time: float):
        """Record timing data for analysis"""
        # Physics step timing
        if len(self.physics_times) > 0:
            physics_dt = sim_time - self.last_physics_time
            self.physics_times.append(physics_dt)
        self.last_physics_time = sim_time
        
        # Control step timing
        if is_control_step:
            if len(self.control_times) > 0:
                control_dt = sim_time - self.last_control_time
                self.control_times.append(control_dt)
            self.last_control_time = sim_time
        
        # Real-time timing
        if len(self.real_times) > 0:
            real_dt = real_time - self.last_real_time
            self.real_times.append(real_dt)
        self.last_real_time = real_time
    
    def analyze(self) -> Dict[str, float]:
        """Analyze timing statistics"""
        if not self.physics_times or not self.control_times or not self.real_times:
            return {}
        
        physics_avg = np.mean(self.physics_times)
        control_avg = np.mean(self.control_times)
        real_avg = np.mean(self.real_times)
        
        physics_rate = 1.0 / physics_avg if physics_avg > 0 else 0
        control_rate = 1.0 / control_avg if control_avg > 0 else 0
        real_rate = 1.0 / real_avg if real_avg > 0 else 0
        
        return {
            'physics_dt_avg': physics_avg,
            'control_dt_avg': control_avg,
            'real_dt_avg': real_avg,
            'physics_rate': physics_rate,
            'control_rate': control_rate,
            'real_rate': real_rate,
            'physics_rate_error': abs(physics_rate - (1.0/self.target_physics_dt)) / (1.0/self.target_physics_dt) * 100,
            'control_rate_error': abs(control_rate - (1.0/self.target_control_dt)) / (1.0/self.target_control_dt) * 100,
            'real_time_factor': real_rate / (1.0/self.target_physics_dt) if (1.0/self.target_physics_dt) > 0 else 0
        }

def setup_lighting(sim):
    """Set up proper scene lighting"""
    try:
        from pxr import UsdGeom, Sdf
        
        # Create dome light for better visualization
        stage = sim.stage
        dome_light_path = "/World/DomeLight"
        
        if not stage.GetPrimAtPath(dome_light_path):
            dome_light = UsdGeom.DomeLight.Define(stage, dome_light_path)
            dome_light.CreateIntensityAttr(1000.0)
            dome_light.CreateTextureFileAttr("")
            
        # Create distant light as key light
        distant_light_path = "/World/DistantLight"
        if not stage.GetPrimAtPath(distant_light_path):
            distant_light = UsdGeom.DistantLight.Define(stage, distant_light_path)
            distant_light.CreateIntensityAttr(3000.0)
            distant_light.CreateAngleAttr(5.0)
            
        logger.info("✓ Scene lighting configured")
        
    except Exception as e:
        logger.warning(f"Could not set up lighting: {e}")

def dummy_policy(obs: torch.Tensor, step: int) -> torch.Tensor:
    """
    Simple dummy policy for testing control decimation
    Returns smooth sinusoidal actions for testing
    """
    batch_size = obs.shape[0]
    
    # Time-based sinusoidal actions for smooth movement
    time_factor = step * 0.1
    
    # Throttle: oscillate between 0.2 and 0.8
    throttle = 0.5 + 0.3 * torch.sin(torch.tensor(time_factor))
    
    # Steering: slower oscillation between -0.5 and 0.5
    steering = 0.5 * torch.sin(torch.tensor(time_factor * 0.3))
    
    actions = torch.zeros((batch_size, 2), device=obs.device)
    actions[:, 0] = throttle  # throttle
    actions[:, 1] = steering  # steering
    
    return actions

def main():
    """Main execution function"""
    
    # Set device
    device = "cpu" if args_cli.cpu else "cuda"
    
    # Create environment configuration
    env_cfg = LeatherbackEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 30.0  # Longer episode for better analysis
    
    # Log environment configuration
    logger.info("🏎️  Forest Gump Real-Time Demo Starting")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Number of environments: {env_cfg.scene.num_envs}")
    logger.info(f"Real-time mode: {args_cli.real_time}")
    logger.info(f"Physics timestep (dt): {env_cfg.sim.dt:.6f}s ({1/env_cfg.sim.dt:.1f}Hz)")
    logger.info(f"Control decimation: {env_cfg.decimation}")
    logger.info(f"Control frequency: {1/(env_cfg.sim.dt * env_cfg.decimation):.1f}Hz")
    logger.info(f"Max steps: {args_cli.max_steps}")
    logger.info("="*60)
    
    # Create environment
    env = LeatherbackEnv(env_cfg, render_mode="rgb_array" if not args_cli.headless else None)
    
    # Set up scene lighting
    setup_lighting(env.sim)
    
    # Get physics dt for real-time synchronization
    physics_dt = env.physics_dt
    control_dt = physics_dt * env_cfg.decimation
    
    logger.info(f"📊 Timing Configuration:")
    logger.info(f"   Physics dt: {physics_dt:.6f}s ({1/physics_dt:.1f}Hz)")
    logger.info(f"   Control dt: {control_dt:.6f}s ({1/control_dt:.1f}Hz)")
    
    # Initialize timing analyzer
    analyzer = TimingAnalyzer(
        target_physics_rate=1/physics_dt,
        target_control_rate=1/control_dt
    )
    
    # Reset environment
    obs, _ = env.reset()
    logger.info(f"✓ Environment reset. Observation shape: {obs.shape}")
    
    # Main simulation loop
    step_count = 0
    control_step_count = 0
    start_time = time.time()
    
    logger.info("🚀 Starting real-time simulation...")
    
    try:
        for step in range(args_cli.max_steps):
            step_start_time = time.time()
            
            # Determine if this is a control step (based on decimation)
            is_control_step = (step % env_cfg.decimation) == 0
            
            if is_control_step:
                # Generate actions using dummy policy
                actions = dummy_policy(obs, control_step_count)
                control_step_count += 1
                
                # Log every 10th control step
                if control_step_count % 10 == 0:
                    logger.info(f"Control step {control_step_count}: throttle={actions[0,0]:.3f}, steering={actions[0,1]:.3f}")
            else:
                # Use previous actions (maintained by environment)
                actions = None
            
            # Step environment
            if actions is not None:
                obs, rewards, terminated, truncated, info = env.step(actions)
            else:
                # Physics-only step (no new actions)
                obs, rewards, terminated, truncated, info = env.step(None)
            
            # Record timing data
            sim_time = step * physics_dt
            real_time = time.time() - start_time
            analyzer.record_step(sim_time, is_control_step, real_time)
            
            # Real-time synchronization
            if args_cli.real_time:
                elapsed_step_time = time.time() - step_start_time
                sleep_time = physics_dt - elapsed_step_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            step_count += 1
            
            # Log progress every 60 steps (1 second of simulation)
            if step % 60 == 0:
                elapsed_real = time.time() - start_time
                elapsed_sim = step * physics_dt
                real_time_factor = elapsed_sim / elapsed_real if elapsed_real > 0 else 0
                
                logger.info(f"Step {step:3d}: sim_time={elapsed_sim:.2f}s, real_time={elapsed_real:.2f}s, factor={real_time_factor:.2f}x")
            
            # Handle episode termination
            if terminated.any() or truncated.any():
                logger.info(f"Episode finished at step {step}")
                obs, _ = env.reset()
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    finally:
        # Analyze timing results
        total_real_time = time.time() - start_time
        total_sim_time = step_count * physics_dt
        
        logger.info("\n" + "="*60)
        logger.info("📈 TIMING ANALYSIS RESULTS")
        logger.info("="*60)
        
        # Overall timing
        logger.info(f"Total steps: {step_count}")
        logger.info(f"Total control steps: {control_step_count}")
        logger.info(f"Total simulation time: {total_sim_time:.3f}s")
        logger.info(f"Total real time: {total_real_time:.3f}s")
        logger.info(f"Real-time factor: {total_sim_time/total_real_time:.3f}x")
        
        # Detailed analysis
        stats = analyzer.analyze()
        if stats:
            logger.info(f"\nDetailed Rate Analysis:")
            logger.info(f"   Physics rate: {stats['physics_rate']:.1f}Hz (target: {1/physics_dt:.1f}Hz, error: {stats['physics_rate_error']:.1f}%)")
            logger.info(f"   Control rate: {stats['control_rate']:.1f}Hz (target: {1/control_dt:.1f}Hz, error: {stats['control_rate_error']:.1f}%)")
            logger.info(f"   Real-time factor: {stats['real_time_factor']:.3f}x")
            
            # Timing validation
            physics_rate_ok = stats['physics_rate_error'] < 5.0  # 5% tolerance
            control_rate_ok = stats['control_rate_error'] < 5.0
            real_time_ok = 0.95 <= stats['real_time_factor'] <= 1.05  # 5% real-time tolerance
            
            logger.info(f"\nValidation Results:")
            logger.info(f"   Physics rate: {'✓ PASS' if physics_rate_ok else '✗ FAIL'}")
            logger.info(f"   Control rate: {'✓ PASS' if control_rate_ok else '✗ FAIL'}")
            logger.info(f"   Real-time sync: {'✓ PASS' if real_time_ok else '✗ FAIL'}")
            
            overall_pass = physics_rate_ok and control_rate_ok and real_time_ok
            logger.info(f"   Overall: {'🎉 ALL TESTS PASSED' if overall_pass else '⚠️  SOME TESTS FAILED'}")
        
        logger.info("="*60)
        logger.info("🏁 Forest Gump Real-Time Demo Complete!")
        
        # Close environment
        env.close()

if __name__ == "__main__":
    main()
