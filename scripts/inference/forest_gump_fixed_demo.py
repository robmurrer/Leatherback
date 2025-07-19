"""
🎯 Fixed Forest Gump Demo with Proper Control Frequency

This script implements the CRITICAL FIX for the Forest Gump instability:
- Matches training control frequency (15Hz) using decimation
- Uses dummy observations for testing without CSV dependency
- Focuses on physics timestep and control frequency investigation

🔧 KEY FIX: Control Decimation
Training uses decimation=4 (effective 15Hz control), but inference was running at 60Hz.
This 4x frequency mismatch is the primary cause of instabilities.

🧪 This demo tests the corrected control frequency to validate the fix.
"""

import os
import numpy as np
import torch
import glob
import time
import logging
from datetime import datetime
from isaacsim import SimulationApp

# Get script directory for relative paths
script_dir = os.path.dirname(__file__)

# Enhanced logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"/home/goat/Documents/GitHub/Leatherback/logs/forest_gump/fixed_demo_{timestamp}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print(f"🎯 Starting FIXED Forest Gump Demo...")
print(f"📝 Debug log: {log_file}")

# 🔧 CRITICAL FIX: Match training control frequency
PHYSICS_DT = 1/60         # 60Hz physics (same as training)
RENDERING_DT = 1/50       # 50Hz rendering
CONTROL_DECIMATION = 4    # Match training decimation=4
EFFECTIVE_CONTROL_HZ = 60 / CONTROL_DECIMATION  # 15Hz control updates

logger.info(f"🔧 CONTROL FREQUENCY FIX:")
logger.info(f"   Physics frequency: {1/PHYSICS_DT:.1f}Hz (dt={PHYSICS_DT:.6f}s)")
logger.info(f"   Control decimation: {CONTROL_DECIMATION}")
logger.info(f"   Effective control frequency: {EFFECTIVE_CONTROL_HZ:.1f}Hz")
logger.info(f"   Training had: 60Hz physics / 4 decimation = 15Hz control")
logger.info(f"   Previous error: Running at 60Hz control (4x too fast!)")

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
import onnxruntime as rt

def find_latest_policy():
    """Find the latest policy ONNX file"""
    base_path = os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "leatherback_direct")
    base_path = os.path.abspath(base_path)
    
    # Get all training run directories
    run_dirs = glob.glob(os.path.join(base_path, "2025-*"))
    if not run_dirs:
        raise FileNotFoundError("No training runs found")
    
    # Sort by timestamp and get the latest
    latest_run = sorted(run_dirs)[-1]
    policy_path = os.path.join(latest_run, "exported", "policy.onnx")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    logger.info(f"🎯 Using policy from: {policy_path}")
    return policy_path

# Load the latest policy
policy_path = find_latest_policy()
logger.info(f"🎯 Loading ONNX model from: {policy_path}")
sess = rt.InferenceSession(policy_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Log model details
input_shape = sess.get_inputs()[0].shape
output_shape = sess.get_outputs()[0].shape
logger.info(f"📊 Model: {input_name} {input_shape} -> {output_name} {output_shape}")

# Global variables for simulation
step_counter = 0
control_counter = 0  # Track control updates separately
start_time = None
max_steps = 900  # Run for 15 seconds at 60Hz (to allow multiple control cycles)
dummy_target_position = [5.0, 0.0]  # Simple target 5m ahead

# Control state variables
current_throttle_action = np.zeros(4)  # Hold action for multiple physics steps
current_steering_action = np.zeros(2)
previous_throttle_state = 0.0  # For observations
previous_steering_state = 0.0

def create_dummy_observations():
    """Create realistic dummy observations for testing"""
    
    # Get robot state
    robot_pos, robot_rot = robot_art.get_world_poses()
    robot_pos = robot_pos[0]  # Extract position array
    robot_rot = robot_rot[0]  # Extract quaternion array (x,y,z,w)
    
    # Get velocities
    robot_velocities = robot_art.get_velocities()
    robot_lin_vel_w = robot_velocities[0, :3]  # World linear velocity
    robot_ang_vel_w = robot_velocities[0, 3:6]  # World angular velocity
    
    # Convert to torch for Isaac Lab computations
    robot_pos_torch = torch.tensor(robot_pos, dtype=torch.float32).unsqueeze(0)
    robot_rot_torch = torch.tensor(robot_rot, dtype=torch.float32).unsqueeze(0)  # Isaac Sim format (x,y,z,w)
    robot_lin_vel_w_torch = torch.tensor(robot_lin_vel_w, dtype=torch.float32).unsqueeze(0)
    
    # Convert quaternion format (x,y,z,w) -> (w,x,y,z) for Isaac Lab
    robot_rot_wxyz = torch.cat([robot_rot_torch[:, 3:4], robot_rot_torch[:, :3]], dim=-1)
    
    # Compute heading (yaw angle)
    qw, qx, qy, qz = robot_rot_wxyz[:, 0], robot_rot_wxyz[:, 1], robot_rot_wxyz[:, 2], robot_rot_wxyz[:, 3]
    heading = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    # Transform linear velocity to body frame using quaternion inverse
    def quat_apply_inverse(quat_wxyz, vec):
        qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
        
        # Conjugate quaternion for inverse
        qx_inv, qy_inv, qz_inv, qw_inv = -qx, -qy, -qz, qw
        
        # Apply quaternion rotation
        cross1_x = qy_inv * vz - qz_inv * vy
        cross1_y = qz_inv * vx - qx_inv * vz  
        cross1_z = qx_inv * vy - qy_inv * vx
        
        temp_x = cross1_x + qw_inv * vx
        temp_y = cross1_y + qw_inv * vy
        temp_z = cross1_z + qw_inv * vz
        
        cross2_x = qy_inv * temp_z - qz_inv * temp_y
        cross2_y = qz_inv * temp_x - qx_inv * temp_z
        cross2_z = qx_inv * temp_y - qy_inv * temp_x
        
        result_x = vx + 2 * cross2_x
        result_y = vy + 2 * cross2_y
        result_z = vz + 2 * cross2_z
        
        return torch.stack([result_x, result_y, result_z], dim=-1)
    
    robot_lin_vel_b = quat_apply_inverse(robot_rot_wxyz, robot_lin_vel_w_torch)
    
    # Compute observations exactly like training
    # Position error (distance to target)
    position_error = torch.norm(torch.tensor([dummy_target_position[0] - robot_pos[0], 
                                            dummy_target_position[1] - robot_pos[1]], dtype=torch.float32))
    
    # Heading error (direction to target)
    target_heading = torch.atan2(dummy_target_position[1] - robot_pos[1], 
                                dummy_target_position[0] - robot_pos[0])
    heading_error = torch.atan2(torch.sin(target_heading - heading[0]), 
                               torch.cos(target_heading - heading[0]))
    
    # Build observation vector exactly like training
    obs = torch.tensor([
        position_error.item(),                    # obs_0: position error
        torch.cos(heading_error).item(),         # obs_1: cos(heading error)
        torch.sin(heading_error).item(),         # obs_2: sin(heading error)
        robot_lin_vel_b[0, 0].item(),           # obs_3: linear vel x (body)
        robot_lin_vel_b[0, 1].item(),           # obs_4: linear vel y (body)
        robot_ang_vel_w[2],                      # obs_5: angular vel z (world)
        previous_throttle_state,                 # obs_6: throttle state (scaled)
        previous_steering_state,                 # obs_7: steering state (scaled)
    ], dtype=np.float32)
    
    return obs.reshape(1, -1)  # Shape: (1, 8)

def on_physics_step(step_size):
    """Physics step callback with FIXED control frequency"""
    global step_counter, control_counter, start_time
    global current_throttle_action, current_steering_action
    global previous_throttle_state, previous_steering_state
    
    step_counter += 1
    
    # Initialize start time
    if start_time is None:
        start_time = time.time()
        logger.info(f"🚀 Physics simulation started with CONTROL DECIMATION FIX")
    
    # Stop after max_steps
    if step_counter > max_steps:
        elapsed_time = time.time() - start_time
        logger.info(f"✅ FIXED Demo completed after {step_counter} steps in {elapsed_time:.2f}s")
        logger.info(f"📊 Physics rate: {step_counter/elapsed_time:.1f} Hz (target: {1/PHYSICS_DT:.1f} Hz)")
        logger.info(f"📊 Control updates: {control_counter} (target: {control_counter/(elapsed_time*EFFECTIVE_CONTROL_HZ)*EFFECTIVE_CONTROL_HZ:.1f} Hz)")
        logger.info(f"📊 Control decimation working: {step_counter/control_counter:.1f}x (target: {CONTROL_DECIMATION}x)")
        simulation_app.close()
        return
    
    # 🔧 CRITICAL FIX: Only update control every CONTROL_DECIMATION steps
    if step_counter % CONTROL_DECIMATION == 0:
        control_counter += 1
        
        # Create dummy observations
        obs = create_dummy_observations()
        
        # Run ONNX inference ONLY at control frequency
        actions = sess.run([output_name], {input_name: obs})[0]
        
        # Extract and bound actions
        throttle_raw = np.clip(actions[0, 0], -5.0, 5.0)
        steering_raw = np.clip(actions[0, 1], -7.5, 7.5)
        
        # Scale actions (same as training environment)
        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 0.75
        
        # Process throttle (applied to 4 wheel joints)
        throttle_action = np.repeat(throttle_raw, 4) * throttle_scale
        current_throttle_action = np.clip(throttle_action, -throttle_max, throttle_max)
        
        # Process steering (applied to 2 steering joints)
        steering_action = np.repeat(steering_raw, 2) * steering_scale
        current_steering_action = np.clip(steering_action, -steering_max, steering_max)
        
        # Update previous actions for next observation (use scaled values like training)
        previous_throttle_state = current_throttle_action[0]
        previous_steering_state = current_steering_action[0]
        
        # Log control updates
        if control_counter <= 5 or control_counter % 10 == 0:
            logger.info(f"🎮 CONTROL UPDATE #{control_counter} (Physics Step {step_counter}):")
            logger.info(f"   ONNX raw: throttle={throttle_raw:.6f}, steering={steering_raw:.6f}")
            logger.info(f"   Scaled: throttle={current_throttle_action[0]:.3f}, steering={current_steering_action[0]:.3f}")
            logger.info(f"   Observations: pos_err={obs[0,0]:.3f}, cos={obs[0,1]:.3f}, sin={obs[0,2]:.3f}")
            
            # Check for extreme actions (stability indicators)
            if abs(throttle_raw) > 3.0:
                logger.warning(f"⚠️ HIGH THROTTLE: {abs(throttle_raw):.3f}")
            if abs(steering_raw) > 3.0:
                logger.warning(f"⚠️ HIGH STEERING: {abs(steering_raw):.3f}")
    
    # Apply CURRENT actions to robot at every physics step
    # (Actions are held constant between control updates)
    robot_art.set_joint_velocities(current_throttle_action, joint_indices=throttle_dof_idx)
    robot_art.set_joint_positions(current_steering_action, joint_indices=steering_dof_idx)
    
    # 🔬 TIMING ANALYSIS - Log every 60 steps (1 second at 60Hz)
    if step_counter % 60 == 0:
        current_time = time.time()
        elapsed = current_time - start_time
        expected_time = step_counter * PHYSICS_DT
        time_drift = elapsed - expected_time
        
        expected_control_updates = elapsed * EFFECTIVE_CONTROL_HZ
        control_frequency_error = control_counter - expected_control_updates
        
        logger.info(f"🕒 TIMING ANALYSIS (Step {step_counter}, t={elapsed:.1f}s):")
        logger.info(f"   Physics timing:")
        logger.info(f"     Actual rate: {step_counter/elapsed:.1f} Hz (target: {1/PHYSICS_DT:.1f} Hz)")
        logger.info(f"     Time drift: {time_drift:.3f}s ({time_drift/expected_time*100:.1f}%)")
        logger.info(f"   Control timing:")
        logger.info(f"     Control updates: {control_counter} (expected: {expected_control_updates:.1f})")
        logger.info(f"     Control rate: {control_counter/elapsed:.1f} Hz (target: {EFFECTIVE_CONTROL_HZ:.1f} Hz)")
        logger.info(f"     Decimation ratio: {step_counter/control_counter:.1f}x (target: {CONTROL_DECIMATION}x)")
        
        # Check for timing issues
        if abs(time_drift) > 0.1:
            logger.warning(f"⚠️ PHYSICS TIMING DRIFT: {time_drift:.3f}s")
        if abs(control_frequency_error) > 2.0:
            logger.warning(f"⚠️ CONTROL FREQUENCY ERROR: {control_frequency_error:.1f} updates")

# Setup world with FIXED physics settings
logger.info(f"🌍 Creating world with FIXED settings:")
logger.info(f"   physics_dt={PHYSICS_DT:.6f}s ({1/PHYSICS_DT:.1f}Hz)")
logger.info(f"   rendering_dt={RENDERING_DT:.6f}s ({1/RENDERING_DT:.1f}Hz)")
my_world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

# Add ground plane
assets_root_path = get_assets_root_path()
ground_prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim.GetReferences().AddReference(asset_path)

# Add basic lighting
from pxr import Sdf
dome_light = define_prim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(2000.0)

# Add Leatherback robot
logger.info("🤖 Loading Leatherback robot...")
robot_usd_path = os.path.join(script_dir, "..", "..", "source", "Leatherback", "Leatherback", 
                             "tasks", "direct", "leatherback", "custom_assets", "leatherback_simple_better.usd")
robot_usd_path = os.path.abspath(robot_usd_path)

lb_prim_path = "/World/Leatherback"
add_reference_to_stage(robot_usd_path, lb_prim_path)
robot_art = Articulation(prim_paths_expr=lb_prim_path, name="Leatherback")

# Initialize world
my_world.reset()

# Setup joint indices
logger.info("🔧 Setting up joint mappings...")
throttle_joint_names = [
    'Wheel__Upright__Rear_Right',
    'Wheel__Upright__Rear_Left', 
    'Wheel__Knuckle__Front_Right',
    'Wheel__Knuckle__Front_Left'
]

steering_joint_names = [
    'Knuckle__Upright__Front_Right',
    'Knuckle__Upright__Front_Left'
]

throttle_dof_idx = []
steering_dof_idx = []

for joint_name in throttle_joint_names:
    if joint_name in robot_art.joint_names:
        throttle_dof_idx.append(robot_art.joint_names.index(joint_name))

for joint_name in steering_joint_names:
    if joint_name in robot_art.joint_names:
        steering_dof_idx.append(robot_art.joint_names.index(joint_name))

logger.info(f"Throttle DOF indices: {throttle_dof_idx}")
logger.info(f"Steering DOF indices: {steering_dof_idx}")

# Position robot at start
start_position = np.array([[0.0, 0.0, 0.05]])
robot_art.set_world_poses(positions=start_position)

# Initialize action arrays
current_throttle_action = np.zeros(4)
current_steering_action = np.zeros(2)

# Add physics callback
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# 🔧 FINAL FIX SUMMARY
logger.info("🔧 CONTROL FREQUENCY FIX SUMMARY:")
logger.info(f"   🚨 PROBLEM: Training used 15Hz control, inference used 60Hz (4x mismatch)")
logger.info(f"   ✅ SOLUTION: Implement decimation={CONTROL_DECIMATION} to match training")
logger.info(f"   🎯 RESULT: {EFFECTIVE_CONTROL_HZ:.1f}Hz control updates, {1/PHYSICS_DT:.1f}Hz physics")
logger.info(f"   📊 EXPECTATION: Stable control, no oscillations, proper action magnitudes")

# Start simulation
logger.info("🚀 Starting FIXED demonstration...")
logger.info(f"Target: Drive toward [{dummy_target_position[0]:.1f}, {dummy_target_position[1]:.1f}]")

try:
    while simulation_app.is_running():
        my_world.step(render=True)
        
except KeyboardInterrupt:
    logger.info("🛑 Demo stopped by user")

logger.info("✅ FIXED demo completed!")
simulation_app.close()
