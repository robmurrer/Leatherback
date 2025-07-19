"""
🧪 Minimal Forest Gump Demo with Static Dummy Observations

This is the simplest possible test to validate the control frequency fix
without requiring a trained ONNX model. Uses predefined dummy observations
and actions to test the decimation timing fix.

🎯 Purpose: Validate that control decimation timing works correctly
📊 Expected: Stable 15Hz control updates, 60Hz physics steps
"""

import os
import numpy as np
import time
import logging
from datetime import datetime
from isaacsim import SimulationApp

# Enhanced logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(__file__)
log_file = f"/home/goat/Documents/GitHub/Leatherback/logs/forest_gump/minimal_demo_{timestamp}.log"
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

print(f"🧪 Starting Minimal Forest Gump Demo...")
print(f"📝 Debug log: {log_file}")

# 🔧 CONTROL FREQUENCY FIX SETTINGS
PHYSICS_DT = 1/60         # 60Hz physics
RENDERING_DT = 1/50       # 50Hz rendering  
CONTROL_DECIMATION = 4    # Match training decimation
EFFECTIVE_CONTROL_HZ = 60 / CONTROL_DECIMATION  # 15Hz control

logger.info(f"🔧 TESTING CONTROL FREQUENCY FIX:")
logger.info(f"   Physics: {1/PHYSICS_DT:.1f}Hz, Control: {EFFECTIVE_CONTROL_HZ:.1f}Hz")
logger.info(f"   Decimation: {CONTROL_DECIMATION}x (only update control every {CONTROL_DECIMATION} physics steps)")

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation

# Global variables
step_counter = 0
control_counter = 0
start_time = None
max_steps = 600  # 10 seconds at 60Hz

# Static dummy actions for testing (realistic values from training data)
DUMMY_ACTIONS = [
    [0.5, 0.1],    # Forward, slight right
    [0.3, -0.2],   # Slower, left turn  
    [0.8, 0.0],    # Faster, straight
    [0.2, 0.3],    # Slow, right turn
    [0.6, -0.1],   # Medium, slight left
]
current_action_idx = 0

# Current action state (held between control updates)
current_throttle_action = np.zeros(4)
current_steering_action = np.zeros(2)

def get_dummy_action():
    """Get next dummy action from predefined sequence"""
    global current_action_idx
    action = DUMMY_ACTIONS[current_action_idx % len(DUMMY_ACTIONS)]
    current_action_idx += 1
    return action

def on_physics_step(step_size):
    """Physics step with FIXED control decimation"""
    global step_counter, control_counter, start_time
    global current_throttle_action, current_steering_action
    
    step_counter += 1
    
    # Initialize start time
    if start_time is None:
        start_time = time.time()
        logger.info(f"🚀 Minimal demo started - testing control decimation")
    
    # Stop after max_steps
    if step_counter > max_steps:
        elapsed_time = time.time() - start_time
        expected_controls = elapsed_time * EFFECTIVE_CONTROL_HZ
        
        logger.info(f"✅ Minimal demo completed:")
        logger.info(f"   Physics steps: {step_counter} in {elapsed_time:.2f}s")
        logger.info(f"   Physics rate: {step_counter/elapsed_time:.1f}Hz (target: {1/PHYSICS_DT:.1f}Hz)")
        logger.info(f"   Control updates: {control_counter} (expected: {expected_controls:.1f})")
        logger.info(f"   Control rate: {control_counter/elapsed_time:.1f}Hz (target: {EFFECTIVE_CONTROL_HZ:.1f}Hz)")
        logger.info(f"   Decimation ratio: {step_counter/control_counter:.1f}x (target: {CONTROL_DECIMATION}x)")
        
        # Validate timing
        physics_error = abs((step_counter/elapsed_time) - (1/PHYSICS_DT))
        control_error = abs((control_counter/elapsed_time) - EFFECTIVE_CONTROL_HZ)
        decimation_error = abs((step_counter/control_counter) - CONTROL_DECIMATION)
        
        logger.info(f"📊 Timing validation:")
        logger.info(f"   Physics rate error: {physics_error:.2f}Hz")
        logger.info(f"   Control rate error: {control_error:.2f}Hz") 
        logger.info(f"   Decimation error: {decimation_error:.2f}x")
        
        if physics_error < 5.0 and control_error < 2.0 and decimation_error < 0.2:
            logger.info("✅ TIMING VALIDATION PASSED - Control decimation working correctly!")
        else:
            logger.warning("⚠️ TIMING VALIDATION ISSUES DETECTED")
            
        simulation_app.close()
        return
    
    # 🔧 CRITICAL FIX: Only update control every CONTROL_DECIMATION steps
    if step_counter % CONTROL_DECIMATION == 0:
        control_counter += 1
        
        # Get dummy action instead of ONNX inference
        dummy_action = get_dummy_action()
        throttle_raw, steering_raw = dummy_action[0], dummy_action[1]
        
        # Scale actions exactly like training environment
        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 0.75
        
        # Process actions
        throttle_action = np.repeat(throttle_raw, 4) * throttle_scale
        current_throttle_action = np.clip(throttle_action, -throttle_max, throttle_max)
        
        steering_action = np.repeat(steering_raw, 2) * steering_scale
        current_steering_action = np.clip(steering_action, -steering_max, steering_max)
        
        # Log control updates
        if control_counter <= 10 or control_counter % 10 == 0:
            logger.info(f"🎮 CONTROL UPDATE #{control_counter} (Step {step_counter}):")
            logger.info(f"   Raw action: [{throttle_raw:.3f}, {steering_raw:.3f}]")
            logger.info(f"   Scaled: throttle={current_throttle_action[0]:.2f}, steering={current_steering_action[0]:.3f}")
    
    # Apply current actions at every physics step (held between control updates)
    robot_art.set_joint_velocities(current_throttle_action, joint_indices=throttle_dof_idx)
    robot_art.set_joint_positions(current_steering_action, joint_indices=steering_dof_idx)
    
    # Log timing every 60 steps (1 second)
    if step_counter % 60 == 0:
        elapsed = time.time() - start_time
        logger.info(f"🕒 t={elapsed:.1f}s: {step_counter} physics steps, {control_counter} control updates")

# Setup minimal world
logger.info("🌍 Setting up minimal world...")
my_world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

# Add ground plane
assets_root_path = get_assets_root_path()
ground_prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim.GetReferences().AddReference(asset_path)

# Add basic lighting
from pxr import Sdf
dome_light = define_prim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(1500.0)

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

logger.info(f"Joint indices: throttle={throttle_dof_idx}, steering={steering_dof_idx}")

# Position robot
start_position = np.array([[0.0, 0.0, 0.05]])
robot_art.set_world_poses(positions=start_position)

# Add physics callback
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

logger.info("🚀 Starting minimal demonstration with dummy actions...")
logger.info(f"🎯 Testing: Control decimation timing fix")
logger.info(f"📊 Expected: 60Hz physics, 15Hz control, 4x decimation ratio")

try:
    while simulation_app.is_running():
        my_world.step(render=True)
        
except KeyboardInterrupt:
    logger.info("🛑 Demo stopped by user")

logger.info("✅ Minimal demo completed!")
simulation_app.close()
