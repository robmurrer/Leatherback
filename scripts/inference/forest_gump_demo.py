"""
🚀 Leatherback ONNX Policy Demo

This script demonstrates a trained Leatherback policy using ONNX inference.
It spawns a fresh environment with waypoints and runs the policy in real-time.

✨ Features:
- Auto-detects latest ONNX policy file
- Fresh environment with randomly generated waypoints
- Real-time ONNX inference
- Clean, simple demonstration script

🎯 Perfect for showcasing trained policies!
"""

import os
import numpy as np
import glob
import time
import logging
from datetime import datetime
from isaacsim import SimulationApp

# Get script directory for relative paths
script_dir = os.path.dirname(__file__)

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Starting Leatherback ONNX Policy Demo...")

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
sess = rt.InferenceSession(policy_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Global variables for simulation
step_counter = 0
target_positions = []
current_target_idx = 0
position_tolerance = 0.15

def generate_waypoints():
    """Generate a simple set of waypoints for demonstration"""
    waypoints = []
    # Create a simple rectangular path
    base_x, base_y = 0.0, 5.0
    
    # Rectangle waypoints
    waypoints.extend([
        [base_x + 2.0, base_y + 0.0],  # Forward
        [base_x + 2.0, base_y + 2.0],  # Right
        [base_x - 2.0, base_y + 2.0],  # Back
        [base_x - 2.0, base_y - 2.0],  # Left
        [base_x + 0.0, base_y - 2.0],  # Forward to center
        [base_x + 0.0, base_y + 0.0],  # Return to start
    ])
    
    logger.info(f"🗺️ Generated {len(waypoints)} waypoints")
    return waypoints

def compute_observations(robot_pos, robot_heading, robot_lin_vel, robot_ang_vel):
    """Compute observations like the training environment"""
    global current_target_idx, target_positions
    
    # Get current target position
    if current_target_idx >= len(target_positions):
        current_target_idx = 0  # Loop back to start
    
    target_pos = target_positions[current_target_idx]
    
    # Position error (distance to target)
    position_error_vec = np.array([target_pos[0] - robot_pos[0], target_pos[1] - robot_pos[1]])
    position_error = np.linalg.norm(position_error_vec)
    
    # Target heading (direction to target)
    target_heading = np.arctan2(position_error_vec[1], position_error_vec[0])
    heading_error = np.arctan2(np.sin(target_heading - robot_heading), np.cos(target_heading - robot_heading))
    
    # Check if we reached the current target
    if position_error < position_tolerance:
        current_target_idx += 1
        logger.info(f"🎯 Reached waypoint {current_target_idx-1}! Moving to next target.")
        if current_target_idx >= len(target_positions):
            logger.info("🏁 Reached all waypoints! Looping back to start.")
            current_target_idx = 0
    
    # Build observation vector (same format as training)
    obs = np.array([
        position_error,                    # obs_0: position error
        np.cos(heading_error),            # obs_1: heading cos
        np.sin(heading_error),            # obs_2: heading sin  
        robot_lin_vel[0],                 # obs_3: linear vel x
        robot_lin_vel[1],                 # obs_4: linear vel y
        robot_ang_vel[2],                 # obs_5: angular vel z
        0.0,                             # obs_6: throttle state (previous action)
        0.0,                             # obs_7: steering state (previous action)
    ], dtype=np.float32)
    
    return obs.reshape(1, -1)  # Shape: (1, 8)

def on_physics_step(step_size):
    """Physics step callback for ONNX inference"""
    global step_counter
    step_counter += 1
    
    # Get robot state
    robot_pos, robot_rot = robot_art.get_world_poses()
    robot_pos = robot_pos[0].numpy()  # Get first (and only) robot position
    robot_lin_vel = robot_art.get_linear_velocities()[0].numpy()
    robot_ang_vel = robot_art.get_angular_velocities()[0].numpy()
    
    # Convert quaternion to heading (simplified)
    robot_rot_np = robot_rot[0].numpy()
    robot_heading = np.arctan2(2.0 * (robot_rot_np[3] * robot_rot_np[2] + robot_rot_np[0] * robot_rot_np[1]), 
                              1.0 - 2.0 * (robot_rot_np[1]**2 + robot_rot_np[2]**2))
    
    # Compute observations
    obs = compute_observations(robot_pos, robot_heading, robot_lin_vel, robot_ang_vel)
    
    # Run ONNX inference
    actions = sess.run([output_name], {input_name: obs})[0]
    
    # Process actions (same scaling as training environment)
    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75
    
    # Process throttle (applied to 4 wheel joints)
    throttle_action = np.repeat(actions[0, 0], 4) * throttle_scale
    throttle_action = np.clip(throttle_action, -throttle_max, throttle_max)
    
    # Process steering (applied to 2 steering joints)
    steering_action = np.repeat(actions[0, 1], 2) * steering_scale
    steering_action = np.clip(steering_action, -steering_max, steering_max)
    
    # Apply actions to robot
    robot_art.set_joint_velocities(throttle_action, joint_indices=throttle_dof_idx)
    robot_art.set_joint_positions(steering_action, joint_indices=steering_dof_idx)
    
    # Log progress every 60 steps (1 second at 60 FPS)
    if step_counter % 60 == 0:
        target_pos = target_positions[current_target_idx]
        distance_to_target = np.linalg.norm([target_pos[0] - robot_pos[0], target_pos[1] - robot_pos[1]])
        logger.info(f"Step {step_counter}: Target {current_target_idx+1}/{len(target_positions)}, Distance: {distance_to_target:.2f}m")

# Setup world and environment
logger.info("🌍 Setting up simulation world...")
my_world = World(stage_units_in_meters=1.0, physics_dt=1/60, rendering_dt=1/50)

# Add ground plane
assets_root_path = get_assets_root_path()
ground_prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim.GetReferences().AddReference(asset_path)

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

# Setup joint indices (same as training environment)
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

# Generate waypoints
target_positions = generate_waypoints()

# Add physics callback
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# Start simulation
logger.info("🚀 Starting demonstration...")
logger.info(f"Target: Waypoint {current_target_idx+1}/{len(target_positions)}")

try:
    while simulation_app.is_running():
        my_world.step(render=True)
        
except KeyboardInterrupt:
    logger.info("🛑 Demo stopped by user")

logger.info("✅ Demo completed!")
simulation_app.close()
