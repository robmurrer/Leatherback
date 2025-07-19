#!/usr/bin/env python3
"""
🚀 Forest Gump Clean Demo

Single, focused demo script that:
- Loads the latest ONNX policy
- Matches training timestep with control decimation (15Hz effective control)
- Navigates waypoints in a rectangular pattern
- Uses proper Isaac Lab math and scaling

This is the definitive Forest Gump demo incorporating all our learnings.
"""

import os
import numpy as np
import torch
import glob
import time
import logging
from datetime import datetime

# Get script directory for relative paths
script_dir = os.path.dirname(__file__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Starting Forest Gump Clean Demo...")

# Training-matched parameters
PHYSICS_DT = 1/60      # 60Hz physics (same as training)
CONTROL_DECIMATION = 4  # Apply action every 4th physics step (15Hz effective control)
RENDERING_DT = 1/50    # 50Hz rendering

# Import Isaac Sim after logging setup
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "physics_dt": PHYSICS_DT,
    "rendering_dt": RENDERING_DT
})

from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
import onnxruntime as rt

# Isaac Lab math utilities
def convert_quat_xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from Isaac Sim (x,y,z,w) to Isaac Lab (w,x,y,z) format."""
    return torch.cat([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], dim=-1)

def quat_apply_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation to vector (transform world to body frame)."""
    qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    
    # Vector components
    vx, vy, vz = vec[:, 0], vec[:, 1], vec[:, 2]
    
    # Apply quaternion inverse rotation formula
    qx_inv, qy_inv, qz_inv = -qx, -qy, -qz  # Conjugate for inverse
    
    # Cross product: q_xyz × v
    cross1_x = qy_inv * vz - qz_inv * vy
    cross1_y = qz_inv * vx - qx_inv * vz  
    cross1_z = qx_inv * vy - qy_inv * vx
    
    # q_xyz × v + q_w * v
    temp_x = cross1_x + qw * vx
    temp_y = cross1_y + qw * vy
    temp_z = cross1_z + qw * vz
    
    # Cross product: q_xyz × temp
    cross2_x = qy_inv * temp_z - qz_inv * temp_y
    cross2_y = qz_inv * temp_x - qx_inv * temp_z
    cross2_z = qx_inv * temp_y - qy_inv * temp_x
    
    # Final result: v + 2 * cross2
    result_x = vx + 2 * cross2_x
    result_y = vy + 2 * cross2_y
    result_z = vz + 2 * cross2_z
    
    return torch.stack([result_x, result_y, result_z], dim=-1)

def compute_heading_w(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Compute yaw heading from quaternion (Isaac Lab implementation)."""
    qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    heading = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return heading

def find_latest_policy():
    """Find the latest policy ONNX file"""
    base_path = os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "leatherback_direct")
    base_path = os.path.abspath(base_path)
    
    run_dirs = glob.glob(os.path.join(base_path, "2025-*"))
    if not run_dirs:
        raise FileNotFoundError("No training runs found")
    
    latest_run = sorted(run_dirs)[-1]
    policy_path = os.path.join(latest_run, "exported", "policy.onnx")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    logger.info(f"🎯 Using policy from: {policy_path}")
    return policy_path

# Load ONNX model
policy_path = find_latest_policy()
sess = rt.InferenceSession(policy_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Global simulation state
step_counter = 0
control_counter = 0
current_target_idx = 0
position_tolerance = 0.15
previous_throttle_action = 0.0
previous_steering_action = 0.0
start_time = None

# Waypoints in a rectangular pattern
waypoints = [
    [2.0, 5.0],   # Forward
    [2.0, 7.0],   # Right  
    [-2.0, 7.0],  # Back
    [-2.0, 3.0],  # Left
    [0.0, 3.0],   # Center
    [0.0, 5.0],   # Start
]

def create_observations():
    """Create observations exactly like training environment"""
    global current_target_idx, previous_throttle_action, previous_steering_action
    
    # Get robot state
    robot_pos, robot_quat = robot_art.get_world_poses()
    robot_lin_vel_w = robot_art.get_linear_velocities()
    robot_ang_vel_w = robot_art.get_angular_velocities()
    
    # Convert to tensors (shape: [1, 3])
    robot_pos = robot_pos.clone().detach()  # [1, 3]
    robot_quat_xyzw = robot_quat.clone().detach()  # [1, 4] in xyzw format
    robot_lin_vel_w = robot_lin_vel_w.clone().detach()  # [1, 3]
    robot_ang_vel_w = robot_ang_vel_w.clone().detach()  # [1, 3]
    
    # Convert quaternion to Isaac Lab format
    robot_quat_wxyz = convert_quat_xyzw_to_wxyz(robot_quat_xyzw)
    
    # Compute heading
    heading = compute_heading_w(robot_quat_wxyz)
    
    # Transform velocity to body frame
    robot_lin_vel_b = quat_apply_inverse(robot_quat_wxyz, robot_lin_vel_w)
    
    # Get current target
    if current_target_idx >= len(waypoints):
        current_target_idx = 0
    
    target_pos = waypoints[current_target_idx]
    target_position = torch.tensor([[target_pos[0], target_pos[1]]], dtype=torch.float32)
    
    # Position error
    position_error_vec = target_position - robot_pos[:, :2]
    position_error = torch.norm(position_error_vec, dim=1)
    
    # Check if reached target
    if position_error.item() < position_tolerance:
        current_target_idx += 1
        logger.info(f"🎯 Reached waypoint! Moving to target {current_target_idx % len(waypoints) + 1}")
        if current_target_idx >= len(waypoints):
            current_target_idx = 0
            logger.info("🔄 Completed loop! Starting again.")
    
    # Target heading and heading error
    target_heading = torch.atan2(
        torch.tensor(target_pos[1] - robot_pos[0, 1], dtype=torch.float32), 
        torch.tensor(target_pos[0] - robot_pos[0, 0], dtype=torch.float32)
    )
    heading_error = torch.atan2(torch.sin(target_heading - heading[0]), 
                               torch.cos(target_heading - heading[0]))
    
    # Build observation vector (same format as training)
    obs = torch.tensor([
        position_error.item(),
        torch.cos(heading_error).item(),
        torch.sin(heading_error).item(),
        robot_lin_vel_b[0, 0].item(),
        robot_lin_vel_b[0, 1].item(),
        robot_ang_vel_w[0, 2].item(),
        previous_throttle_action,
        previous_steering_action,
    ], dtype=torch.float32)
    
    return obs.reshape(1, -1)

def on_physics_step(step_size):
    """Physics step callback with proper control decimation"""
    global step_counter, control_counter, start_time
    global previous_throttle_action, previous_steering_action
    
    step_counter += 1
    
    if start_time is None:
        start_time = time.time()
        logger.info("🚀 Starting navigation...")
        logger.info(f"🎯 Target 1/{len(waypoints)}: [{waypoints[0][0]:.1f}, {waypoints[0][1]:.1f}]")
    
    # Control decimation: only run ONNX inference every 4th step (15Hz effective)
    if step_counter % CONTROL_DECIMATION == 0:
        control_counter += 1
        
        # Create observations
        obs = create_observations()
        obs_numpy = obs.detach().cpu().numpy()
        
        # Run ONNX inference
        actions = sess.run([output_name], {input_name: obs_numpy})[0]
        
        # Extract and clip raw actions
        throttle_raw = np.clip(actions[0, 0], -5.0, 5.0)
        steering_raw = np.clip(actions[0, 1], -7.5, 7.5)
        
        # Scale actions (same as training)
        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 0.75
        
        # Process throttle for 4 wheel joints
        throttle_action = np.repeat(throttle_raw, 4) * throttle_scale
        throttle_action = np.clip(throttle_action, -throttle_max, throttle_max)
        
        # Process steering for 2 steering joints
        steering_action = np.repeat(steering_raw, 2) * steering_scale
        steering_action = np.clip(steering_action, -steering_max, steering_max)
        
        # Apply actions (with safety checks)
        if len(throttle_dof_idx) > 0:
            robot_art.set_joint_velocities(throttle_action, joint_indices=throttle_dof_idx)
        else:
            logger.warning("⚠️ Skipping throttle control - no throttle joints found")
            
        if len(steering_dof_idx) > 0:
            robot_art.set_joint_positions(steering_action, joint_indices=steering_dof_idx)
        else:
            logger.warning("⚠️ Skipping steering control - no steering joints found")
        
        # Store for next observation
        previous_throttle_action = throttle_raw
        previous_steering_action = steering_raw
        
        # Log progress every 4 seconds (60 control steps)
        if control_counter % 60 == 0:
            elapsed = time.time() - start_time
            target = waypoints[current_target_idx]
            robot_pos = robot_art.get_world_poses()[0][0].cpu().numpy()
            distance = np.linalg.norm([target[0] - robot_pos[0], target[1] - robot_pos[1]])
            logger.info(f"Control step {control_counter} ({elapsed:.1f}s): Target {current_target_idx+1}, Distance: {distance:.2f}m")

# Setup world
logger.info("🌍 Setting up world...")
world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)

# Add ground
ground_prim = define_prim("/World/GroundPlane", "Xform")
ground_prim.GetReferences().AddReference("http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Environments/Grid/default_environment.usd")

# Add robot
logger.info("🤖 Loading Leatherback...")
robot_usd_path = os.path.join(script_dir, "..", "..", "source", "Leatherback", "Leatherback", 
                             "tasks", "direct", "leatherback", "custom_assets", "leatherback_simple_better.usd")
robot_usd_path = os.path.abspath(robot_usd_path)

if not os.path.exists(robot_usd_path):
    raise FileNotFoundError(f"Robot USD not found: {robot_usd_path}")

robot_prim_path = "/World/Leatherback"
add_reference_to_stage(robot_usd_path, robot_prim_path)
robot_art = Articulation(prim_path=robot_prim_path, name="Leatherback")

# Initialize world
world.reset()

# Setup joint indices (same as training)
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

# Get joint names (with error checking)
all_joint_names = robot_art.dof_names
if all_joint_names is None:
    logger.error("❌ Robot joint names not available. Robot may not be properly initialized.")
    # Try alternative methods
    all_joint_names = getattr(robot_art, 'joint_names', None)
    if all_joint_names is None:
        logger.error("❌ Could not get joint names from robot. Using fallback indices.")
        # Fallback to hardcoded indices based on typical robot setup
        throttle_dof_idx = [0, 1, 2, 3]  # First 4 joints are typically wheels
        steering_dof_idx = [4, 5]        # Next 2 joints are typically steering
    else:
        logger.info(f"✅ Found joint names via alternative method: {all_joint_names}")
else:
    logger.info(f"✅ Available joints: {all_joint_names}")

if all_joint_names is not None:
    throttle_dof_idx = []
    steering_dof_idx = []

    for joint_name in throttle_joint_names:
        if joint_name in all_joint_names:
            throttle_dof_idx.append(all_joint_names.index(joint_name))
        else:
            logger.warning(f"⚠️ Throttle joint '{joint_name}' not found in robot")

    for joint_name in steering_joint_names:
        if joint_name in all_joint_names:
            steering_dof_idx.append(all_joint_names.index(joint_name))
        else:
            logger.warning(f"⚠️ Steering joint '{joint_name}' not found in robot")

logger.info(f"🔧 Throttle DOF indices: {throttle_dof_idx}")
logger.info(f"🔧 Steering DOF indices: {steering_dof_idx}")

# Validate that we have the required joints
if len(throttle_dof_idx) == 0:
    logger.error("❌ No throttle joints found! Robot control will not work.")
if len(steering_dof_idx) == 0:
    logger.error("❌ No steering joints found! Robot control will not work.")

if len(throttle_dof_idx) > 0 and len(steering_dof_idx) > 0:
    logger.info("✅ Joint mapping successful - robot control ready!")
else:
    logger.warning("⚠️ Joint mapping incomplete - robot may not respond properly")

# Position robot at origin
start_position = torch.tensor([[0.0, 5.0, 0.05]], dtype=torch.float32)
robot_art.set_world_poses(positions=start_position)

# Add physics callback
world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# Run simulation
logger.info("🚀 Starting demonstration...")
logger.info(f"Physics: {1/PHYSICS_DT:.0f}Hz, Control: {60/CONTROL_DECIMATION:.0f}Hz (decimation={CONTROL_DECIMATION})")

try:
    while simulation_app.is_running():
        world.step(render=True)
        
except KeyboardInterrupt:
    logger.info("🛑 Demo stopped by user")

elapsed_total = time.time() - start_time if start_time else 0
logger.info(f"✅ Demo completed! Total time: {elapsed_total:.1f}s, Control steps: {control_counter}")
simulation_app.close()
