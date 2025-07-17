import os
import numpy as np
import pandas as pd
import glob
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
import carb
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from leatherback.policy.example.leatherback import LeatherbackPolicy
from isaacsim.storage.native import get_assets_root_path

script_dir = os.path.dirname(__file__)

def find_latest_policy():
    """Find the latest policy ONNX file"""
    # Use relative path from script directory
    base_path = os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "leatherback_direct")
    base_path = os.path.abspath(base_path)
    
    # Get all training run directories
    run_dirs = glob.glob(os.path.join(base_path, "2025-*"))
    if not run_dirs:
        raise FileNotFoundError("No training runs found")
    
    # Sort by timestamp (directory name) and get the latest
    latest_run = sorted(run_dirs)[-1]
    policy_path = os.path.join(latest_run, "exported", "policy.onnx")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    print(f"Using policy from: {policy_path}")
    return policy_path

def find_latest_observations():
    """Find and load the latest observation data"""
    # Use relative paths from script directory
    obs_paths = [
        os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "leatherback_direct", "*", "csv_logs", "observations_actions_*.csv"),
        os.path.join(script_dir, "..", "..", "source", "Leatherback", "logs", "observations", "observations_*.csv")
    ]
    
    all_obs_files = []
    for pattern in obs_paths:
        pattern = os.path.abspath(pattern)
        all_obs_files.extend(glob.glob(pattern))
    
    if not all_obs_files:
        raise FileNotFoundError("No observation files found")
    
    # Sort by modification time and get the latest
    latest_obs_file = max(all_obs_files, key=os.path.getmtime)
    print(f"Using observations from: {latest_obs_file}")
    
    # Load the CSV data
    df = pd.read_csv(latest_obs_file)
    return df

# Load latest policy and observations
policy_path = find_latest_policy()
obs_df = find_latest_observations()

print(f"Loaded {len(obs_df)} observation rows")
print(f"Observation columns: {obs_df.columns.tolist()}")

# Global observation index
obs_index = 0
#/home/goat/Documents/GitHub/Leatherback/source/Leatherback/Leatherback/tasks/direct/leatherback/custom_assets/leatherback_simple_better.usd
#relative_path = os.path.join("..", "leatherback")
relative_path = os.path.join("../..", "source", "Leatherback", "Leatherback", "tasks", "direct", "leatherback", "custom_assets")
full_path = os.path.abspath(os.path.join(script_dir, relative_path))
usd_path = os.path.abspath(os.path.join(full_path, "leatherback_simple_better.usd"))

import onnxruntime as rt
sess = rt.InferenceSession(policy_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

def get_observation_data():
    """Get the next observation from the loaded data"""
    global obs_index
    
    if obs_index >= len(obs_df):
        obs_index = 0  # Loop back to beginning
    
    row = obs_df.iloc[obs_index]
    obs_index += 1
    
    # Extract observation based on file format
    if 'obs_0' in obs_df.columns:
        # RSL-RL format: obs_0 to obs_7
        obs = [row[f'obs_{i}'] for i in range(8)]
    else:
        # Source format: position_error, heading_error_cos, heading_error_sin, root_lin_vel_x, root_lin_vel_y, root_ang_vel_z, throttle_state, steering_state
        obs = [
            row['position_error'],
            row['heading_error_cos'], 
            row['heading_error_sin'],
            row['root_lin_vel_x'],
            row['root_lin_vel_y'],
            0.0,  # vel_z (not in source format)
            0.0,  # omega_x (not in source format) 
            row['root_ang_vel_z']
        ]
    
    return np.array([obs], dtype=np.float32)

# pos_error, heading_error_cos, heading_error_sin, vel_x, vel_y, vel_z, omega_x, omega_y
def on_physics_step(step_size) -> None:
    # Get real observation data from loaded files
    X_test = get_observation_data()
    print(f"Observation: {X_test}")
    
    # Run ONNX inference
    pred_onx = sess.run([label_name], {input_name: X_test})[0]
    print("prediction: " + pred_onx.__str__())
    
    # Action processing (same as in the training environment)
    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    #print(f"Raw actions: {pred_onx}")

    # Process throttle action (index 0, applied to 4 wheel joints)
    throttle_action = np.repeat(pred_onx[:, 0], 4).reshape((-1, 4)) * throttle_scale
    throttle_action = np.clip(throttle_action, -throttle_max, throttle_max)
    
    # Process steering action (index 1, applied to 2 steering joints)  
    steering_action = np.repeat(pred_onx[:, 1], 2).reshape((-1, 2)) * steering_scale
    steering_action = np.clip(steering_action, -steering_max, steering_max)
    
    #print(f"Throttle action: {throttle_action}")
    #print(f"Steering action: {steering_action}")
    
    # Apply actions to the robot
    robot_art.set_joint_velocities(throttle_action.flatten(), joint_indices=_throttle_dof_idx)
    robot_art.set_joint_positions(steering_action.flatten(), joint_indices=_steering_dof_idx)


my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()

prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
prim.GetReferences().AddReference(asset_path)


lb_prim_path = "/World/Leatherback"
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path, lb_prim_path)

from isaacsim.core.prims import Articulation
robot_art = Articulation(prim_paths_expr=lb_prim_path, name="Leatherback")

my_world.reset() # required to have joints available
robot_art.set_world_poses(positions=np.array([[0,0,0.5]]))

# Get joint information for action mapping
print("joint names: ", robot_art.joint_names)

# Joint name references
wheel_rr = 'Wheel__Upright__Rear_Right'
wheel_rl = 'Wheel__Upright__Rear_Left'
knuckle_fr = 'Knuckle__Upright__Front_Right'
knuckle_fl = 'Knuckle__Upright__Front_Left'
wheel_knuckle_fr = 'Wheel__Knuckle__Front_Right'
wheel_knuckle_fl = 'Wheel__Knuckle__Front_Left'

# Define DOF indices for throttle (wheel joints) and steering (knuckle joints)
throttle_joint_names = [wheel_rr, wheel_rl, wheel_knuckle_fr, wheel_knuckle_fl]
steering_joint_names = [knuckle_fr, knuckle_fl]

# Get the indices for these joints
_throttle_dof_idx = []
_steering_dof_idx = []

for joint_name in throttle_joint_names:
    if joint_name in robot_art.joint_names:
        _throttle_dof_idx.append(robot_art.joint_names.index(joint_name))

for joint_name in steering_joint_names:
    if joint_name in robot_art.joint_names:
        _steering_dof_idx.append(robot_art.joint_names.index(joint_name))

print(f"Throttle DOF indices: {_throttle_dof_idx}")
print(f"Steering DOF indices: {_steering_dof_idx}")


my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()
