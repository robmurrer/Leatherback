import os
import numpy as np
import pandas as pd
import glob
import time
import logging
from datetime import datetime
from isaacsim import SimulationApp

# Get script directory first for relative paths
script_dir = os.path.dirname(__file__)

# Setup logging
log_dir = os.path.join(script_dir, "..", "..", "logs", "forest_gump")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"forest_gump_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Console output for critical messages only
    ]
)

# Set console handler to only show INFO and above
console_handler = None
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        console_handler = handler
        break

if console_handler:
    console_handler.setLevel(logging.INFO)  # Only show INFO and above on console

logger = logging.getLogger(__name__)
logger.info(f"Logging initialized. Log file: {log_file}")
print(f"Debug output will be logged to: {log_file}")  # Print to console so user knows where to find logs

# Enhanced tolerances for floating point comparisons
OBSERVATION_EPSILON = 1e-6
ACTION_EPSILON = 1e-4
POSITION_EPSILON = 1e-3  # Tolerance for position deviations
POSITION_CHECK_INTERVAL = 10  # Check position every N steps

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
import carb
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from leatherback.policy.example.leatherback import LeatherbackPolicy
from isaacsim.storage.native import get_assets_root_path

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
    
    logger.info(f"Using policy from: {policy_path}")
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
    logger.info(f"Using observations from: {latest_obs_file}")
    
    # Load the CSV data
    df = pd.read_csv(latest_obs_file)
    
    # Validate that the CSV has the expected format
    expected_obs_cols = [
        "obs_0_position_error",
        "obs_1_heading_cos", 
        "obs_2_heading_sin",
        "obs_3_linear_vel_x",
        "obs_4_linear_vel_y", 
        "obs_5_angular_vel_z",
        "obs_6_throttle_state",
        "obs_7_steering_state"
    ]
    expected_action_cols = [
        "action_0_throttle",
        "action_1_steering"
    ]
    expected_position_cols = ["pos_x", "pos_y", "pos_z"]
    expected_reset_cols = ["terminated", "truncated", "episode_reset"]
    required_cols = ['timestep', 'sim_time'] + expected_obs_cols + expected_action_cols + expected_position_cols + expected_reset_cols
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")
    
    logger.info(f"CSV validation passed: {len(df)} rows, columns: {df.columns.tolist()}")
    return df

# Load latest policy and observations
policy_path = find_latest_policy()
obs_df = find_latest_observations()

logger.info(f"Loaded {len(obs_df)} observation rows")

# Global observation index and simulation timing
obs_index = 0
start_time = None  # Will be set when simulation starts
current_obs_data = None  # Cache current observation data
previous_position = None  # Track previous position for reset detection
reset_threshold = 15.0  # Distance threshold to detect reset (increased)
position_deviation_threshold = 1.0  # Max allowed deviation from logged position (increased for debugging)
step_counter = 0  # Track physics steps
position_deviations = []  # Track position deviations
max_position_deviation = 0.0  # Track maximum position deviation
high_deviation_count = 0  # Track number of high deviation events
#/home/goat/Documents/GitHub/Leatherback/source/Leatherback/Leatherback/tasks/direct/leatherback/custom_assets/leatherback_simple_better.usd
#relative_path = os.path.join("..", "leatherback")
relative_path = os.path.join("../..", "source", "Leatherback", "Leatherback", "tasks", "direct", "leatherback", "custom_assets")
full_path = os.path.abspath(os.path.join(script_dir, relative_path))
usd_path = os.path.abspath(os.path.join(full_path, "leatherback_simple_better.usd"))

import onnxruntime as rt
sess = rt.InferenceSession(policy_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

def get_observation_data(sim_time):
    """Get observation data for the current simulation time, game loop style"""
    global obs_index, current_obs_data
    
    # If we don't have current observation data or need to advance to next observation
    if current_obs_data is None or sim_time >= current_obs_data['next_sim_time']:
        # Check if we've reached the end of the CSV data
        if obs_index >= len(obs_df):
            logger.info("📋 Reached end of CSV data - stopping simulation")
            return None, None, None, None, None, None  # Signal to stop simulation
        
        row = obs_df.iloc[obs_index]
        
        # Determine when to advance to next observation
        next_sim_time = row['sim_time'] + (obs_df.iloc[obs_index + 1]['sim_time'] - row['sim_time']) if obs_index + 1 < len(obs_df) else float('inf')
        
        current_obs_data = {
            'obs': [
                row["obs_0_position_error"],
                row["obs_1_heading_cos"], 
                row["obs_2_heading_sin"],
                row["obs_3_linear_vel_x"],
                row["obs_4_linear_vel_y"], 
                row["obs_5_angular_vel_z"],
                row["obs_6_throttle_state"],
                row["obs_7_steering_state"]
            ],
            'logged_actions': [row["action_0_throttle"], row["action_1_steering"]],
            'position': [row["pos_x"], row["pos_y"], row["pos_z"]],
            'reset_info': {
                'terminated': row["terminated"], 
                'truncated': row["truncated"], 
                'episode_reset': row["episode_reset"]
            },
            'timestep': row['timestep'],
            'sim_time': row['sim_time'],
            'next_sim_time': next_sim_time
        }
        
        logger.info(f"📊 Advanced to CSV row {obs_index}, timestep {row['timestep']}, sim_time {row['sim_time']:.3f}s")
        logger.info(f"  Position: [{row['pos_x']:.3f}, {row['pos_y']:.3f}, {row['pos_z']:.3f}]")
        logger.info(f"  Actions: throttle={row['action_0_throttle']:.6f}, steering={row['action_1_steering']:.6f}")
        logger.info(f"  Observations: pos_err={row['obs_0_position_error']:.3f}, vel_x={row['obs_3_linear_vel_x']:.3f}, vel_y={row['obs_4_linear_vel_y']:.3f}")
        # Log reset information when it occurs
        if row["episode_reset"]:
            logger.info(f"  🔄 RESET EVENT: terminated={row['terminated']}, truncated={row['truncated']}")
        obs_index += 1
    
    # Return the current observation data
    return (np.array([current_obs_data['obs']], dtype=np.float32), 
            current_obs_data['timestep'], 
            current_obs_data['sim_time'], 
            current_obs_data['logged_actions'],
            current_obs_data['position'],
            current_obs_data['reset_info'])

# pos_error, heading_error_cos, heading_error_sin, vel_x, vel_y, vel_z, omega_x, omega_y
def on_physics_step(step_size) -> None:
    global start_time, previous_position, step_counter, position_deviations, max_position_deviation, high_deviation_count
    
    # Initialize start time on first call
    if start_time is None:
        start_time = time.perf_counter()
        logger.info(f"Physics step initialized at time: {start_time:.6f}")
    
    step_counter += 1
    
    # Calculate current simulation time using actual elapsed time
    current_sim_time = time.perf_counter() - start_time
    
    # Get observation data for current sim time (will reuse same data until time threshold reached)
    result = get_observation_data(current_sim_time)
    
    # Check if we've reached the end of CSV data
    if result[0] is None:
        logger.info("📋 End of CSV data reached - stopping simulation")
        simulation_app.close()
        return
    
    X_test, timestep, csv_sim_time, logged_actions, position, reset_info = result
    
    # Get current actual robot position for comparison
    actual_position = robot_art.get_world_poses()[0][0]
    
    # Log every step for first 50 steps, then every 10 steps
    if step_counter <= 50 or step_counter % 10 == 0:
        logger.info(f"Step {step_counter}: sim_time={current_sim_time:.3f}s, csv_time={csv_sim_time:.3f}s, timestep={timestep}")
        logger.info(f"  Expected pos: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        logger.info(f"  Actual pos:   [{actual_position[0]:.3f}, {actual_position[1]:.3f}, {actual_position[2]:.3f}]")
        position_dev = np.linalg.norm(np.array(position[:3]) - actual_position)
        logger.info(f"  Position deviation: {position_dev:.3f}m")
        logger.info(f"  Expected actions: throttle={logged_actions[0]:.6f}, steering={logged_actions[1]:.6f}")
    
    # Check for reset using proper CSV flags instead of position-based detection
    if reset_info['episode_reset']:
        logger.info(f"🔄 Episode reset detected from CSV data!")
        logger.info(f"  Reset reason: terminated={reset_info['terminated']}, truncated={reset_info['truncated']}")
        logger.info(f"  At step {step_counter}, CSV timestep {timestep}, sim_time {csv_sim_time:.3f}s")
        logger.info(f"  This indicates the end of an episode - stopping simulation")
        simulation_app.close()
        return
    
    # Check position deviation periodically
    if step_counter % POSITION_CHECK_INTERVAL == 0:
        position_deviation = np.linalg.norm(np.array(position[:3]) - actual_position)
        position_deviations.append(position_deviation)
        
        logger.info(f"🔍 Position check at step {step_counter}:")
        logger.info(f"  Expected: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        logger.info(f"  Actual:   [{actual_position[0]:.3f}, {actual_position[1]:.3f}, {actual_position[2]:.3f}]")
        logger.info(f"  Deviation: {position_deviation:.3f}m")
        
        if position_deviation > max_position_deviation:
            max_position_deviation = position_deviation
            logger.info(f"  🔺 New maximum deviation: {max_position_deviation:.3f}m")
            
        if position_deviation > position_deviation_threshold:
            high_deviation_count += 1
            logger.warning(f"⚠️ HIGH DEVIATION #{high_deviation_count} DETECTED (continuing simulation)")
            logger.warning(f"Step: {step_counter}, CSV Timestep: {timestep}")
            logger.warning(f"Expected position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
            logger.warning(f"Actual position:   [{actual_position[0]:.3f}, {actual_position[1]:.3f}, {actual_position[2]:.3f}]")
            logger.warning(f"Deviation: {position_deviation:.3f}m (threshold: {position_deviation_threshold}m)")
            logger.warning(f"Current sim time: {current_sim_time:.3f}s, CSV sim time: {csv_sim_time:.3f}s")
            logger.warning(f"Time difference: {abs(current_sim_time - csv_sim_time):.3f}s")
            logger.warning(f"Maximum deviation so far: {max_position_deviation:.3f}m")
            logger.warning(f"Average deviation: {np.mean(position_deviations):.3f}m")
            
            # Log robot state
            joint_positions = robot_art.get_joint_positions()
            joint_velocities = robot_art.get_joint_velocities()
            logger.warning(f"Joint positions: {joint_positions}")
            logger.warning(f"Joint velocities: {joint_velocities}")
            logger.warning("Continuing simulation...")
    
    # Run ONNX inference
    pred_onx = sess.run([label_name], {input_name: X_test})[0]
    
    # Log inference results for first 50 steps or every 10 steps
    if step_counter <= 50 or step_counter % 10 == 0:
        logger.info(f"  ONNX prediction: throttle={pred_onx[0, 0]:.6f}, steering={pred_onx[0, 1]:.6f}")
    
    # Assert that predicted actions match logged actions
    predicted_actions = [pred_onx[0, 0], pred_onx[0, 1]]
    for i, (pred, logged) in enumerate(zip(predicted_actions, logged_actions)):
        if abs(pred - logged) >= ACTION_EPSILON:
            logger.error(f"❌ Action {i} mismatch: predicted={pred:.6f}, logged={logged:.6f}, diff={abs(pred - logged):.6f}")
            logger.error(f"Step: {step_counter}, CSV timestep: {timestep}, sim_time: {current_sim_time:.3f}s")
            assert False, f"Action {i} mismatch: predicted={pred}, logged={logged}"
    
    # Log action validation for first 50 steps or every 10 steps
    if step_counter <= 50 or step_counter % 10 == 0:
        logger.info(f"  ✓ Actions validated: throttle={predicted_actions[0]:.6f}, steering={predicted_actions[1]:.6f}")
    
    # Action processing (same as in the training environment)
    throttle_scale = 10
    throttle_max = 50
    steering_scale = 0.1
    steering_max = 0.75

    # Process throttle action (index 0, applied to 4 wheel joints)
    throttle_action = np.repeat(pred_onx[:, 0], 4).reshape((-1, 4)) * throttle_scale
    throttle_action = np.clip(throttle_action, -throttle_max, throttle_max)
    
    # Process steering action (index 1, applied to 2 steering joints)  
    steering_action = np.repeat(pred_onx[:, 1], 2).reshape((-1, 2)) * steering_scale
    steering_action = np.clip(steering_action, -steering_max, steering_max)
    
    # Log scaled actions for first 50 steps or every 10 steps
    if step_counter <= 50 or step_counter % 10 == 0:
        logger.info(f"  Scaled actions: throttle={throttle_action.flatten()}, steering={steering_action.flatten()}")
    
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

# Get the starting position from the first CSV row
first_row = obs_df.iloc[0]
# Use the actual CSV position values (including Z)
start_position = np.array([[first_row["pos_x"], first_row["pos_y"], first_row["pos_z"]]])
logger.info(f"Setting robot starting position to: [{start_position[0,0]:.3f}, {start_position[0,1]:.3f}, {start_position[0,2]:.3f}]")
logger.info(f"First CSV row details:")
logger.info(f"  Timestep: {first_row['timestep']}, Sim time: {first_row['sim_time']:.3f}s")
logger.info(f"  Actions: throttle={first_row['action_0_throttle']:.6f}, steering={first_row['action_1_steering']:.6f}")
logger.info(f"  Position error: {first_row['obs_0_position_error']:.3f}")
logger.info(f"  Velocities: x={first_row['obs_3_linear_vel_x']:.3f}, y={first_row['obs_4_linear_vel_y']:.3f}, z={first_row['obs_5_angular_vel_z']:.3f}")
robot_art.set_world_poses(positions=start_position)

# Get joint information for action mapping
logger.info("joint names: " + str(robot_art.joint_names))

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

logger.info(f"Throttle DOF indices: {_throttle_dof_idx}")
logger.info(f"Steering DOF indices: {_steering_dof_idx}")


my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

logger.info("Starting simulation...")
logger.info(f"Position deviation threshold: {position_deviation_threshold}m")
logger.info(f"Position check interval: {POSITION_CHECK_INTERVAL} steps")
logger.info(f"Action epsilon: {ACTION_EPSILON}")
logger.info(f"Reset threshold: {reset_threshold}m")

while simulation_app.is_running():
    my_world.step(render=True)

logger.info("Simulation ended")
if position_deviations:
    logger.info(f"📊 Final Position Deviation Statistics:")
    logger.info(f"  Average deviation: {np.mean(position_deviations):.4f}m")
    logger.info(f"  Maximum deviation: {max_position_deviation:.4f}m")
    logger.info(f"  High deviation events: {high_deviation_count} (threshold: {position_deviation_threshold}m)")
    logger.info(f"  Total position checks: {len(position_deviations)}")
    logger.info(f"  Total simulation steps: {step_counter}")
    logger.info(f"  CSV rows processed: {obs_index}")
    
    # Calculate percentage of high deviation events
    high_deviation_percentage = (high_deviation_count / len(position_deviations)) * 100 if len(position_deviations) > 0 else 0
    logger.info(f"  High deviation percentage: {high_deviation_percentage:.1f}%")
    
    # Log distribution of deviations
    import numpy as np
    deviations_array = np.array(position_deviations)
    logger.info(f"  Deviation percentiles: 50th={np.percentile(deviations_array, 50):.3f}m, 90th={np.percentile(deviations_array, 90):.3f}m, 95th={np.percentile(deviations_array, 95):.3f}m")

simulation_app.close()
