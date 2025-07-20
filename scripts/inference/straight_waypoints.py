import numpy as np
import os
import numpy
import onnxruntime as rt

script_dir = os.path.dirname(os.path.abspath(__file__))
target_positions = np.array([[-10.24000072479248, 0.0], [-7.680000305175781, 0.0], [-5.12000036239624, 0.0], [-2.56000018119812, 0.0], [0.0, 0.0], [2.56000018119812, 0.0], [5.12000036239624, 0.0], [7.680000305175781, 0.0], [10.24000072479248, 0.0], [12.800000190734863, 0.0]])

policy_path = "/home/goat/Documents/GitHub/Leatherback/logs/rsl_rl/leatherback_direct/2025-07-20_10-59-06/exported/policy.onnx"
sess = rt.InferenceSession(policy_path, providers=rt.get_available_providers())
print(sess)
for inp in sess.get_inputs():
    print(f"Input Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

import numpy as np

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
X_test = np.array([[0.972418, 0.999365, 0.035642, 2.553681, -0.078560, -0.574272, 50.000000, -0.164952]])
expected_action = (101.636024, 21.282106)

print(input_name, label_name)
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)
print(f"Expected: {expected_action}, Predicted: {pred_onx}")

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
#from omni.isaac.core.articulations import Articulation # this doesn't fuckin work
from isaacsim.core.prims import Articulation


from isaacsim.core.utils.rotations import quat_to_rot_matrix

import torch

from pxr import UsdLux, Gf, UsdGeom
# Add lighting to illuminate everything
def add_lighting():
    # Add distant light (primary directional lighting)
    distant_light_prim = define_prim("/World/DistantLight", "DistantLight")
    distant_light = UsdLux.DistantLight(distant_light_prim)
    distant_light.CreateIntensityAttr(3000.0)  # Strong intensity
    distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))  # White light
    distant_light.CreateAngleAttr(0.53)
    
    # Rotate the light to shine down at an angle
    xformable = UsdGeom.Xformable(distant_light_prim)
    xformable.AddRotateXOp().Set(45.0)  # 45 degrees down
    xformable.AddRotateYOp().Set(45.0)  # 45 degrees to the side
    
    # Add dome light for ambient fill lighting
    dome_light_prim = define_prim("/World/DomeLight", "DomeLight")
    dome_light = UsdLux.DomeLight(dome_light_prim)
    dome_light.CreateIntensityAttr(800.0)  # Moderate ambient intensity
    dome_light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))  # Slightly cool ambient

 # region Observation
# This need to be replaced to something similar like the IsaacLab get observations
def _compute_observation(robot, command):
    throttle_scale = 1 # when set to 2 it trains but the cars are flying, 3 you get NaNs
    throttle_max = 5 #50.0 # throttle_max = 60.0
    """Multiplier for the steering position. The action is in the range [-1, 1]"""
    steering_scale = 0.1 # steering_scale = math.pi / 4.0
    steering_max = 0.75
    # this is using the articulation type 
    # from isaacsim.core.prims import SingleArticulation
    # self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
    lin_vel_I = robot.get_linear_velocities()
    ang_vel_I = robot.get_angular_velocities()
    # position, orientation = prim.get_world_pose()
    pos_IB, q_IB = robot.get_world_poses() 
    
    R_IB = quat_to_rot_matrix(q_IB)
    R_BI = R_IB.transpose()
    lin_vel_b = np.matmul(R_BI, lin_vel_I)
    ang_vel_b = np.matmul(R_BI, ang_vel_I)
    gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

    # region pos_error
    _position_error_vector = command - pos_IB
    _position_error = np.linalg.norm(_position_error_vector) # , axis=-1
    FORWARD_VEC_B = np.array([1.0, 0.0, 0.0]) # Do I need _root_physx_view ?
    quat = q_IB.reshape(-1, 4)
    vec = FORWARD_VEC_B.reshape(-1, 3)
    xyz = quat[:, 1:]
    # In torch: t = xyz.cross(vec, dim=-1) * 2
    # Cross product: t = 2 * cross(xyz, vec)
    t = 2 * np.cross(xyz, vec)

    # forward_w = (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)
    # Compute rotated vector: forward_w = vec + w * t + cross(xyz, t)
    w = quat[:, 0:1]  # shape (N, 1)
    forward_w = vec + w * t + np.cross(xyz, t)

    # heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    # Get heading from world-frame forward vector (X and Y components)
    heading_w = np.arctan2(forward_w[:, 1], forward_w[:, 0])  # shape (N,)

    target_heading_w = np.arctan2(command[1]-pos_IB[1], command[0]-pos_IB[0])
    _heading_error = target_heading_w - heading_w

    global _previous_action
    obs = np.zeros(8)
    # Position Error
    obs[:1] = _position_error
    # Heading error
    obs[1:2] = np.cos(_heading_error)[:, np.newaxis]
    obs[2:3] = np.sin(_heading_error)[:, np.newaxis]
    # Linear Velocity X and Y
    obs[3:4] = lin_vel_b[0]
    obs[4:5] = lin_vel_b[1]
    # Angular Velocity vZ
    obs[5:6] = ang_vel_b[2]
    # _throttle_state
    throttle_action = _previous_action[0]*throttle_scale
    _throttle_state = np.clip(throttle_action, -throttle_max, throttle_max*0.1)
    obs[6:7] = _throttle_state # self._previous_action[0]
    # _steering_state
    steering_action = _previous_action[1]*steering_scale
    _steering_state = np.clip(steering_action, -steering_max, steering_max)
    # which joint is steering and will this return the right value ?
    # current_joint_pos = self.robot.get_joint_positions()
    obs[7:] = _steering_state # self._previous_action[1]

    return obs

#from isaacsim.core.api.robots.robot import Robot

def on_physics_step(step_size):
    obs = _compute_observation(robot_art, target_positions[0])
    print("Observation: ", obs)
    wheel_vel = 16
    robot_art.set_joint_velocities([[wheel_vel, wheel_vel, 0, 0, wheel_vel, wheel_vel]])


world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
ground_prim = define_prim("/World/GroundPlane", "Xform")
ground_prim.GetReferences().AddReference("http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.1/Isaac/Environments/Grid/default_environment.usd")
robot_usd_path = os.path.join(script_dir, "..", "..", "source", "Leatherback", "Leatherback", 
                             "tasks", "direct", "leatherback", "custom_assets", "leatherback_simple_better.usd")
robot_usd_path = os.path.abspath(robot_usd_path)
robot_prim_path = "/World/Leatherback"
robot_name = "Leatherback"

add_reference_to_stage(robot_usd_path, robot_prim_path)
robot_art = Articulation(prim_paths_expr=robot_prim_path, name=robot_name)

add_lighting()

start_position = torch.tensor([[-15.996741, 0.00038, 0.024182]], dtype=torch.float32)
robot_art.set_world_poses(positions=start_position)

world.reset()

print("joint names: ", robot_art.joint_names)

world.add_physics_callback("physics_step", callback_fn=on_physics_step)

while simulation_app.is_running():
    world.step(render=True)
    

simulation_app.close()