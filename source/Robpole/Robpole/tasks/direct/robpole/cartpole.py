# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
#from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# USD path with proper resolution for cross-platform compatibility
USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "cartpole_v0.usd")


CARTPOLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"], effort_limit=400.0, velocity_limit=100.0, stiffness=0.0, damping=0.0
        ),
    },
)
"""Configuration for a simple Cartpole robot."""
