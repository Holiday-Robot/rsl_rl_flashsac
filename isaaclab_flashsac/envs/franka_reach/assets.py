# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license),
# with modifications by Holiday Robotics (BSD-3-Clause license).

"""Franka Research 3 articulation for the reach task."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # the FR3 SDK's low-level controller applies gravity compensation automatically
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            "fr3_finger_joint.*": 0.04,
        }
    ),
    actuators={
        "fr3_arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-7]"],
            # documented ceilings — never raise them
            effort_limit_sim={
                "fr3_joint1": 20.0,
                "fr3_joint2": 20.0,
                "fr3_joint3": 18.0,
                "fr3_joint4": 18.0,
                "fr3_joint5": 16.0,
                "fr3_joint6": 14.0,
                "fr3_joint7": 12.0,
            },
            velocity_limit_sim={
                "fr3_joint1": 2.62,
                "fr3_joint2": 2.62,
                "fr3_joint3": 2.62,
                "fr3_joint4": 2.62,
                "fr3_joint5": 5.26,
                "fr3_joint6": 4.18,
                "fr3_joint7": 5.26,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
