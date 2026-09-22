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

"""XHand1 right hand (RoboEra, 12 DoF) articulation for the in-hand reorient task."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

_XHAND_RIGHT_URDF_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "xhand_right" / "xhand_right.urdf")

# Gains and joint dynamics follow dexscrew's XHand1 PhysX setup
# (x-robotics-lab/dexscrew: configs/task/XHandHoraScrewDriver.yaml, xhand_hora.py _parse_hand_dof_props).
XHAND_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_XHAND_RIGHT_URDF_PATH,
        fix_base=True,
        # folds the massless tip and ee links into their parents; fingertips are the distal phalanges
        merge_fixed_joints=True,
        # gains: placeholder written into the USD, overwritten by the actuator cfg below at init
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=3.0, damping=0.01)
        ),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,  # deg/s: 64 rad/s
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # dexscrew's contact_offset 0.002 m is not applied: the URDF importer writes instanceable
        # collision prims, which reject collision_props (Isaac Lab's Allegro cfg skips it for the same reason)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # palm normal (+y of right_hand_link) up, fingers (-z of right_hand_link) along world +x
        rot=(0.5, 0.5, -0.5, -0.5),
        # fingers half curled and the thumb opposed so the dropped cube lands in a cage:
        # 3/64 zero-action drops fall in 3 s versus 9/64 with the hand flat
        joint_pos={
            "right_hand_(index|mid|ring|pinky)_joint[12]": 0.8,
            "right_hand_index_bend_joint": 0.0,
            "right_hand_thumb_bend_joint": 0.9,
            "right_hand_thumb_rota_joint[12]": 0.5,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # URDF effort limits, N·m: 1.1 on the thumb bend and proximal flexions, 0.4 elsewhere
            effort_limit_sim={
                "right_hand_thumb_bend_joint": 1.1,
                "right_hand_thumb_rota_joint1": 1.1,
                "right_hand_(index|mid|ring|pinky)_joint1": 1.1,
                "right_hand_thumb_rota_joint2": 0.4,
                "right_hand_index_bend_joint": 0.4,
                "right_hand_(index|mid|ring|pinky)_joint2": 0.4,
            },
            stiffness=3.0,
            damping=0.01,
            armature=0.001,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
