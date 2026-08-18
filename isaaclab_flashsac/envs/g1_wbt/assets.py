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

"""Unitree G1 tracking profile: articulation config and body/joint groups."""

from __future__ import annotations

from typing import Any

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab_flashsac.envs.g1_dreamwaq.assets import G1_BEYONDMIMIC_ACTION_SCALE, G1_BEYONDMIMIC_CFG

ANCHOR_BODY_NAME = "torso_link"
LOCAL_REWARD_ANCHOR_BODY_NAME = "pelvis"

TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

END_EFFECTOR_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

ANTI_SHAKE_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

FEET_JOINT_NAMES = [".*ankle.*"]
BASE_COM_BODY_NAME = "torso_link"

CONTACT_PENALTY_ALLOWED_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

UNDESIRED_CONTACT_BODY_REGEX = r"^" + "".join(f"(?!{body}$)" for body in CONTACT_PENALTY_ALLOWED_BODIES) + r".+$"


G1_TRACKING_CFG: ArticulationCfg = G1_BEYONDMIMIC_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
)
"""29-DoF G1 articulation for motion tracking, with the WBT initial state."""


# Same derivation (0.25 * effort_limit / stiffness) over the same actuator table as the
# reference's G1_ACTION_SCALE, so the values are identical.
G1_TRACKING_ACTION_SCALE: dict[str, float] = G1_BEYONDMIMIC_ACTION_SCALE


def apply_g1_tracking_profile(env_cfg: Any) -> None:
    """Apply Unitree G1 body and joint groups to the generic tracking cfg."""
    env_cfg.commands.motion.anchor_body_name = ANCHOR_BODY_NAME
    env_cfg.commands.motion.body_names = list(TRACKED_BODY_NAMES)

    env_cfg.rewards.motion_ee_body_pos.params["body_names"] = list(END_EFFECTOR_BODY_NAMES)
    env_cfg.rewards.motion_ee_body_pos.params["anchor_body_name"] = LOCAL_REWARD_ANCHOR_BODY_NAME
    env_cfg.rewards.anti_shake_ang_vel.params["body_names"] = list(ANTI_SHAKE_BODY_NAMES)
    env_cfg.rewards.feet_acc.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(FEET_JOINT_NAMES))
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=[UNDESIRED_CONTACT_BODY_REGEX]
    )

    env_cfg.terminations.ee_body_pos.params["body_names"] = list(END_EFFECTOR_BODY_NAMES)
    env_cfg.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE_COM_BODY_NAME)
