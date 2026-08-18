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

"""FR3 reach environment configuration."""

import math

import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg

from isaaclab_flashsac.mdp import rewards as flashsac_rews

from .assets import FR3_CFG


@configclass
class Fr3ReachEnvCfg(FrankaReachEnvCfg):
    """FR3 reach task."""

    def __post_init__(self):
        super().__post_init__()

        # -- robot
        self.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        for term in (
            self.rewards.end_effector_position_tracking,
            self.rewards.end_effector_position_tracking_fine_grained,
            self.rewards.end_effector_orientation_tracking,
        ):
            term.params["asset_cfg"].body_names = ["fr3_hand"]

        self.commands.ee_pose.body_name = "fr3_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)  # end-effector points down z

        # -- action
        self.actions.arm_action = reach_mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["fr3_joint.*"], scale=0.05, use_zero_offset=True
        )

        # -- tracking
        self.rewards.end_effector_orientation_tracking_fine_grained = RewTerm(
            func=flashsac_rews.orientation_command_error_tanh,
            weight=1.0,
            params={
                "std": 0.5,
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=["fr3_hand"]),
            },
        )
        self.rewards.end_effector_position_tracking_fine_grained.weight = 1.0

        # -- smoothness
        self.rewards.action_l2 = RewTerm(func=reach_mdp.action_l2, weight=-1e-4)
        self.curriculum.action_l2 = CurrTerm(func=reach_mdp.modify_reward_weight, params={"term_name": "action_l2"})

        for term_name in ("action_rate", "joint_vel", "action_l2"):
            params = getattr(self.curriculum, term_name).params
            params["weight"] = -1e-2
            params["num_steps"] = 10000

        # -- goal gate
        self.rewards.stay_at_goal = RewTerm(
            func=flashsac_rews.stay_at_goal,
            weight=-1.0,
            params={
                "command_name": "ee_pose",
                "position_threshold": 0.01,
                "orientation_threshold": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=["fr3_hand"]),
            },
        )


@configclass
class Fr3ReachEnvCfg_PLAY(Fr3ReachEnvCfg):
    """Play variant: smaller scene, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
