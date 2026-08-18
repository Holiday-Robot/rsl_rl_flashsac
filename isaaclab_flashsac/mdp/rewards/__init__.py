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

"""Reward term functions, grouped by role.

- :mod:`.tracking`: velocity-command/gait-tracking rewards (exponential-kernel style).
- :mod:`.motion_tracking`: reference-motion (BeyondMimic WBT) tracking rewards.
- :mod:`.regularization`: smoothness/effort/behavior-shaping penalties.
- :mod:`.safety`: contact-sensor-driven hardware/collision-protection penalties.

Reward functions already provided by ``isaaclab.envs.mdp`` (``track_lin_vel_xy_yaw_frame_exp``,
``track_ang_vel_z_world_exp``, ``lin_vel_z_l2``, ``ang_vel_xy_l2``, ``flat_orientation_l2``,
``joint_acc_l2``, ``joint_deviation_l1``, ``joint_pos_limits``,
``is_terminated``, ``feet_slide``) are used directly from env_cfg.py and are not duplicated here.
"""

from isaaclab_flashsac.mdp.rewards.motion_tracking import (
    anti_shake_ang_vel_l2,
    motion_global_anchor_orientation_error_exp,
    motion_global_anchor_position_error_exp,
    motion_global_body_angular_velocity_error_exp,
    motion_global_body_linear_velocity_error_exp,
    motion_local_body_position_error_exp,
    motion_relative_body_orientation_error_exp,
    motion_relative_body_position_error_exp,
)
from isaaclab_flashsac.mdp.rewards.regularization import (
    action_rate_l2,
    body_orientation_l2,
    energy,
    feet_air_time_variance_penalty,
    stand_still,
    stay_at_goal,
)
from isaaclab_flashsac.mdp.rewards.safety import (
    body_force,
    feet_impact_velocity_delta_penalty,
    feet_stumble,
    feet_too_near_humanoid,
    feet_yaw_drag,
    fly,
)
from isaaclab_flashsac.mdp.rewards.tracking import (
    feet_air_time_with_inplace,
    orientation_command_error_tanh,
    orthogonal_velocity_exp,
)

__all__ = [
    "action_rate_l2",
    "anti_shake_ang_vel_l2",
    "body_force",
    "body_orientation_l2",
    "energy",
    "feet_air_time_variance_penalty",
    "feet_air_time_with_inplace",
    "feet_impact_velocity_delta_penalty",
    "feet_stumble",
    "feet_too_near_humanoid",
    "feet_yaw_drag",
    "fly",
    "motion_global_anchor_orientation_error_exp",
    "motion_global_anchor_position_error_exp",
    "motion_global_body_angular_velocity_error_exp",
    "motion_global_body_linear_velocity_error_exp",
    "motion_local_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "orientation_command_error_tanh",
    "orthogonal_velocity_exp",
    "stand_still",
    "stay_at_goal",
]
