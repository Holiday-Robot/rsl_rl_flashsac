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

"""Motion-tracking observation terms.

Each term is a ManagerBased observation function ``func(env, command_name) -> torch.Tensor``
returning a tensor of shape ``(num_envs, dim)``. They read the active
:class:`~isaaclab_flashsac.mdp.commands.motion.MotionCommand` term from the command
manager and expose the robot/motion anchor and per-body state in the relevant reference
frames. Unlike :mod:`..locomotion`, these terms call ``isaaclab.utils.math`` at runtime and
therefore require a real Isaac Lab install (they are not re-exported at the
:mod:`isaaclab_flashsac.mdp.obs` package level, which must stay importable without it).
"""

from isaaclab_flashsac.mdp.obs.motion.commands import (
    motion_anchor_ori_b,
    motion_anchor_pos_b,
    robot_anchor_ang_vel_w,
    robot_anchor_lin_vel_w,
    robot_anchor_ori_w,
    robot_body_ori_b,
    robot_body_pos_b,
)

__all__ = [
    "motion_anchor_ori_b",
    "motion_anchor_pos_b",
    "robot_anchor_ang_vel_w",
    "robot_anchor_lin_vel_w",
    "robot_anchor_ori_w",
    "robot_body_ori_b",
    "robot_body_pos_b",
]
