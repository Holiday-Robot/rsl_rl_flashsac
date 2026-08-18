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

"""Motion-tracking observation terms read off the active motion command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_flashsac.mdp.commands.motion import MotionCommand


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference motion anchor orientation expressed in the robot anchor frame.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        The first two columns of the reference anchor rotation matrix in the
        robot anchor frame, flattened to shape ``(num_envs, 6)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference motion anchor position expressed in the robot anchor frame.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        The reference anchor position in the robot anchor frame of
        shape ``(num_envs, 3)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor angular velocity in the world frame.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        The anchor angular velocity of shape ``(num_envs, 3)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_ang_vel_w.view(env.num_envs, -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor linear velocity in the world frame.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        The anchor linear velocity of shape ``(num_envs, 3)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_lin_vel_w.view(env.num_envs, -1)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot anchor orientation in the world frame as a 6D rotation feature.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        The first two columns of the anchor rotation matrix, flattened to
        shape ``(num_envs, 6)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot body orientations expressed in the robot anchor frame.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        The first two columns of each tracked-body rotation matrix in the
        anchor frame, flattened to shape ``(num_envs, 6 * num_bodies)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot body positions expressed in the robot anchor frame.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to query.

    Returns
    -------
        Tracked-body positions in the anchor frame, flattened to
        shape ``(num_envs, 3 * num_bodies)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)
