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

"""Motion-tracking reward terms.

These reward terms penalize the deviation between the robot and the reference
motion, both in the global anchor frame and in the relative body frame. Each
term returns an exponential tracking reward of shape ``(num_envs,)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_flashsac.mdp.commands.motion import MotionCommand


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Resolves a list of body names to their indices in the command body set.

    Args:
        command: The motion command term holding the configured body names.
        body_names: The body names to select, or ``None`` to select all bodies.

    Returns
    -------
        The list of indices into ``command.cfg.body_names`` matching the request.
    """
    if body_names is None:
        return list(range(len(command.cfg.body_names)))

    requested_names = list(body_names)
    if len(set(requested_names)) != len(requested_names):
        duplicates = sorted({name for name in requested_names if requested_names.count(name) > 1})
        raise ValueError(f"Duplicate reward body_names are not allowed: {duplicates}")

    available_names = list(command.cfg.body_names)
    missing = [name for name in requested_names if name not in available_names]
    if missing:
        raise ValueError(f"Tracking body_names are not tracked by the motion command: {missing}")

    return [i for i, name in enumerate(available_names) if name in requested_names]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Exponential reward on the global anchor position error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Exponential reward on the global anchor orientation error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the relative body position error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    body_offsets: list[list[float]] | None = None,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    """Exponential reward on anchor-local body point position error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.
        body_offsets: The local point offsets for each tracked body, or ``None`` to use body origins.
        anchor_body_name: The body frame used as the local anchor, or ``None`` to use the command anchor.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    if not body_indexes:
        raise ValueError(f"No body names matched reward body_names={body_names!r}")

    num_bodies = len(body_indexes)
    selected_body_names = [command.cfg.body_names[i] for i in body_indexes]
    if body_offsets is None:
        offsets = torch.zeros((1, num_bodies, 3), dtype=command.body_pos_w.dtype, device=command.body_pos_w.device)
    else:
        if body_names is None:
            ordered_offsets = body_offsets
        else:
            offset_by_name = dict(zip(body_names, body_offsets, strict=True))
            ordered_offsets = [offset_by_name[name] for name in selected_body_names]
        offsets = torch.tensor(ordered_offsets, dtype=command.body_pos_w.dtype, device=command.body_pos_w.device).view(
            1, num_bodies, 3
        )
    offsets = offsets.expand(command.body_pos_w.shape[0], -1, -1)

    ref_pos_w = command.body_pos_w[:, body_indexes] + quat_apply(command.body_quat_w[:, body_indexes], offsets)
    robot_pos_w = command.robot_body_pos_w[:, body_indexes] + quat_apply(
        command.robot_body_quat_w[:, body_indexes], offsets
    )

    if anchor_body_name is None:
        ref_anchor_pos_w = command.anchor_pos_w
        ref_anchor_quat_w = command.anchor_quat_w
        robot_anchor_pos_w = command.robot_anchor_pos_w
        robot_anchor_quat_w = command.robot_anchor_quat_w
    else:
        anchor_index = command.cfg.body_names.index(anchor_body_name)
        ref_anchor_pos_w = command.body_pos_w[:, anchor_index]
        ref_anchor_quat_w = command.body_quat_w[:, anchor_index]
        robot_anchor_pos_w = command.robot_body_pos_w[:, anchor_index]
        robot_anchor_quat_w = command.robot_body_quat_w[:, anchor_index]

    ref_anchor_quat_w = ref_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_w = robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    ref_pos_local = quat_apply(quat_inv(ref_anchor_quat_w), ref_pos_w - ref_anchor_pos_w[:, None, :])
    robot_pos_local = quat_apply(quat_inv(robot_anchor_quat_w), robot_pos_w - robot_anchor_pos_w[:, None, :])
    error = torch.sum(torch.square(robot_pos_local - ref_pos_local), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalty on excessive reference-relative selected-body angular velocity.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        threshold: The angular-velocity error deadzone in rad/s.
        body_names: The bodies to penalize, or ``None`` to penalize all bodies.

    Returns
    -------
        The penalty of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.linalg.norm(
        command.robot_body_ang_vel_w[:, body_indexes] - command.body_ang_vel_w[:, body_indexes], dim=-1
    )
    return torch.square(torch.relu(error - threshold)).mean(dim=-1)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the relative body orientation error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the global body linear-velocity error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on the global body angular-velocity error.

    Args:
        env: The environment instance.
        command_name: The name of the motion command term.
        std: The standard deviation of the exponential kernel.
        body_names: The bodies to track, or ``None`` to track all bodies.

    Returns
    -------
        The reward of shape ``(num_envs,)``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)
