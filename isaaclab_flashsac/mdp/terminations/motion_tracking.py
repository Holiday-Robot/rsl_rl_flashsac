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

"""Motion-tracking termination terms.

Note the func-vs-field indirection (matching WBT): the env config's ``anchor_pos`` term
uses the ``bad_anchor_pos_z_only`` func and ``ee_body_pos`` uses the
``bad_motion_body_pos_z_only`` func. ``time_out`` is the IsaacLab builtin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_flashsac.mdp.commands.motion import MotionCommand

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from isaaclab_flashsac.mdp.rewards.motion_tracking import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when the anchor position error exceeds ``threshold``.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to read the anchor pose from.
        threshold: Maximum allowed Euclidean anchor position error (in meters).

    Returns
    -------
        Boolean tensor of shape ``(num_envs,)`` that is ``True`` for environments whose
        anchor position error exceeds ``threshold``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when the vertical (z) anchor position error exceeds ``threshold``.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to read the anchor pose from.
        threshold: Maximum allowed absolute z-axis anchor position error (in meters).

    Returns
    -------
        Boolean tensor of shape ``(num_envs,)`` that is ``True`` for environments whose
        vertical anchor position error exceeds ``threshold``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate when the anchor orientation error exceeds ``threshold``.

    The error is measured as the absolute difference of the z-component of the projected
    gravity vector expressed in the motion anchor frame versus the robot anchor frame.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration used to fetch the gravity vector.
        command_name: Name of the motion command term to read the anchor pose from.
        threshold: Maximum allowed absolute projected-gravity z error.

    Returns
    -------
        Boolean tensor of shape ``(num_envs,)`` that is ``True`` for environments whose
        anchor orientation error exceeds ``threshold``.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate when any tracked body position error exceeds ``threshold``.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to read body poses from.
        threshold: Maximum allowed per-body Euclidean position error (in meters).
        body_names: Optional subset of body names to check; ``None`` uses all tracked bodies.

    Returns
    -------
        Boolean tensor of shape ``(num_envs,)`` that is ``True`` for environments where any
        tracked body position error exceeds ``threshold``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate when any tracked body's vertical (z) position error exceeds ``threshold``.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term to read body poses from.
        threshold: Maximum allowed per-body absolute z-axis position error (in meters).
        body_names: Optional subset of body names to check; ``None`` uses all tracked bodies.

    Returns
    -------
        Boolean tensor of shape ``(num_envs,)`` that is ``True`` for environments where any
        tracked body's vertical position error exceeds ``threshold``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
