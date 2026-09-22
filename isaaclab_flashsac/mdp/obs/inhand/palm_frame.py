# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Licensed under BSD-3-Clause.

"""In-hand reorientation observation terms expressed in the palm frame (the robot root link)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, random_orientation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.manipulation.inhand.mdp import InHandReOrientationCommand


def object_pos_in_palm(
    env: ManagerBasedRLEnv,
    injection_prob: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object position in the palm frame, m, shape ``(num_envs, 3)``.

    Args:
        env: The environment instance.
        injection_prob: Per-step chance an env's value is replaced by a random position in
            [-0.5, 0.5] m, so the policy does not over-trust the tracker (wuji-mjlab). 0 disables it.
        object_cfg: The tracked object.
        robot_cfg: The hand; its root link is the palm.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    pos = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    return _inject(pos, torch.empty_like(pos).uniform_(-0.5, 0.5), injection_prob)


def goal_orientation_error_6d(
    env: ManagerBasedRLEnv,
    command_name: str,
    injection_prob: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object-to-goal rotation in the palm frame as 6D, shape ``(num_envs, 6)``.

    Args:
        env: The environment instance.
        command_name: The in-hand reorientation command holding the goal quaternion.
        injection_prob: Per-step chance an env's value is replaced by a random rotation. 0 disables it.
        object_cfg: The tracked object.
        robot_cfg: The hand; its root link is the palm.
    """
    command: InHandReOrientationCommand = env.command_manager.get_term(command_name)
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    error = rotation_error_6d(obj.data.root_quat_w, command.command[:, 3:7], robot.data.root_quat_w)
    random_error = matrix_from_quat(random_orientation(env.num_envs, env.device)).reshape(-1, 9)[:, 3:]
    return _inject(error, random_error, injection_prob)


def body_pos_in_palm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Selected body positions in the palm frame, m, flattened to ``(num_envs, 3 * num_bodies)``.

    Args:
        env: The environment instance.
        asset_cfg: The hand with ``body_names`` selecting the bodies (the fingertips).
    """
    robot = env.scene[asset_cfg.name]
    rel = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w[:, None]
    palm_quat = robot.data.root_quat_w[:, None].expand(-1, rel.shape[1], -1)
    return quat_apply_inverse(palm_quat, rel).flatten(1)


def rotation_error_6d(object_quat: torch.Tensor, goal_quat: torch.Tensor, palm_quat: torch.Tensor) -> torch.Tensor:
    """Second and third rows of the object-to-goal rotation matrix, both quaternions taken into the palm frame.

    Args:
        object_quat: Object orientation (N, 4), world, (w, x, y, z).
        goal_quat: Goal orientation (N, 4), world, (w, x, y, z).
        palm_quat: Palm orientation (N, 4), world, (w, x, y, z).
    """
    palm_inv = quat_conjugate(palm_quat)
    object_in_palm = quat_mul(palm_inv, object_quat)
    goal_in_palm = quat_mul(palm_inv, goal_quat)
    error = quat_mul(object_in_palm, quat_conjugate(goal_in_palm))
    return matrix_from_quat(error).reshape(-1, 9)[:, 3:]


def _inject(value: torch.Tensor, replacement: torch.Tensor, prob: float) -> torch.Tensor:
    """Replace each env's row of ``value`` with ``replacement`` with probability ``prob``."""
    if prob <= 0.0:
        return value
    mask = torch.rand((value.shape[0], 1), device=value.device) < prob
    return torch.where(mask, replacement, value)
