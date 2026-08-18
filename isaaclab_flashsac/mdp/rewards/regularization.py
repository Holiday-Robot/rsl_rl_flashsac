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

"""Smoothness/effort/behavior-shaping reward terms for the G1 DreamWaQ velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize mechanical power, ``||tau * qdot||``."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)


def body_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat orientation of a specific body (e.g. torso) via projected gravity."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance across legs in last air time and last contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def stand_still(
    env: ManagerBasedRLEnv,
    threshold: float = 0.1,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint deviation from default when the command is small (near-zero)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < threshold)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the action rate measured on the policy's *normalized* action."""
    delta = env.action_manager.action - env.action_manager.prev_action
    action_scale = getattr(env, "_action_scale", None)
    if action_scale is not None:
        delta = delta / action_scale
    return torch.sum(torch.square(delta), dim=1)


def stay_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    position_threshold: float = 0.05,
    orientation_threshold: float | None = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize end-effector speed once the pose command is met, so the arm parks at the target.

    Outside the tolerance the term is zero and the tracking rewards do the approach.

    Args:
        env: The manager-based RL environment.
        command_name: Pose command term, read as ``[pos_b (3), quat_b (4)]``.
        position_threshold: Goal radius in metres.
        orientation_threshold: Goal orientation tolerance in radians, or ``None`` to gate on
            position alone. Tighter than the policy's steady-state error leaves the term dead.
        asset_cfg: The asset and the single body to measure.

    Returns
    -------
        Linear speed of the tracked body, zeroed outside the goal tolerance. Shape: (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore[index]
    command = env.command_manager.get_command(command_name)

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w

    des_pos_w, _ = math_utils.combine_frame_transforms(root_pos_w, root_quat_w, command[:, :3])
    curr_pos_w = asset.data.body_pos_w[:, body_id]
    in_position = torch.linalg.norm(curr_pos_w - des_pos_w, dim=1) < position_threshold

    at_goal = in_position
    if orientation_threshold is not None:
        des_quat_w = math_utils.quat_mul(root_quat_w, command[:, 3:7])
        curr_quat_w = asset.data.body_quat_w[:, body_id]
        at_goal = in_position & (math_utils.quat_error_magnitude(curr_quat_w, des_quat_w) < orientation_threshold)

    speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, body_id], dim=1)
    return speed * at_goal
