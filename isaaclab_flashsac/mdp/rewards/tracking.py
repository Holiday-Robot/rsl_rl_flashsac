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

"""Command/gait-tracking reward terms for the G1 DreamWaQ velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def orthogonal_velocity_exp(
    env: ManagerBasedRLEnv,
    scale: float = 1.5,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for minimizing velocity orthogonal to the command direction (Lee et al. 2020, eq. 14).

    Penalizes sideways drift while allowing velocity along the commanded direction. For stop
    commands (small command norm) all planar velocity is penalized.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Velocity in the yaw-aligned body frame.
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    vel_xy = vel_yaw[:, :2]

    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)

    # For non-zero commands: orthogonal velocity = v - projection of v onto the command direction.
    cmd_dir = cmd_xy / (cmd_norm + 1e-8)
    vel_parallel = torch.sum(vel_xy * cmd_dir, dim=1, keepdim=True) * cmd_dir
    vel_orthogonal = vel_xy - vel_parallel
    v_o_sq = torch.sum(vel_orthogonal**2, dim=1)

    # For stop commands (small command norm): penalize all planar velocity.
    is_stop = cmd_norm.squeeze(-1) < 0.1
    v_o_sq = torch.where(is_stop, torch.sum(vel_xy**2, dim=1), v_o_sq)

    return torch.exp(-scale * v_o_sq)


def feet_air_time_with_inplace(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    t_swing_target: float = 0.25,
    t_stance_cmd_norm: float = 0.5,
    t_stance_range: tuple[float, float] = (0.1, 0.5),
    inplace_upper_bound: float = 0.3,
    inplace_t_diff_range: tuple[float, float] = (-0.3, 0.3),
    t_swing_stance_tol: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of desired swing/stance times, with dedicated handling of in-place commands.

    Three cases are handled per foot:
      1. In-place commands (low command norm): reward stance minus swing time.
      2. In-contact: reward achieving the target stance time.
      3. In-air: reward achieving the target swing time.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    t_swing = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    t_stance = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_filtered = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    )

    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    inplace_cmd = (cmd_norm < inplace_upper_bound).unsqueeze(1)

    # Case 1: in-place - prefer longer stance, shorter swing.
    inplace_rew = torch.clip(t_stance - t_swing, min=inplace_t_diff_range[0], max=inplace_t_diff_range[1])

    # Target stance time from command (inverse relationship: faster command -> shorter stance).
    t_stance_target = torch.clip(
        t_stance_cmd_norm / cmd_norm,
        min=t_stance_range[0],
        max=t_stance_range[1],
    ).unsqueeze(1)

    # Case 2: in-contact - reward achieving target stance time.
    in_contact_rew = torch.min(t_stance, t_stance_target) * (t_stance < (t_stance_target + t_swing_stance_tol))

    # Case 3: in-air - reward achieving target swing time.
    no_contact_rew = torch.min(t_swing, torch.tensor(t_swing_target, device=env.device)) * (
        t_swing < (t_swing_target + t_swing_stance_tol)
    )

    airtime_rew = inplace_cmd * inplace_rew + ~inplace_cmd * (
        in_contact_rew * contact_filtered + no_contact_rew * ~contact_filtered
    )

    return torch.sum(airtime_rew, dim=1)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward orientation tracking of a pose command with a tanh kernel.

    The orientation counterpart of ``position_command_error_tanh``, giving orientation a shaped
    bonus rather than a linear penalty whose gradient goes flat at the goal.

    Args:
        env: The manager-based RL environment.
        std: Tanh kernel width in radians. The reward is ~0.76 at an error of ``std``.
        command_name: Pose command term, read as ``[pos_b (3), quat_b (4)]``.
        asset_cfg: The asset and the single tracked body.

    Returns
    -------
        ``1 - tanh(orientation_error / std)`` in [0, 1]. Shape: (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore[index]
    command = env.command_manager.get_command(command_name)

    des_quat_w = math_utils.quat_mul(asset.data.root_quat_w, command[:, 3:7])
    curr_quat_w = asset.data.body_quat_w[:, body_id]
    return 1 - torch.tanh(math_utils.quat_error_magnitude(curr_quat_w, des_quat_w) / std)
