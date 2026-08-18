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

"""Contact-sensor-driven hardware/collision-protection reward terms for the G1 DreamWaQ task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize having no foot in contact with the ground (a flight phase)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces (large horizontal contact force relative to vertical)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    ).float()


def feet_yaw_drag(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize yaw angular velocity of feet while in contact (discourage pivoting/scrubbing)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > contact_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    return torch.sum(torch.abs(body_ang_vel) * contacts, dim=1)


def body_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    """Penalize excessive vertical contact force above ``threshold`` (clamped to ``max_reward``)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_too_near_humanoid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    """Penalize feet that get closer together (laterally, in the base frame) than ``threshold``."""
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    # Vector from left foot to right foot in world frame, expressed in the base frame.
    feet_diff = feet_pos[:, 1, :] - feet_pos[:, 0, :]
    feet_diff_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, feet_diff)

    lateral_distance = torch.abs(feet_diff_b[:, 1])
    return (threshold - lateral_distance).clamp(min=0)


def feet_impact_velocity_delta_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    delta_v_max: float = 1.0,
) -> torch.Tensor:
    """Penalize sudden changes in feet z-velocity (impact suppression): ``sum_i min(dv_z^2, dv_max^2)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_feet_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]

    # First call: initialize the buffer and return zero.
    if not hasattr(env, "_prev_feet_vel_z"):
        env._prev_feet_vel_z = current_feet_vel_z.clone()
        return torch.zeros(env.num_envs, device=env.device)

    delta_vel_z = current_feet_vel_z - env._prev_feet_vel_z
    env._prev_feet_vel_z = current_feet_vel_z.clone()

    delta_v_max_sq = delta_v_max**2
    penalty = torch.minimum(torch.square(delta_vel_z), torch.tensor(delta_v_max_sq, device=env.device))
    return torch.sum(penalty, dim=1)
