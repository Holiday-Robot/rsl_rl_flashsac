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

"""Feet contact observation term for the G1 DreamWaQ velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5) -> torch.Tensor:
    """Binary foot contact flags from the contact sensor (one flag per selected body, 2 dims for G1).

    Args:
        env: The environment instance.
        sensor_cfg: Contact sensor configuration selecting the feet bodies.
        threshold: Force-magnitude threshold above which a foot is considered in contact. The
            reference implementation's ``FeetContactWrapper`` hardcodes this at 0.5 N (distinct
            from the 1.0 N ``contact_threshold`` used by the reward terms' own contact checks).

    Returns
    -------
        A ``(num_envs, num_selected_bodies)`` float tensor of 0/1 contact flags.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: (num_envs, history_length, num_bodies, 3); take the latest frame (index 0).
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > threshold).float()
