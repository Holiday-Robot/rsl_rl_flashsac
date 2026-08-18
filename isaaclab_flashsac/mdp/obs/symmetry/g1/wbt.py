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

"""Left-right symmetry augmentation for the G1 whole-body tracking observation groups.

Positions and velocities flip sign about the sagittal (x-z) plane; orientations travel as the
first two rotation-matrix columns, whose mirror is ``_ROT6D_SIGN``. Mirroring gives a valid
transition of the same MDP: the physics, the norm-based tracking rewards, and the G1's joint
limits are all left-right symmetric.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from isaaclab_flashsac.mdp.obs.symmetry.g1.base import (
    get_symmetry_cache,
    switch_g1_joints_left_right,
    switch_tracked_bodies_left_right,
)

__all__ = ["compute_symmetric_states"]

# Sign flips about the sagittal (x-z) plane.
_LIN_VEL_SIGN = [1.0, -1.0, 1.0]
_ANG_VEL_SIGN = [-1.0, 1.0, -1.0]
_POS_SIGN = [1.0, -1.0, 1.0]
_ROT6D_SIGN = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]


def _sign(values: list[float], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=like.device, dtype=like.dtype)


def _mirror_rot6d(x: torch.Tensor) -> torch.Tensor:
    """Mirror flattened first-two-columns rotation blocks (``(..., 6k)`` for any ``k``)."""
    assert x.shape[-1] % 6 == 0, f"rot6d width {x.shape[-1]} is not a multiple of 6"
    signs = _sign(_ROT6D_SIGN, x).repeat(x.shape[-1] // 6)
    return x * signs


def _mirror_command(env: Any, cmd: torch.Tensor) -> torch.Tensor:
    """Mirror the ``[ref joint_pos(n) | ref joint_vel(n)]`` motion command block."""
    n = cmd.shape[-1] // 2
    ref_pos, ref_vel = torch.split(cmd, [n, n], dim=-1)
    ref_pos = switch_g1_joints_left_right(env, ref_pos)
    ref_vel = switch_g1_joints_left_right(env, ref_vel)
    return torch.cat([ref_pos, ref_vel], dim=-1)


def _mirror_policy(env: Any, x: torch.Tensor) -> torch.Tensor:
    """Mirror a ``policy``-layout tensor, for ``n`` joints.

    ``(B, 5n + 15)``: ``[command(2n) | anchor_pos(3) | anchor_ori(6) | lin_vel(3) | ang_vel(3) |
    joint_pos(n) | joint_vel(n) | actions(n)]``, where ``command`` is ``[ref joint_pos(n) |
    ref joint_vel(n)]``. The without-state-estimation variant drops ``anchor_pos`` and
    ``lin_vel`` (``5n + 9``); the two are told apart by width.
    """
    n = get_symmetry_cache(env).num_joints
    full_width = 5 * n + 15
    wose_width = 5 * n + 9
    if x.shape[-1] == full_width:
        cmd, anchor_pos, anchor_ori, lin_vel, ang_vel, jpos, jvel, act = torch.split(
            x, [2 * n, 3, 6, 3, 3, n, n, n], dim=-1
        )
        anchor_pos = anchor_pos * _sign(_POS_SIGN, x)
        lin_vel = lin_vel * _sign(_LIN_VEL_SIGN, x)
        head = [
            _mirror_command(env, cmd),
            anchor_pos,
            _mirror_rot6d(anchor_ori),
            lin_vel,
        ]
    elif x.shape[-1] == wose_width:
        cmd, anchor_ori, ang_vel, jpos, jvel, act = torch.split(x, [2 * n, 6, 3, n, n, n], dim=-1)
        head = [_mirror_command(env, cmd), _mirror_rot6d(anchor_ori)]
    else:
        raise NotImplementedError(
            f"policy width {x.shape[-1]} matches neither the full ({full_width}) nor the WoSE "
            f"({wose_width}) tracking layout for {n} joints."
        )
    ang_vel = ang_vel * _sign(_ANG_VEL_SIGN, x)
    jpos = switch_g1_joints_left_right(env, jpos)
    jvel = switch_g1_joints_left_right(env, jvel)
    act = switch_g1_joints_left_right(env, act)
    return torch.cat([*head, ang_vel, jpos, jvel, act], dim=-1)


def _mirror_critic(env: Any, x: torch.Tensor) -> torch.Tensor:
    """Mirror a ``critic``-layout tensor: the ``policy`` layout with ``body_pos(3N)`` and
    ``body_ori(6N)`` inserted after ``anchor_ori``, for ``N`` tracked bodies.
    """
    n = get_symmetry_cache(env).num_joints
    num_bodies = get_symmetry_cache(env).num_bodies
    expected = 5 * n + 15 + 9 * num_bodies
    assert x.shape[-1] == expected, (
        f"critic width {x.shape[-1]} does not match the layout ({expected} = 5*{n} + 15 + 9*{num_bodies})."
    )
    cmd, anchor_pos, anchor_ori, body_pos, body_ori, lin_vel, ang_vel, jpos, jvel, act = torch.split(
        x, [2 * n, 3, 6, 3 * num_bodies, 6 * num_bodies, 3, 3, n, n, n], dim=-1
    )
    body_pos = switch_tracked_bodies_left_right(env, body_pos, dim_per_body=3)
    body_pos = body_pos * _sign(_POS_SIGN, x).repeat(num_bodies)
    body_ori = _mirror_rot6d(switch_tracked_bodies_left_right(env, body_ori, dim_per_body=6))
    return torch.cat(
        [
            _mirror_command(env, cmd),
            anchor_pos * _sign(_POS_SIGN, x),
            _mirror_rot6d(anchor_ori),
            body_pos,
            body_ori,
            lin_vel * _sign(_LIN_VEL_SIGN, x),
            ang_vel * _sign(_ANG_VEL_SIGN, x),
            switch_g1_joints_left_right(env, jpos),
            switch_g1_joints_left_right(env, jvel),
            switch_g1_joints_left_right(env, act),
        ],
        dim=-1,
    )


_OBS_GROUP_TRANSFORMS = {
    "policy": _mirror_policy,
    "critic": _mirror_critic,
}


@torch.no_grad()
def compute_symmetric_states(
    env: Any,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augment tracking observations/actions with their left-right mirrored counterparts.

    Returns ``[original; mirrored]`` stacked along the batch dimension (2x the input batch size),
    matching the ``rsl_rl.extensions.Symmetry`` data-augmentation-function contract (this function
    is used directly as ``symmetry_cfg["data_augmentation_func"]``).

    Args:
        env: The tracking environment (or RSL-RL VecEnv wrapper) the joint/body symmetry caches
            are resolved against.
        obs: TensorDict with a subset of the "policy"/"critic" groups. ``None`` skips
            observation augmentation.
        actions: ``(B, num_joints)`` action tensor. ``None`` skips action augmentation.

    Returns
    -------
        A tuple ``(obs_aug, actions_aug)``; either element is ``None`` if the respective input
        was ``None``.

    Raises
    ------
    NotImplementedError
        If ``obs`` contains a group with no defined symmetry transform, or a group width that
        matches no known tracking observation layout.
    """
    obs_aug = None
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            transform = _OBS_GROUP_TRANSFORMS.get(key)
            if transform is None:
                raise NotImplementedError(f"No left-right symmetry transform defined for the obs group '{key}'.")
            obs_aug[key][:batch_size] = obs[key]
            obs_aug[key][batch_size : 2 * batch_size] = transform(env, obs[key])

    actions_aug = None
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(2 * batch_size, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = switch_g1_joints_left_right(env, actions)

    return obs_aug, actions_aug
