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

"""Left-right symmetry augmentation for the G1 DreamWaQ observation groups.

Group layouts, with ``n`` robot joints, ``F`` history frames, and ``S`` height-scan rays:

- ``current``    ``(B, 12 + 3n)``: ``[lin_vel(3) | ang_vel(3) | grav(3) | joint_pos(n) |
  joint_vel(n) | action(n) | command(3)]``.
- ``measurable`` ``(B, F * (6 + 3n))``: ``F`` frames (oldest first) of ``current`` without
  lin_vel/command, each term block flattened frame-major.
- ``critic``     ``(B, 12 + 3n + 2 [+ S])``: ``current`` plus ``feet_contact(2)`` and, on rough
  terrain, ``height_scan(S)``.

``n`` and ``F`` are derived from the joint cache and the tensor's own width; nothing is
hardcoded except the assertions that check the derived widths line up.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from isaaclab_flashsac.mdp.obs.symmetry.g1.base import get_symmetry_cache, switch_g1_joints_left_right

__all__ = ["compute_symmetric_states"]

# Sign flips about the sagittal (x-z) plane for the 3-vector observation terms.
_LIN_VEL_SIGN = [1.0, -1.0, 1.0]
_ANG_VEL_SIGN = [-1.0, 1.0, -1.0]
_GRAV_SIGN = [1.0, -1.0, 1.0]
_COMMAND_SIGN = [1.0, -1.0, -1.0]  # (vx, vy, wz) -> (vx, -vy, -wz)

# Height-scan ray grid: GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) with the default
# ordering="xy" flattens row-major to (num_y, num_x) -- lateral rows of forward columns. The y
# samples are symmetric about 0, so a left-right mirror is exactly a flip of the lateral axis.
_SCAN_RESOLUTION = 0.1
_SCAN_SIZE = (1.6, 1.0)  # (length/x, width/y)
_SCAN_NUM_X = round(_SCAN_SIZE[0] / _SCAN_RESOLUTION) + 1  # 17 rays along x (forward)
_SCAN_NUM_Y = round(_SCAN_SIZE[1] / _SCAN_RESOLUTION) + 1  # 11 rays along y (lateral)


def _sign(values: list[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


def _mirror_current_layout(env: Any, x: torch.Tensor) -> torch.Tensor:
    """Mirror a ``(..., 12 + 3n)`` ``current``-layout tensor about the sagittal (x-z) plane."""
    cache = get_symmetry_cache(env)
    n = cache.num_joints
    expected = 12 + 3 * n
    assert x.shape[-1] == expected, f"expected last dim {expected} (12 + 3*{n} joints), got {x.shape[-1]}"

    lin_vel, ang_vel, grav, jpos, jvel, act, cmd = torch.split(x, [3, 3, 3, n, n, n, 3], dim=-1)
    lin_vel = lin_vel * _sign(_LIN_VEL_SIGN, x.device, x.dtype)
    ang_vel = ang_vel * _sign(_ANG_VEL_SIGN, x.device, x.dtype)
    grav = grav * _sign(_GRAV_SIGN, x.device, x.dtype)
    jpos = switch_g1_joints_left_right(env, jpos)
    jvel = switch_g1_joints_left_right(env, jvel)
    act = switch_g1_joints_left_right(env, act)
    cmd = cmd * _sign(_COMMAND_SIGN, x.device, x.dtype)
    return torch.cat([lin_vel, ang_vel, grav, jpos, jvel, act, cmd], dim=-1)


def _mirror_measurable(env: Any, x: torch.Tensor) -> torch.Tensor:
    """Mirror a ``(B, F * (6 + 3n))`` ``measurable``-layout tensor (F stacked history frames)."""
    cache = get_symmetry_cache(env)
    n = cache.num_joints
    per_frame = 6 + 3 * n  # ang_vel(3) + grav(3) + joint_pos(n) + joint_vel(n) + action(n)
    total = x.shape[-1]
    assert total % per_frame == 0, f"measurable width {total} is not a multiple of per-frame {per_frame}"
    num_frames = total // per_frame
    batch = x.shape[0]

    ang_w, grav_w, joint_w = 3 * num_frames, 3 * num_frames, n * num_frames
    ang, grav, jpos, jvel, act = torch.split(x, [ang_w, grav_w, joint_w, joint_w, joint_w], dim=-1)

    # Each block is frame-major ([frame0 | frame1 | ... | frame_{F-1}], oldest-first); reshape to
    # (B, num_frames, term_dim), transform per-frame, flatten back.
    ang = (ang.reshape(batch, num_frames, 3) * _sign(_ANG_VEL_SIGN, x.device, x.dtype)).reshape(batch, ang_w)
    grav = (grav.reshape(batch, num_frames, 3) * _sign(_GRAV_SIGN, x.device, x.dtype)).reshape(batch, grav_w)
    jpos = switch_g1_joints_left_right(env, jpos.reshape(batch, num_frames, n)).reshape(batch, joint_w)
    jvel = switch_g1_joints_left_right(env, jvel.reshape(batch, num_frames, n)).reshape(batch, joint_w)
    act = switch_g1_joints_left_right(env, act.reshape(batch, num_frames, n)).reshape(batch, joint_w)
    return torch.cat([ang, grav, jpos, jvel, act], dim=-1)


def _mirror_height_scan(x: torch.Tensor) -> torch.Tensor:
    """Mirror a flattened height-scan ray grid about the robot's lateral (y) axis."""
    expected = _SCAN_NUM_X * _SCAN_NUM_Y
    if x.shape[-1] != expected:
        raise NotImplementedError(
            f"height-scan symmetry expects the {_SCAN_NUM_Y}x{_SCAN_NUM_X}={expected}-ray grid "
            f"(GridPatternCfg(resolution={_SCAN_RESOLUTION}, size={_SCAN_SIZE}), ordering='xy'), "
            f"got {x.shape[-1]} rays; the ordering of another grid is not derived here."
        )
    batch = x.shape[0]
    grid = x.reshape(batch, _SCAN_NUM_Y, _SCAN_NUM_X)
    return grid.flip(dims=[1]).reshape(batch, expected)


def _mirror_critic(env: Any, x: torch.Tensor) -> torch.Tensor:
    """Mirror a ``(B, 12 + 3n + 2 [+ S])`` ``critic``-layout tensor."""
    cache = get_symmetry_cache(env)
    n = cache.num_joints
    current_width = 12 + 3 * n
    assert x.shape[-1] >= current_width + 2, (
        f"critic width {x.shape[-1]} is smaller than current + feet_contact ({current_width + 2})"
    )

    current_part = _mirror_current_layout(env, x[..., :current_width])
    feet_contact = x[..., current_width : current_width + 2].flip(dims=[-1])
    parts = [current_part, feet_contact]

    scan = x[..., current_width + 2 :]
    if scan.shape[-1] > 0:
        parts.append(_mirror_height_scan(scan))
    return torch.cat(parts, dim=-1)


_OBS_GROUP_TRANSFORMS = {
    "current": _mirror_current_layout,
    "measurable": _mirror_measurable,
    "critic": _mirror_critic,
}


@torch.no_grad()
def compute_symmetric_states(
    env: Any,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augment observations/actions with their left-right mirrored counterparts.

    Returns ``[original; mirrored]`` stacked along the batch dimension (2x the input batch size),
    matching the ``rsl_rl.extensions.Symmetry`` data-augmentation-function contract (this function
    is used directly as ``symmetry_cfg["data_augmentation_func"]``).

    Args:
        env: The environment (or RSL-RL VecEnv wrapper) the joint symmetry cache is built
            against.
        obs: TensorDict with a subset of the "current"/"measurable"/"critic" groups. ``None``
            skips observation augmentation.
        actions: ``(B, num_joints)`` action tensor. ``None`` skips action augmentation.

    Returns
    -------
        A tuple ``(obs_aug, actions_aug)``; either element is ``None`` if the respective input
        was ``None``.

    Raises
    ------
    NotImplementedError
        If ``obs`` contains a group with no defined symmetry transform, or a ``critic`` height
        scan whose ray grid does not match the supported (derived) geometry.
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
