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

"""Left-right symmetry tables for the Unitree G1, shared by the per-task mirrors.

Joints and tracked bodies are resolved by name, so the tables survive a change in the asset's
index order. The per-task observation-group mirrors live in :mod:`.dreamwaq` and :mod:`.wbt`.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

__all__ = [
    "G1SymmetryCache",
    "get_symmetry_cache",
    "switch_g1_joints_left_right",
    "switch_tracked_bodies_left_right",
]

LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [name.replace("left_", "right_") for name in LEFT_LEG_JOINTS]
# Sign flip per leg joint, in the order above (same for both sides).
LEG_SIGNS = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0]

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [name.replace("left_", "right_") for name in LEFT_ARM_JOINTS]
# Sign flip per arm joint, in the order above (same for both sides).
ARM_SIGNS = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_SIGNS = [-1.0, -1.0, 1.0]


def _resolve_robot(env: Any) -> Any:
    """Return the articulation to query, unwrapping an RSL-RL VecEnv wrapper if needed."""
    scene_env = env if hasattr(env, "scene") else env.unwrapped
    return scene_env.scene["robot"]


def _find_joints_ordered(robot: Any, names: list[str]) -> list[int]:
    """Resolve joint indices by name, asserting the returned order matches ``names``.

    ``find_joints`` returns matches in asset order, which coincides with the query order for
    the G1 groups above -- asserted rather than assumed.
    """
    indices, found_names = robot.find_joints(names)
    assert list(found_names) == list(names), (
        f"find_joints returned a different order than queried (queried {names}, got "
        f"{list(found_names)}); update the joint tables before trusting the mirror."
    )
    return list(indices)


def _resolve_command(env: Any) -> Any:
    """Return the motion command term, unwrapping an RSL-RL VecEnv wrapper if needed."""
    scene_env = env if hasattr(env, "command_manager") else env.unwrapped
    return scene_env.command_manager.get_term("motion")


def _left_right_permutation(names: list[str]) -> list[int]:
    """Index permutation swapping ``left_*``/``right_*`` entries; centerline names map to self."""
    perm: list[int] = []
    for name in names:
        if name.startswith("left_"):
            counterpart = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            counterpart = "left_" + name.removeprefix("right_")
        else:
            perm.append(names.index(name))
            continue
        assert counterpart in names, (
            f"'{name}' has no left/right counterpart '{counterpart}' in {names}; the mirror transform needs both sides."
        )
        perm.append(names.index(counterpart))
    return perm


class G1SymmetryCache:
    """Left-right permutation/sign tables for one environment.

    Joints come from the robot articulation, which every G1 env has. Tracked bodies come from
    the motion command term, which only the tracking task has, so they are resolved on first
    use rather than in ``__init__``.
    """

    def __init__(self, env: Any) -> None:
        """Resolve and cache the joint permutation/sign tensors.

        Args:
            env: The environment (or VecEnv wrapper) exposing the robot articulation, either
                directly via ``.scene["robot"]`` or via ``.unwrapped.scene["robot"]``.
        """
        robot = _resolve_robot(env)
        self.device = env.device
        num_joints = len(robot.data.joint_names)
        # Weak: the table below holds this cache, so a strong ref here would keep the env
        # alive forever and defeat the weak keying.
        self._env_ref = weakref.ref(env)
        self._body_perm: torch.Tensor | None = None
        self._num_bodies = 0

        left_leg = _find_joints_ordered(robot, LEFT_LEG_JOINTS)
        right_leg = _find_joints_ordered(robot, RIGHT_LEG_JOINTS)
        left_arm = _find_joints_ordered(robot, LEFT_ARM_JOINTS)
        right_arm = _find_joints_ordered(robot, RIGHT_ARM_JOINTS)
        waist = _find_joints_ordered(robot, WAIST_JOINTS)

        # Legs + arms + waist must cover every joint exactly once; anything else is a robot
        # variant these tables were not built for.
        all_indices = left_leg + right_leg + left_arm + right_arm + waist
        assert sorted(all_indices) == list(range(num_joints)), (
            f"expected legs+arms+waist to cover all {num_joints} joints exactly once, got "
            f"{sorted(all_indices)}; this is not the 29-DoF G1 layout."
        )

        # ``output[..., i] = input[..., joint_perm[i]]`` swaps left<->right (identity elsewhere);
        # the sign is applied after the permutation.
        joint_perm = torch.arange(num_joints, device=self.device)
        joint_perm[left_leg] = torch.tensor(right_leg, device=self.device)
        joint_perm[right_leg] = torch.tensor(left_leg, device=self.device)
        joint_perm[left_arm] = torch.tensor(right_arm, device=self.device)
        joint_perm[right_arm] = torch.tensor(left_arm, device=self.device)
        self.joint_perm = joint_perm

        joint_sign = torch.ones(num_joints, device=self.device, dtype=torch.float32)
        leg_signs = torch.tensor(LEG_SIGNS, device=self.device, dtype=torch.float32)
        joint_sign[left_leg] = leg_signs
        joint_sign[right_leg] = leg_signs
        arm_signs = torch.tensor(ARM_SIGNS, device=self.device, dtype=torch.float32)
        joint_sign[left_arm] = arm_signs
        joint_sign[right_arm] = arm_signs
        joint_sign[waist] = torch.tensor(WAIST_SIGNS, device=self.device, dtype=torch.float32)
        self.joint_sign = joint_sign

        self.num_joints = num_joints

    def _resolve_bodies(self) -> None:
        env = self._env_ref()
        assert env is not None, "the environment this symmetry cache was built for is gone"
        body_names = list(_resolve_command(env).cfg.body_names)
        self._body_perm = torch.tensor(_left_right_permutation(body_names), dtype=torch.long)
        self._num_bodies = len(body_names)

    @property
    def body_perm(self) -> torch.Tensor:
        """Left-right permutation over the motion command's tracked-body list."""
        if self._body_perm is None:
            self._resolve_bodies()
        assert self._body_perm is not None
        return self._body_perm

    @property
    def num_bodies(self) -> int:
        """Number of tracked bodies in the motion command."""
        if self._body_perm is None:
            self._resolve_bodies()
        return self._num_bodies


# Keyed by the env object itself, so an entry is dropped once its env is collected.
_SYMMETRY_CACHE: weakref.WeakKeyDictionary[Any, G1SymmetryCache] = weakref.WeakKeyDictionary()


def get_symmetry_cache(env: Any) -> G1SymmetryCache:
    """Get or create the per-env symmetry cache for ``env``."""
    cache = _SYMMETRY_CACHE.get(env)
    if cache is None:
        cache = G1SymmetryCache(env)
        _SYMMETRY_CACHE[env] = cache
    return cache


def switch_g1_joints_left_right(env: Any, joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right joints and flip their signs.

    Args:
        env: The environment the joint symmetry cache is built against.
        joint_data: Joint data tensor of shape ``(..., num_joints)`` (e.g. joint pos/vel/action,
            optionally with extra leading dims such as a stacked history axis).

    Returns
    -------
        The transformed tensor, same shape as ``joint_data``.
    """
    cache = get_symmetry_cache(env)
    assert joint_data.shape[-1] == cache.num_joints, (
        f"expected last dim {cache.num_joints}, got {joint_data.shape[-1]}."
    )
    # The cache lives on env.device, but replay batches may not (buffer_device="cpu" with a
    # CUDA env) -- follow the data.
    perm = cache.joint_perm.to(joint_data.device)
    sign = cache.joint_sign.to(joint_data.device)
    return joint_data[..., perm] * sign


def switch_tracked_bodies_left_right(env: Any, body_data: torch.Tensor, dim_per_body: int) -> torch.Tensor:
    """Swap left/right tracked-body blocks of a flattened per-body tensor.

    Args:
        env: The environment the body symmetry cache is built against.
        body_data: Tensor of shape ``(B, num_bodies * dim_per_body)``, bodies in the motion
            command's ``body_names`` order, ``dim_per_body`` consecutive values per body.
        dim_per_body: Width of one body block (3 for positions, 6 for orientations).

    Returns
    -------
        The permuted tensor, same shape as ``body_data``.
    """
    cache = get_symmetry_cache(env)
    batch = body_data.shape[0]
    expected = cache.num_bodies * dim_per_body
    assert body_data.shape[-1] == expected, (
        f"expected last dim {expected} ({cache.num_bodies} bodies x {dim_per_body}), got {body_data.shape[-1]}."
    )
    perm = cache.body_perm.to(body_data.device)
    return body_data.reshape(batch, cache.num_bodies, dim_per_body)[:, perm].reshape(batch, expected)
