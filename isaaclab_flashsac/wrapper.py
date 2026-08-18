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

"""RSL-RL vecenv wrapper that exposes pre-reset observations for truncated episodes."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from tensordict import TensorDict


class FlashSACVecEnvWrapper(RslRlVecEnvWrapper):
    """Vecenv wrapper providing ``extras["time_outs_obs"]`` and physics-based action bounds.

    Isaac Lab resets done environments inside ``step()`` and recomputes observations afterwards,
    so the returned observation of a done environment is already the post-reset frame. FlashSAC
    bootstraps truncated (timed-out) episodes from the true final observation instead; this
    wrapper caches it by intercepting ``_reset_idx`` — the last point where the simulator still
    holds the terminal state — and republishes it as ``extras["time_outs_obs"]``.

    Additionally, ``action_bias``/``action_scale`` are computed from the robot's soft joint
    position limits (relative to the default pose, divided by the action scale) and consumed by
    ``FlashSAC.construct_algorithm`` for the Tanh policy's affine action scaling.
    """

    action_bias: torch.Tensor | None
    action_scale: torch.Tensor | None

    def __init__(
        self,
        env: Any,
        clip_actions: float | None = None,
        action_bound: str = "joint_limit",
        action_bound_scale: float = 1.0,
    ) -> None:
        super().__init__(env, clip_actions)

        self._final_obs_buf: dict[str, torch.Tensor] | None = None
        base_env = self.unwrapped
        if hasattr(base_env, "_reset_idx") and hasattr(base_env, "observation_manager"):
            base_env._reset_idx = partial(self._reset_idx_with_final_obs, base_env, base_env._reset_idx)
        else:
            warnings.warn(
                "FlashSACVecEnvWrapper: the environment exposes no _reset_idx/observation_manager;"
                " extras['time_outs_obs'] will not be provided and truncated episodes bootstrap"
                " from post-reset observations.",
                stacklevel=2,
            )

        if action_bound == "scalar":
            # +-action_bound_scale bounds
            self.action_bias = torch.zeros(self.num_actions, device=self.device)
            self.action_scale = torch.full((self.num_actions,), action_bound_scale, device=self.device)
            base_env._action_scale = self.action_scale
            return
        if action_bound != "joint_limit":
            raise ValueError(f"Unknown action_bound '{action_bound}' (expected 'joint_limit' or 'scalar').")
        self.action_bias, self.action_scale = self._compute_action_scaling(base_env)
        if self.action_bias is not None and self.action_bias.numel() != self.num_actions:
            warnings.warn(
                f"FlashSACVecEnvWrapper: robot has {self.action_bias.numel()} joints but the env"
                f" has {self.num_actions} actions; joint-limit action scaling does not apply and"
                f" FlashSAC falls back to scalar bounds (±{action_bound_scale}).",
                stacklevel=2,
            )
            self.action_bias, self.action_scale = None, None
        if self.action_bias is None:
            self.action_bias = torch.zeros(self.num_actions, device=self.device)
            self.action_scale = torch.full((self.num_actions,), action_bound_scale, device=self.device)

        # Read by `mdp.rewards.regularization.action_rate_l2` to recover the raw [-1, 1] action.
        base_env._action_scale = self.action_scale

    @staticmethod
    def _compute_action_scaling(base_env: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute the affine action (bias, range) from the robot's soft joint position limits.

        Zero bias and a symmetric per-joint range of max(upper, lower) distance from the default
        joint position to the soft limits, divided by the action manager's action scale, so that
        tanh(0) is exactly the default pose and ±1 reaches the farther soft limit (the nearer one
        can be overshot).
        """
        if not hasattr(base_env, "scene") or "robot" not in base_env.scene.keys():
            warnings.warn(
                "FlashSACVecEnvWrapper: no 'robot' in env.scene; action_bias/action_scale are not"
                " computed and FlashSAC falls back to identity action scaling.",
                stacklevel=2,
            )
            return None, None

        robot = base_env.scene["robot"]
        lower_limits = robot.data.soft_joint_pos_limits[0, :, 0]
        upper_limits = robot.data.soft_joint_pos_limits[0, :, 1]
        default_pos = robot.data.default_joint_pos[0]

        # Action scale from the action manager: scalar, or per-joint (dict cfg ->
        # the term's applied ``_scale`` tensor, scattered onto robot joint order).
        term_action_scale: float | torch.Tensor = 1.0
        if hasattr(base_env, "action_manager"):
            for term in base_env.action_manager._terms.values():
                if hasattr(term.cfg, "scale"):
                    if isinstance(term.cfg.scale, (float, int)):
                        term_action_scale = term.cfg.scale
                        break
                    applied = getattr(term, "_scale", None)
                    if isinstance(applied, torch.Tensor):
                        vec = torch.ones_like(default_pos)
                        ids = term._joint_ids
                        row = applied[0].to(device=default_pos.device, dtype=default_pos.dtype)
                        if isinstance(ids, slice):
                            vec = row.clone()
                        else:
                            vec[torch.as_tensor(ids, device=default_pos.device)] = row
                        term_action_scale = vec
                        break
                    raise NotImplementedError(
                        "FlashSACVecEnvWrapper: action scale is neither a scalar nor an applied per-joint tensor."
                    )

        upper = torch.abs(upper_limits - default_pos) / term_action_scale
        lower = torch.abs(lower_limits - default_pos) / term_action_scale
        action_bias = torch.zeros_like(upper)
        action_scale = torch.maximum(upper, lower)

        # Joints without finite soft limits (e.g. continuous joints) fall back to identity
        # scaling.
        finite = torch.isfinite(action_bias) & torch.isfinite(action_scale) & (action_scale > 0)
        if not bool(finite.all()):
            warnings.warn(
                f"FlashSACVecEnvWrapper: {int((~finite).sum())} joint(s) have no finite soft"
                " position limits; using identity action scaling for them.",
                stacklevel=2,
            )
            action_bias = torch.where(finite, action_bias, torch.zeros_like(action_bias))
            action_scale = torch.where(finite, action_scale, torch.ones_like(action_scale))

        print("FlashSACVecEnvWrapper: computed physics-based action scaling.")
        print(f"  action scale: {term_action_scale}, bias: {action_bias}, range: {action_scale}")
        return action_bias, action_scale

    def _reset_idx_with_final_obs(
        self,
        base_env: Any,
        orig_reset_idx: Any,
        env_ids: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Cache the terminal observations of the environments about to be reset."""
        if env_ids is None:
            env_ids = slice(None)
        terminal = base_env.observation_manager.compute()
        # Only flat (concatenated) observation groups are cached; nested term dicts are skipped.
        terminal = {key: value for key, value in terminal.items() if isinstance(value, torch.Tensor)}
        if self._final_obs_buf is None:
            self._final_obs_buf = {key: torch.zeros_like(value) for key, value in terminal.items()}
        for key, value in terminal.items():
            self._final_obs_buf[key][env_ids] = value[env_ids]
        return orig_reset_idx(env_ids, *args, **kwargs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment, adding ``time_outs_obs`` to the extras when available.

        The policy emits normalized [-1, 1] actions; the affine joint-limit/scalar scaling is
        applied here, at the env boundary, as ``clamp(a, -1, 1) * action_bounds``.
        """
        if (
            self.action_bias is not None
            and self.action_scale is not None
            and self.action_scale.numel() == actions.shape[-1]
        ):
            actions = self.action_bias + self.action_scale * torch.clamp(actions, -1.0, 1.0)
        obs, rew, dones, extras = super().step(actions)
        if "time_outs" in extras and self._final_obs_buf is not None:
            extras["time_outs_obs"] = TensorDict(
                {key: value.clone() for key, value in self._final_obs_buf.items()},
                batch_size=[self.num_envs],
            )
        return obs, rew, dones, extras
