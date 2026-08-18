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

"""Sim2real deployment config writer and export self-verification for the G1 DreamWaQ policy.

Writes a deployment-runtime-compatible ``config.yaml`` next to the exported ``policy.pt``/``cenet.pt``
TorchScript files (mirrors the reference robot-interface's config key names), and
provides a round-trip check (:func:`verify_exported_pair`) that the two exported networks compose to
reproduce the live policy's action -- the exact composition the deployment side performs at
inference time.
"""

from __future__ import annotations

import copy
import os
from datetime import datetime
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any

import torch
import yaml

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab_flashsac.rl_cfg import FlashSACRunnerCfg


def _resolve_action_term_joints(unwrapped_env: Any) -> tuple[list[int], list[str]]:
    """Return the policy-order joint ``(ids, names)`` resolved by the ``joint_pos`` action term.

    The action term restricts the articulation's full joint set (which may include joints the
    policy never controls, e.g. G1's Dex3 hand joints) down to the ones it actually actuates, in
    resolution order. This resolved order -- not ``robot.data.joint_names`` -- is the policy's
    joint-order contract.
    """
    action_term = unwrapped_env.action_manager.get_term("joint_pos")
    joint_ids = action_term._joint_ids
    joint_names = list(action_term._joint_names)
    if isinstance(joint_ids, slice):
        # The term matched every joint on the articulation (e.g. joint_names=[".*"]).
        num_joints = unwrapped_env.scene["robot"].data.default_joint_pos.shape[-1]
        joint_ids = list(range(num_joints))
    else:
        joint_ids = list(joint_ids)
    return joint_ids, joint_names


def _resolve_action_scale(action_term: Any, num_joints: int) -> torch.Tensor:
    """Read the ``joint_pos`` action term's scale as a per-robot-joint vector.

    Mirrors ``isaaclab_flashsac.wrapper.FlashSACVecEnvWrapper._compute_action_scaling``:
    a scalar cfg scale broadcasts to every joint; a dict cfg scale is read from the
    term's applied ``_scale`` tensor (env row 0) and scattered onto the robot's
    joint ordering via the term's ``_joint_ids``.
    """
    cfg_scale = action_term.cfg.scale
    if isinstance(cfg_scale, (float, int)):
        return torch.full((num_joints,), float(cfg_scale))
    applied = getattr(action_term, "_scale", None)
    if isinstance(applied, torch.Tensor):
        vec = torch.ones(num_joints, dtype=applied.dtype)
        ids = action_term._joint_ids
        row = applied[0].detach().cpu()
        if isinstance(ids, slice):
            vec = row.clone()
        else:
            vec[torch.as_tensor(ids)] = row
        return vec
    raise NotImplementedError("deploy_export: action scale is neither a scalar nor an applied per-joint tensor.")


def _build_dof_params(robot: Any, joint_ids: list[int], joint_names: list[str], action_scale: torch.Tensor) -> dict:
    """Build the per-joint ``dof_params`` block, keyed by full joint name in policy order.

    Uses the actuator-model values (``joint_stiffness``/``joint_damping``, as applied to the
    simulation) rather than the USD defaults (``default_joint_stiffness``/``default_joint_damping``),
    and the soft joint position limits (a sub-region of the hard USD limits) as ``lower``/``upper``.
    """
    default_pos = robot.data.default_joint_pos[0, joint_ids]
    stiffness = robot.data.joint_stiffness[0, joint_ids]
    damping = robot.data.joint_damping[0, joint_ids]
    effort_limits = robot.data.joint_effort_limits[0, joint_ids]
    lower_limits = robot.data.soft_joint_pos_limits[0, joint_ids, 0]
    upper_limits = robot.data.soft_joint_pos_limits[0, joint_ids, 1]

    dof_params = {}
    for i, name in enumerate(joint_names):
        dof_params[name] = {
            "default": float(default_pos[i]),
            "p": float(stiffness[i]),
            "d": float(damping[i]),
            "scale": float(action_scale[joint_ids[i]]),
            "torque": float(effort_limits[i]),
            "lower": float(lower_limits[i]),
            "upper": float(upper_limits[i]),
        }
    return dof_params


def _build_provenance(agent_cfg: FlashSACRunnerCfg) -> dict:
    """Build the export provenance block: run name, git SHA, export date, and library versions."""
    # Lazy import: gitpython is an undeclared dependency (only present transitively via wandb),
    # and is only needed here at export time, not for importing this module/package.
    from git import Repo

    repo = Repo(os.path.dirname(__file__), search_parent_directories=True)
    git_sha = repo.head.commit.hexsha

    return {
        "run_name": agent_cfg.run_name,
        "git_sha": git_sha,
        "export_date": datetime.now().isoformat(),
        "versions": {
            "isaaclab": importlib_metadata.version("isaaclab"),
            "rsl_rl": importlib_metadata.version("rsl-rl-lib"),
            # str(): torch.__version__ is a TorchVersion (str subclass) that some PyYAML versions
            # cannot represent via the plain-str representer.
            "torch": str(torch.__version__),
            "mujoco": "n/a (training export)",
        },
    }


def write_deploy_config(env: Any, agent_cfg: FlashSACRunnerCfg, path: str, filename: str = "config.yaml") -> None:
    """Write the deployment-runtime-compatible config next to the exported networks.

    Mirrors the key names read by the reference robot-interface's config schema
    (``policy``/``cenet``/``obs.scales``/``clip``), plus this sim2real path's additions: an
    explicit ``joint_order`` (the policy-order joint-name contract the deployment runtime uses to
    build its SDK<->policy index maps), per-joint ``lower``/``upper`` soft limits under
    ``dof_params``, ``action_bound``, and a ``provenance`` block.

    Args:
        env: The (possibly wrapped) play/training environment; ``env.unwrapped`` must expose
            ``scene["robot"]``, ``action_manager``, and ``observation_manager``.
        agent_cfg: The resolved runner configuration for this run (a
            :class:`~isaaclab_flashsac.rl_cfg.FlashSACRunnerCfg` with a DreamWaQ actor cfg).
        path: Directory to write the config into (created if missing).
        filename: Output file name. Defaults to ``"config.yaml"``.
    """
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    joint_ids, joint_names = _resolve_action_term_joints(unwrapped)
    num_joints = (
        len(robot.data.joint_names) if hasattr(robot.data, "joint_names") else robot.data.default_joint_pos.shape[1]
    )
    action_scale = _resolve_action_scale(unwrapped.action_manager.get_term("joint_pos"), num_joints)

    obs = unwrapped.observation_manager.compute()
    current_dim = obs["current"].shape[-1]
    measurable_dim = obs["measurable"].shape[-1]
    history_length = unwrapped.observation_manager.cfg.measurable.history_length
    measurement_size = measurable_dim // history_length
    dof_vel_scale = unwrapped.observation_manager.cfg.measurable.joint_vel.scale

    latent_dim = agent_cfg.actor.cenet_latent_dim  # type: ignore[attr-defined]
    estimation_dim = agent_cfg.actor.cenet_estimation_dim  # type: ignore[attr-defined]

    data = {
        "task_name": agent_cfg.experiment_name,
        "policy": {
            "input_size": current_dim + latent_dim,
            "command_size": 3,
            "num_actions": len(joint_names),
        },
        "cenet": {
            "measurement_size": measurement_size,
            "framestack": history_length,
            "latent_size": latent_dim,
            "estimation_size": estimation_dim,
        },
        "obs": {
            "scales": {
                "linvel": 1.0,
                "angvel": 1.0,
                "action": 1.0,
                "dof_vel": float(dof_vel_scale),
            }
        },
        "clip": {"observation": 100.0, "action": 100.0},
        "runtime": {"policy_rate": 50, "low_level_rate": 200, "action_upsampling": True},
        "action_bound": "joint_limits",
        "joint_order": joint_names,
        "dof_params": _build_dof_params(robot, joint_ids, joint_names, action_scale),
        "provenance": _build_provenance(agent_cfg),
    }

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def verify_exported_pair(policy: Any, export_dir: str, obs: TensorDict, atol: float = 1e-4) -> None:
    """Verify that the exported ``policy.pt`` + ``cenet.pt`` reproduce the live policy's action.

    Loads the two freshly-exported TorchScript modules back on CPU and replays the deployment-side
    composition: ``cenet(measurable)`` -> splice the estimation into the leading dims of ``current``
    -> ``policy_jit(cat([current_spliced, latent]))``. This is compared against a CPU copy of
    ``policy`` run directly on the same observation slice (``policy.forward`` performs the identical
    splice internally, via ``flatten_obs``). ``policy`` itself is never moved or mutated -- it may
    still be in use for rollout on its original device.

    Args:
        policy: The live policy (e.g. as returned by ``runner.alg.get_policy()``), exposing
            ``obs_groups``, ``estimator_obs_groups``, ``cenet_estimation_dim``.
        export_dir: Directory containing the just-exported ``policy.pt`` and ``cenet.pt``.
        obs: A batched observation ``TensorDict`` (e.g. captured from ``env.get_observations()``).
            Only the first row is used, moved to CPU internally.
        atol: Absolute tolerance for the comparison. Defaults to 1e-4.

    Raises
    ------
    RuntimeError
        If the exported pair's composed action disagrees with the reference beyond ``atol``.
    """
    policy_jit = torch.jit.load(os.path.join(export_dir, "policy.pt")).eval()
    cenet_jit = torch.jit.load(os.path.join(export_dir, "cenet.pt")).eval()
    reference_policy = copy.deepcopy(policy).cpu().eval()

    obs_cpu = obs[:1].cpu()

    with torch.inference_mode():
        measurable = torch.cat([obs_cpu[group] for group in policy.estimator_obs_groups], dim=-1)
        cenet_out = cenet_jit(measurable)
        estimation = cenet_out[:, : policy.cenet_estimation_dim]
        latent = cenet_out[:, policy.cenet_estimation_dim :]

        current = torch.cat([obs_cpu[group] for group in policy.obs_groups], dim=-1).clone()
        current[:, : policy.cenet_estimation_dim] = estimation
        exported_action = policy_jit(torch.cat([current, latent], dim=-1))

        reference_action = reference_policy(obs_cpu)

    if not torch.allclose(exported_action, reference_action, atol=atol):
        raise RuntimeError("exported pair does not reproduce act_inference")
