# Copyright (c) 2021-2026, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from RSL-RL Project (BSD-3-Clause license),
# with modifications by Holiday Robotics (BSD-3-Clause license).

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from rsl_rl_flashsac.modules.flash_sac_layers import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    FlashSACEncoder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


def _resolve_obs_dim(obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
    """Select active observation groups and compute the flattened observation dimension."""
    active_obs_groups = obs_groups[obs_set]
    obs_dim = 0
    for obs_group in active_obs_groups:
        assert len(obs[obs_group].shape) == 2, "The FlashSAC model only supports 1D observations."
        obs_dim += obs[obs_group].shape[-1]
    return active_obs_groups, obs_dim


class FlashSACActor(nn.Module):
    """FlashSAC actor: BatchNorm-embedded residual MLP trunk with a Tanh-squashed Gaussian head.

    Observation normalization is handled by the BatchNorm layers inside the network (CrossQ-style),
    so no ``EmpiricalNormalization`` is used. Batch statistics are only updated on forward passes
    with ``training=True`` (during learning updates); rollout and inference use running statistics.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_blocks: int = 2,
        hidden_dim: int = 128,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        **kwargs,
    ) -> None:
        """Initialize the FlashSAC actor model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor").
            output_dim: Dimension of the action space.
            num_blocks: Number of residual FlashSAC blocks in the trunk.
            hidden_dim: Hidden dimension of the trunk.
            log_std_min: Lower bound of the Tanh-normalized log standard deviation.
            log_std_max: Upper bound of the Tanh-normalized log standard deviation.
        """
        super().__init__()

        self.obs_groups, self.obs_dim = _resolve_obs_dim(obs, obs_groups, obs_set)
        self.output_dim = output_dim

        self.embedder = FlashSACEmbedder(input_dim=self.obs_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(
            hidden_dim=hidden_dim,
            action_dim=output_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        # Last action std computed during rollout (for logging only)
        self.last_action_std: torch.Tensor = torch.zeros(output_dim)

    @property
    def action_bias(self) -> torch.Tensor:
        return self.predictor.action_bias

    @property
    def action_scale(self) -> torch.Tensor:
        return self.predictor.action_scale

    @property
    def output_std(self) -> torch.Tensor:
        return self.last_action_std

    def set_action_scaling(self, action_bias: torch.Tensor, action_scale: torch.Tensor) -> None:
        """Populate the affine action scaling buffers (called by ``FlashSAC.construct_algorithm``)."""
        safe_action_scale = torch.clamp(action_scale.abs(), min=1e-6)
        self.predictor.action_bias.copy_(action_bias)
        self.predictor.action_scale.copy_(action_scale)
        self.predictor.log_action_scale.copy_(torch.log(safe_action_scale).sum())

    def flatten_obs(self, obs: TensorDict, training: bool = False) -> torch.Tensor:
        """Concatenate the active observation groups into a flat tensor."""
        del training  # stateless in the base model; used by the DreamWaQ variant
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)

    def get_mean_and_std(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Trunk + head forward returning the pre-squash Gaussian mean and std."""
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def sample_action_logp(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a squashed, scaled action with its corrected log-probability of shape (B,)."""
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info["log_prob"]

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference forward pass returning Tanh-squashed, scaled actions.

        Always uses BatchNorm running statistics (``training=False``). With ``stochastic_output``
        the pre-squash action is sampled with fresh Gaussian noise; otherwise the mean is used.
        """
        flat_obs = self.flatten_obs(obs)
        mean, std = self.get_mean_and_std(flat_obs, training=False)
        x_t = mean + std * torch.randn_like(mean) if stochastic_output else mean
        # Normalized [-1, 1] action; the env wrapper applies the affine scaling (deploy
        # exports bake it instead - see FlashSACActorJit).
        return torch.tanh(x_t)

    def update_normalization(self, obs: TensorDict) -> None:
        """No-op: normalization is handled by BatchNorm layers inside the network."""

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchFlashSACActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxFlashSACActor(self, verbose)


class FlashSACDreamwaqActor(FlashSACActor):
    """FlashSAC actor with a DreamWaQ-style CENet velocity estimator.

    The CENet consumes ``obs_groups["estimator"]`` (measurable obs history) and
    outputs ``[lin-vel estimation | latent]``. ``flatten_obs`` splices the
    estimation into the first ``cenet_estimation_dim`` dims of the actor obs
    and appends the latent, so the trunk input is ``current_dim + latent_dim``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_blocks: int = 2,
        hidden_dim: int = 128,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        cenet_num_blocks: int = 1,
        cenet_hidden_dim: int = 128,
        cenet_latent_dim: int = 32,
        cenet_estimation_dim: int = 3,
        **kwargs,
    ) -> None:
        """Initialize the FlashSAC DreamWaQ actor model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
                Must include an ``"estimator"`` entry (measurable obs history for the CENet).
            obs_set: Observation set to use for this model (e.g., "actor").
            output_dim: Dimension of the action space.
            num_blocks: Number of residual FlashSAC blocks in the trunk.
            hidden_dim: Hidden dimension of the trunk.
            log_std_min: Lower bound of the Tanh-normalized log standard deviation.
            log_std_max: Upper bound of the Tanh-normalized log standard deviation.
            cenet_num_blocks: Number of residual FlashSAC blocks in the CENet.
            cenet_hidden_dim: Hidden dimension of the CENet.
            cenet_latent_dim: Dimension of the CENet's unsupervised latent (appended to the trunk input).
            cenet_estimation_dim: Dimension of the CENet's supervised estimation (e.g., linear velocity),
                spliced into the first dims of the actor observation.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            **kwargs,
        )
        self.cenet_estimation_dim = cenet_estimation_dim
        self.cenet_latent_dim = cenet_latent_dim
        self.estimator_obs_groups, estimator_dim = _resolve_obs_dim(obs, obs_groups, "estimator")
        # The base embedder was built for self.obs_dim; the trunk consumes
        # self.obs_dim + latent_dim, so rebuild it (blocks/head are dim-agnostic).
        self.embedder = FlashSACEmbedder(input_dim=self.obs_dim + cenet_latent_dim, hidden_dim=hidden_dim)
        self.cenet = FlashSACEncoder(
            n_input=estimator_dim,
            hidden_dim=cenet_hidden_dim,
            latent_dim=cenet_estimation_dim + cenet_latent_dim,
            num_blocks=cenet_num_blocks,
        )

    def flatten_obs(self, obs: TensorDict, training: bool = False) -> torch.Tensor:
        """Run the CENet and splice its estimation/latent into the flattened actor observation."""
        cenet_obs = torch.cat([obs[obs_group] for obs_group in self.estimator_obs_groups], dim=-1)
        cenet_out = self.cenet(cenet_obs, training=training)
        estimation = cenet_out[:, : self.cenet_estimation_dim]
        latent = cenet_out[:, self.cenet_estimation_dim :]
        actor_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1).clone()
        actor_obs[:, : self.cenet_estimation_dim] = estimation
        return torch.cat([actor_obs, latent], dim=-1)

    def cenet_as_jit(self) -> nn.Module:
        """Return a version of the CENet estimator compatible with Torch JIT export."""
        return _TorchFlashSACCenet(self)


class FlashSACDoubleCritic(nn.Module):
    """Double-Q critic for Clipped Double Q-learning (https://arxiv.org/pdf/1802.09477v3).

    Fuses N parallel critic networks into single batched operations with a categorical
    (distributional) value head. All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ) -> None:
        super().__init__()
        self.num_qs = num_qs

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACCritic(nn.Module):
    """FlashSAC critic model holding the online double critic and its EMA target copy.

    The target network parameters are updated only by Polyak averaging
    (``soft_update_target_networks``). Its BatchNorm running statistics evolve through its
    own forward passes with ``training=True`` during the critic update (CrossQ-style).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = 1,
        num_actions: int = 0,
        num_blocks: int = 2,
        hidden_dim: int = 256,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        num_qs: int = 2,
        **kwargs,
    ) -> None:
        """Initialize the FlashSAC critic model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "critic").
            output_dim: Unused (kept for interface compatibility; Q output is scalar per ensemble member).
            num_actions: Dimension of the action space (concatenated with observations).
            num_blocks: Number of residual FlashSAC blocks in the trunk.
            hidden_dim: Hidden dimension of the trunk.
            num_bins: Number of bins of the categorical value distribution.
            min_v: Minimum value of the categorical support.
            max_v: Maximum value of the categorical support.
            num_qs: Number of ensembled Q-networks.
        """
        super().__init__()

        self.obs_groups, self.obs_dim = _resolve_obs_dim(obs, obs_groups, obs_set)
        self.num_actions = num_actions
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v

        self.critic = FlashSACDoubleCritic(
            num_blocks=num_blocks,
            input_dim=self.obs_dim + num_actions,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            num_qs=num_qs,
        )

        # EMA target copy — parameters updated only via soft_update_target_networks()
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def flatten_obs(self, obs: TensorDict, training: bool = False) -> torch.Tensor:
        """Concatenate the active observation groups into a flat tensor."""
        del training  # stateless in the base model; used by the DreamWaQ variant
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)

    def evaluate(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Online critic forward: returns (qs of shape (num_qs, B), info with categorical log-probs)."""
        return self.critic(observations, actions, training)

    def evaluate_target(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Target critic forward: returns (qs of shape (num_qs, B), info with categorical log-probs)."""
        return self.critic_target(observations, actions, training)

    def init_target_networks(self) -> None:
        """Initialize the target network with the current online critic parameters and buffers."""
        self.critic_target.load_state_dict(self.critic.state_dict())

    @torch.no_grad()
    def soft_update_target_networks(self, tau: float) -> None:
        """Polyak-average online parameters into the target: ``target = (1 - tau) * target + tau * online``."""
        target_params: list[torch.Tensor] = list(self.critic_target.parameters())
        online_params: list[torch.Tensor] = list(self.critic.parameters())
        torch._foreach_lerp_(target_params, online_params, tau)

    def update_normalization(self, obs: TensorDict) -> None:
        """No-op: normalization is handled by BatchNorm layers inside the network."""

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass


class FlashSACTemperature(nn.Module):
    """Learnable SAC temperature parameterized in log space."""

    def __init__(self, initial_value: float = 0.01) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor([initial_value], dtype=torch.float32)))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)


##################################################
# Export helpers — JIT and ONNX for FlashSAC Actor
##################################################


class _TorchFlashSACActor(nn.Module):
    """Exportable FlashSAC actor model for JIT.

    Runs the trunk with BatchNorm running statistics and returns the deterministic
    (mean) Tanh-squashed, scaled action.
    """

    def __init__(self, model: FlashSACActor) -> None:
        super().__init__()
        self.embedder = copy.deepcopy(model.embedder)
        self.encoder = copy.deepcopy(model.encoder)
        self.post_norm = copy.deepcopy(model.post_norm)
        self.mean_weight = nn.Parameter(model.predictor.mean_w.w.weight.detach().clone())
        self.mean_bias = nn.Parameter(model.predictor.mean_bias.detach().clone())
        self.action_bias = nn.Parameter(model.predictor.action_bias.detach().clone())
        self.action_scale = nn.Parameter(model.predictor.action_scale.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x, False)
        for block in self.encoder:
            x = block(x, False)
        x = self.post_norm(x)
        mean = F.linear(x, self.mean_weight, self.mean_bias)
        return self.action_scale * torch.tanh(mean) + self.action_bias

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TorchFlashSACCenet(nn.Module):
    """TorchScript-friendly CENet export: deterministic trunk, no training flag."""

    def __init__(self, policy: FlashSACDreamwaqActor) -> None:
        super().__init__()
        cenet = copy.deepcopy(policy.cenet)
        self.embedder = cenet.embedder
        self.encoder = cenet.encoder
        self.post_norm = cenet.post_norm
        self.predictor_weight = nn.Parameter(cenet.predictor.w.weight.detach().clone())
        self.predictor_bias = nn.Parameter(cenet.predictor_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x, False)
        for block in self.encoder:
            x = block(x, False)
        x = self.post_norm(x)
        return F.linear(x, self.predictor_weight, self.predictor_bias)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFlashSACActor(nn.Module):
    """Exportable FlashSAC actor model for ONNX."""

    is_recurrent: bool = False
    action_bias: torch.Tensor
    action_scale: torch.Tensor

    def __init__(self, model: FlashSACActor, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.embedder = copy.deepcopy(model.embedder)
        self.encoder = copy.deepcopy(model.encoder)
        self.post_norm = copy.deepcopy(model.post_norm)
        self.mean_weight = nn.Parameter(model.predictor.mean_w.w.weight.detach().clone())
        self.mean_bias = nn.Parameter(model.predictor.mean_bias.detach().clone())
        self.register_buffer("action_bias", model.predictor.action_bias.detach().clone())
        self.register_buffer("action_scale", model.predictor.action_scale.detach().clone())
        # Derive the dummy input size from the actual trunk (embedder), not from re-resolving
        # obs group dims: DreamWaQ variants rebuild the embedder for obs_dim + cenet_latent_dim,
        # so model.obs_dim alone would be stale/wrong here.
        self.input_size = self.embedder.norm.running_mean.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x, False)
        for block in self.encoder:
            x = block(x, False)
        x = self.post_norm(x)
        mean = F.linear(x, self.mean_weight, self.mean_bias)
        return self.action_scale * torch.tanh(mean) + self.action_bias

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
