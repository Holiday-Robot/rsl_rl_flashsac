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

# NOTE: no `from __future__ import annotations` here — TorchScript cannot resolve stringified
# class-level attribute annotations when these layers are scripted for policy export.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Safe version of the Tanh log-det-Jacobian that avoids numerical issues.

    log(1 - tanh^2(x)) = log(4) - 2*log(exp(x) + exp(-x)) = 2*(log(2) - x - softplus(-2x))
    https://github.com/google-deepmind/distrax/issues/216
    """
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class UnitLinear(nn.Module):
    """Linear layer without bias whose weight rows are re-normalized to unit norm after each optimizer step."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)

    def normalize_parameters(self) -> None:
        """Normalize along dim=-1 (input features) to keep output features as unit vectors.

        Called after initialization and after every optimizer.step().
        """
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """BatchNorm whose affine parameters are jointly re-normalized to norm sqrt(d) after each step."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        """Normalize scale and bias parameters to joint norm sqrt(d)."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    """RMSNorm whose scale parameters are re-normalized to norm sqrt(d) after each step."""

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    """Input embedder: BatchNorm over raw observations followed by a unit-normalized linear layer."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class FlashSACBlock(nn.Module):
    """Residual MLP block with BatchNorm and unit-normalized linear layers."""

    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class NormalTanhPolicy(nn.Module):
    """Gaussian policy head with Tanh squashing and affine action scaling.

    The log-std head output is squashed with tanh into [log_std_min, log_std_max] for stability.
    ``action_bias``/``action_scale``/``log_action_scale`` buffers are populated by
    ``FlashSAC.construct_algorithm()`` after construction (see ``FlashSACActor.set_action_scaling``).
    """

    action_bias: torch.Tensor
    action_scale: torch.Tensor
    log_action_scale: torch.Tensor

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))

        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.register_buffer("action_bias", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("log_action_scale", torch.zeros(()))

    def get_mean_and_std(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # Use functional linear for AMP
        mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)

        # normalize log-stds for stability
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw_log_std))
        std = torch.exp(log_std)

        return mean, std

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std = self.get_mean_and_std(x, training)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        # Normalized [-1, 1] action (official FlashSAC convention): the replay buffer and the
        # critic operate in tanh space; the affine bias/range scaling is applied at the env
        # boundary (FlashSACVecEnvWrapper.step) and baked into the deploy exports only.
        action = torch.tanh(raw_action)

        # Compute log probability (accounting for tanh and affine scaling via Jacobian correction)
        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        log_prob = log_prob.sum(1)

        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return action, info


# -------------------------------------
# Ensembled layers for the Double Critic
# -------------------------------------


class EnsembleUnitLinear(nn.Module):
    """Batched linear layer over N ensemble members; all computation uses (N, batch, dim) layout."""

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, B, in] @ [N, in, out] -> [N, B, out]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                B = x.shape[1]
                # Cast to float32 for running stats (BatchNorm uses float32 even in AMP)
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (B / (B - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class EnsembleCategoricalValue(nn.Module):
    """Distributional (categorical) value head: Q is the expectation over fixed value bins."""

    bin_values: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ) -> None:
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(value, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return value, info


class FlashSACEncoder(nn.Module):
    """FlashSAC encoder trunk: embedder + N residual blocks + unit-linear head.

    Used as the DreamWaQ CENet: output (B, latent_dim) is split by the consumer
    into a lin-vel estimation head and an unsupervised latent.
    """

    def __init__(self, n_input: int, hidden_dim: int, latent_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.embedder = FlashSACEmbedder(n_input, hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = UnitLinear(hidden_dim, latent_dim)
        self.predictor_bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        x = self.embedder(x, training=training)
        for block in self.encoder:
            x = block(x, training=training)
        x = self.post_norm(x)
        return self.predictor(x) + self.predictor_bias

    def normalize_parameters(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if module is not self and hasattr(module, "normalize_parameters"):
                    module.normalize_parameters()  # type: ignore[operator]
