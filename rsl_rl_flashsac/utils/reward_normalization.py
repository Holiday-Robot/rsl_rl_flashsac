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

import torch
import torch.distributed as dist


def _dist_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    done = torch.logical_or(terminated, truncated).float()
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return rewards / denominator


def _update_mean_var_count_from_batch_moments(
    sample_mean: torch.Tensor,
    sample_var: torch.Tensor,
    sample_count: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update the mean, var and count using the previous mean, var, count and batch values."""
    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return new_mean, new_var, total_count


def _get_batch_moments(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = torch.tensor(float(samples.shape[0]), dtype=samples.dtype, device=samples.device)
    sample_sum = torch.sum(samples, dim=0)
    sample_sumsq = torch.sum(torch.square(samples), dim=0)

    if _dist_world_size() > 1:
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_sumsq, op=dist.ReduceOp.SUM)

    sample_mean = sample_sum / sample_count
    sample_var = torch.clamp(sample_sumsq / sample_count - torch.square(sample_mean), min=0.0)
    return sample_mean, sample_var, sample_count


class RunningMeanStd:
    """Tracks the mean, variance and count of values.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        device: torch.device | str,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        """Update the mean, var and count from a batch of samples (all-reduced across ranks)."""
        sample_shape = tuple(self.mean.shape)
        samples = x.reshape(-1, *sample_shape) if sample_shape else x.reshape(-1)
        sample_mean, sample_var, sample_count = _get_batch_moments(samples)
        self.mean, self.var, self.count = _update_mean_var_count_from_batch_moments(
            sample_mean=sample_mean,
            sample_var=sample_var,
            sample_count=sample_count,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )


class RewardNormalizer:
    """Normalize rewards using the variance of a running estimate of the discounted returns.

    In policy gradient methods, the update rule often involves the term ∇log π(a|s)⋅G_t, where G_t
    is the return from time t. Scaling G_t to have unit variance is an effective variance reduction
    technique. The denominator is lower-bounded by ``G_r_max / G_max`` so that the largest observed
    return magnitude maps to at most ``G_max`` after normalization.
    """

    def __init__(
        self,
        gamma: float,
        G_max: float,
        device: torch.device | str,
        epsilon: float = 1e-8,
    ) -> None:
        self.gamma = gamma
        # running estimates of the discounted return (broadcasts to num_envs on first update)
        self.G_r = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_r_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)
        self.G_max = G_max
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self.G_r, self.G_r_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            G_r_max=self.G_r_max,
            gamma=self.gamma,
        )
        if _dist_world_size() > 1:
            dist.all_reduce(self.G_r_max, op=dist.ReduceOp.MAX)
        self.G_rms.update(self.G_r)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            G_r_max=self.G_r_max,
            G_max=self.G_max,
            eps=self.epsilon,
        )

    def state_dict(self) -> dict:
        return {
            "G_r": self.G_r,
            "G_r_max": self.G_r_max,
            "G_rms_mean": self.G_rms.mean,
            "G_rms_var": self.G_rms.var,
            "G_rms_count": self.G_rms.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.G_r = state["G_r"].to(self.device)
        self.G_r_max = state["G_r_max"].to(self.device)
        self.G_rms.mean = state["G_rms_mean"].to(self.device)
        self.G_rms.var = state["G_rms_var"].to(self.device)
        self.G_rms.count = state["G_rms_count"].to(self.device)
