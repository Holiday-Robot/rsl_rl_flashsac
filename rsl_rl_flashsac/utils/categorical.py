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

"""Distributional (categorical) TD-target helpers for the FlashSAC critic."""

from __future__ import annotations

import torch


def select_min_q_log_probs(next_qs: torch.Tensor, next_q_log_probs: torch.Tensor) -> torch.Tensor:
    """Select the categorical log-probs of the min-Q critic per sample.

    Args:
        next_qs: Q values of shape (num_qs, B).
        next_q_log_probs: Categorical log-probs of shape (num_qs, B, num_bins).

    Returns
    -------
        Selected log-probs of shape (B, num_bins).
    """
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]
    return selected


def compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,) — true terminations only; time-outs bootstrap
    actor_entropy: torch.Tensor,  # (B,)
    gamma: float,  # n-step discount gamma^n_step
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    """Project the entropy-regularized Bellman target onto the categorical support (C51-style)."""
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # Compute target value buckets
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    # Distribute probability mass to the two neighboring bins
    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    frac = b - lower.float()

    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs
