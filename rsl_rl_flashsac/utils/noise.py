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

"""Temporally correlated exploration noise helpers (zeta-distributed repeat lengths)."""

from __future__ import annotations

import torch


def build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
    """Build the truncated zeta (power-law) CDF over noise repeat lengths 1..max_n."""
    ns = torch.arange(1, max_n + 1, dtype=torch.float32)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


def sample_integer_from_cdf(cdf: torch.Tensor) -> int:
    """Sample an integer in 1..len(cdf) from the given CDF."""
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return int(idx.item()) + 1
