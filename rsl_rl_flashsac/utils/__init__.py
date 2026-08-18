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

"""Helper functions."""

from .categorical import compute_categorical_td_target, select_min_q_log_probs
from .lr_scheduler import warmup_cosine_decay_scheduler
from .noise import build_truncated_zeta_cdf, sample_integer_from_cdf
from .reward_normalization import RewardNormalizer, RunningMeanStd
from .utils import resolve_callable, resolve_compile_mode

__all__ = [
    "RewardNormalizer",
    "RunningMeanStd",
    "build_truncated_zeta_cdf",
    "compute_categorical_td_target",
    "resolve_callable",
    "resolve_compile_mode",
    "sample_integer_from_cdf",
    "select_min_q_log_probs",
    "warmup_cosine_decay_scheduler",
]
