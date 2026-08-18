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

"""Learning rate schedules."""

from __future__ import annotations

import math
from collections.abc import Callable


def warmup_cosine_decay_scheduler(
    init_value: float, peak_value: float, end_value: float, warmup_steps: int, decay_steps: int
) -> Callable[[int], float]:
    """Linear warmup from init to peak, then cosine decay to end (optax-style: decay_steps is total length)."""

    def scheduler(step: int) -> float:
        if step < warmup_steps:
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        elif step < decay_steps:
            progress = (step - warmup_steps) / (decay_steps - warmup_steps)
            return end_value + (peak_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return end_value

    return scheduler
