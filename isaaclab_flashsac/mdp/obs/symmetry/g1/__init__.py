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

"""Left-right symmetry for the Unitree G1.

:mod:`.base` holds the joint/body tables both tasks share; :mod:`.dreamwaq` and :mod:`.wbt`
hold the per-task observation-group mirrors, each exposing ``compute_symmetric_states``.
"""

from isaaclab_flashsac.mdp.obs.symmetry.g1 import base, dreamwaq, wbt
from isaaclab_flashsac.mdp.obs.symmetry.g1.base import (
    G1SymmetryCache,
    get_symmetry_cache,
    switch_g1_joints_left_right,
    switch_tracked_bodies_left_right,
)

__all__ = [
    "G1SymmetryCache",
    "base",
    "dreamwaq",
    "get_symmetry_cache",
    "switch_g1_joints_left_right",
    "switch_tracked_bodies_left_right",
    "wbt",
]
