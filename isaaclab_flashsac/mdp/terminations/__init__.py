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

"""Termination term functions, grouped by role.

- :mod:`.motion_tracking`: reference-motion divergence terminations (anchor pose and
  tracked-body position error thresholds).

Termination functions already provided by ``isaaclab.envs.mdp`` (``time_out``) are used
directly from env_cfg.py and are not duplicated here.
"""

from isaaclab_flashsac.mdp.terminations.motion_tracking import (
    bad_anchor_ori,
    bad_anchor_pos,
    bad_anchor_pos_z_only,
    bad_motion_body_pos,
    bad_motion_body_pos_z_only,
)

__all__ = [
    "bad_anchor_ori",
    "bad_anchor_pos",
    "bad_anchor_pos_z_only",
    "bad_motion_body_pos",
    "bad_motion_body_pos_z_only",
]
