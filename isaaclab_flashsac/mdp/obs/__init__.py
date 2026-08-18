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

"""Observation term functions.

Terms live in the :mod:`.locomotion` and :mod:`.motion` subpackages; import the one you need
(``from isaaclab_flashsac.mdp.obs import locomotion``). Nothing is re-exported here because
:mod:`.motion` calls ``isaaclab.utils.math`` at runtime, and this package must stay importable
without an Isaac Lab install.
"""
