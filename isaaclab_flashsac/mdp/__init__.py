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

"""FlashSAC-specific MDP term library for Isaac Lab manager-based environments.

Subpackages are split by term kind, each grouped by role:

- :mod:`isaaclab_flashsac.mdp.obs`: observation term functions, under ``obs/locomotion/``
  and ``obs/motion/``.
- :mod:`isaaclab_flashsac.mdp.rewards`: reward term functions, grouped by role under
  ``rewards/{tracking,motion_tracking,regularization,safety}.py``.
- :mod:`isaaclab_flashsac.mdp.terminations`: termination term functions
  (``terminations/motion_tracking.py``).
- :mod:`isaaclab_flashsac.mdp.events`: domain-randomization event functions
  (``events/domain_rand.py``).
- :mod:`isaaclab_flashsac.mdp.commands`: command terms (``commands/motion.py``).

Each subpackage re-exports its functions, so an env_cfg can import one and use its terms flat
(e.g. ``rewards.orthogonal_velocity_exp``, ``obs.locomotion.feet_contact``).
"""

from isaaclab_flashsac.mdp import obs


def _isaaclab_missing(exc: ModuleNotFoundError) -> bool:
    """True when the missing module is Isaac Lab itself (not an isaaclab_flashsac bug)."""
    return exc.name is not None and (exc.name == "isaaclab" or exc.name.startswith("isaaclab."))


try:
    # These subpackages use isaaclab at runtime (not just for type hints), unlike obs/, so they
    # cannot be imported without a real Isaac Lab install. Degrade gracefully so CPU-only
    # dev/test environments (no isaaclab) can still import isaaclab_flashsac.mdp.obs.* (e.g. the
    # left-right symmetry transforms), which need no isaaclab at all.
    from isaaclab_flashsac.mdp import commands, events, rewards, terminations
except ModuleNotFoundError as exc:
    # Only degrade when Isaac Lab itself is absent; a genuine bug inside them must surface.
    if not _isaaclab_missing(exc):
        raise
    commands = None  # type: ignore[assignment]
    events = None  # type: ignore[assignment]
    rewards = None  # type: ignore[assignment]
    terminations = None  # type: ignore[assignment]

__all__ = ["commands", "events", "obs", "rewards", "terminations"]
