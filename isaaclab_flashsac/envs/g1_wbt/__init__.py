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

"""G1 whole-body motion-tracking environment module.

BeyondMimic-style motion tracking. This package holds the task assembly
only (:mod:`.env_cfg` and the G1 profile in :mod:`.assets`); the term functions live in
:mod:`isaaclab_flashsac.mdp` (``commands.motion``, ``obs.motion``,
``rewards.motion_tracking``, ``terminations.motion_tracking``, ``events.domain_rand``) and
the multi-clip loader in :mod:`isaaclab_flashsac.utils.motion_loader`.

The env-config classes are registered by lazy string entry points in
:mod:`isaaclab_flashsac.envs` (``gym.register``), so importing this subpackage is deferred
until ``gym.make``/``parse_env_cfg`` resolves the entry point. This ``__init__`` therefore
stays import-light (no ``isaaclab``/``gym`` imports).
"""

from __future__ import annotations

from typing import Any


def apply_motion_files(env_cfg: Any, motion_files: list[str] | None) -> None:
    """Write CLI-provided motion clips into a tracking env config, in place.

    ``commands.motion.motion_files`` is left ``MISSING`` in the env config (the clip paths
    are runtime data, not part of the task definition), so the train/play/eval scripts call
    this right after ``parse_env_cfg``. No-op for tasks without a ``commands.motion`` term.

    Args:
        env_cfg: The parsed environment configuration (duck-typed; not imported here).
        motion_files: One or more ``.npz`` clip paths or a directory of clips, or ``None``.

    Raises
    ------
        ValueError: If a tracking task is launched without ``motion_files``, or
            ``motion_files`` is given for a task that has no motion command.
    """
    commands = getattr(env_cfg, "commands", None)
    motion = getattr(commands, "motion", None) if commands is not None else None
    if motion is None:
        if motion_files:
            raise ValueError("--motion_files was given, but this task has no motion command.")
        return
    if not motion_files:
        raise ValueError("Tracking tasks require --motion_files (one or more .npz clips or a directory).")
    motion.motion_files = list(motion_files)
