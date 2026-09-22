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

"""Event (domain randomization) term functions.

Re-exports ``isaaclab.envs.mdp.events`` so an env_cfg reaches every event term through this one
module; the terms defined here shadow upstream ones of the same name.

- :mod:`.domain_rand`: joint default position and rigid-body center-of-mass randomization.
"""

from isaaclab.envs.mdp.events import *  # noqa: F401, F403

from isaaclab_flashsac.mdp.events.domain_rand import randomize_joint_default_pos, randomize_rigid_body_com

__all__ = ["randomize_joint_default_pos", "randomize_rigid_body_com"]
