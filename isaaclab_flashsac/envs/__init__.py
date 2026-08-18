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

"""Out-of-tree Isaac Lab environment registrations for FlashSAC.

Env configs are passed as lazy string entry points, so importing this package pulls in nothing
from Isaac Lab; the heavy imports fire when ``gym.make``/``parse_env_cfg`` resolves an entry
point, after ``AppLauncher`` has started. The ``scripts/{train,play,eval}.py`` entry points
import it next to ``import isaaclab_tasks``.
"""

try:
    import gymnasium as gym
except ModuleNotFoundError:
    # gymnasium comes with Isaac Lab, not with this package: CPU-only dev/test environments
    # import this package for its import-light submodules and skip registration
    gym = None  # type: ignore[assignment]

if gym is not None:
    gym.register(
        id="Isaac-Velocity-Rough-G1-Dreamwaq-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_dreamwaq.env_cfg:G1RoughDreamwaqEnvCfg"},
    )

    gym.register(
        id="Isaac-Velocity-Rough-G1-Dreamwaq-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_dreamwaq.env_cfg:G1RoughDreamwaqEnvCfg_PLAY"},
    )

    gym.register(
        id="Isaac-Velocity-Flat-G1-Dreamwaq-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_dreamwaq.env_cfg:G1FlatDreamwaqEnvCfg"},
    )

    gym.register(
        id="Isaac-Velocity-Flat-G1-Dreamwaq-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_dreamwaq.env_cfg:G1FlatDreamwaqEnvCfg_PLAY"},
    )

    gym.register(
        id="Isaac-Tracking-Flat-G1-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_wbt.env_cfg:G1FlatTrackingEnvCfg"},
    )

    gym.register(
        id="Isaac-Tracking-Flat-G1-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_wbt.env_cfg:G1FlatTrackingEnvCfg_PLAY"},
    )

    gym.register(
        id="Isaac-Tracking-Flat-G1-WoSE-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.g1_wbt.env_cfg:G1FlatTrackingWoStateEstimationEnvCfg"},
    )

    gym.register(
        id="Isaac-Tracking-Flat-G1-WoSE-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_flashsac.envs.g1_wbt.env_cfg:G1FlatTrackingWoStateEstimationEnvCfg_PLAY"
        },
    )

    # Overrides isaaclab_tasks' own registration of the stock reach ids, so
    # `--task Isaac-Reach-Franka-v0` runs the FR3 instead of the Panda. Ordering is what makes it
    # work: the entry points import isaaclab_tasks before this package.
    gym.registry.pop("Isaac-Reach-Franka-v0", None)  # skip gymnasium's "already in registry" warning
    gym.register(
        id="Isaac-Reach-Franka-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.franka_reach.env_cfg:Fr3ReachEnvCfg"},
    )

    gym.registry.pop("Isaac-Reach-Franka-Play-v0", None)
    gym.register(
        id="Isaac-Reach-Franka-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isaaclab_flashsac.envs.franka_reach.env_cfg:Fr3ReachEnvCfg_PLAY"},
    )
