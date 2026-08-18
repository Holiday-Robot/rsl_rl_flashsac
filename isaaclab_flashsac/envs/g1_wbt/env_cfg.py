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

"""G1 whole-body motion-tracking environment configuration (BeyondMimic-style).

The generic :class:`TrackingEnvCfg` assembles the scene, motion command, two observation
groups, the flat single-critic reward set, terminations, and WBT domain randomization;
:class:`G1FlatTrackingEnvCfg` binds the Unitree G1 articulation and body/joint groups
(``assets.py``). Reward weights/stds, termination thresholds, command ranges, and sim
settings follow the reference's effective training config (``configs/env/
isaaclab_tracking.yaml``); the only value where that yaml overrides the reference's code
default is the ``action_rate_l2`` weight (-0.1 here, -1.0 in the upstream rewards_cfg).

``commands.motion.motion_files`` is intentionally left ``MISSING``: the train/play/eval
scripts fill it from the ``--motion_files`` CLI argument after ``parse_env_cfg``.

The privileged observation group is exposed under the name ``critic`` so the runner's
``obs_groups`` maps the asymmetric actor/critic split directly (see ``rl_cfg.py``).
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_flashsac.mdp import events as track_events
from isaaclab_flashsac.mdp import rewards as track_rews
from isaaclab_flashsac.mdp import terminations as track_terms
from isaaclab_flashsac.mdp.commands import MotionCommandCfg
from isaaclab_flashsac.mdp.obs import motion as track_obs

from .assets import G1_TRACKING_ACTION_SCALE, G1_TRACKING_CFG, apply_g1_tracking_profile

# Velocity perturbation range shared by the motion command resampling and the
# ``push_robot`` interval event.
VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


##
# Scene
##


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    """Flat-terrain scene with a legged robot.

    The robot articulation is left as ``MISSING`` so that the concrete robot config is
    injected by the per-robot env config.
    """

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots (injected by the per-robot env config)
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # contact forces sensor (used by the safety reward term)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    The ``anchor_body_name`` and ``body_names`` fields of the motion command are injected
    by the per-robot env config (``apply_g1_tracking_profile``), and ``motion_files`` is
    supplied at runtime via the ``--motion_files`` CLI argument.
    """

    motion = MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (order preserved to match WBT)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy (actor) group."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=track_obs.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b = ObsTerm(
            func=track_obs.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged observations for the critic group (no noise corruption)."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=track_obs.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=track_obs.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=track_obs.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=track_obs.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class RewardsCfg:
    """Flat single-critic reward terms (summed to one scalar by the RewardManager).

    Weights/stds are the reference's effective training values
    (``configs/env/isaaclab_tracking.yaml``). ``body_names``/``anchor_body_name`` params
    left ``None`` here are injected by ``apply_g1_tracking_profile``.
    """

    # -- motion tracking
    motion_global_anchor_pos = RewTerm(
        func=track_rews.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=track_rews.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=track_rews.motion_relative_body_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_ee_body_pos = RewTerm(
        func=track_rews.motion_local_body_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.1,
            "body_names": None,
            "body_offsets": None,
            "anchor_body_name": None,
        },
    )
    motion_body_ori = RewTerm(
        func=track_rews.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=track_rews.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=track_rews.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    # -- regularization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    anti_shake_ang_vel = RewTerm(
        func=track_rews.anti_shake_ang_vel_l2,
        weight=-5.0e-3,
        params={"command_name": "motion", "threshold": 1.5, "body_names": None},
    )
    feet_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # -- safety
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*"]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Note the func-vs-field indirection (matching WBT): ``anchor_pos`` uses the
    ``bad_anchor_pos_z_only`` func and ``ee_body_pos`` uses ``bad_motion_body_pos_z_only``.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=track_terms.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )
    anchor_ori = DoneTerm(
        func=track_terms.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=track_terms.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": None,
        },
    )


@configclass
class EventCfg:
    """WBT domain randomization events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=track_events.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=track_events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (none)."""

    pass


##
# Environment configurations
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Generic configuration for the motion-tracking environment (robot injected per-robot)."""

    # Scene settings
    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class G1FlatTrackingEnvCfg(TrackingEnvCfg):
    """Flat-terrain motion-tracking config for the Unitree G1.

    Binds the G1 articulation, action scale, and body/joint groups to the generic
    tracking assembler.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = G1_TRACKING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_TRACKING_ACTION_SCALE
        apply_g1_tracking_profile(self)


@configclass
class G1FlatTrackingEnvCfg_PLAY(G1FlatTrackingEnvCfg):
    """Play variant (this repo's convention; upstream has no play cfg): smaller scene,
    no observation corruption, no random pushes.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class G1FlatTrackingWoStateEstimationEnvCfg(G1FlatTrackingEnvCfg):
    """Without-state-estimation variant.

    Disables the policy observations that depend on base-frame state estimation (the
    motion anchor position in the base frame and the base linear velocity).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatTrackingWoStateEstimationEnvCfg_PLAY(G1FlatTrackingWoStateEstimationEnvCfg):
    """Play variant of the without-state-estimation config."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
