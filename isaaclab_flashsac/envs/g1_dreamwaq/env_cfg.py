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

"""G1 DreamWaQ velocity environment configuration."""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from isaaclab_flashsac.mdp import rewards as loco_rews
from isaaclab_flashsac.mdp.obs import locomotion as loco_obs

from .assets import G1_BEYONDMIMIC_ACTION_SCALE, G1_BEYONDMIMIC_CFG
from .terrains import DREAMWAQ_ROUGH_TERRAINS_CFG

##
# Observations
##


@configclass
class DreamwaqObservationsCfg:
    """Three-group DreamWaQ observations: actor (current), CENet (measurable), critic (privileged)."""

    @configclass
    class CurrentCfg(ObsGroup):
        # Order is the deploy contract - do not reorder.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class MeasurableCfg(ObsGroup):
        # current[3:96] for 29-DoF G1: no lin_vel, no commands; 5-frame history for the CENet.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # Corruption ON: the CENet estimator consumes this history on the real robot,
            # so it must be trained against sensor noise (sim2real).
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        # First term MUST be ground-truth lin vel (CENet supervision target).
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        feet_contact = ObsTerm(
            func=loco_obs.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    current: CurrentCfg = CurrentCfg()
    measurable: MeasurableCfg = MeasurableCfg()
    critic: CriticCfg = CriticCfg()


##
# Rewards
##


@configclass
class G1DreamwaqRewards:
    """Stock ``Isaac-Velocity-Rough-G1-v0`` reward set on the 29-DoF BeyondMimic asset.

    Joint regexes are remapped for the 29-DoF asset (no fingers, torso -> waist). Tune
    freely - this class is the single source of truth for the active rewards.
    """

    # -- task tracking
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # Orthogonal velocity reward (reduces lateral drift) - from Lee et al. 2020 equation 14
    orthogonal_velocity = RewTerm(func=loco_rews.orthogonal_velocity_exp, weight=1.0, params={"scale": 1.5})

    # -- base motion / orientation penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    body_orientation_l2 = RewTerm(
        func=loco_rews.body_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # -- effort penalties
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    energy = RewTerm(func=loco_rews.energy, weight=-1e-3)
    action_rate_l2 = RewTerm(func=loco_rews.action_rate_l2, weight=-0.5)

    # -- gait shaping
    feet_air_time = RewTerm(
        func=loco_rews.feet_air_time_with_inplace,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "t_swing_target": 0.4,
            "t_stance_cmd_norm": 0.5,
            "t_stance_range": (0.1, 0.5),
            # Matches stand_still's threshold so both "the command is zero" tests agree.
            "inplace_upper_bound": 0.1,
            "inplace_t_diff_range": (-0.3, 0.3),
            "t_swing_stance_tol": 0.05,
            "contact_threshold": 1.0,
        },
    )
    feet_air_time_variance = RewTerm(  # may not be suitable for learning unstructured gaits, so use when necessary
        func=loco_rews.feet_air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_too_near = RewTerm(
        func=loco_rews.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"), "threshold": 0.3},
    )
    feet_stumble = RewTerm(
        func=loco_rews.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    feet_yaw_drag = RewTerm(
        func=loco_rews.feet_yaw_drag,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
            "contact_threshold": 1.0,
        },
    )
    feet_force = RewTerm(
        func=loco_rews.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 700,
            "max_reward": 400,
        },
    )
    feet_impact_velocity_delta = RewTerm(
        func=loco_rews.feet_impact_velocity_delta_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "delta_v_max": 1.0,
        },
    )

    # -- joint limits / deviations
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
    )
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle.*"])
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])},
    )
    stand_still = RewTerm(
        func=loco_rews.stand_still,
        weight=-1.0,
        params={"threshold": 0.1},
    )

    # -- safety & termination
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=loco_rews.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


##
# Environment configurations
##


@configclass
class G1RoughDreamwaqEnvCfg(G1RoughEnvCfg):
    """G1 DreamWaQ rough-terrain env: 3 obs groups + DreamWaQ rewards + domain randomization."""

    observations: DreamwaqObservationsCfg = DreamwaqObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The deploy contract needs the true 29-DoF hardware asset, not the 37-DoF
        # G1_MINIMAL_CFG the parent assigns; its naming matches, so no spawn overrides.
        self.scene.robot = G1_BEYONDMIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Rough-terrain curriculum: inverted-stair variants, boxes, random rough, waves, and
        # double pits on a 10x10 grid (replaces the stock Isaac Lab set).
        self.scene.terrain.terrain_generator = DREAMWAQ_ROUGH_TERRAINS_CFG

        # Terminate on torso contact only.
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link"]

        # Assigned after super().__post_init__() so the parent's mutations cannot clobber it.
        self.rewards = G1DreamwaqRewards()

        # Per-joint action scale (0.25 * effort_limit / stiffness) replaces the inherited scalar
        # 0.5; the inherited joint_names=[".*"] already covers all 29 joints.
        self.actions.joint_pos.scale = G1_BEYONDMIMIC_ACTION_SCALE

        # -- command ranges: wider than the parent's, with a fixed (non-heading) yaw-rate command
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # -- domain randomization
        # Physics material randomization (base default is a fixed 0.8/0.6/0.0).
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 0.8)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.005)

        # Base mass randomization on the torso
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"),
                "mass_distribution_params": (-5.0, 5.0),
                "operation": "add",
            },
        )

        # Random pushes
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )

        # Reset-state randomization
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)


@configclass
class G1RoughDreamwaqEnvCfg_PLAY(G1RoughDreamwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # Spawn randomly in the grid instead of by terrain levels.
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable observation corruption and pushes for play.
        self.observations.current.enable_corruption = False
        self.observations.measurable.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1FlatDreamwaqEnvCfg(G1RoughDreamwaqEnvCfg):
    """G1 DreamWaQ flat-terrain env (same rewards/DR; flat terrain, no height scan)."""

    def __post_init__(self):
        super().__post_init__()

        # Flatten terrain.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # No height scan on flat terrain.
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        # No terrain curriculum.
        self.curriculum.terrain_levels = None


@configclass
class G1FlatDreamwaqEnvCfg_PLAY(G1FlatDreamwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Disable observation corruption and pushes for play.
        self.observations.current.enable_corruption = False
        self.observations.measurable.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
