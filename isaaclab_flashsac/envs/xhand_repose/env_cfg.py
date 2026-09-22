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

"""XHand1 cube reorientation environment configuration."""

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.inhand.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg import InHandObjectEnvCfg

from isaaclab_flashsac.mdp.obs import inhand as inhand_obs

from .assets import XHAND_RIGHT_CFG

# dex_cube_instanceable.usd is a 0.065 m cube; 0.8 gives mujoco_playground's 0.052 m XHand cube
CUBE_SCALE = 0.8

# distal phalanges: the URDF tip links are fixed-joint children merged into these at import
FINGERTIP_BODY_NAMES = [
    "right_hand_thumb_rota_link2",
    "right_hand_index_rota_link2",
    "right_hand_mid_link2",
    "right_hand_ring_link2",
    "right_hand_pinky_link2",
]

##
# Observations
##


@configclass
class XHandObservationsCfg:
    """Actor sees proprioception and the tracked cube pose; the critic adds velocities and fingertips."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Order is the deploy contract - do not reorder. Noise and injection follow wuji-mjlab.
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Unoise(n_min=-0.06, n_max=0.06))
        object_pos = ObsTerm(
            func=inhand_obs.object_pos_in_palm,
            noise=Unoise(n_min=-0.008, n_max=0.008),
            params={"injection_prob": 0.02},
        )
        goal_orientation_error = ObsTerm(
            func=inhand_obs.goal_orientation_error_6d,
            noise=Gnoise(std=0.05),
            params={"command_name": "object_pose", "injection_prob": 0.02},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        object_pos = ObsTerm(func=inhand_obs.object_pos_in_palm, params={"injection_prob": 0.0})
        goal_orientation_error = ObsTerm(
            func=inhand_obs.goal_orientation_error_6d,
            params={"command_name": "object_pose", "injection_prob": 0.0},
        )
        object_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, scale=0.2, params={"asset_cfg": SceneEntityCfg("object")})
        fingertip_pos = ObsTerm(
            func=inhand_obs.body_pos_in_palm,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODY_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Environment configurations
##


@configclass
class XHandReposeCubeEnvCfg(InHandObjectEnvCfg):
    """Isaac Lab's in-hand reorientation task on the XHand1 with a non-privileged actor and delta actions."""

    observations: XHandObservationsCfg = XHandObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- physics: 200 Hz simulation, 20 Hz control (dexscrew)
        self.sim.dt = 1.0 / 200.0
        self.decimation = 10
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0

        # -- robot
        self.scene.robot = XHAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.clone_in_fabric = True

        # -- object: mujoco_playground's XHand cube, spawned 1.4 cm above the palm plane (z = 0.525 m)
        self.scene.object.spawn.scale = (CUBE_SCALE, CUBE_SCALE, CUBE_SCALE)
        self.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.108)  # kg
        self.scene.object.init_state.pos = (0.07, 0.005, 0.565)  # env frame, m

        # -- goal: the cube settles 2 cm below its spawn; 0.2 rad is Allegro's tolerance (Shadow uses 0.1)
        self.commands.object_pose.init_pos_offset = (0.0, 0.0, -0.02)
        self.commands.object_pose.orientation_success_threshold = 0.2
        self.commands.object_pose.goal_pose_visualizer_cfg.markers["goal"].scale = (CUBE_SCALE, CUBE_SCALE, CUBE_SCALE)

        # -- action: targets move by up to 0.05 rad per step from the measured joint position (dexscrew's step)
        self.actions.joint_pos = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.05, use_zero_offset=True
        )


@configclass
class XHandReposeCubeEnvCfg_PLAY(XHandReposeCubeEnvCfg):
    """Play variant: smaller scene, clean observations."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.observations.policy.object_pos.params["injection_prob"] = 0.0
        self.observations.policy.goal_orientation_error.params["injection_prob"] = 0.0
