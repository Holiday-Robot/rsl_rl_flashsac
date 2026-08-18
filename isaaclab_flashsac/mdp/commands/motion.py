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

"""Motion-tracking command term."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from isaaclab_flashsac.utils.motion_loader import MotionLoader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionCommand(CommandTerm):
    """Command term that drives the robot toward a reference motion clip.

    The term pools one or more motion clips through :class:`MotionLoader`, samples a
    reference frame per environment (adaptive bin-based sampling for ``balance_mode``
    ``"frame"``, or per-clip uniform sampling for ``balance_mode`` ``"motion"``), and
    exposes the reference body/joint targets used by the tracking observations and rewards.
    """

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        body_indexes, found_body_names = self.robot.find_bodies(self.cfg.body_names, preserve_order=True)
        if len(found_body_names) != len(self.cfg.body_names):
            missing = [name for name in self.cfg.body_names if name not in found_body_names]
            raise ValueError(f"MotionCommand body_names are missing from robot body list: {missing}")
        self.body_indexes = torch.tensor(body_indexes, dtype=torch.long, device=self.device)

        self.motion = MotionLoader(
            self.cfg.resolved_motion_files,
            body_indexes=self.body_indexes,
            device=self.device,
            balance_mode=self.cfg.balance_mode,
            body_names=self.cfg.body_names,
            joint_names=self.robot.joint_names,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, 0] + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, 0]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _set_uniform_sampling_metrics(self) -> None:
        self.metrics["sampling_entropy"][:] = 1.0 if self.bin_count > 1 else 0.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / float(self.bin_count)
        self.metrics["sampling_top1_bin"][:] = 0.0

    def _uniform_frame_sampling(self, env_ids: Sequence[int]) -> None:
        self.time_steps[env_ids] = self.motion.sample_start_frames(len(env_ids), balance_mode="frame").to(
            device=self.device
        )
        self._set_uniform_sampling_metrics()

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if getattr(self._env, "eval_mode", False):
            self._uniform_frame_sampling(env_ids)
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            # Accumulate failures until _update_command folds this buffer into the EMA.
            # A reset wave can be followed by a clip-boundary resample in the same env step.
            self._current_bin_failed += torch.bincount(fail_bins, minlength=self.bin_count).float()

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.balance_mode == "frame":
            # Adaptive bin-based sampling over the pooled timeline (verbatim WBT logic).
            self._adaptive_sampling(env_ids)
        else:
            # Per-clip balanced sampling: pick start frames respecting ``balance_mode``.
            self.time_steps[env_ids] = self.motion.sample_start_frames(
                len(env_ids), balance_mode=self.cfg.balance_mode
            ).to(device=self.device)

        root_pos = self.root_pos_w.clone()
        root_ori = self.root_quat_w.clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # IsaacLab calls CommandTerm.reset() during env reset, but does not call
        # compute() again before the first reward/termination of the next step.
        # Keep relative body targets in sync with the freshly written state.
        self._update_relative_body_targets()

    def _update_command(self):
        self.time_steps += 1
        # Resample at the per-clip end, not the global pooled end: an episode that crosses an internal
        # clip boundary must restart (else it silently indexes the next pooled clip's frames -> a
        # pose/root/velocity discontinuity). For a single clip clip_end_of_frame == time_step_total,
        # so this is a no-op (byte-identical to WBT). NOTE: in balance_mode="frame" the adaptive start
        # sampler still draws over the whole pooled timeline (uniform-over-all-frames, by design); only
        # the playback boundary is per-clip here.
        env_ids = torch.where(self.time_steps >= self.motion.clip_end_of_frame(self.time_steps))[0]
        self._resample_command(env_ids)

        self._update_relative_body_targets()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _update_relative_body_targets(self):
        # Match Holosoma's reset-frame guard: immediately after reset, use the
        # root/pelvis frame instead of a configured child reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(-1)
        ref_pos_w = torch.where(use_root, self.root_pos_w, self.anchor_pos_w)
        ref_quat_w = torch.where(use_root, self.root_quat_w, self.anchor_quat_w)
        robot_ref_pos_w = torch.where(use_root, self.robot_root_pos_w, self.robot_anchor_pos_w)
        robot_ref_quat_w = torch.where(use_root, self.robot_root_quat_w, self.robot_anchor_quat_w)

        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_ref_pos_w_repeat.clone()
        delta_pos_w[..., 2] = ref_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_ref_quat_w_repeat, quat_inv(ref_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - ref_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_anchor_visualizer"):
                # Base global pose -> 3-axis frame markers (goal = reference, current = robot).
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )
                # Base velocity -> 1D arrows (goal = green, current = blue).
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                # Tracked body positions -> spheres (goal = green, current = blue).
                goal_body_cfg = _sphere_marker_cfg(self.cfg.viz_sphere_radius, self.cfg.viz_goal_color)
                cur_body_cfg = _sphere_marker_cfg(self.cfg.viz_sphere_radius, self.cfg.viz_current_color)
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(cur_body_cfg.replace(prim_path="/Visuals/Command/current/" + name))
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(goal_body_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "goal_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # Base global pose -> 3-axis frames.
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        # Base velocity -> 1D arrows at the base position (world-frame xy velocity).
        goal_scale, goal_quat = self._resolve_xy_velocity_to_arrow(
            self.anchor_lin_vel_w[:, :2], self.goal_vel_visualizer
        )
        cur_scale, cur_quat = self._resolve_xy_velocity_to_arrow(
            self.robot_anchor_lin_vel_w[:, :2], self.current_vel_visualizer
        )
        # Lift the velocity arrows above the head so they read clearly and don't overlap the base frame.
        head_offset = torch.tensor([0.0, 0.0, self.cfg.viz_vel_height_offset], device=self.device)
        self.goal_vel_visualizer.visualize(self.anchor_pos_w + head_offset, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(self.robot_anchor_pos_w + head_offset, cur_quat, cur_scale)

        # Tracked body positions -> spheres.
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, visualizer: VisualizationMarkers
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a world-frame XY velocity to (arrow_scale, arrow_quat) for a 1D arrow marker.

        The arrow length scales with speed and points along the velocity heading. Velocities are
        already in the world frame, so (unlike IsaacLab's base-frame velocity command) no extra
        base->world rotation is applied.
        """
        default_scale = visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat


def _sphere_marker_cfg(radius: float, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Single-sphere VisualizationMarkers cfg for reference-motion visualization."""
    return VisualizationMarkersCfg(
        prim_path="/Visuals/Command/pose",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command.

    Supports multi-dataset training: ``motion_files`` may be a list of ``.npz`` paths, a
    directory of ``.npz`` clips, or a single path. ``motion_file`` is kept as a back-compat
    single-clip alias that, when set, is promoted to ``[motion_file]``.
    """

    class_type: type = MotionCommand

    asset_name: str = MISSING  # type: ignore[assignment]

    motion_files: list[str] = MISSING  # type: ignore[assignment]
    """Reference motion clip(s): a list of ``.npz`` paths, a directory of ``.npz``, or a single path."""

    motion_file: str | None = None
    """Back-compat single-clip alias. If set, it is used as ``[motion_file]`` for ``motion_files``."""

    balance_mode: str = "frame"
    """Multi-dataset sampling strategy: ``"frame"`` (uniform over frames) or ``"motion"`` (uniform over clips)."""

    anchor_body_name: str = MISSING  # type: ignore[assignment]
    body_names: list[str] = MISSING  # type: ignore[assignment]

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # Visualization. Base global pose -> 3-axis frame triad; base velocity -> 1D arrow (goal = green,
    # current/robot = blue); per-body tracked positions -> spheres (goal = green, robot = blue).
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/vel_goal"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/vel_current"
    )
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    viz_sphere_radius: float = 0.04
    viz_goal_color: tuple[float, float, float] = (0.1, 1.0, 0.1)
    viz_current_color: tuple[float, float, float] = (0.1, 0.4, 1.0)
    viz_vel_height_offset: float = 0.7  # raise the base-velocity arrows this far (m) above the torso anchor

    @property
    def resolved_motion_files(self) -> str | list[str]:
        """Resolve ``motion_files`` honoring the ``motion_file`` single-clip alias.

        Returns
        -------
            The single-clip alias as ``[motion_file]`` when ``motion_file`` is set, otherwise
            ``motion_files`` unchanged (a list of paths, a directory, or a single path).
        """
        if self.motion_file is not None:
            return [self.motion_file]
        return self.motion_files
