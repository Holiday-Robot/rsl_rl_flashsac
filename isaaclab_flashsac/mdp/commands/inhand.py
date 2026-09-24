# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Licensed under BSD-3-Clause.

"""In-hand reorientation commands: axis frames in the debug visualization, and a fixed-axis shifting goal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.inhand.mdp import (
    InHandReOrientationCommand,
    InHandReOrientationCommandCfg,
)


class ReorientWithFramesCommand(InHandReOrientationCommand):
    """Isaac Lab's goal-cube marker plus an axis frame on the object and on the goal marker."""

    cfg: ReorientWithFramesCommandCfg

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and not hasattr(self, "object_frame_visualizer"):
            frame_cfg = self.cfg.frame_visualizer_cfg
            self.object_frame_visualizer = VisualizationMarkers(
                frame_cfg.replace(prim_path=f"{frame_cfg.prim_path}/object")
            )
            self.goal_frame_visualizer = VisualizationMarkers(
                frame_cfg.replace(prim_path=f"{frame_cfg.prim_path}/goal")
            )
        if hasattr(self, "object_frame_visualizer"):
            self.object_frame_visualizer.set_visibility(debug_vis)
            self.goal_frame_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        self.object_frame_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        self.goal_frame_visualizer.visualize(marker_pos, self.quat_command_w)


@configclass
class ReorientWithFramesCommandCfg(InHandReOrientationCommandCfg):
    """Configuration for :class:`ReorientWithFramesCommand`."""

    class_type: type = ReorientWithFramesCommand

    frame_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/frame",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.04, 0.04, 0.04),  # axis length, m: about one cube edge
            )
        },
    )
    """The axis-frame marker drawn on the object and on the goal marker (x red, y green, z blue)."""


class RotateAboutAxisCommand(ReorientWithFramesCommand):
    """Goal = the object's current orientation rotated by ``angle`` about an axis.

    The axis is fixed in the palm frame, or a fresh random direction per goal. Reached goals shift
    another ``angle``, so the policy learns continuous finger gaiting rather than one static
    reorientation (arXiv 2601.02778, "Constrained Rotation for Focused Learning").

    Each environment carries its own angle, which grows by ``angle_step`` every time it reaches a goal
    and never shrinks: a 90 deg goal from the start left the delta-action policies at zero successes,
    and POISE (arXiv 2609.13761) reports 6.2 % without its goal-range curriculum against 59.5 % with it.
    """

    cfg: RotateAboutAxisCommandCfg

    def __init__(self, cfg: RotateAboutAxisCommandCfg, env):
        super().__init__(cfg, env)
        # per-env goal angle, rad; starts at the easy end of the range
        self._angle = torch.full((self.num_envs,), cfg.angle_range[0], device=self.device)
        # a copy: the base class zeroes every metric on reset, and the curriculum has to survive that
        self.metrics["goal_angle"] = self._angle.clone()

    def _update_command(self):
        reached = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        self._angle = torch.where(reached, self._angle + self.cfg.angle_step, self._angle).clamp(
            max=self.cfg.angle_range[1]
        )
        self.metrics["goal_angle"][:] = self._angle
        super()._update_command()

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.axis is None:
            axis_w = torch.nn.functional.normalize(torch.randn((len(env_ids), 3), device=self.device), dim=-1)
        else:
            axis_palm = torch.tensor(self.cfg.axis, device=self.device).expand(len(env_ids), 3)
            axis_w = math_utils.quat_apply(self.robot.data.root_quat_w[env_ids], axis_palm)
        angle = self._angle[env_ids]
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(angle, axis_w), self.object.data.root_quat_w[env_ids]
        )
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat

    @property
    def robot(self):
        return self._env.scene[self.cfg.robot_name]


@configclass
class RotateAboutAxisCommandCfg(ReorientWithFramesCommandCfg):
    """Configuration for :class:`RotateAboutAxisCommand`."""

    class_type: type = RotateAboutAxisCommand

    robot_name: str = MISSING  # type: ignore[assignment]
    """The hand; its root link frame defines ``axis``."""

    axis: tuple[float, float, float] | None = MISSING  # type: ignore[assignment]
    """Rotation axis in the hand root frame, unit vector. None draws a uniformly random direction per goal."""

    angle_range: tuple[float, float] = MISSING  # type: ignore[assignment]
    """Goal rotation from the object's current orientation, rad: the starting angle and the ceiling."""

    angle_step: float = MISSING  # type: ignore[assignment]
    """How much an environment's goal angle grows each time it reaches a goal, rad."""
