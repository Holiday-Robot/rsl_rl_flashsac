# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Licensed under BSD-3-Clause.

"""Delta joint position action: the command integrates the action, as in POISE (arXiv 2609.13761)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass


class DeltaJointPositionAction(JointAction):
    r"""Integrate the joint position command by the action, then clip it to the joint limits.

    .. math::

        q^{cmd}_{t+1} = \mathrm{clip}(q^{cmd}_t + s\,a_t,\ q_{min},\ q_{max})

    Isaac Lab's :class:`RelativeJointPositionAction` integrates the *measured* position instead, so the
    command can never lead it by more than one step and the joint torque is capped at ``kp * s``.

    ``max_command_lead`` bounds how far the command may run ahead of the measured position, which caps the
    torque at ``kp * max_command_lead``; ``None`` keeps only the joint limits, as POISE has it.
    """

    cfg: DeltaJointPositionActionCfg

    def __init__(self, cfg: DeltaJointPositionActionCfg, env):
        super().__init__(cfg, env)
        # joint position command, rad; reset to the measured position
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._lower, self._upper = soft_limits[..., 0], soft_limits[..., 1]

    @property
    def processed_actions(self) -> torch.Tensor:
        """The joint position command the articulation is tracking, rad."""
        return self._command

    def process_actions(self, actions: torch.Tensor):
        """Integrate the command once per control step (``apply_actions`` runs per physics step)."""
        super().process_actions(actions)
        command = self._command + self._processed_actions
        if self.cfg.max_command_lead is not None:
            measured = self._asset.data.joint_pos[:, self._joint_ids]
            lead = self.cfg.max_command_lead
            command = torch.clamp(command, measured - lead, measured + lead)
        self._command = torch.clamp(command, self._lower, self._upper)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._command[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]


@configclass
class DeltaJointPositionActionCfg(JointActionCfg):
    """Configuration for :class:`DeltaJointPositionAction`; ``scale`` is the step size in rad."""

    class_type: type = DeltaJointPositionAction

    max_command_lead: float | None = MISSING  # type: ignore[assignment]
    """How far the command may lead the measured joint position, rad; it caps the torque at ``kp`` times
    this. ``None`` leaves the command bounded only by the joint limits, as POISE has it."""
