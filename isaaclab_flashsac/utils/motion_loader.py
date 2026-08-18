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

"""Multi-dataset motion loader for the tracking task.

This module is import-light on purpose: it depends only on ``numpy`` and
``torch`` so it can be unit-tested without IsaacLab/IsaacSim installed. It must
NOT import ``isaaclab`` (or anything that transitively imports it).

The :class:`MotionLoader` extends the BeyondMimic (whole_body_tracking) loader
to pool multiple motion clips into a single timeline while recording per-clip
boundaries. This enables multi-dataset training with two balancing strategies
(uniform over frames or uniform over clips) and lets the command term terminate
or resample at clip boundaries.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence

import numpy as np
import torch

# Keys expected in each motion ``.npz`` clip.
_MOTION_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

#: Valid balancing modes for :meth:`MotionLoader.sample_start_frames`.
BALANCE_MODES: tuple[str, ...] = ("frame", "motion")


class MotionLoader:
    """Pooled multi-clip motion data with per-frame indexing.

    Loads one or more motion clips and concatenates their frames into a single
    pooled timeline. Per-clip boundaries are recorded so that a global frame
    index can be mapped back to its originating clip and to that clip's
    exclusive end boundary.

    The per-frame property API (``joint_pos``, ``joint_vel``, ``body_pos_w``,
    ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w``) matches the
    BeyondMimic ``MotionLoader`` so that :class:`MotionCommand` can index it by a
    frame-index tensor unchanged. When a clip contains ``body_names`` or
    ``joint_names`` metadata, arrays are reordered by name into the requested
    robot order instead of assuming that the stored array order already matches.

    Attributes
    ----------
        fps: Frames per second of the (first) loaded clip.
        device: Torch device the pooled tensors live on.
        balance_mode: Default balancing mode used by ``sample_start_frames``.
        joint_pos: Pooled joint positions, shape ``(num_frames, num_joints)``.
        joint_vel: Pooled joint velocities, shape ``(num_frames, num_joints)``.
        time_step_total: Total number of pooled frames.
        num_clips: Number of loaded clips.
        clip_starts: Global start frame index of each clip, ``long[num_clips]``.
        clip_lengths: Frame count of each clip, ``long[num_clips]``.
        motion_files: Resolved list of ``.npz`` files that were loaded.
    """

    def __init__(
        self,
        motion_files: str | Sequence[str],
        body_indexes: Sequence[int] | torch.Tensor | None = None,
        device: str = "cpu",
        balance_mode: str = "frame",
        body_names: Sequence[str] | None = None,
        joint_names: Sequence[str] | None = None,
    ) -> None:
        """Load and pool one or more motion clips.

        Args:
            motion_files: A single ``.npz`` path, a list of ``.npz`` paths, or a
                directory containing ``.npz`` clips (globbed, sorted).
            body_indexes: Fallback indices selecting which bodies the
                ``body_*_w`` properties expose when a clip has no ``body_names``
                metadata.
            device: Torch device for the pooled tensors.
            balance_mode: Default sampling strategy, ``"frame"`` or ``"motion"``.
            body_names: Body names to expose in output order. Preferred over
                ``body_indexes`` when clips contain ``body_names`` metadata.
            joint_names: Joint names to expose in output order when clips
                contain ``joint_names`` metadata.

        Raises
        ------
            ValueError: If ``balance_mode`` is invalid or no clips are found.
            FileNotFoundError: If a provided path does not exist.
            KeyError: If a clip is missing a required data key.
        """
        if balance_mode not in BALANCE_MODES:
            raise ValueError(f"Invalid balance_mode {balance_mode!r}; expected one of {BALANCE_MODES}.")

        self.device = device
        self.balance_mode = balance_mode
        self.body_names = list(body_names) if body_names is not None else None
        self.joint_names = list(joint_names) if joint_names is not None else None

        self.motion_files = self._resolve_motion_files(motion_files)
        if len(self.motion_files) == 0:
            raise ValueError(f"No motion clips found for {motion_files!r}.")

        # Per-clip pooled tensors plus boundary bookkeeping.
        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_w_list: list[torch.Tensor] = []
        body_quat_w_list: list[torch.Tensor] = []
        body_lin_vel_w_list: list[torch.Tensor] = []
        body_ang_vel_w_list: list[torch.Tensor] = []
        clip_starts: list[int] = []
        clip_lengths: list[int] = []
        fps_values: list[float] = []

        cursor = 0
        for path in self.motion_files:
            data = np.load(path)
            for key in _MOTION_KEYS:
                if key not in data:
                    raise KeyError(f"Motion clip {path!r} is missing required key {key!r}.")

            fps_values.append(float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else float("nan"))

            joint_indexes = self._resolve_joint_indexes(data, path)
            body_selection = self._resolve_body_indexes(data, path, body_indexes)

            joint_pos_np = data["joint_pos"]
            joint_vel_np = data["joint_vel"]
            if joint_indexes is not None:
                joint_pos_np = joint_pos_np[:, joint_indexes]
                joint_vel_np = joint_vel_np[:, joint_indexes]

            jp = torch.tensor(joint_pos_np, dtype=torch.float32, device=device)
            jv = torch.tensor(joint_vel_np, dtype=torch.float32, device=device)
            bp = torch.tensor(data["body_pos_w"][:, body_selection], dtype=torch.float32, device=device)
            bq = torch.tensor(data["body_quat_w"][:, body_selection], dtype=torch.float32, device=device)
            blv = torch.tensor(data["body_lin_vel_w"][:, body_selection], dtype=torch.float32, device=device)
            bav = torch.tensor(data["body_ang_vel_w"][:, body_selection], dtype=torch.float32, device=device)

            length = jp.shape[0]
            joint_pos_list.append(jp)
            joint_vel_list.append(jv)
            body_pos_w_list.append(bp)
            body_quat_w_list.append(bq)
            body_lin_vel_w_list.append(blv)
            body_ang_vel_w_list.append(bav)
            clip_starts.append(cursor)
            clip_lengths.append(length)
            cursor += length

        # FPS is taken from the first clip for API compatibility with the
        # single-clip loader (clips are assumed to share an fps).
        self.fps = fps_values[0]

        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self._body_pos_w = torch.cat(body_pos_w_list, dim=0)
        self._body_quat_w = torch.cat(body_quat_w_list, dim=0)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)

        self.time_step_total = self.joint_pos.shape[0]
        self.num_clips = len(self.motion_files)
        self.clip_starts = torch.tensor(clip_starts, dtype=torch.long, device=device)
        self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long, device=device)
        # Exclusive global end boundary of each clip (one past its last frame),
        # i.e. the global start index of the next clip. Used by the boundary
        # helpers so ``time_step >= clip_end`` resamples at the boundary,
        # mirroring how ``time_step_total`` (the global pooled end) is used.
        self._clip_ends = self.clip_starts + self.clip_lengths

    @staticmethod
    def _resolve_motion_files(motion_files: str | Sequence[str]) -> list[str]:
        """Resolve the ``motion_files`` argument to a concrete list of paths.

        Args:
            motion_files: A single path, a list of paths, or a directory.

        Returns
        -------
            A sorted list of absolute/relative ``.npz`` file paths.

        Raises
        ------
            FileNotFoundError: If a referenced file or directory is missing.
        """
        if isinstance(motion_files, str):
            candidates: list[str] = [motion_files]
        else:
            candidates = list(motion_files)

        resolved: list[str] = []
        for entry in candidates:
            if os.path.isdir(entry):
                matched = sorted(glob.glob(os.path.join(entry, "*.npz")))
                resolved.extend(matched)
            elif os.path.isfile(entry):
                resolved.append(entry)
            else:
                raise FileNotFoundError(f"Motion path does not exist: {entry!r}")
        return resolved

    def _resolve_joint_indexes(self, data: np.lib.npyio.NpzFile, path: str) -> list[int] | None:
        """Resolve joint reorder indices for one clip.

        Returns ``None`` when no reordering is needed.
        """
        joint_dim = int(data["joint_pos"].shape[1])
        if data["joint_vel"].shape[1] != joint_dim:
            raise ValueError(
                f"Motion clip {path!r} has inconsistent joint dimensions: "
                f"joint_pos={data['joint_pos'].shape}, joint_vel={data['joint_vel'].shape}."
            )

        if self.joint_names is None:
            return None

        if "joint_names" not in data:
            if joint_dim != len(self.joint_names):
                raise ValueError(
                    f"Motion clip {path!r} has {joint_dim} joints but {len(self.joint_names)} names were requested, "
                    "and the clip has no 'joint_names' metadata for name-based mapping."
                )
            return None

        available_names = self._decode_names(data["joint_names"])
        self._validate_name_count(available_names, joint_dim, "joint", path)
        return self._indexes_by_name(self.joint_names, available_names, "joint", path)

    def _resolve_body_indexes(
        self,
        data: np.lib.npyio.NpzFile,
        path: str,
        fallback_body_indexes: Sequence[int] | torch.Tensor | None,
    ) -> list[int]:
        """Resolve body reorder indices for one clip."""
        body_dim = int(data["body_pos_w"].shape[1])
        for key in ("body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            if data[key].shape[1] != body_dim:
                raise ValueError(
                    f"Motion clip {path!r} has inconsistent body dimensions: "
                    f"body_pos_w={data['body_pos_w'].shape}, {key}={data[key].shape}."
                )

        if self.body_names is not None and "body_names" in data:
            available_names = self._decode_names(data["body_names"])
            self._validate_name_count(available_names, body_dim, "body", path)
            return self._indexes_by_name(self.body_names, available_names, "body", path)

        if fallback_body_indexes is None:
            raise KeyError(
                f"Motion clip {path!r} has no 'body_names' metadata and no fallback body_indexes were provided."
            )

        indexes = self._to_index_list(fallback_body_indexes)
        if any(index < 0 or index >= body_dim for index in indexes):
            raise IndexError(f"Motion clip {path!r} body_indexes {indexes!r} exceed body dimension {body_dim}.")
        return indexes

    @staticmethod
    def _decode_names(names: np.ndarray) -> list[str]:
        """Decode string metadata from numpy arrays into Python strings."""
        decoded: list[str] = []
        for name in np.asarray(names).reshape(-1).tolist():
            if isinstance(name, bytes):
                decoded.append(name.decode("utf-8"))
            else:
                decoded.append(str(name))
        return decoded

    @staticmethod
    def _to_index_list(indexes: Sequence[int] | torch.Tensor) -> list[int]:
        """Convert a sequence or tensor of indices to plain Python ints."""
        if isinstance(indexes, torch.Tensor):
            return [int(index) for index in indexes.detach().cpu().reshape(-1).tolist()]
        return [int(index) for index in indexes]

    @staticmethod
    def _validate_name_count(names: Sequence[str], expected_count: int, kind: str, path: str) -> None:
        """Validate that metadata count matches the array axis it describes."""
        if len(names) != expected_count:
            raise ValueError(
                f"Motion clip {path!r} has {expected_count} {kind}s in arrays but {len(names)} {kind}_names entries."
            )

    @staticmethod
    def _indexes_by_name(requested: Sequence[str], available: Sequence[str], kind: str, path: str) -> list[int]:
        """Return indices mapping requested names into available names."""
        if len(set(available)) != len(available):
            raise ValueError(f"Motion clip {path!r} has duplicate {kind}_names metadata.")

        index_by_name = {name: i for i, name in enumerate(available)}
        missing = [name for name in requested if name not in index_by_name]
        if missing:
            raise KeyError(
                f"Motion clip {path!r} is missing requested {kind} names {missing!r}. "
                f"Available names: {list(available)!r}"
            )
        return [index_by_name[name] for name in requested]

    # ------------------------------------------------------------------
    # Per-frame property API (mirrors the BeyondMimic single-clip loader).
    # ------------------------------------------------------------------
    @property
    def body_pos_w(self) -> torch.Tensor:
        """Pooled body positions for the selected bodies, ``(T, B, 3)``."""
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Pooled body orientations for the selected bodies, ``(T, B, 4)``."""
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Pooled body linear velocities for the selected bodies, ``(T, B, 3)``."""
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Pooled body angular velocities for the selected bodies, ``(T, B, 3)``."""
        return self._body_ang_vel_w

    # ------------------------------------------------------------------
    # Clip-boundary helpers.
    # ------------------------------------------------------------------
    def clip_id_of_frame(self, global_idx: torch.Tensor) -> torch.Tensor:
        """Return the clip id owning each global frame index.

        Args:
            global_idx: ``long`` tensor of global frame indices.

        Returns
        -------
            ``long`` tensor (same shape as ``global_idx``) of clip ids.
        """
        global_idx = torch.as_tensor(global_idx, dtype=torch.long, device=self.device)
        # ``clip_starts`` is sorted ascending; right-side bucketize then -1 gives
        # the index of the last start that is <= global_idx.
        clip_ids = torch.bucketize(global_idx, self.clip_starts, right=True) - 1
        return clip_ids.clamp_(0, self.num_clips - 1)

    def clip_end_of_frame(self, global_idx: torch.Tensor) -> torch.Tensor:
        """Return the exclusive global end boundary of each frame's clip.

        The returned value is one past the clip's last frame (the global start
        index of the next clip), so ``time_step >= clip_end_of_frame(...)``
        detects a clip-boundary crossing the same way ``time_step >=
        time_step_total`` detects the pooled-timeline end.

        Args:
            global_idx: ``long`` tensor of global frame indices.

        Returns
        -------
            ``long`` tensor (same shape as ``global_idx``) of exclusive end
            boundaries.
        """
        clip_ids = self.clip_id_of_frame(global_idx)
        return self._clip_ends[clip_ids]

    # ------------------------------------------------------------------
    # Sampling.
    # ------------------------------------------------------------------
    def sample_start_frames(
        self,
        n: int,
        generator: torch.Generator | None = None,
        balance_mode: str | None = None,
    ) -> torch.Tensor:
        """Sample ``n`` global start-frame indices.

        Args:
            n: Number of start frames to draw.
            generator: Optional torch RNG for reproducibility.
            balance_mode: Override for the loader's default balancing mode.
                ``"frame"`` draws uniformly over all pooled frames; ``"motion"``
                draws a clip uniformly then a frame uniformly within that clip.

        Returns
        -------
            ``long[n]`` tensor of global start-frame indices.

        Raises
        ------
            ValueError: If ``balance_mode`` is invalid.
        """
        mode = balance_mode if balance_mode is not None else self.balance_mode
        if mode not in BALANCE_MODES:
            raise ValueError(f"Invalid balance_mode {mode!r}; expected one of {BALANCE_MODES}.")

        if mode == "frame":
            return torch.randint(
                0,
                self.time_step_total,
                (n,),
                generator=generator,
                dtype=torch.long,
                device=self.device,
            )

        # "motion": uniform over clips, then uniform within the chosen clip.
        clip_ids = torch.randint(
            0,
            self.num_clips,
            (n,),
            generator=generator,
            dtype=torch.long,
            device=self.device,
        )
        lengths = self.clip_lengths[clip_ids]
        starts = self.clip_starts[clip_ids]
        # Uniform offset in [0, length) per sampled clip.
        offsets = (torch.rand((n,), generator=generator, device=self.device) * lengths.float()).long()
        offsets = torch.minimum(offsets, lengths - 1)
        return starts + offsets
