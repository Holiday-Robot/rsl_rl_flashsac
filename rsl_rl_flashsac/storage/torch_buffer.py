# Copyright (c) 2021-2026, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, Holiday Robotics
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from RSL-RL Project (BSD-3-Clause license),
# with modifications by Holiday Robotics (BSD-3-Clause license).

from __future__ import annotations

import os
from collections import deque
from typing import Any

import torch
from tensordict import TensorDict

# A transition/batch is a dict with keys: observation, action, reward, terminated, truncated,
# next_observation. Observations are TensorDicts of observation groups; the rest are tensors.
Batch = dict[str, Any]


class TorchUniformBuffer:
    """Uniform experience replay buffer using PyTorch tensors.

    Transitions are stored in flat slots of total size ``max_length`` (across all environments);
    each ``add()`` call inserts one batch of ``num_envs`` transitions. N-step returns are
    aggregated at insertion time from a deque of the last ``n_step`` vector-env transitions.
    Observations are stored per observation group and returned as TensorDicts on sampling.
    """

    def __init__(
        self,
        obs: TensorDict,
        num_actions: int,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device: str,
        obs_storage_dtype: torch.dtype | None = None,
        store_groups: list[str] | None = None,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            obs: Initial observation TensorDict (used to infer group shapes).
            num_actions: Dimension of the action space.
            n_step: Number of steps for n-step returns.
            gamma: Discount factor for n-step returns.
            max_length: Maximum number of transitions (across all environments).
            min_length: Minimum number of transitions before sampling is allowed.
            sample_batch_size: Batch size returned by ``sample()``.
            device: Device on which the buffer storage lives (may differ from the training device).
            obs_storage_dtype: Optional storage dtype for observations (cast back to float32 on sampling).
            store_groups: Observation groups to store. Defaults to all groups in ``obs``.
        """
        self._n_step = n_step
        self._gamma = gamma
        self._max_length = max_length
        self._min_length = min_length
        self._sample_batch_size = sample_batch_size
        self._num_actions = num_actions
        self._device = torch.device(device)
        self._obs_storage_dtype = obs_storage_dtype

        self._store_groups = list(store_groups) if store_groups is not None else list(obs.keys())
        self._obs_shapes = {key: tuple(obs[key].shape[1:]) for key in self._store_groups}
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def _make_obs_storage(self, pin: bool) -> dict[str, torch.Tensor]:
        dtype = self._obs_storage_dtype or torch.float32
        return {
            key: torch.empty((self._max_length, *shape), dtype=dtype, device=self._device, pin_memory=pin)
            for key, shape in self._obs_shapes.items()
        }

    def reset(self) -> None:
        """Clear the buffer and re-allocate storage."""
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        self._observations = self._make_obs_storage(pin)
        self._next_observations = self._make_obs_storage(pin)
        self._actions = torch.empty((m, self._num_actions), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0

    def _to_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Copy a tensor to the buffer device (cloned so later in-place env updates cannot alias)."""
        return value.detach().to(self._device, copy=True)

    def _obs_to_dict(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        return {key: self._to_tensor(obs[key]) for key in self._store_groups}

    def _copy_transition(self, transition: Batch) -> dict[str, Any]:
        return {
            "observation": self._obs_to_dict(transition["observation"]),
            "action": self._to_tensor(transition["action"]),
            "reward": self._to_tensor(transition["reward"]).float(),
            "terminated": self._to_tensor(transition["terminated"]).float(),
            "truncated": self._to_tensor(transition["truncated"]).float(),
            "next_observation": self._obs_to_dict(transition["next_observation"]),
        }

    def _get_n_step_prev_transition(self) -> dict[str, Any]:
        """Aggregate the deque into the n-step transition anchored at its oldest element.

        Computes the n-step return, done status, and next observation: rewards after the first
        episode end in the window are dropped, and the next observation / done flags are taken
        from the first episode-ending step.
        """
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        # clone last transition
        n_step_reward = curr_transition["reward"].clone()
        n_step_terminated = curr_transition["terminated"].clone()
        n_step_truncated = curr_transition["truncated"].clone()
        n_step_next_observation = {key: value.clone() for key, value in curr_transition["next_observation"].items()}

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n,)
            terminated = transition["terminated"]  # (n,)
            truncated = transition["truncated"]  # (n,)

            # compute n-step return
            done = (terminated.bool() | truncated.bool()).float()
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            for key in n_step_next_observation:
                n_step_next_observation[key][done_mask] = transition["next_observation"][key][done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation

        return n_step_prev_transition

    def _insert_indices(self, add_batch_size: int) -> Any:
        end_idx = self._current_idx + add_batch_size
        if end_idx <= self._max_length:
            # Contiguous slice — avoids scatter and tensor allocation
            return slice(self._current_idx, end_idx)
        return (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length

    def add(self, transition: Batch) -> None:
        """Insert one vector-env transition; writes the n-step-aggregated transition once warm."""
        self._n_step_transitions.append(self._copy_transition(transition))

        if len(self._n_step_transitions) < self._n_step:
            return

        n_step_prev_transition = self._get_n_step_prev_transition()
        add_batch_size = len(n_step_prev_transition["reward"])
        idxs = self._insert_indices(add_batch_size)

        for key, storage in self._observations.items():
            storage[idxs] = n_step_prev_transition["observation"][key].to(storage.dtype)
        for key, storage in self._next_observations.items():
            storage[idxs] = n_step_prev_transition["next_observation"][key].to(storage.dtype)
        self._actions[idxs] = n_step_prev_transition["action"]
        self._rewards[idxs] = n_step_prev_transition["reward"]
        self._terminateds[idxs] = n_step_prev_transition["terminated"]
        self._truncateds[idxs] = n_step_prev_transition["truncated"]

        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def _obs_batch(self, storage: dict[str, torch.Tensor], idxs: torch.Tensor) -> TensorDict:
        cast = self._obs_storage_dtype is not None
        return TensorDict(
            {key: value[idxs].float() if cast else value[idxs] for key, value in storage.items()},
            batch_size=[idxs.shape[0]],
            device=self._device,
        )

    def sample(self, sample_idxs: torch.Tensor | None = None) -> Batch:
        """Sample a uniform batch of transitions (observations returned as TensorDicts)."""
        if sample_idxs is None:
            idxs = torch.randint(0, self._num_in_buffer, (self._sample_batch_size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device, dtype=torch.long)

        return {
            "observation": self._obs_batch(self._observations, idxs),
            "action": self._actions[idxs],
            "reward": self._rewards[idxs],
            "terminated": self._terminateds[idxs],
            "truncated": self._truncateds[idxs],
            "next_observation": self._obs_batch(self._next_observations, idxs),
        }

    def save(self, path: str) -> None:
        """Save buffer contents and metadata to the given file path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = self._num_in_buffer
        dataset: dict[str, Any] = {
            "observation": {key: value[:n] for key, value in self._observations.items()},
            "action": self._actions[:n],
            "reward": self._rewards[:n],
            "terminated": self._terminateds[:n],
            "truncated": self._truncateds[:n],
            "next_observation": {key: value[:n] for key, value in self._next_observations.items()},
            "num_in_buffer": self._num_in_buffer,
            "current_idx": self._current_idx,
        }
        torch.save(dataset, path)

    def load(self, path: str) -> None:
        """Load buffer contents and metadata from the given file path."""
        dataset = torch.load(path, map_location=self._device, weights_only=False)
        n = dataset["num_in_buffer"]

        for key, storage in self._observations.items():
            storage[:n] = dataset["observation"][key]
        for key, storage in self._next_observations.items():
            storage[:n] = dataset["next_observation"][key]
        self._actions[:n] = dataset["action"]
        self._rewards[:n] = dataset["reward"]
        self._terminateds[:n] = dataset["terminated"]
        self._truncateds[:n] = dataset["truncated"]

        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        # Note: _n_step_transitions is intentionally not saved/loaded.
        # At most (n_step - 1) in-flight transitions are lost, which is negligible.
        self._n_step_transitions.clear()


class MemoryEfficientTorchUniformBuffer(TorchUniformBuffer):
    """Store only observations and reconstruct n-step next observations by index.

    The newest ``n_step`` vector-env batches are not sampled because their future observation
    slots have not been written yet. Episode ends keep a sparse copy of final next observations
    because the following observation slot may already contain a reset observation.
    """

    def reset(self) -> None:
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        self._observations = self._make_obs_storage(pin)
        self._actions = torch.empty((m, self._num_actions), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0
        self._add_batch_size: int | None = None
        self._episode_end_next_observations: dict[int, dict[str, torch.Tensor]] = {}

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append(self._copy_transition(transition))

        if len(self._n_step_transitions) < self._n_step:
            return

        n_step_prev_transition = self._get_n_step_prev_transition()
        add_batch_size = len(n_step_prev_transition["reward"])
        if self._add_batch_size is None:
            self._add_batch_size = add_batch_size
            if self._n_step * add_batch_size >= self._max_length:
                raise ValueError("max_length must be larger than n_step * add_batch_size")
        elif add_batch_size != self._add_batch_size:
            raise ValueError("MemoryEfficientTorchUniformBuffer requires a constant add batch size")

        end_idx = self._current_idx + add_batch_size
        idx_tensor: torch.Tensor | None = None
        idxs: Any = slice(self._current_idx, end_idx)
        if end_idx > self._max_length:
            idx_tensor = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length
            idxs = idx_tensor

        for key, storage in self._observations.items():
            storage[idxs] = n_step_prev_transition["observation"][key].to(storage.dtype)
        self._actions[idxs] = n_step_prev_transition["action"]
        self._rewards[idxs] = n_step_prev_transition["reward"]
        self._terminateds[idxs] = n_step_prev_transition["terminated"]
        self._truncateds[idxs] = n_step_prev_transition["truncated"]

        # Overwritten slots no longer belong to the episode ends they were recorded for
        if self._episode_end_next_observations:
            if end_idx > self._max_length:
                assert idx_tensor is not None
                for idx in idx_tensor.detach().cpu().tolist():
                    self._episode_end_next_observations.pop(int(idx), None)
            else:
                for idx in range(self._current_idx, end_idx):
                    self._episode_end_next_observations.pop(idx, None)

        episode_end_mask = n_step_prev_transition["terminated"].bool() | n_step_prev_transition["truncated"].bool()
        if episode_end_mask.any():
            if end_idx > self._max_length:
                assert idx_tensor is not None
                episode_end_idxs = idx_tensor[episode_end_mask].detach().cpu().tolist()
            else:
                episode_end_positions = episode_end_mask.nonzero(as_tuple=False).squeeze(-1)
                episode_end_idxs = (episode_end_positions + self._current_idx).detach().cpu().tolist()
            episode_end_obs = {
                key: value[episode_end_mask].to(self._observations[key].dtype)
                for key, value in n_step_prev_transition["next_observation"].items()
            }
            for row, idx in enumerate(episode_end_idxs):
                self._episode_end_next_observations[int(idx)] = {
                    key: value[row].detach().clone() for key, value in episode_end_obs.items()
                }

        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        return (
            self._num_in_buffer >= self._min_length
            and self._add_batch_size is not None
            and self._num_in_buffer > self._n_step * self._add_batch_size
        )

    def sample(self, sample_idxs: torch.Tensor | None = None) -> Batch:
        assert self._add_batch_size is not None
        if sample_idxs is None:
            sample_high = self._num_in_buffer - self._n_step * self._add_batch_size
            idxs = torch.randint(0, sample_high, (self._sample_batch_size,), device=self._device)
            if self._num_in_buffer == self._max_length:
                idxs = (idxs + self._current_idx) % self._max_length
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device, dtype=torch.long)

        batch: Batch = {
            "observation": self._obs_batch(self._observations, idxs),
            "action": self._actions[idxs],
            "reward": self._rewards[idxs],
            "terminated": self._terminateds[idxs],
            "truncated": self._truncateds[idxs],
        }

        next_idxs = (idxs + self._n_step * self._add_batch_size) % self._max_length
        next_observation = self._obs_batch(self._observations, next_idxs)
        if self._episode_end_next_observations:
            hits = [
                (pos, obs)
                for pos, idx in enumerate(idxs.detach().cpu().tolist())
                if (obs := self._episode_end_next_observations.get(int(idx))) is not None
            ]
            if hits:
                positions = torch.as_tensor([pos for pos, _ in hits], device=self._device)
                cast = self._obs_storage_dtype is not None
                for key in self._store_groups:
                    stacked = torch.stack([obs[key] for _, obs in hits]).to(self._device)
                    next_observation[key][positions] = stacked.float() if cast else stacked
        batch["next_observation"] = next_observation

        return batch

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = self._num_in_buffer
        torch.save(
            {
                "observation": {key: value[:n] for key, value in self._observations.items()},
                "action": self._actions[:n],
                "reward": self._rewards[:n],
                "terminated": self._terminateds[:n],
                "truncated": self._truncateds[:n],
                "num_in_buffer": self._num_in_buffer,
                "current_idx": self._current_idx,
                "add_batch_size": self._add_batch_size,
                "episode_end_next_observations": self._episode_end_next_observations,
            },
            path,
        )

    def load(self, path: str) -> None:
        dataset = torch.load(path, map_location=self._device, weights_only=False)
        n = dataset["num_in_buffer"]

        for key, storage in self._observations.items():
            storage[:n] = dataset["observation"][key]
        self._actions[:n] = dataset["action"]
        self._rewards[:n] = dataset["reward"]
        self._terminateds[:n] = dataset["terminated"]
        self._truncateds[:n] = dataset["truncated"]
        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        self._add_batch_size = dataset["add_batch_size"]
        self._episode_end_next_observations = dataset["episode_end_next_observations"]
        self._n_step_transitions.clear()
