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
import time

import torch
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan
from rsl_rl.utils.logger import Logger

from rsl_rl_flashsac.algorithms import FlashSAC
from rsl_rl_flashsac.utils import resolve_callable


class OffPolicyRunner(OnPolicyRunner):
    """Off-policy runner for training with FlashSAC.

    Inherits checkpointing, policy export, and multi-GPU setup from the upstream
    :class:`rsl_rl.runners.OnPolicyRunner`; overrides construction (to resolve FlashSAC
    classes) and the learning loop (off-policy update cadence and windowed logging).
    """

    alg: FlashSAC
    """The FlashSAC algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Resolve the symmetry augmentation config (injects the env under symmetry_cfg["env"])
        # before algorithm construction, matching the official rsl_rl PPO/OnPolicyRunner pattern.
        self.cfg["algorithm"] = resolve_symmetry_config(self.cfg["algorithm"], self.env)

        # Create the algorithm (classes resolved from rsl_rl_flashsac first, then rsl_rl)
        alg_class: type[FlashSAC] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self.start_training = self.cfg.get("start_training", 0)
        self.log_interval = self.cfg.get("log_interval", 20)
        # Training wall-clock bookkeeping (persisted into checkpoints for time-axis eval curves)
        self._wall_time_offset = 0.0
        self._learn_start_time: float | None = None
        self.loaded_wall_time: float | None = None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        self._learn_start_time = time.time()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        # A non-positive save_interval derives ~10 checkpoints per run (9 periodic + final)
        save_interval = self.cfg["save_interval"]
        if save_interval <= 0:
            save_interval = max(1, num_learning_iterations // 10)

        # Off-policy iterations are short (typically one env step); logging is windowed
        # over `log_interval` iterations to keep console output and writer traffic sane.
        window_collect_time = 0.0
        window_learn_time = 0.0
        window_iters = 0

        for it in range(start_it, total_it):
            start = time.time()
            # Rollout under no_grad (NOT inference_mode: the compiled actor forward uses CUDA
            # graphs, and inference-mode tensors interacting with the CUDA-graph memory pool
            # corrupt its liveness bookkeeping — the official FlashSAC also samples under no_grad)
            with torch.no_grad():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs) if self.alg.can_start_training() else self.alg.act_random(obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    next_obs, rewards, dones = (
                        next_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards=None)
                    obs = next_obs

                stop = time.time()
                collect_time = stop - start
                start = stop

            # Update policy (the algorithm additionally gates on the replay buffer warm-up)
            loss_dict = self.alg.update() if it >= self.start_training else {}

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            window_collect_time += collect_time
            window_learn_time += learn_time
            window_iters += 1

            # Log information
            if (it % self.log_interval == 0) or (it == total_it - 1):
                # Pass per-iteration averages so the logger's FPS stays correct...
                avg_collect = window_collect_time / window_iters
                avg_learn = window_learn_time / window_iters
                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=avg_collect,
                    learn_time=avg_learn,
                    loss_dict=loss_dict,
                    learning_rate=self.alg.actor_learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=None,
                )
                # ...and correct the logger's running totals for the skipped iterations
                # (logger.log() accumulates one iteration's steps/time per call).
                if self.logger.writer is not None:
                    collection_size = self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size
                    self.logger.tot_timesteps += collection_size * (window_iters - 1)
                    self.logger.tot_time += (window_collect_time + window_learn_time) - (avg_collect + avg_learn)
                    self.logger.writer.add_scalar("Policy/temperature", self.alg.alpha, it)
                    # Cumulative environment steps (all envs, all ranks) — usable as an
                    # alternative x-axis for sample-efficiency curves in W&B.
                    self.logger.writer.add_scalar("Train/env_steps", self.logger.tot_timesteps, it)
                    self.logger.writer.add_scalar("Train/wall_time", self._elapsed_wall_time(), it)
                window_collect_time = 0.0
                window_learn_time = 0.0
                window_iters = 0

            # Save model
            if self.logger.writer is not None and it % save_interval == 0 and it != 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def export_cenet_to_jit(self, path: str, filename: str = "cenet.pt") -> None:
        """Export the policy's CENet (DreamWaQ variant) as a TorchScript file."""
        policy = self.alg.get_policy()
        if not hasattr(policy, "cenet_as_jit"):
            raise AttributeError(f"{type(policy).__name__} has no CENet; nothing to export.")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        jit_module = policy.cenet_as_jit().to("cpu")  # type: ignore[operator]
        scripted = torch.jit.script(jit_module)
        scripted.save(os.path.join(path, filename))

    def _elapsed_wall_time(self) -> float:
        """Cumulative training wall-clock seconds (across resumes)."""
        running = time.time() - self._learn_start_time if self._learn_start_time is not None else 0.0
        return self._wall_time_offset + running

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state, including the elapsed training wall time."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["wall_time"] = self._elapsed_wall_time()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state, restoring the wall-time offset when present."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        self.loaded_wall_time = loaded_dict.get("wall_time")
        if self.loaded_wall_time is not None:
            self._wall_time_offset = self.loaded_wall_time
        return loaded_dict["infos"]
