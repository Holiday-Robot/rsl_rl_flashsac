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

"""DreamWaQ variant of FlashSAC: adds the CENet estimation loss."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict

from rsl_rl_flashsac.algorithms.flash_sac import FlashSAC
from rsl_rl_flashsac.models import FlashSACActor, FlashSACCritic, FlashSACDreamwaqActor
from rsl_rl_flashsac.storage import MemoryEfficientTorchUniformBuffer, TorchUniformBuffer
from rsl_rl_flashsac.utils import resolve_callable


class FlashSACDreamwaq(FlashSAC):
    """FlashSAC with a DreamWaQ CENet supervised by privileged lin vel.

    The CENet lives inside the actor model and already receives actor-loss
    gradients through ``flatten_obs``; this class adds the supervision term
    ``MSE(actor_obs[:, :d], critic_obs[:, :d])`` where the critic obs group
    must start with the ground-truth base linear velocity.
    """

    # Narrowed from FlashSACActor: this class reads the CENet-specific attributes.
    actor: FlashSACDreamwaqActor

    def __init__(self, *args, cenet_loss_coeff: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cenet_loss_coeff = cenet_loss_coeff
        self._cenet_loss_sum = 0.0
        self._cenet_update_count = 0

    def _update_actor(
        self,
        actor_obs: torch.Tensor,
        actor_next_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the actor. Returns (actor_loss, entropy)."""
        with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self.use_amp):
            # Forward both current and next observations so BatchNorm sees the same
            # distribution as the critic update (only the first half enters the loss).
            actor_obs_all = torch.cat([actor_obs, actor_next_obs], dim=0)
            actions_all, log_probs_all = self._actor_sample(actor_obs_all, training=True)
            actions = torch.chunk(actions_all, 2, dim=0)[0]
            log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

            # Disable critic gradients to prevent CUDA graph overwriting
            self.critic.critic.requires_grad_(False)
            qs, _ = self._critic_eval(critic_obs, actions, training=False)
            q = torch.minimum(qs[0], qs[1])
            self.critic.critic.requires_grad_(True)

            temp_value = self.temperature().detach()
            actor_loss = (log_probs * temp_value - q).mean()

            # dreamwaq: CENet supervision, added right after actor_loss is first computed.
            # dreamwaq: actor_obs[:, :est_dim] is the CENet estimation spliced in by flatten_obs;
            # dreamwaq: critic_obs[:, :est_dim] is the ground-truth lin vel from the critic obs group.
            est_dim = self.actor.cenet_estimation_dim  # dreamwaq: added
            cenet_loss = F.mse_loss(actor_obs[:, :est_dim], critic_obs[:, :est_dim].detach())  # dreamwaq: added
            actor_loss = actor_loss + self.cenet_loss_coeff * cenet_loss  # dreamwaq: added
            self._cenet_loss_sum += cenet_loss.item()  # dreamwaq: added
            self._cenet_update_count += 1  # dreamwaq: added

            if self.actor_bc_alpha > 0:
                # https://arxiv.org/abs/2306.02451
                q_abs = torch.abs(q).mean().detach()
                bc_loss = ((actions - actions_batch) ** 2).mean()
                actor_loss = actor_loss + self.actor_bc_alpha * q_abs * bc_loss

            entropy = -log_probs.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.grad_scaler.scale(actor_loss).backward()
            # Average scaled grads before unscale so an AMP overflow on any rank is seen by every rank
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor_parameters)
            self.grad_scaler.unscale_(self.actor_optimizer)
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            actor_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.actor_parameters)
            self.actor_optimizer.step()
        self.actor_scheduler.step()

        # Weight normalization after the optimizer step
        with torch.no_grad():
            self._actor_normalize()

        return actor_loss.detach(), entropy.detach()

    def update(self) -> dict:
        """Perform FlashSAC updates with CENet statistics tied to actor updates."""
        self._cenet_loss_sum = 0.0
        self._cenet_update_count = 0

        # Wait until the buffer holds enough transitions (also required for the memory-efficient
        # buffer, whose newest n_step batches cannot be sampled yet)
        if not self.replay_buffer.can_sample():
            return {}

        mean_actor_loss = 0.0
        mean_critic_loss = 0.0
        mean_temperature_loss = 0.0
        mean_entropy = 0.0
        num_actor_updates = 0

        gamma_n = self.gamma**self.n_steps

        for _ in range(self.num_learning_epochs * self.num_mini_batches):
            self._mark_cudagraph_step()
            batch = self._sample_batch()

            obs_batch = batch["observation"].to(self.device, non_blocking=True)
            next_obs_batch = batch["next_observation"].to(self.device, non_blocking=True)
            actions_batch = batch["action"].to(self.device, non_blocking=True)
            rewards = batch["reward"].to(self.device, non_blocking=True)
            terminated = batch["terminated"].to(self.device, non_blocking=True)

            critic_obs = self.critic.flatten_obs(obs_batch)
            critic_next_obs = self.critic.flatten_obs(next_obs_batch)

            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer.normalize_rewards(rewards)

            # CENet BatchNorm statistics update only with the delayed actor update.
            if self.update_step % self.actor_update_period == 0:
                # dreamwaq: one CENet forward over current+next, so BatchNorm sees the joint batch.
                actor_cenet_obs_all = TensorDict(
                    {
                        group: torch.cat([obs_batch[group], next_obs_batch[group]], dim=0)
                        for group in (*self.actor.obs_groups, *self.actor.estimator_obs_groups)
                    },
                    batch_size=[2 * obs_batch.batch_size[0]],
                )
                actor_obs, actor_next_obs = torch.chunk(
                    self.actor.flatten_obs(actor_cenet_obs_all, training=True), 2, dim=0
                )
                actor_loss, entropy = self._update_actor(actor_obs, actor_next_obs, critic_obs, actions_batch)
                temperature_loss = self._update_temperature(entropy)
                mean_actor_loss += actor_loss.item()
                mean_temperature_loss += temperature_loss.item()
                mean_entropy += entropy.item()
                num_actor_updates += 1

            # Use the updated actor/CENet for the critic target without changing CENet BN stats.
            with torch.no_grad():
                critic_actor_next_obs = self.actor.flatten_obs(next_obs_batch, training=False)
            critic_loss = self._update_critic(
                critic_obs, critic_next_obs, critic_actor_next_obs, actions_batch, rewards, terminated, gamma_n
            )
            mean_critic_loss += critic_loss.item()

            with torch.no_grad():
                self._ema_update()

            self.update_step += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "actor": mean_actor_loss / max(num_actor_updates, 1),
            "critic": mean_critic_loss / num_updates,
            "temperature": mean_temperature_loss / max(num_actor_updates, 1),
            "entropy": mean_entropy / max(num_actor_updates, 1),
            "cenet": self._cenet_loss_sum / max(self._cenet_update_count, 1),
        }

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> FlashSACDreamwaq:  # dreamwaq: return type narrowed to FlashSACDreamwaq
        """Construct the FlashSACDreamwaq algorithm with actor, critic, and replay buffer.

        Args:
            obs: Initial observations from the environment.
            env: The vectorized environment.
            cfg: Configuration dictionary.
            device: Device to place models on.

        Returns
        -------
            Initialized FlashSACDreamwaq algorithm instance.
        """
        # Resolve class callables
        alg_class: type[FlashSAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FlashSACActor] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FlashSACCritic] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        cfg["obs_groups"] = resolve_obs_groups(
            obs, cfg["obs_groups"], ["actor", "critic", "estimator"]
        )  # dreamwaq: added "estimator" to the default sets

        # Logger compatibility (FlashSAC does not support RND)
        cfg["algorithm"].setdefault("rnd_cfg", None)

        # Initialize the actor
        actor: FlashSACActor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"FlashSAC Actor: {actor}")

        # Action scaling is provided by the environment (FlashSACVecEnvWrapper computes it
        # from the robot joint limits); fall back to identity scaling when absent.
        action_bias = getattr(env, "action_bias", None)
        action_scale = getattr(env, "action_scale", None)
        if action_bias is None or action_scale is None:
            warnings.warn(
                "FlashSAC: env exposes no action_bias/action_scale (see FlashSACVecEnvWrapper);"
                " using identity action scaling.",
                stacklevel=2,
            )
            action_bias = torch.zeros(env.num_actions, device=device)
            action_scale = torch.ones(env.num_actions, device=device)
        actor.set_action_scaling(action_bias.to(device), action_scale.to(device))

        # Initialize the critic
        critic: FlashSACCritic = critic_class(
            obs, cfg["obs_groups"], "critic", 1, num_actions=env.num_actions, **cfg["critic"]
        ).to(device)
        print(f"FlashSAC Critic: {critic}")

        # Initialize the replay buffer (stores only the observation groups the models consume)
        alg_cfg = cfg["algorithm"]
        store_groups = sorted(
            set(cfg["obs_groups"]["actor"])
            | set(cfg["obs_groups"]["critic"])
            | set(cfg["obs_groups"]["estimator"])  # dreamwaq: also store the CENet's estimator obs groups
        )
        buffer_obs_dtype = alg_cfg.get("buffer_obs_dtype")
        buffer_class = (
            MemoryEfficientTorchUniformBuffer
            if alg_cfg.get("buffer_optimize_memory_usage", True)
            else TorchUniformBuffer
        )
        replay_buffer = buffer_class(
            obs=obs,
            num_actions=env.num_actions,
            n_step=alg_cfg.get("n_steps", 1),
            gamma=alg_cfg.get("gamma", 0.99),
            max_length=alg_cfg.get("replay_buffer_size", 1_000_000),
            min_length=alg_cfg.get("buffer_min_length", 10_000),
            sample_batch_size=alg_cfg.get("mini_batch_size", 2048),
            device=alg_cfg.get("buffer_device") or device,
            obs_storage_dtype=getattr(torch, buffer_obs_dtype) if buffer_obs_dtype is not None else None,
            store_groups=store_groups,
        )

        # Resolve the learning rate schedule length from the training run length if not given
        if cfg["algorithm"].get("learning_rate_decay_steps") is None:
            max_iterations = cfg.get("max_iterations")
            if max_iterations is None:
                raise ValueError(
                    "FlashSAC: Set algorithm.learning_rate_decay_steps or runner max_iterations to "
                    "resolve the learning rate schedule length."
                )
            num_updates_per_iteration = cfg["algorithm"].get("num_learning_epochs", 1) * cfg["algorithm"].get(
                "num_mini_batches", 1
            )
            cfg["algorithm"]["learning_rate_decay_steps"] = int(max_iterations) * num_updates_per_iteration

        # Initialize the algorithm
        alg: FlashSACDreamwaq = alg_class(  # type: ignore[assignment]  # dreamwaq: alg_class resolves to FlashSACDreamwaq (annotation updated)
            actor, critic, replay_buffer, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg.get("multi_gpu")
        )

        return alg
