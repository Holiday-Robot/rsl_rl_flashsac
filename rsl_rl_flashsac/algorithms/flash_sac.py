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

import math
import warnings
from collections.abc import Callable, Iterable

import torch
import torch.optim as optim
from rsl_rl.env import VecEnv
from rsl_rl.extensions import Symmetry
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict

from rsl_rl_flashsac.models import FlashSACActor, FlashSACCritic, FlashSACTemperature
from rsl_rl_flashsac.storage import MemoryEfficientTorchUniformBuffer, TorchUniformBuffer
from rsl_rl_flashsac.utils import (
    build_truncated_zeta_cdf,
    compute_categorical_td_target,
    resolve_callable,
    resolve_compile_mode,
    sample_integer_from_cdf,
    select_min_q_log_probs,
    warmup_cosine_decay_scheduler,
)
from rsl_rl_flashsac.utils.reward_normalization import RewardNormalizer


class FlashSAC:
    """FlashSAC: fast and stable off-policy RL for high-dimensional control.

    Compared to vanilla SAC it uses a BatchNorm-embedded residual trunk with unit-norm weight
    normalization after every optimizer step, a distributional (categorical) double critic
    trained with cross-entropy, return-based reward normalization, temporally correlated
    exploration noise (zeta-distributed repeat lengths), a warmup-cosine learning rate
    schedule, and optional AMP (fp16) and torch.compile.
    """

    actor: FlashSACActor
    """The actor model."""

    critic: FlashSACCritic
    """The critic model (holds the online and EMA target double critic)."""

    def __init__(
        self,
        actor: FlashSACActor,
        critic: FlashSACCritic,
        replay_buffer: TorchUniformBuffer,
        replay_buffer_size: int = 1_000_000,
        buffer_min_length: int = 10_000,
        buffer_optimize_memory_usage: bool = True,
        buffer_device: str | None = None,
        buffer_obs_dtype: str | None = None,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        mini_batch_size: int = 2048,
        learning_rate_init: float = 3e-4,
        learning_rate_peak: float = 3e-4,
        learning_rate_end: float = 1.5e-4,
        learning_rate_warmup_steps: int = 0,
        learning_rate_decay_steps: int | None = None,
        actor_bc_alpha: float = 0.0,
        actor_noise_zeta_mu: float = 2.0,
        actor_noise_zeta_max: int = 16,
        actor_update_period: int = 2,
        critic_target_update_tau: float = 0.01,
        temp_initial_value: float = 0.01,
        temp_target_sigma: float = 0.15,
        temp_target_entropy: float | None = None,
        gamma: float = 0.99,
        n_steps: int = 1,
        normalize_reward: bool = True,
        normalized_G_max: float = 5.0,
        use_compile: bool = True,
        compile_mode: str = "auto",
        use_amp: bool = True,
        device: str = "cpu",
        # Unsupported extensions (accepted for config compatibility with SAC)
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the FlashSAC algorithm.

        Args:
            actor: The FlashSAC actor model.
            critic: The FlashSAC critic model.
            replay_buffer: An instance of the TorchUniformBuffer (or subclass) replay buffer.
            replay_buffer_size: Max replay buffer size in total transitions across all environments
                (consumed by ``construct_algorithm``).
            buffer_min_length: Minimum number of transitions before updates start
                (consumed by ``construct_algorithm``).
            buffer_optimize_memory_usage: Use the memory-efficient buffer that stores observations
                only once and reconstructs next observations by index (consumed by ``construct_algorithm``).
            buffer_device: Device for the buffer storage; None uses the training device
                (consumed by ``construct_algorithm``).
            buffer_obs_dtype: Optional torch dtype name for observation storage, e.g. "bfloat16"
                (consumed by ``construct_algorithm``).
            num_learning_epochs: How many epochs to run each update.
            num_mini_batches: How many mini-batches (gradient updates) to run per epoch.
            mini_batch_size: Mini-batch size for updates.
            learning_rate_init: Learning rate at the start of the warmup.
            learning_rate_peak: Learning rate after warmup (also the optimizer base LR).
            learning_rate_end: Learning rate at the end of the cosine decay.
            learning_rate_warmup_steps: Number of update steps for the linear warmup.
            learning_rate_decay_steps: Total schedule length in update steps. If None, resolved by
                ``construct_algorithm`` from max_iterations * num_learning_epochs * num_mini_batches.
            actor_bc_alpha: BC regularization coefficient (https://arxiv.org/abs/2306.02451).
            actor_noise_zeta_mu: Zeta distribution exponent for exploration noise repeat lengths.
            actor_noise_zeta_max: Maximum noise repeat length.
            actor_update_period: Actor/temperature update period relative to critic updates.
            critic_target_update_tau: EMA coefficient for the target critic.
            temp_initial_value: Initial temperature value.
            temp_target_sigma: Target per-dimension physical action std used to derive the
                target entropy.
            temp_target_entropy: Explicit target entropy. If None, derived from temp_target_sigma.
            gamma: Discount factor.
            n_steps: Number of steps for n-step returns (consumed by ``construct_algorithm``).
            normalize_reward: Whether to normalize rewards with the running return scale.
            normalized_G_max: Maximum magnitude of the normalized return (should match the critic
                categorical support, i.e. critic min_v/max_v = ∓normalized_G_max).
            use_compile: Whether to torch.compile the network forward passes and update helpers.
            compile_mode: torch.compile mode ('auto' resolves based on the torch version).
            use_amp: Whether to use fp16 automatic mixed precision for actor/critic updates.
            device: Device for training.
            rnd_cfg: Not supported by FlashSAC; must be None.
            symmetry_cfg: ``rsl_rl.extensions.Symmetry`` kwargs (``env``/``data_augmentation_func``/
                ``use_data_augmentation``/``use_mirror_loss``/``mirror_loss_coeff``), or None to
                disable. Resolved into ``self.symmetry`` and consumed by ``_sample_batch``.
                ``use_mirror_loss=True`` is not supported (see the constructor body).
            multi_gpu_cfg: Optional dictionary of multi-GPU configuration parameters.
        """
        if rnd_cfg:
            raise NotImplementedError("FlashSAC does not support RND. Please set rnd_cfg to null.")
        self.rnd = None  # runner compatibility

        # Symmetry data augmentation (see _sample_batch). The mirror loss is not supported: it
        # needs the actor's mean-only forward (as used by rsl_rl.extensions.Symmetry.compute_loss)
        # plumbed alongside sample_action_logp, which FlashSAC's stochastic actor does not
        # currently expose.
        if symmetry_cfg is not None and symmetry_cfg.get("use_mirror_loss"):
            raise NotImplementedError(
                "FlashSAC does not support the symmetry mirror loss; set use_mirror_loss=False "
                "(use_data_augmentation is supported)."
            )
        self.symmetry: Symmetry | None = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Store actor, critic and temperature
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.temperature = FlashSACTemperature(temp_initial_value).to(device)

        # Replay buffer
        self.replay_buffer = replay_buffer
        self.replay_buffer_size = replay_buffer_size

        # Pending transition (observations and actions recorded in act())
        self._last_obs: TensorDict | None = None
        self._last_actions: torch.Tensor | None = None

        # Hyperparameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.n_steps = n_steps
        self.actor_bc_alpha = actor_bc_alpha
        self.actor_update_period = actor_update_period
        self.critic_target_update_tau = critic_target_update_tau
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.compile_mode = resolve_compile_mode(compile_mode)
        self.update_step = 0

        self._device_type = torch.device(device).type

        # Target entropy: entropy of an independent Gaussian with std temp_target_sigma per action
        # dimension, fixed in PHYSICAL action space (the log-prob carries the affine Jacobian, so
        # with wide bounds the normalized std shrinks as 1/range and the physical noise stays
        # ≈ temp_target_sigma).
        if temp_target_entropy is None:
            action_dim = self.actor.output_dim
            self.temp_target_entropy = 0.5 * action_dim * math.log(2 * math.pi * math.e * temp_target_sigma**2)
        else:
            self.temp_target_entropy = temp_target_entropy

        # Optimizers (base LR = peak; the LambdaLR scales relative to it)
        use_fused = self._device_type == "cuda" and torch.cuda.is_available()
        self.actor_parameters = [p for p in self.actor.parameters() if p.requires_grad]
        self.critic_parameters = [p for p in self.critic.critic.parameters() if p.requires_grad]
        self.temperature_parameters = list(self.temperature.parameters())
        self.actor_optimizer = optim.Adam(self.actor_parameters, lr=learning_rate_peak, fused=use_fused)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=learning_rate_peak, fused=use_fused)
        self.temperature_optimizer = optim.Adam(self.temperature_parameters, lr=learning_rate_peak, fused=use_fused)

        # Warmup-cosine learning rate schedule
        if learning_rate_decay_steps is None:
            raise ValueError("learning_rate_decay_steps must be set (or resolvable from max_iterations in the config).")
        warmup_cosine_decay_lr = warmup_cosine_decay_scheduler(
            init_value=learning_rate_init,
            peak_value=learning_rate_peak,
            end_value=learning_rate_end,
            warmup_steps=learning_rate_warmup_steps,
            decay_steps=learning_rate_decay_steps,
        )
        lr_lambda = lambda step: warmup_cosine_decay_lr(step) / learning_rate_peak  # noqa: E731
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lr_lambda)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lr_lambda)
        self.temperature_scheduler = optim.lr_scheduler.LambdaLR(self.temperature_optimizer, lr_lambda=lr_lambda)

        # Init target networks
        self.critic.init_target_networks()

        # Grad scaler for FP16 AMP
        self.grad_scaler = torch.amp.GradScaler(self._device_type, enabled=use_amp)

        # Weight normalization functions (normalize network parameters after init and every step)
        self._actor_normalize = self._build_weight_normalize_fn(self.actor)
        self._critic_normalize = self._build_weight_normalize_fn(self.critic.critic)
        self._target_critic_normalize = self._build_weight_normalize_fn(self.critic.critic_target)
        with torch.no_grad():
            self._actor_normalize()
            self._critic_normalize()
            self._target_critic_normalize()

        # EMA update function for the target critic
        target_params: list[torch.Tensor] = list(self.critic.critic_target.parameters())
        online_params: list[torch.Tensor] = list(self.critic.critic.parameters())
        tau = critic_target_update_tau

        def _ema_update_fn() -> None:
            torch._foreach_lerp_(target_params, online_params, tau)

        self._ema_update = _ema_update_fn

        # Compile hot paths
        if use_compile:
            self._actor_mean_std = torch.compile(self.actor.get_mean_and_std, mode=self.compile_mode)
            self._actor_sample = torch.compile(self.actor.sample_action_logp, mode=self.compile_mode)
            self._critic_eval = torch.compile(self.critic.evaluate, mode=self.compile_mode)
            self._critic_eval_target = torch.compile(self.critic.evaluate_target, mode=self.compile_mode)
            # The standalone TD helpers use default-mode compile (as in the reference
            # implementation): CUDA-graph modes would trip over consuming the target
            # critic's CUDA-graph outputs ("overwritten by a subsequent run").
            self._select_min_q = torch.compile(select_min_q_log_probs)
            self._compute_td_target = torch.compile(compute_categorical_td_target)
            self._actor_normalize = torch.compile(self._actor_normalize, mode=self.compile_mode)
            self._critic_normalize = torch.compile(self._critic_normalize, mode=self.compile_mode)
            self._ema_update = torch.compile(self._ema_update, mode=self.compile_mode)
        else:
            self._actor_mean_std = self.actor.get_mean_and_std
            self._actor_sample = self.actor.sample_action_logp
            self._critic_eval = self.critic.evaluate
            self._critic_eval_target = self.critic.evaluate_target
            self._select_min_q = select_min_q_log_probs
            self._compute_td_target = compute_categorical_td_target

        # Noise repetition (zeta distribution) for temporally correlated exploration
        self._zeta_cdf = build_truncated_zeta_cdf(mu=actor_noise_zeta_mu, max_n=actor_noise_zeta_max).to(device)
        self._noise_repeat_n = 1
        self._noise_repeat_count = 0
        self._cached_noise: torch.Tensor | None = None

        # Reward normalizer
        self.normalize_reward = normalize_reward
        self.reward_normalizer = (
            RewardNormalizer(gamma=gamma, G_max=normalized_G_max, device=device) if normalize_reward else None
        )

    @staticmethod
    def _build_weight_normalize_fn(module: torch.nn.Module) -> Callable[[], None]:
        norm_modules = [m for m in module.modules() if hasattr(m, "normalize_parameters")]

        def _weight_normalize_fn() -> None:
            for m in norm_modules:
                m.normalize_parameters()  # type: ignore[operator]

        return _weight_normalize_fn

    @property
    def alpha(self) -> float:
        """Current temperature value (for logging)."""
        with torch.no_grad():
            return self.temperature().item()

    @property
    def actor_learning_rate(self) -> float:
        """Current actor learning rate (for logging)."""
        return self.actor_optimizer.param_groups[0]["lr"]

    def _mark_cudagraph_step(self) -> None:
        """Mark a CUDA-graph step boundary: outputs of previous compiled invocations may be
        overwritten from here on (everything retained across steps is cloned explicitly).
        """
        if self.use_compile and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    def can_start_training(self) -> bool:
        """Return whether the replay buffer has enough transitions for learning."""
        return self.replay_buffer.can_sample()

    def act_random(self, obs: TensorDict) -> torch.Tensor:
        """Select uniform normalized [-1, 1] actions without advancing the actor's exploration state."""
        actions = torch.empty((obs.batch_size[0], self.actor.output_dim), device=self.device).uniform_(-1.0, 1.0)
        self._last_obs = obs
        self._last_actions = actions
        return actions

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Select a stochastic action with temporally correlated (repeated) exploration noise."""
        with torch.no_grad():
            self._mark_cudagraph_step()
            flat_obs = self.actor.flatten_obs(obs)
            mean, std = self._actor_mean_std(flat_obs, training=False)

            # Re-initialize the shared noise after the sampled number of steps
            if (
                self._cached_noise is None
                or self._cached_noise.shape != mean.shape
                or self._noise_repeat_count >= self._noise_repeat_n
            ):
                self._cached_noise = torch.randn_like(mean)
                self._noise_repeat_n = sample_integer_from_cdf(self._zeta_cdf)
                self._noise_repeat_count = 0
            self._noise_repeat_count += 1

            # Normalized [-1, 1] action; the env wrapper applies the affine scaling.
            actions = torch.tanh(mean + std * self._cached_noise)

            # Clone: `std` may live in a CUDA-graph memory pool (compiled actor forward);
            # retaining it across invocations would corrupt CUDA-graph bookkeeping.
            self.actor.last_action_std = std.detach().clone()
            self._last_obs = obs
            self._last_actions = actions
            return actions

    def process_env_step(self, next_obs: TensorDict, rew: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Process a single environment step and store the transition in the replay buffer."""
        # Time-outs mark truncations (bootstrap continues) even when the true final observation
        # is unavailable — envs without time_outs_obs (e.g. Direct envs) then bootstrap from the
        # post-reset observation, matching the official FlashSAC behavior. Zeroing time_outs here
        # instead would turn every truncation into a termination and cut the bootstrap exactly at
        # the highest-value states (long successful episodes ending by time-out).
        if "time_outs" in extras:
            time_outs = extras["time_outs"].int().to(self.device)
        else:
            time_outs = torch.zeros_like(dones, device=self.device)
        if "time_outs_obs" in extras:
            time_outs_obs = extras["time_outs_obs"].to(self.device)
            true_next_obs = {}
            mask = time_outs.squeeze(-1).bool()

            for key in time_outs_obs.keys():
                true_next_obs[key] = torch.where(mask[:, None], time_outs_obs[key], next_obs[key])
            true_next_obs = TensorDict(true_next_obs, batch_size=next_obs.batch_size)
        else:
            true_next_obs = next_obs

        # Only true terminations cut the bootstrap; time-outs (truncations) continue it
        dones = dones.float().reshape(-1)
        truncated = time_outs.float().reshape(-1)
        terminated = (dones - truncated).clamp(min=0.0)

        # Update the reward normalization statistics (episode ends of any kind reset the return)
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(
                reward=rew.reshape(-1),
                terminated=terminated,
                truncated=truncated,
            )

        # Record transition and insert into replay buffer
        if self._last_obs is None or self._last_actions is None:
            raise RuntimeError("FlashSAC.process_env_step() called before act().")
        self.replay_buffer.add(
            {
                "observation": self._last_obs,
                "action": self._last_actions,
                "reward": rew.reshape(-1),
                "terminated": terminated,
                "truncated": truncated,
                "next_observation": true_next_obs,
            }
        )
        self._last_obs = None
        self._last_actions = None
        self.actor.reset(dones)

    def _sample_batch(self) -> dict:
        """Sample one mini-batch from the replay buffer, applying symmetry augmentation.

        Adapted from the reference FlashSAC's ``_sample_and_prepare_batches`` (which draws one
        large batch and augments/splits it once for several updates): rsl_rl_flashsac instead
        samples one mini-batch per update iteration in the ``update()`` loop, so this augments
        (and doubles) that single mini-batch in its place. Augmentation itself is unchanged from
        the reference: the observations, next-observations, and actions are mirrored via
        ``self.symmetry.data_augmentation_func`` (appending ``[original; mirrored]`` along the
        batch dimension), and the remaining per-transition tensors are repeated to match.

        Returns
        -------
            The sampled (and, if enabled, symmetry-augmented) batch dict, with the same keys as
            ``TorchUniformBuffer.sample()``.
        """
        batch = self.replay_buffer.sample()
        if self.symmetry is not None and self.symmetry.use_data_augmentation:
            augment = self.symmetry.data_augmentation_func
            obs, actions = augment(env=self.symmetry.env, obs=batch["observation"], actions=batch["action"])
            next_obs, _ = augment(env=self.symmetry.env, obs=batch["next_observation"], actions=None)
            num_aug = obs.batch_size[0] // batch["observation"].batch_size[0]
            batch = {
                "observation": obs,
                "action": actions,
                "reward": batch["reward"].repeat(num_aug),
                "terminated": batch["terminated"].repeat(num_aug),
                "truncated": batch["truncated"].repeat(num_aug),
                "next_observation": next_obs,
            }
        return batch

    def update(self) -> dict:
        """Perform FlashSAC updates, returning mean losses."""
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

            # Flatten observation groups outside the compiled forward passes.
            actor_obs = self.actor.flatten_obs(obs_batch, training=True)
            actor_next_obs = self.actor.flatten_obs(next_obs_batch, training=True)
            critic_obs = self.critic.flatten_obs(obs_batch)
            critic_next_obs = self.critic.flatten_obs(next_obs_batch)

            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer.normalize_rewards(rewards)

            # Actor and temperature update (delayed), then critic update with the updated actor
            if self.update_step % self.actor_update_period == 0:
                actor_loss, entropy = self._update_actor(actor_obs, actor_next_obs, critic_obs, actions_batch)
                temperature_loss = self._update_temperature(entropy)
                mean_actor_loss += actor_loss.item()
                mean_temperature_loss += temperature_loss.item()
                mean_entropy += entropy.item()
                num_actor_updates += 1

            critic_loss = self._update_critic(
                critic_obs, critic_next_obs, actor_next_obs, actions_batch, rewards, terminated, gamma_n
            )
            mean_critic_loss += critic_loss.item()

            # EMA update of the target critic
            with torch.no_grad():
                self._ema_update()

            self.update_step += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "actor": mean_actor_loss / max(num_actor_updates, 1),
            "critic": mean_critic_loss / num_updates,
            "temperature": mean_temperature_loss / max(num_actor_updates, 1),
            "entropy": mean_entropy / max(num_actor_updates, 1),
        }
        return loss_dict

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

    def _update_temperature(self, entropy: torch.Tensor) -> torch.Tensor:
        """Update the temperature towards the target entropy. Returns the temperature loss."""
        temperature_value = self.temperature().clone()
        temperature_loss = temperature_value * (entropy.detach() - self.temp_target_entropy).mean()

        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters(self.temperature_parameters)
        self.temperature_optimizer.step()
        self.temperature_scheduler.step()

        return temperature_loss.detach()

    def _update_critic(
        self,
        critic_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
        actor_next_obs: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        gamma_n: float,
    ) -> torch.Tensor:
        """Update the critic with the categorical cross-entropy TD loss. Returns the critic loss."""
        with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                next_actions, next_log_probs = self._actor_sample(actor_next_obs, training=False)
                # Clone to prevent CUDA graph overwriting
                next_actions = next_actions.clone()
                next_log_probs = next_log_probs.clone()

                temp_value = self.temperature()
                next_actor_entropy = temp_value * next_log_probs

                # Joint forward over (obs, action) and (next_obs, next_action) so BatchNorm
                # statistics cover both halves (CrossQ-style; target BN stats update here).
                obs_all = torch.cat([critic_obs, critic_next_obs], dim=0)
                act_all = torch.cat([actions_batch, next_actions], dim=0)

                qs_all, q_infos_all = self._critic_eval_target(obs_all, act_all, training=True)
                next_qs = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
                next_q_log_probs = self._select_min_q(next_qs, next_q_log_probs)

                target_probs = self._compute_td_target(
                    next_q_log_probs,
                    rewards,
                    terminated,
                    next_actor_entropy,
                    gamma_n,
                    self.critic.num_bins,
                    self.critic.min_v,
                    self.critic.max_v,
                )

            _pred_qs_all, pred_q_infos = self._critic_eval(obs_all, act_all, training=True)
            pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]

            ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (num_qs, B)
            critic_loss = ce_loss.mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.grad_scaler.scale(critic_loss).backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.critic_parameters)
            self.grad_scaler.unscale_(self.critic_optimizer)
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            critic_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.critic_parameters)
            self.critic_optimizer.step()
        self.critic_scheduler.step()

        with torch.no_grad():
            self._critic_normalize()

        return critic_loss.detach()

    def train_mode(self) -> None:
        """Set actor, critic and temperature to training mode."""
        self.actor.train()
        self.critic.train()
        self.temperature.train()

    def eval_mode(self) -> None:
        """Set actor, critic and temperature to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.temperature.eval()

    def get_policy(self) -> FlashSACActor:
        """Get the policy model (actor)."""
        return self.actor

    def save(self) -> dict:
        """Return a dict of all model and training states for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "temperature_state_dict": self.temperature.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "temperature_optimizer_state_dict": self.temperature_optimizer.state_dict(),
            "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
            "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
            "temperature_scheduler_state_dict": self.temperature_scheduler.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            "update_step": self.update_step,
            "reward_normalizer_state_dict": (
                self.reward_normalizer.state_dict() if self.reward_normalizer is not None else None
            ),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict.

        Args:
            loaded_dict: Dictionary of saved model states.
            load_cfg: Dictionary specifying which components to load. If None, loads all.
            strict: Whether to strictly enforce state dict key matching.

        Returns
        -------
            Whether the iteration counter should be restored.
        """
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}

        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
            self.temperature.load_state_dict(loaded_dict["temperature_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.temperature_optimizer.load_state_dict(loaded_dict["temperature_optimizer_state_dict"])
            self.actor_scheduler.load_state_dict(loaded_dict["actor_scheduler_state_dict"])
            self.critic_scheduler.load_state_dict(loaded_dict["critic_scheduler_state_dict"])
            self.temperature_scheduler.load_state_dict(loaded_dict["temperature_scheduler_state_dict"])
            self.grad_scaler.load_state_dict(loaded_dict["grad_scaler_state_dict"])
            self.update_step = loaded_dict.get("update_step", 0)
            if self.reward_normalizer is not None and loaded_dict.get("reward_normalizer_state_dict") is not None:
                self.reward_normalizer.load_state_dict(loaded_dict["reward_normalizer_state_dict"])
        return load_cfg.get("iteration", False)

    def clear_storage(self) -> None:
        """Clear the replay buffer."""
        if self.replay_buffer is not None:
            self.replay_buffer.reset()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FlashSAC:
        """Construct the FlashSAC algorithm with actor, critic, and replay buffer.

        Args:
            obs: Initial observations from the environment.
            env: The vectorized environment.
            cfg: Configuration dictionary.
            device: Device to place models on.

        Returns
        -------
            Initialized FlashSAC algorithm instance.
        """
        # Resolve class callables
        alg_class: type[FlashSAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FlashSACActor] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FlashSACCritic] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

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
        store_groups = sorted(set(cfg["obs_groups"]["actor"]) | set(cfg["obs_groups"]["critic"]))
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
        alg: FlashSAC = alg_class(
            actor, critic, replay_buffer, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg.get("multi_gpu")
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters from rank 0 to all GPUs."""
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.temperature.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.temperature.load_state_dict(model_params[2])

    def reduce_parameters(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Collect gradients from the provided params and average them across all GPUs."""
        params = list(params)
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return

        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
