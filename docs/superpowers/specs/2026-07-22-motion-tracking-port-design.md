# G1 motion-tracking task port — design

Date: 2026-07-22
Source: https://github.com/leeyngdo/FlashSAC @ `a03550c` (`flash_rl/envs/isaaclab_envs/tracking`
and its supporting `mdp`/`utils`/`robots` modules).
Mode: autonomous run — assumptions below were taken conservatively instead of interactive
brainstorming; each is reversible and called out in the PR summary.

## Goal

Port the BeyondMimic-style Unitree G1 whole-body motion-tracking task (adaptive-sampling
`MotionCommand`, multi-clip `MotionLoader`, tracking observations/rewards/terminations,
WBT domain randomization) into `isaaclab_flashsac`, registered as out-of-tree gym tasks and
trainable via `flashsac-train`, following this repo's conventions.

## Non-goals

- The Hydra/OmegaConf config override layer (`tracking/overrides.py`, the
  `OBS_TERMS`/`REW_TERMS`/`TERM_TERMS` registries). This repo configures envs through
  configclasses and `rl_cfg.py`; term functions are referenced directly, as
  `envs/g1_dreamwaq/env_cfg.py` already does.
- `robots/actuator.py` (unused by the tracking task), `utils/video.py` and
  `utils/action_bounds.py` (this repo's `play.py` records video via gym wrappers and
  `FlashSACVecEnvWrapper` already implements joint-limit action bounds).
- `feet_contact_time` (registered upstream but not used by any tracking reward term).
- Symmetry augmentation (the reference tracking task does not use it).

## Stated assumptions (autonomous)

1. **Faithful numerics, adapted layout.** All math (command resampling, adaptive
   sampling, relative body targets, reward/termination functions, DR events, loader
   pooling) is ported verbatim; only packaging and config plumbing change.
2. **Robot articulation reuses the vendored USD.** The reference spawns a URDF with
   cylinder-to-capsule conversion (`unitree_description/urdf/g1/main.urdf`) that is *not
   shipped* in either repo. This port instead reuses the vendored
   `isaaclab_flashsac/assets/g1_29dof/g1_29dof_rev_1_0.usd` (same 29-DoF G1, same
   joint/body names, byte-identical actuator table in `g1_dreamwaq/assets.py`) with the
   WBT initial state (base z = 0.76, WBT default joint angles, soft limit factor 0.9).
   Known deviation: collision geometry is the USD's mesh approximation rather than
   capsules, so contact-dependent numerics can differ from upstream.
3. **Effective training values, not code defaults.** Reward weights/stds, termination
   thresholds, and adaptive-sampling parameters follow the reference's actual training
   config (`configs/env/isaaclab_tracking.yaml`), which overrides one code default:
   `action_rate_l2` weight is **-0.1** (yaml) not -1.0 (rewards_cfg.py).
4. **Gym ids preserved**: `Isaac-Tracking-Flat-G1-v0` and `Isaac-Tracking-Flat-G1-WoSE-v0`,
   plus `-Play-` variants added per this repo's convention (smaller scene, corruption and
   pushes off — an addition, upstream has no play cfg).
5. **Motion clips via CLI.** `--motion_files` (one or more `.npz` paths or a directory) is
   added to `train.py`/`play.py`/`eval.py` and written into
   `env_cfg.commands.motion.motion_files` after `parse_env_cfg`. Launching a tracking task
   without it fails with a clear error. Clip format: BeyondMimic `.npz`
   (`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`,
   `body_ang_vel_w`, optional `fps`/`joint_names`/`body_names` metadata).

## Layout

The MDP term functions live in the shared `isaaclab_flashsac/mdp` term library (obs terms
one function per file, the other kinds grouped by role, per the mdp package conventions);
the env package holds only the task assembly:

```
isaaclab_flashsac/
  utils/motion_loader.py              # MotionLoader — numpy/torch only, unit-testable
                                      # without Isaac Lab
  mdp/
    commands/motion_command.py        # MotionCommand + MotionCommandCfg (+ marker helpers)
    obs/motion/<term>.py              # motion_anchor_{pos,ori}_b, robot_anchor_ori_w,
                                      # robot_anchor_{lin,ang}_vel_w, robot_body_{pos,ori}_b
    rewards/motion_tracking.py        # motion_*_error_exp terms, anti_shake_ang_vel_l2,
                                      # _get_body_indexes
    terminations/motion_tracking.py   # bad_anchor_pos[_z_only], bad_anchor_ori,
                                      # bad_motion_body_pos[_z_only]
    events/domain_rand.py             # randomize_joint_default_pos, randomize_rigid_body_com
  envs/tracking/
    __init__.py                       # import-light docstring + apply_motion_files helper
    assets.py                         # G1_TRACKING_CFG (vendored USD + WBT init state)
                                      # + body-name groups
    env_cfg.py                        # scene/observations/commands/rewards/terminations/
                                      # events cfgs, TrackingEnvCfg assembler,
                                      # G1FlatTrackingEnvCfg (+ WoSE, + _PLAY)
```

Isaac Lab builtin terms are referenced as `mdp.<name>` via
`import isaaclab.envs.mdp as mdp`; local terms via explicit module imports (repo
convention, no star-import aggregation package). `isaaclab_flashsac.mdp` degrades
gracefully when Isaac Lab is absent: the runtime-isaaclab subpackages
(`commands`/`events`/`rewards`/`terminations`) resolve to `None` and only import-light
`obs` terms are re-exported eagerly.

## Wiring

- `envs/__init__.py`: four lazy-entry-point `gym.register` calls
  (`Isaac-Tracking-Flat-G1{,-WoSE}{,-Play}-v0`).
- `rl_cfg.py`: `G1FlatTrackingFlashSACCfg` and `G1FlatTrackingWoSEFlashSACCfg` —
  unified FlashSAC hyperparameters plus
  `obs_groups={"actor": ["policy"], "critic": ["critic"]}` (the env's privileged group is
  named `critic` so the asymmetric split matches upstream) and
  `action_bound="joint_limit"` (yaml: `action_bound.type: joint_limit, fraction: 1.0`).
- `scripts/{train,play,eval}.py`: `--motion_files` passthrough as in assumption 5.

## Key invariants carried over (documented in code)

- Resampling happens at the *per-clip* end, not the pooled end (`clip_end_of_frame`).
- `_update_relative_body_targets` is re-run inside `_resample_command` because Isaac Lab
  does not call `compute()` again before the first post-reset reward/termination.
- Adaptive sampling accumulates failures in `_current_bin_failed` and folds them into the
  EMA once per `_update_command`.
- The reset-frame guard uses the root frame instead of the configured anchor body on the
  first step after reset (`episode_length_buf == 0`).
- `MotionLoader` must stay import-light (numpy/torch only; no isaaclab).

## Testing

- `tests/unit/utils/test_motion_loader.py` — real unit tests (no stubs needed):
  file resolution (single/list/dir), pooling and clip boundaries
  (`clip_id_of_frame`/`clip_end_of_frame`), both balance modes with a seeded generator,
  name-based joint/body reordering, error paths (missing keys, bad names, bad mode).
- `tests/unit/test_rl_cfg_tasks.py` — extend: tracking ids registered, play alias maps to
  the same cfg, obs_groups/action_bound values, experiment-name derivation, and a
  `construct_algorithm` smoke with `policy`/`critic` groups. The existing
  "12 benchmark tasks" filter is updated to also exclude `Tracking` ids.
- `tests/integration/test_isaaclab_integration.py` — tracking cfg roundtrip
  (importorskip; full env construction needs Isaac Sim + motion data and stays out of CI).
- Verification: `bin/lint` and `bin/test` must pass; training smoke run requires an Isaac
  Lab machine + motion clips and is left to the user (documented in the PR summary).
