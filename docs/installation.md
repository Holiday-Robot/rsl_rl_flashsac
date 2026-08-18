# Installation

## Development setup (no Isaac Lab)

The `rsl_rl_flashsac` algorithm package works without Isaac Lab.

Prerequisites: [uv](https://docs.astral.sh/uv/), Python >= 3.10.

```bash
git clone https://github.com/leeyngdo/rsl_rl_flashsac
cd rsl_rl_flashsac
uv sync
bin/lint    # ruff format -> check -> mypy
```

## Training setup (Isaac Lab)

### Option A — one-shot install script (recommended)

```bash
bash scripts/install.sh
conda activate flashsac-env
```

The script creates a conda env (`flashsac-env`, Python 3.11), installs the system GL/Vulkan
dependencies, PyTorch (CUDA 12.8 wheels), Isaac Sim 5.1.0 (pip), Isaac Lab v2.3.1 (cloned to
`~/third_party/IsaacLab`), and both repo packages in editable mode, then runs an import sanity
check. It is idempotent (safe to re-run). Versions, env name, and the Isaac Lab checkout
location are overridable — see the header of `scripts/install.sh`. Use `SKIP_APT=1` on machines
without sudo where the system libraries are already present.

### Option B — existing Isaac Lab environment

1. Install Isaac Lab per the
   [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   (Isaac Sim ships Python 3.10/3.11 — this repo supports >= 3.10).
2. Install both packages into the Isaac Lab python environment:

   ```bash
   # e.g. ./isaaclab.sh -p -m pip ... when using the Isaac Lab launcher
   pip install -e .
   ```

### First run (both options)

Dry-run the pipeline with a tiny training run before long experiments:

```bash
flashsac-train --task Isaac-Velocity-Flat-G1-v0 --num_envs 64 --max_iterations 200 --headless
```

Checkpoints, TensorBoard logs, and the dumped `env.yaml`/`agent.yaml` land in
`logs/flashsac/<experiment_name>/<timestamp>/`. Then, benchmark-scale training and playback:

```bash
flashsac-train --task Isaac-Velocity-Rough-G1-v0 --num_envs 1024 --headless
flashsac-play  --task Isaac-Velocity-Rough-G1-Play-v0 --num_envs 32 --export_policy
```

## Version notes

- `rsl-rl-lib >= 5.4.2` (the runner targets the 5.4 Logger API; older 5.x is not supported).
- torch >= 2.5; `torch.compile` mode `auto` picks `max-autotune` on torch >= 2.9.
- Record simulator/framework/CUDA versions with results; the dumped `agent.yaml` captures the
  full hyperparameter set for each run.

## Note: uv environment location

`bin/lint` pins `UV_PROJECT_ENVIRONMENT` to the repo-local `.venv` so a globally
exported `UV_PROJECT_ENVIRONMENT` (pointing at another project's venv) is never touched. To place
the venv elsewhere (e.g. on a larger disk), export `FLASHSAC_VENV=/path/to/venv`.
