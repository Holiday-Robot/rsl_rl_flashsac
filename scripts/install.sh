#!/usr/bin/env bash
# One-shot environment installation for rsl_rl_flashsac, uv-based.
#
# Creates a uv-managed venv and installs Isaac Sim + Isaac Lab from PyPI (no git clone,
# following the official FlashSAC setup) plus both packages of this repo in editable mode.
# Idempotent: safe to re-run.
#
# Usage:
#   bash scripts/install.sh
#   source <venv>/bin/activate   # printed at the end
#
# Overridable via environment variables:
#   FLASHSAC_ISAAC_VENV  venv path              (default: <repo>/.venv-isaaclab)
#   PYTHON_VERSION       python version         (default: 3.11 — required by isaacsim 5.1)
#   ISAACSIM_VERSION     Isaac Sim pip version  (default: 5.1.0)
#   ISAACLAB_VERSION     Isaac Lab pip version  (default: 2.3.0)
#   TORCH_VERSION        torch version          (default: 2.9.1 — required for Blackwell GPUs,
#                                                see https://github.com/isaac-sim/IsaacLab/issues/4371)
#   TORCHVISION_VERSION  torchvision version    (default: 0.24.1)
#   SKIP_APT=1           skip system package installation
#
# NOTE: running Isaac Sim requires accepting the NVIDIA EULA on first launch
# (export OMNI_KIT_ACCEPT_EULA=YES for headless runs).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FLASHSAC_ISAAC_VENV="${FLASHSAC_ISAAC_VENV:-${REPO_DIR}/.venv-isaaclab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-2.3.0}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"

echo "[flashsac] repo: ${REPO_DIR}"
echo "[flashsac] venv: ${FLASHSAC_ISAAC_VENV} (python ${PYTHON_VERSION})"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# System dependencies (X/GL/Vulkan libs required by Isaac Sim)
if [ "${SKIP_APT:-0}" != "1" ]; then
    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    ${SUDO} apt-get update && ${SUDO} apt-get install -y --no-install-recommends \
        build-essential curl wget bzip2 \
        libx11-6 libxext6 libxrender1 libxrandr2 libxi6 libxtst6 \
        libxcomposite1 libxcursor1 libxdamage1 libxfixes3 libxinerama1 \
        libxss1 libnss3 libasound2 libpulse0 libegl1 \
        libglu1-mesa vulkan-tools libxt6 libgl1 \
        git ca-certificates &&
        ${SUDO} apt-get clean && ${SUDO} rm -rf /var/lib/apt/lists/*
else
    echo "[flashsac] SKIP_APT=1 — skipping system packages"
fi

# Create the venv (idempotent). Isolated from any global UV_PROJECT_ENVIRONMENT.
unset UV_PROJECT_ENVIRONMENT
[ -d "${FLASHSAC_ISAAC_VENV}" ] || uv venv --python "${PYTHON_VERSION}" "${FLASHSAC_ISAAC_VENV}"
VENV_PY="${FLASHSAC_ISAAC_VENV}/bin/python"

# Isaac Sim + Isaac Lab from PyPI (same recipe as the official FlashSAC repo). isaacsim pins an
# older torch/torchvision, so the Blackwell-compatible versions are forced via uv overrides
# (equivalent to FlashSAC's [tool.uv] override-dependencies; see IsaacLab issue #4371).
OVERRIDES_FILE="$(mktemp)"
trap 'rm -f "${OVERRIDES_FILE}"' EXIT
cat > "${OVERRIDES_FILE}" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
flatdict==4.0.0
EOF
# flatdict: isaaclab pins 4.0.1, whose sdist does not build on modern setuptools
# (https://github.com/isaac-sim/IsaacLab/issues/4577)
uv pip install --python "${VENV_PY}" \
    "isaacsim[all,extscache]==${ISAACSIM_VERSION}" \
    "isaaclab[isaacsim]==${ISAACLAB_VERSION}" \
    --override "${OVERRIDES_FILE}" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match

# rsl_rl_flashsac (both packages from the single root project)
uv pip install --python "${VENV_PY}" -e "${REPO_DIR}"

# Sanity check (imports that do not require the simulator to be running)
"${VENV_PY}" - <<'EOF'
import sys

print("Using:", sys.executable)
import rsl_rl_flashsac

print("rsl_rl_flashsac:", rsl_rl_flashsac.__version__)
from rsl_rl_flashsac.algorithms import FlashSAC  # noqa: F401
from rsl_rl_flashsac.runners import OffPolicyRunner  # noqa: F401

print("FlashSAC / OffPolicyRunner import: OK")
# NOTE: importing isaaclab triggers the Isaac Sim kernel bootstrap (EULA prompt), and the
# bundled isaaclab_tasks / isaaclab_rl only become importable after AppLauncher starts the
# simulator — so only package metadata is checked here.
from importlib.metadata import version

for pkg in ["isaacsim", "isaaclab", "rsl-rl-lib", "torch"]:
    print(f"{pkg}: {version(pkg)}")
EOF

echo "All installation steps completed successfully!"
echo
echo "Next steps:"
echo "  source ${FLASHSAC_ISAAC_VENV}/bin/activate"
echo "  export OMNI_KIT_ACCEPT_EULA=YES   # accept the NVIDIA EULA for headless runs"
echo "  # tiny-step dry-run before long training:"
echo "  flashsac-train --task Isaac-Velocity-Flat-G1-v0 --num_envs 64 --max_iterations 200 --headless"
