#!/bin/bash
# ==============================================================================
# HPC Leipzig Cluster Setup - Genesis Drone Landing Project
#
# Creates a conda environment with all dependencies for RL training
# on the HPC cluster (A30/V100 GPUs with CUDA 12.6).
#
# Usage:
#   source setup_env.sh          # first time: creates env + installs everything
#   source setup_env.sh --load   # subsequent sessions: load modules + activate
# ==============================================================================

set -e
source "$(dirname "${BASH_SOURCE[0]}")/_output.sh"

ENV_NAME="ba_v04"
ENV_DIR="$HOME/.conda/envs/$ENV_NAME"

# ==============================================================================
# 1. MODULE LOADS
# ==============================================================================

module purge
module load Anaconda3
module load CUDA/12.6.0

eval "$(conda shell.bash hook)"

# Pin Vulkan loader to NVIDIA ICD only.
# HPC nodes have 5 ICDs (Intel, lavapipe, radeon, NVIDIA) registered system-wide.
# Without this, Madrona's vkCreateInstance hits an incompatible driver and abort()s.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
elif [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
elif [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

# ==============================================================================
# 2. QUICK RELOAD MODE
# ==============================================================================

if [ "$1" = "--load" ]; then
    if [ ! -d "$ENV_DIR" ]; then
        fail "Conda env '$ENV_NAME' not found. Run without --load first."
        return 1 2>/dev/null || exit 1
    fi
    conda activate "$ENV_NAME"
    sleep 5
    export PYTHONNOUSERSITE=1

    banner "Environment Ready" "$GREEN"
    section "Status"
    info "Conda env" "$ENV_NAME"
    info "Python" "$(python --version 2>&1)"
    info "PyTorch" "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
    info "CUDA" "$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
    echo ""
    return 0 2>/dev/null || exit 0
fi

# ==============================================================================
# 3. FULL SETUP
# ==============================================================================

banner "HPC Environment Setup" "$CYAN"

section "Modules loaded"
module list 2>&1 | while IFS= read -r line; do echo -e "   ${DIM}${line}${RESET}"; done

# ==============================================================================
# 4. CREATE CONDA ENVIRONMENT
# ==============================================================================

section "Conda environment" "$SYM_GEAR"
if conda env list | grep -q "^$ENV_NAME "; then
    warn "Env '$ENV_NAME' already exists - activating"
else
    spin_start "Creating env '$ENV_NAME' with Python 3.13..."
    conda create -y -q --name "$ENV_NAME" python=3.13 > /dev/null
    spin_stop $?
fi

conda activate "$ENV_NAME"
sleep 5

# Prevent pip from installing to ~/.local (user site-packages).
# Without this, packages can leak outside the conda env on HPC systems.
export PIP_USER=false
export PYTHONNOUSERSITE=1

spin_run_check "Upgrading pip..." python -m pip install -q --upgrade pip
python -m pip install -q --no-deps packaging 2>/dev/null

# ==============================================================================
# 5. PYTORCH (cu126 wheels)
# ==============================================================================

section "PyTorch" "$SYM_GEAR"
spin_run_check "Installing PyTorch + torchvision (cu126)..." \
    python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ==============================================================================
# 6. CORE DEPENDENCIES
# ==============================================================================

section "Core dependencies" "$SYM_GEAR"
spin_run_check "Installing numpy, scipy, matplotlib, wandb, etc..." \
    python -m pip install -q \
    numpy \
    scipy \
    pyyaml \
    matplotlib \
    pillow \
    tqdm \
    tensorboard \
    wandb \
    requests

# ==============================================================================
# 7. RL LIBRARIES
# ==============================================================================

section "RL libraries" "$SYM_GEAR"
spin_run_check "Installing rsl-rl-lib, stable-baselines3, tensordict..." \
    python -m pip install -q \
    rsl-rl-lib==5.0.1 \
    stable-baselines3 \
    tensordict

# ==============================================================================
# 8. GENESIS + SIMULATION DEPENDENCIES
# ==============================================================================

section "Genesis simulator" "$SYM_GEAR"
spin_run_check "Installing genesis-world..." \
    python -m pip install -q genesis-world==0.4.3

# Remove the unrelated 'genesis' package if it got pulled in as a transitive
# dependency — it conflicts with genesis-world's 'import genesis' namespace.
if python -m pip show genesis &>/dev/null && python -m pip show genesis-world &>/dev/null; then
    spin_run "Removing conflicting 'genesis' package..." python -m pip uninstall -y -q genesis
fi

# ==============================================================================
# 9. VERIFICATION
# ==============================================================================

banner "Setup Complete" "$GREEN"

section "Verification"
info "Python" "$(python --version 2>&1)"
info "PyTorch" "$(python -c 'import torch; print(torch.__version__)')"
info "CUDA" "$(python -c 'import torch; print(torch.cuda.is_available())')"
info "GPU count" "$(python -c 'import torch; print(torch.cuda.device_count())')"
info "Genesis" "$(python -c 'import genesis; print(genesis.__version__)' 2>/dev/null || echo 'import failed')"
echo ""
hint "For future sessions: source ~/genesis_v04/hpc/setup_env.sh --load"
