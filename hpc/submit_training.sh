#!/bin/bash
# ==============================================================================
# Interactive SLURM job submission for Genesis RL training
#
# Usage:
#   ./submit_training.sh              # fully interactive
# ==============================================================================

set -e
source "$(dirname "$0")/_output.sh"

# --- Prompt helper ---
# Usage: ask "Label" "default_value"  →  sets REPLY (or default if empty)
ask() {
    local label="$1"
    local default="$2"
    local input
    printf "   ${DIM}%-20s${RESET}${CYAN}[${default}]${RESET}: " "$label"
    read -r input
    REPLY="${input:-$default}"
}

# --- Prompt yes/no ---
# Usage: ask_yn "Label" "Y"  →  REPLY is "true" or "false"
ask_yn() {
    local label="$1"
    local default="$2"  # Y or n
    local hint
    if [ "$default" = "Y" ]; then
        hint="Y/n"
    else
        hint="y/N"
    fi
    local input
    printf "   ${DIM}%-20s${RESET}${CYAN}[${hint}]${RESET}: " "$label"
    read -r input
    input="${input:-$default}"
    case "$input" in
        [Yy]*) REPLY="true" ;;
        *)     REPLY="false" ;;
    esac
}

banner "Submit Training Job" "$MAGENTA"

# ╔══════════════════════════════════════╗
# ║  1. Prototype selection              ║
# ╚══════════════════════════════════════╝
section "Prototype"
echo -e "   ${WHITE}1)${RESET} global_coordinate"
echo -e "   ${WHITE}2)${RESET} obstacle_avoidance"
printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Select"
read -r proto_choice
case "${proto_choice:-1}" in
    2) PROTOTYPE="obstacle_avoidance" ;;
    *) PROTOTYPE="global_coordinate" ;;
esac

# --- Per-prototype defaults ---
case "$PROTOTYPE" in
    obstacle_avoidance)
        DEF_BATCH=32
        DEF_ITERS=15000
        ;;
    *)
        DEF_BATCH=4096
        DEF_ITERS=401
        ;;
esac
DEF_EXP_NAME="genesis-${PROTOTYPE}"

# ╔══════════════════════════════════════╗
# ║  2. Experiment name                  ║
# ╚══════════════════════════════════════╝
section "Experiment"
ask "Name" "$DEF_EXP_NAME"
EXP_NAME="$REPLY"

# ╔══════════════════════════════════════╗
# ║  3. Training parameters              ║
# ╚══════════════════════════════════════╝
section "Training Parameters"
ask "Batch size" "$DEF_BATCH"
BATCH="$REPLY"

ask "Max iterations" "$DEF_ITERS"
ITERS="$REPLY"

ask "Seed" "0"
SEED="$REPLY"

ask_yn "W&B logging" "Y"
USE_WANDB="$REPLY"

# ╔══════════════════════════════════════╗
# ║  4. Cluster resources                ║
# ╚══════════════════════════════════════╝
section "Cluster Resources"
echo -e "   ${DIM}Partitions: paula (A30), clara (V100), clara-long (V100 10d)${RESET}"
ask "Partition" "paula"
PARTITION="$REPLY"

# Set GPU type default based on partition
case "$PARTITION" in
    paula)      DEF_GPU="a30" ;;
    clara*)     DEF_GPU="v100" ;;
    *)          DEF_GPU="a30" ;;
esac

ask "GPU type" "$DEF_GPU"
GPU_TYPE="$REPLY"

ask "GPU count" "1"
GPU_COUNT="$REPLY"

ask "Memory" "64G"
MEM="$REPLY"

ask "CPUs" "16"
CPUS="$REPLY"

ask "Time limit (hours)" "12"
HOURS="$REPLY"

# ╔══════════════════════════════════════╗
# ║  5. Confirmation                     ║
# ╚══════════════════════════════════════╝

# --- Resolve paths ---
GENESIS_DIR="$HOME/genesis_v04"
PROTO_DIR="${GENESIS_DIR}/prototyp_${PROTOTYPE}"
LOG_DIR="${GENESIS_DIR}/logs"

# Pick training script
TRAIN_SCRIPT="train_rl.py"
if [ "$USE_WANDB" = "true" ] && [ -f "${PROTO_DIR}/train_rl_wb.py" ]; then
    TRAIN_SCRIPT="train_rl_wb.py"
fi

# Build training command args
TRAIN_ARGS="-e ${EXP_NAME} -B ${BATCH} --max_iterations ${ITERS}"
if [ "$SEED" != "0" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --seed ${SEED}"
fi

JOB_NAME="${EXP_NAME}"

section "Summary"
info "Prototype" "prototyp_${PROTOTYPE}"
info "Experiment" "$EXP_NAME"
info "Script" "$TRAIN_SCRIPT"
info "Command" "python ${TRAIN_SCRIPT} ${TRAIN_ARGS}"
echo ""
info "Partition" "$PARTITION"
info "GPU" "${GPU_TYPE} x${GPU_COUNT}"
info "Memory" "$MEM"
info "CPUs" "$CPUS"
info "Time limit" "${HOURS}h"
echo ""
info "Batch size" "$BATCH envs"
info "Iterations" "$ITERS"
info "Seed" "$SEED"
info "W&B logging" "$([ "$USE_WANDB" = "true" ] && echo 'enabled' || echo 'disabled')"

echo ""
printf "   ${BOLD}Submit this job?${RESET} ${CYAN}[Y/n]${RESET}: "
read -r confirm
case "${confirm:-Y}" in
    [Nn]*) echo -e "\n   ${DIM}Cancelled.${RESET}\n"; exit 0 ;;
esac

# --- Ensure log directory exists on HPC ---
section "Submitting" "$SYM_ROCKET"
spin_run "Creating log directory..." mkdir -p "${LOG_DIR}"

# --- Generate and submit job script ---
JOBSCRIPT=$(mktemp /tmp/genesis_job_XXXXXX.sh)

cat > "$JOBSCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPU_TYPE}:${GPU_COUNT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${HOURS}:00:00
#SBATCH --output=${LOG_DIR}/slurm-%j.out
#SBATCH --error=${LOG_DIR}/slurm-%j.err
#SBATCH --mail-type=END,FAIL

# --- Environment ---
module purge
module load Anaconda3
module load CUDA/12.6.0
eval "\$(conda shell.bash hook)"
conda activate ba_v04

# Pin Vulkan to NVIDIA ICD — prevents Madrona abort() from incompatible ICDs
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

# --- Training ---
cd ${PROTO_DIR}
echo "=== Starting training: \$(date) ==="
echo "Host: \$(hostname), GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiment: ${EXP_NAME}, Seed: ${SEED}"

python ${TRAIN_SCRIPT} ${TRAIN_ARGS}

echo "=== Training complete: \$(date) ==="
SLURM_EOF

cp "$JOBSCRIPT" "${LOG_DIR}/${JOB_NAME}.sh"
rm "$JOBSCRIPT"

spin_run "Submitting to SLURM..." sbatch "${LOG_DIR}/${JOB_NAME}.sh"

done_banner "Job submitted: ${EXP_NAME}"
hint "Monitor with: squeue -u \$USER"
hint "Logs: ${LOG_DIR}/slurm-<jobid>.out"
