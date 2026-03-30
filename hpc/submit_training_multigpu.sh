#!/bin/bash
# ==============================================================================
# Interactive SLURM job submission for multi-GPU Genesis RL training
#
# Uses torchrun to launch one process per GPU. Each process runs its own
# Genesis scene; rsl-rl handles gradient all-reduce via NCCL.
#
# Usage:
#   ./submit_training_multigpu.sh        # fully interactive
# ==============================================================================

set -e
source "$(dirname "$0")/_output.sh"

# --- Prompt helper ---
ask() {
    local label="$1"
    local default="$2"
    local input
    printf "   ${DIM}%-20s${RESET}${CYAN}[${default}]${RESET}: " "$label"
    read -r input
    REPLY="${input:-$default}"
}

ask_yn() {
    local label="$1"
    local default="$2"
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

banner "Submit Multi-GPU Training" "$MAGENTA"

# ╔══════════════════════════════════════╗
# ║  1. Prototype selection              ║
# ╚══════════════════════════════════════╝
section "Prototype"
echo -e "   ${WHITE}1)${RESET} global_coordinate"
echo -e "   ${WHITE}2)${RESET} obstacle_avoidance"
printf "   ${DIM}%-20s${RESET}${CYAN}[2]${RESET}: " "Select"
read -r proto_choice
case "${proto_choice:-2}" in
    1) PROTOTYPE="global_coordinate" ;;
    *) PROTOTYPE="obstacle_avoidance" ;;
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
DEF_EXP_NAME="genesis-${PROTOTYPE}-multigpu"

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
ask "Envs per GPU" "$DEF_BATCH"
BATCH="$REPLY"

ask "Max iterations" "$DEF_ITERS"
ITERS="$REPLY"

# ╔══════════════════════════════════════╗
# ║  4. Cluster resources                ║
# ╚══════════════════════════════════════╝
section "Cluster Resources"
echo -e "   ${DIM}Partitions: paula (A30), clara (V100), clara-long (V100 10d)${RESET}"
echo -e "   ${DIM}Note: paula has 4x A30 per node, clara has 4x V100 per node${RESET}"
ask "Partition" "paula"
PARTITION="$REPLY"

case "$PARTITION" in
    paula)      DEF_GPU="a30" ;;
    clara*)     DEF_GPU="v100" ;;
    *)          DEF_GPU="a30" ;;
esac

ask "GPU type" "$DEF_GPU"
GPU_TYPE="$REPLY"

ask "Number of GPUs" "2"
NUM_GPUS="$REPLY"

ask "Memory" "64G"
MEM="$REPLY"

# CPUs: scale with GPU count for data loading
DEF_CPUS=$(( NUM_GPUS * 8 ))
ask "CPUs" "$DEF_CPUS"
CPUS="$REPLY"

ask "Time limit (hours)" "12"
HOURS="$REPLY"

# ╔══════════════════════════════════════╗
# ║  5. Confirmation                     ║
# ╚══════════════════════════════════════╝

GENESIS_DIR="$HOME/genesis_v04"
PROTO_DIR="${GENESIS_DIR}/prototyp_${PROTOTYPE}"
LOG_DIR="${GENESIS_DIR}/logs"
TRAIN_SCRIPT="train_rl_multigpu.py"

TOTAL_ENVS=$(( BATCH * NUM_GPUS ))

section "Summary"
info "Prototype" "prototyp_${PROTOTYPE}"
info "Experiment" "$EXP_NAME"
info "Script" "$TRAIN_SCRIPT"
info "Launch" "torchrun --nproc_per_node=${NUM_GPUS}"
echo ""
info "Partition" "$PARTITION"
info "GPU" "${GPU_TYPE} x${NUM_GPUS}"
info "Memory" "$MEM"
info "CPUs" "$CPUS"
info "Time limit" "${HOURS}h"
echo ""
info "Envs/GPU" "$BATCH"
info "Total envs" "$TOTAL_ENVS"
info "Iterations" "$ITERS"

echo ""
printf "   ${BOLD}Submit this job?${RESET} ${CYAN}[Y/n]${RESET}: "
read -r confirm
case "${confirm:-Y}" in
    [Nn]*) echo -e "\n   ${DIM}Cancelled.${RESET}\n"; exit 0 ;;
esac

# --- Ensure log directory exists ---
section "Submitting" "$SYM_ROCKET"
spin_run "Creating log directory..." mkdir -p "${LOG_DIR}"

# --- Generate and submit job script ---
JOBSCRIPT=$(mktemp /tmp/genesis_multigpu_XXXXXX.sh)

cat > "$JOBSCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPU_TYPE}:${NUM_GPUS}
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

# Pin Vulkan to NVIDIA ICD
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

# NCCL tuning
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL

# --- Training ---
cd ${PROTO_DIR}
echo "=== Starting multi-GPU training: \$(date) ==="
echo "Host: \$(hostname)"
echo "GPUs: \$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"
echo "Experiment: ${EXP_NAME}, Envs/GPU: ${BATCH}, Total: ${TOTAL_ENVS}"

torchrun \\
    --standalone \\
    --nproc_per_node=${NUM_GPUS} \\
    ${TRAIN_SCRIPT} \\
    -e ${EXP_NAME} \\
    -B ${BATCH} \\
    --max_iterations ${ITERS}

echo "=== Training complete: \$(date) ==="
SLURM_EOF

cp "$JOBSCRIPT" "${LOG_DIR}/${EXP_NAME}.sh"
rm "$JOBSCRIPT"

spin_run "Submitting to SLURM..." sbatch "${LOG_DIR}/${EXP_NAME}.sh"

done_banner "Job submitted: ${EXP_NAME}"
hint "Monitor with: squeue -u \$USER"
hint "Logs: ${LOG_DIR}/slurm-<jobid>.out"
hint "Effective batch: ${TOTAL_ENVS} envs (${BATCH} x ${NUM_GPUS} GPUs)"
