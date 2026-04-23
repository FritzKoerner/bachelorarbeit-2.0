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
echo -e "   ${WHITE}3)${RESET} corridor_navigation"
printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Select"
read -r proto_choice
case "${proto_choice:-1}" in
    2) PROTOTYPE="obstacle_avoidance" ;;
    3) PROTOTYPE="corridor_navigation" ;;
    *) PROTOTYPE="global_coordinate" ;;
esac

# ╔══════════════════════════════════════╗
# ║  1b. Env version                     ║
# ╚══════════════════════════════════════╝
# corridor_navigation has no v1/v2 split — single placement strategy.
ENV_VERSION="v1"
if [ "$PROTOTYPE" != "corridor_navigation" ]; then
    section "Env Version"
    echo -e "   ${WHITE}1)${RESET} v1  (distance + time penalties)"
    echo -e "   ${WHITE}2)${RESET} v2  (progress + close rewards, no dt-scaling)"
    printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Select"
    read -r version_choice
    case "${version_choice:-1}" in
        2) ENV_VERSION="v2" ;;
        *) ENV_VERSION="v1" ;;
    esac
fi

# ╔══════════════════════════════════════╗
# ║  1c. Scenario (obstacle_avoidance)   ║
# ╚══════════════════════════════════════╝
SCENARIO="default"
if [ "$PROTOTYPE" = "obstacle_avoidance" ]; then
    section "Scenario"
    echo -e "   ${WHITE}1)${RESET} default  ${DIM}(±5 m square spawn, fixed 10 m altitude, current placement)${RESET}"
    echo -e "   ${WHITE}2)${RESET} hard     ${DIM}(10 m ring + ±5 m jitter, 5–10 m altitude, 4-row vineyard, 3 m cubes)${RESET}"
    printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Select"
    read -r scenario_choice
    case "${scenario_choice:-1}" in
        2) SCENARIO="hard" ;;
        *) SCENARIO="default" ;;
    esac
fi

# --- Per-prototype defaults ---
case "$PROTOTYPE" in
    obstacle_avoidance)
        DEF_BATCH=256
        DEF_ITERS=8001
        DEF_CURRICULUM=300
        ;;
    corridor_navigation)
        DEF_BATCH=256
        DEF_ITERS=1001
        DEF_CURRICULUM=301
        ;;
    *)
        DEF_BATCH=4096
        DEF_ITERS=401
        DEF_CURRICULUM=0
        ;;
esac
DEF_EXP_NAME="genesis-${PROTOTYPE}"
if [ "$ENV_VERSION" = "v2" ]; then
    DEF_EXP_NAME="${DEF_EXP_NAME}-v2"
fi
if [ "$SCENARIO" = "hard" ]; then
    DEF_EXP_NAME="${DEF_EXP_NAME}-hard"
fi

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

# Curriculum iterations supported by obstacle_avoidance + corridor_navigation
CURRICULUM_ITERS="0"
case "$PROTOTYPE" in
    obstacle_avoidance | corridor_navigation)
        ask "Curriculum iters" "$DEF_CURRICULUM"
        CURRICULUM_ITERS="$REPLY"
        ;;
esac

# Adaptive LR supported by obstacle_avoidance + corridor_navigation
ADAPTIVE_LR="false"
DESIRED_KL="0.01"
case "$PROTOTYPE" in
    obstacle_avoidance | corridor_navigation)
        ask_yn "Adaptive LR" "n"
        ADAPTIVE_LR="$REPLY"
        if [ "$ADAPTIVE_LR" = "true" ]; then
            ask "Target KL" "0.01"
            DESIRED_KL="$REPLY"
        fi
        ;;
esac

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

TRAIN_SCRIPT="train_rl_wb.py"

# Build training command args
TRAIN_ARGS="-e ${EXP_NAME} -B ${BATCH} --max_iterations ${ITERS}"
if [ "$ENV_VERSION" = "v2" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --env-v2"
fi
if [ "$SCENARIO" = "hard" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --scenario hard"
fi
case "$PROTOTYPE" in
    obstacle_avoidance | corridor_navigation)
        TRAIN_ARGS="${TRAIN_ARGS} --curriculum-iterations ${CURRICULUM_ITERS}"
        ;;
esac
if [ "$ADAPTIVE_LR" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --adaptive-lr --desired-kl ${DESIRED_KL}"
fi

JOB_NAME="${EXP_NAME}"

section "Summary"
info "Prototype" "prototyp_${PROTOTYPE}"
if [ "$PROTOTYPE" != "corridor_navigation" ]; then
    info "Env version" "$ENV_VERSION"
fi
if [ "$PROTOTYPE" = "obstacle_avoidance" ]; then
    info "Scenario" "$SCENARIO"
fi
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
case "$PROTOTYPE" in
    obstacle_avoidance | corridor_navigation)
        info "Curriculum iters" "$CURRICULUM_ITERS"
        ;;
esac
if [ "$ADAPTIVE_LR" = "true" ]; then
    info "LR schedule" "adaptive (KL target ${DESIRED_KL})"
else
    info "LR schedule" "fixed"
fi

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
echo "Experiment: ${EXP_NAME}"

python ${TRAIN_SCRIPT} ${TRAIN_ARGS}

echo "=== Training complete: \$(date) ==="
SLURM_EOF

cp "$JOBSCRIPT" "${LOG_DIR}/${JOB_NAME}.sh"
rm "$JOBSCRIPT"

spin_run "Submitting to SLURM..." sbatch "${LOG_DIR}/${JOB_NAME}.sh"

done_banner "Job submitted: ${EXP_NAME}"
hint "Monitor with: squeue -u \$USER"
hint "Logs: ${LOG_DIR}/slurm-<jobid>.out"
