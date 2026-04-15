#!/bin/bash
# ==============================================================================
# Interactive SLURM job submission for Genesis RL evaluation
#
# Runs eval on HPC so the renderer matches training (BatchRenderer on paula,
# Rasterizer on clara). Eval is lightweight — short time, few resources.
#
# Usage:
#   ./submit_evaluation.sh              # fully interactive
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

banner "Submit Evaluation Job" "$BLUE"

# --- Paths ---
GENESIS_DIR="$HOME/genesis_v04"
LOG_DIR="${GENESIS_DIR}/logs"

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
PROTO_DIR="${GENESIS_DIR}/prototyp_${PROTOTYPE}"

# ╔══════════════════════════════════════╗
# ║  2. Experiment name                  ║
# ╚══════════════════════════════════════╝
section "Experiment"

# List available experiments on HPC (if directory exists)
RUNS_DIR="${PROTO_DIR}/logs"
if [ -d "$RUNS_DIR" ]; then
    echo -e "   ${DIM}Available runs:${RESET}"
    for d in "$RUNS_DIR"/*/; do
        [ -d "$d" ] || continue
        run_name=$(basename "$d")
        # Count checkpoints
        ckpt_count=$(ls "$d"/model_*.pt 2>/dev/null | wc -l)
        if [ "$ckpt_count" -gt 0 ]; then
            latest=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
            echo -e "   ${WHITE}  ${run_name}${RESET} ${DIM}(${ckpt_count} checkpoints, latest: ${latest})${RESET}"
        fi
    done
    echo ""
fi

DEF_EXP_NAME="genesis-${PROTOTYPE}"
ask "Experiment name" "$DEF_EXP_NAME"
EXP_NAME="$REPLY"

# Verify experiment directory exists
EXP_DIR="${RUNS_DIR}/${EXP_NAME}"
if [ -d "$EXP_DIR" ]; then
    # List available checkpoints
    echo -e "   ${DIM}Checkpoints:${RESET}"
    for ckpt in "$EXP_DIR"/model_*.pt; do
        [ -f "$ckpt" ] || continue
        iter=$(basename "$ckpt" | sed 's/model_\([0-9]*\)\.pt/\1/')
        echo -e "   ${DIM}  ${iter}${RESET}"
    done | sort -t$'\t' -k1 -n
    echo ""
fi

# ╔══════════════════════════════════════╗
# ║  3. Checkpoint                       ║
# ╚══════════════════════════════════════╝
section "Checkpoint"
ask "Iteration (empty=latest)" ""
CKPT="$REPLY"

# ╔══════════════════════════════════════╗
# ║  4. Eval parameters                  ║
# ╚══════════════════════════════════════╝
section "Eval Parameters"
ask "Num envs" "50"
NUM_ENVS="$REPLY"

ask "Num episodes" "100"
NUM_EPISODES="$REPLY"

ask_yn "W&B logging" "Y"
USE_WANDB="$REPLY"

# ╔══════════════════════════════════════╗
# ║  5. Cluster resources                ║
# ╚══════════════════════════════════════╝
section "Cluster Resources"
echo -e "   ${DIM}Use the same partition as training to match the renderer!${RESET}"
echo -e "   ${DIM}paula (A30) = BatchRenderer, clara (V100) = Rasterizer${RESET}"
ask "Partition" "paula"
PARTITION="$REPLY"

case "$PARTITION" in
    paula)      DEF_GPU="a30" ;;
    clara*)     DEF_GPU="v100" ;;
    *)          DEF_GPU="a30" ;;
esac

ask "GPU type" "$DEF_GPU"
GPU_TYPE="$REPLY"

ask "Memory" "32G"
MEM="$REPLY"

ask "CPUs" "8"
CPUS="$REPLY"

ask "Time limit (hours)" "1"
HOURS="$REPLY"

# ╔══════════════════════════════════════╗
# ║  6. Build command                    ║
# ╚══════════════════════════════════════╝

# Pick eval script
EVAL_SCRIPT="eval_rl_wb.py"
if [ "$USE_WANDB" = "false" ] && [ -f "${PROTO_DIR}/eval_rl.py" ]; then
    EVAL_SCRIPT="eval_rl.py"
fi

EVAL_ARGS="-e ${EXP_NAME} --num_envs ${NUM_ENVS} --num_episodes ${NUM_EPISODES}"
if [ -n "$CKPT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --ckpt ${CKPT}"
fi

JOB_NAME="${EXP_NAME}-eval"

# ╔══════════════════════════════════════╗
# ║  7. Confirmation                     ║
# ╚══════════════════════════════════════╝
section "Summary"
info "Prototype" "prototyp_${PROTOTYPE}"
info "Experiment" "$EXP_NAME"
info "Script" "$EVAL_SCRIPT"
info "Command" "python ${EVAL_SCRIPT} ${EVAL_ARGS}"
echo ""
info "Checkpoint" "${CKPT:-latest}"
info "Num envs" "$NUM_ENVS"
info "Num episodes" "$NUM_EPISODES"
info "W&B logging" "$([ "$USE_WANDB" = "true" ] && echo 'enabled' || echo 'disabled')"
echo ""
info "Partition" "$PARTITION"
info "GPU" "${GPU_TYPE} x1"
info "Memory" "$MEM"
info "CPUs" "$CPUS"
info "Time limit" "${HOURS}h"

echo ""
printf "   ${BOLD}Submit this job?${RESET} ${CYAN}[Y/n]${RESET}: "
read -r confirm
case "${confirm:-Y}" in
    [Nn]*) echo -e "\n   ${DIM}Cancelled.${RESET}\n"; exit 0 ;;
esac

# ╔══════════════════════════════════════╗
# ║  8. Submit                           ║
# ╚══════════════════════════════════════╝
section "Submitting" "$SYM_ROCKET"
spin_run "Creating log directory..." mkdir -p "${LOG_DIR}"

JOBSCRIPT=$(mktemp /tmp/genesis_eval_XXXXXX.sh)

cat > "$JOBSCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPU_TYPE}:1
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

# --- Evaluation ---
cd ${PROTO_DIR}
echo "=== Starting evaluation: \$(date) ==="
echo "Host: \$(hostname), GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Experiment: ${EXP_NAME}"

python ${EVAL_SCRIPT} ${EVAL_ARGS}

echo "=== Evaluation complete: \$(date) ==="
SLURM_EOF

cp "$JOBSCRIPT" "${LOG_DIR}/${JOB_NAME}.sh"
rm "$JOBSCRIPT"

spin_run "Submitting to SLURM..." sbatch "${LOG_DIR}/${JOB_NAME}.sh"

done_banner "Job submitted: ${JOB_NAME}"
hint "Monitor with: squeue -u \$USER"
hint "Logs: ${LOG_DIR}/slurm-<jobid>.out"
hint "Results: ${PROTO_DIR}/logs/${EXP_NAME}/eval_stats.png"
