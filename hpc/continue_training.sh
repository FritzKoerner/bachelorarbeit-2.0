#!/bin/bash
# ==============================================================================
# Interactive SLURM job submission for continuing obstacle avoidance training
#
# Scans existing training runs, lets you pick one and a checkpoint, then
# submits a SLURM job that resumes from that point.
#
# Usage:
#   ./continue_training.sh              # fully interactive
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
    if [ "$default" = "Y" ]; then hint="Y/n"; else hint="y/N"; fi
    local input
    printf "   ${DIM}%-20s${RESET}${CYAN}[${hint}]${RESET}: " "$label"
    read -r input
    input="${input:-$default}"
    case "$input" in
        [Yy]*) REPLY="true" ;;
        *)     REPLY="false" ;;
    esac
}

banner "Continue Training Job" "$MAGENTA"

# ╔══════════════════════════════════════╗
# ║  Paths                               ║
# ╚══════════════════════════════════════╝
GENESIS_DIR="$HOME/genesis_v04"
PROTO_DIR="${GENESIS_DIR}/prototyp_obstacle_avoidance"
LOGS_DIR="${PROTO_DIR}/logs"
SLURM_LOG_DIR="${GENESIS_DIR}/logs"

if [ ! -d "$LOGS_DIR" ]; then
    fail "No logs directory found at ${LOGS_DIR}"
    echo -e "   ${DIM}Run a training first or sync results to the HPC.${RESET}"
    exit 1
fi

# ╔══════════════════════════════════════╗
# ║  1. Discover training runs           ║
# ╚══════════════════════════════════════╝
section "Available Training Runs"
echo -e "   ${DIM}Scanning ${LOGS_DIR}/ ...${RESET}\n"

# Collect runs that have at least one model_*.pt checkpoint
RUNS=()
RUN_INFO=()
for run_dir in "$LOGS_DIR"/*/; do
    [ -d "$run_dir" ] || continue
    run_name="$(basename "$run_dir")"

    # Find all checkpoint iterations, sorted numerically
    ckpts=()
    for pt_file in "$run_dir"/model_*.pt; do
        [ -f "$pt_file" ] || continue
        iter=$(basename "$pt_file" | sed 's/model_\([0-9]*\)\.pt/\1/')
        ckpts+=("$iter")
    done

    [ ${#ckpts[@]} -eq 0 ] && continue

    # Sort numerically
    IFS=$'\n' sorted=($(sort -n <<<"${ckpts[*]}")); unset IFS
    latest="${sorted[-1]}"
    count=${#sorted[@]}

    RUNS+=("$run_name")
    RUN_INFO+=("latest: ${latest}, ${count} checkpoints")
done

if [ ${#RUNS[@]} -eq 0 ]; then
    fail "No training runs with checkpoints found in ${LOGS_DIR}/"
    exit 1
fi

# Display numbered list
for i in "${!RUNS[@]}"; do
    num=$((i + 1))
    printf "   ${WHITE}%2d)${RESET} %-35s ${DIM}[%s]${RESET}\n" "$num" "${RUNS[$i]}" "${RUN_INFO[$i]}"
done

echo ""
printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Select run"
read -r run_choice
run_choice="${run_choice:-1}"

# Validate selection
if ! [[ "$run_choice" =~ ^[0-9]+$ ]] || [ "$run_choice" -lt 1 ] || [ "$run_choice" -gt ${#RUNS[@]} ]; then
    fail "Invalid selection: ${run_choice}"
    exit 1
fi

RUN_NAME="${RUNS[$((run_choice - 1))]}"
RUN_DIR="${LOGS_DIR}/${RUN_NAME}"
echo -e "   ${SYM_CHECK} Selected: ${GREEN}${RUN_NAME}${RESET}"

# ╔══════════════════════════════════════╗
# ║  2. Show checkpoints & select        ║
# ╚══════════════════════════════════════╝
section "Checkpoint Selection"

# Gather and sort checkpoints
CKPTS=()
for pt_file in "$RUN_DIR"/model_*.pt; do
    [ -f "$pt_file" ] || continue
    iter=$(basename "$pt_file" | sed 's/model_\([0-9]*\)\.pt/\1/')
    CKPTS+=("$iter")
done
IFS=$'\n' CKPTS=($(sort -n <<<"${CKPTS[*]}")); unset IFS
LATEST_CKPT="${CKPTS[-1]}"

# Show range summary
if [ ${#CKPTS[@]} -le 10 ]; then
    echo -e "   ${DIM}Available:${RESET} ${CKPTS[*]}"
else
    # Show first 3 ... last 5
    first3="${CKPTS[0]}, ${CKPTS[1]}, ${CKPTS[2]}"
    last5="${CKPTS[-5]}, ${CKPTS[-4]}, ${CKPTS[-3]}, ${CKPTS[-2]}, ${CKPTS[-1]}"
    echo -e "   ${DIM}Available:${RESET} ${first3}, ..., ${last5}"
fi
echo -e "   ${DIM}Latest:${RESET}    ${WHITE}${LATEST_CKPT}${RESET}"
echo ""

ask "Resume from iteration" "$LATEST_CKPT"
RESUME_ITER="$REPLY"

# Validate checkpoint exists
RESUME_FILE="${RUN_DIR}/model_${RESUME_ITER}.pt"
if [ ! -f "$RESUME_FILE" ]; then
    fail "Checkpoint not found: ${RESUME_FILE}"
    exit 1
fi
echo -e "   ${SYM_CHECK} Checkpoint: ${GREEN}model_${RESUME_ITER}.pt${RESET}"

# ╔══════════════════════════════════════╗
# ║  3. Auto-detect action space & env   ║
# ╚══════════════════════════════════════╝
section "Run Configuration"

# Try to auto-detect from cfgs.pkl using Python
CFGS_FILE="${RUN_DIR}/cfgs.pkl"
ACTION_SPACE=""
ENV_VERSION="v1"

if [ -f "$CFGS_FILE" ]; then
    # Use Python to parse cfgs.pkl and extract metadata
    DETECT_RESULT=$(python3 -c "
import pickle, types, sys

# Stub DictConfig so unpickle works without full imports
class DictConfig(dict):
    def to_dict(self): return dict(self)

for mod_name in ['train_rl_wb', 'train_rl_discrete_wb', 'train_rl_simple_discrete_wb']:
    mod = types.ModuleType(mod_name)
    mod.DictConfig = DictConfig
    sys.modules[mod_name] = mod

try:
    cfgs = pickle.load(open('$CFGS_FILE', 'rb'))
    env_cfg, obs_cfg, reward_cfg, train_cfg = cfgs

    # Detect env version from reward keys
    scales = reward_cfg.get('reward_scales', {})
    env_ver = 'v2' if 'progress' in scales else 'v1'

    # Detect action space from wandb_project or experiment name
    project = train_cfg.get('wandb_project', '')
    exp = train_cfg.get('experiment_name', '')
    combined = project + ' ' + exp
    if 'simple-discrete' in combined or 'simple_discrete' in combined:
        action = 'simple-discrete'
    elif 'discrete' in combined:
        action = 'discrete'
    else:
        action = 'continuous'

    print(f'{action}|{env_ver}')
except Exception as e:
    print(f'error|{e}', file=sys.stderr)
    print('|')
" 2>/dev/null || echo "|")

    IFS='|' read -r DETECTED_ACTION DETECTED_ENV <<< "$DETECT_RESULT"
    if [ -n "$DETECTED_ACTION" ]; then
        ACTION_SPACE="$DETECTED_ACTION"
    fi
    if [ -n "$DETECTED_ENV" ]; then
        ENV_VERSION="$DETECTED_ENV"
    fi
fi

# If auto-detection failed, ask the user
if [ -z "$ACTION_SPACE" ]; then
    warn "Could not auto-detect action space from cfgs.pkl"
    echo -e "   ${WHITE}1)${RESET} continuous       (4D float in [-1,1])"
    echo -e "   ${WHITE}2)${RESET} discrete         (4x3 ternary: stay/neg/pos per axis)"
    echo -e "   ${WHITE}3)${RESET} simple-discrete  (1-of-9: fwd/bwd/l/r/up/dn/yaw/halt)"
    printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Select"
    read -r action_choice
    case "${action_choice:-1}" in
        2) ACTION_SPACE="discrete" ;;
        3) ACTION_SPACE="simple-discrete" ;;
        *) ACTION_SPACE="continuous" ;;
    esac
    # Also ask env version for continuous
    if [ "$ACTION_SPACE" = "continuous" ]; then
        echo -e "   ${WHITE}1)${RESET} v1  (distance + time penalties)"
        echo -e "   ${WHITE}2)${RESET} v2  (progress + close rewards, no dt-scaling)"
        printf "   ${DIM}%-20s${RESET}${CYAN}[1]${RESET}: " "Env version"
        read -r version_choice
        case "${version_choice:-1}" in
            2) ENV_VERSION="v2" ;;
            *) ENV_VERSION="v1" ;;
        esac
    fi
fi

info "Action space" "$ACTION_SPACE"
info "Env version" "$ENV_VERSION"

# ╔══════════════════════════════════════╗
# ║  4. Training parameters              ║
# ╚══════════════════════════════════════╝
section "Training Parameters"

# Default: double the latest checkpoint
DEF_MAX_ITERS=$(( LATEST_CKPT * 2 ))
[ "$DEF_MAX_ITERS" -lt $(( RESUME_ITER + 1000 )) ] && DEF_MAX_ITERS=$(( RESUME_ITER + 1000 ))

ask "New max iterations" "$DEF_MAX_ITERS"
MAX_ITERS="$REPLY"

if [ "$MAX_ITERS" -le "$RESUME_ITER" ]; then
    fail "Max iterations (${MAX_ITERS}) must be greater than resume iteration (${RESUME_ITER})"
    exit 1
fi

ADDITIONAL=$(( MAX_ITERS - RESUME_ITER ))
echo -e "   ${DIM}Will train ${ADDITIONAL} additional iterations (${RESUME_ITER} → ${MAX_ITERS})${RESET}"

ask "Batch size" "256"
BATCH="$REPLY"

# Adaptive LR + learning rate override (only wired into continuous train_rl_wb.py)
ADAPTIVE_LR="false"
DESIRED_KL=""
LEARNING_RATE=""
if [ "$ACTION_SPACE" = "continuous" ]; then
    ask_yn "Adaptive LR (KL)" "n"
    ADAPTIVE_LR="$REPLY"
    if [ "$ADAPTIVE_LR" = "true" ]; then
        ask "Desired KL" "0.02"
        DESIRED_KL="$REPLY"
    fi
    ask "Learning rate" "0.0005"
    LEARNING_RATE="$REPLY"
fi

# Optional rename: fork the experiment into a new log dir with copied checkpoints
if [ "$ADAPTIVE_LR" = "true" ]; then
    DEFAULT_NEW_NAME="${RUN_NAME}-adaptiveKL"
else
    DEFAULT_NEW_NAME="${RUN_NAME}-continued"
fi
ask "New exp name (blank=keep)" "$DEFAULT_NEW_NAME"
NEW_RUN_NAME="$REPLY"

# ╔══════════════════════════════════════╗
# ║  5. Cluster resources                ║
# ╚══════════════════════════════════════╝
section "Cluster Resources"
echo -e "   ${DIM}Partitions: paula (A30), clara (V100), clara-long (V100 10d)${RESET}"

ask "Partition" "paula"
PARTITION="$REPLY"

case "$PARTITION" in
    paula)  DEF_GPU="a30" ;;
    clara*) DEF_GPU="v100" ;;
    *)      DEF_GPU="a30" ;;
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
# ║  6. Build command                    ║
# ╚══════════════════════════════════════╝

# Pick the right training script
case "$ACTION_SPACE" in
    simple-discrete) TRAIN_SCRIPT="train_rl_simple_discrete_wb.py" ;;
    discrete)        TRAIN_SCRIPT="train_rl_discrete_wb.py" ;;
    *)               TRAIN_SCRIPT="train_rl_wb.py" ;;
esac

# Fork into a new experiment dir if renamed, otherwise continue in place.
# Forking copies all model_*.pt from the source dir so eval/plotting has full
# history, and leaves cfgs.pkl to be rewritten by train_rl_wb.py at launch
# (so it reflects the new flags — adaptive-lr, etc).
if [ -n "$NEW_RUN_NAME" ] && [ "$NEW_RUN_NAME" != "$RUN_NAME" ]; then
    EFFECTIVE_RUN_NAME="$NEW_RUN_NAME"
    NEW_RUN_DIR="${LOGS_DIR}/${NEW_RUN_NAME}"
    if [ -d "$NEW_RUN_DIR" ]; then
        echo -e "   ${DIM}Target dir already exists:${RESET} $NEW_RUN_DIR"
        ask_yn "Overwrite?" "n"
        if [ "$REPLY" != "true" ]; then
            echo -e "   ${DIM}Cancelled.${RESET}\n"; exit 0
        fi
        rm -rf "$NEW_RUN_DIR"
    fi
    mkdir -p "$NEW_RUN_DIR"
    echo -e "   ${DIM}Copying checkpoints:${RESET} ${RUN_DIR} -> ${NEW_RUN_DIR}"
    cp "$RUN_DIR"/model_*.pt "$NEW_RUN_DIR"/
else
    EFFECTIVE_RUN_NAME="$RUN_NAME"
fi

# Resume path relative to proto dir (points at the ckpt in the effective dir)
RESUME_PATH="logs/${EFFECTIVE_RUN_NAME}/model_${RESUME_ITER}.pt"

# Build args
TRAIN_ARGS="-e ${EFFECTIVE_RUN_NAME} -B ${BATCH} --max_iterations ${MAX_ITERS} --resume ${RESUME_PATH}"
if [ "$ENV_VERSION" = "v2" ] && [ "$ACTION_SPACE" = "continuous" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --env-v2"
fi
if [ "$ADAPTIVE_LR" = "true" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --adaptive-lr --desired-kl ${DESIRED_KL}"
fi
if [ -n "$LEARNING_RATE" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --learning-rate ${LEARNING_RATE}"
fi

JOB_NAME="continue-${EFFECTIVE_RUN_NAME}-from-${RESUME_ITER}"

# ╔══════════════════════════════════════╗
# ║  7. Confirmation                     ║
# ╚══════════════════════════════════════╝
section "Summary"
info "Run" "$RUN_NAME"
if [ "$EFFECTIVE_RUN_NAME" != "$RUN_NAME" ]; then
    info "Forked to" "$EFFECTIVE_RUN_NAME"
fi
info "Resume from" "iteration ${RESUME_ITER}"
info "Train to" "iteration ${MAX_ITERS} (+${ADDITIONAL})"
info "Action space" "$ACTION_SPACE"
info "Env version" "$ENV_VERSION"
if [ "$ADAPTIVE_LR" = "true" ]; then
    info "Adaptive LR" "enabled (desired_kl=${DESIRED_KL})"
fi
if [ -n "$LEARNING_RATE" ]; then
    info "Learning rate" "$LEARNING_RATE"
fi
info "Script" "$TRAIN_SCRIPT"
info "Command" "python ${TRAIN_SCRIPT} ${TRAIN_ARGS}"
echo ""
info "Partition" "$PARTITION"
info "GPU" "${GPU_TYPE} x${GPU_COUNT}"
info "Memory" "$MEM"
info "CPUs" "$CPUS"
info "Time limit" "${HOURS}h"
info "Batch size" "${BATCH} envs"

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
spin_run "Creating log directory..." mkdir -p "${SLURM_LOG_DIR}"

JOBSCRIPT=$(mktemp /tmp/genesis_continue_XXXXXX.sh)

cat > "$JOBSCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus=${GPU_TYPE}:${GPU_COUNT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${HOURS}:00:00
#SBATCH --output=${SLURM_LOG_DIR}/slurm-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/slurm-%j.err
#SBATCH --mail-type=END,FAIL

# --- Environment ---
module purge
module load Anaconda3
module load CUDA/12.6.0
eval "\$(conda shell.bash hook)"
conda activate ba_v04

# Pin Vulkan to NVIDIA ICD
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

# --- Training ---
cd ${PROTO_DIR}
echo "=== Continuing training: \$(date) ==="
echo "Host: \$(hostname), GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Run: ${RUN_NAME}, resuming from iteration ${RESUME_ITER} → ${MAX_ITERS}"

python ${TRAIN_SCRIPT} ${TRAIN_ARGS}

echo "=== Training complete: \$(date) ==="
SLURM_EOF

cp "$JOBSCRIPT" "${SLURM_LOG_DIR}/${JOB_NAME}.sh"
rm "$JOBSCRIPT"

spin_run "Submitting to SLURM..." sbatch "${SLURM_LOG_DIR}/${JOB_NAME}.sh"

done_banner "Job submitted: ${JOB_NAME}"
hint "Monitor with: squeue -u \$USER"
hint "Logs: ${SLURM_LOG_DIR}/slurm-<jobid>.out"
