#!/bin/bash
# ==============================================================================
# Request an interactive GPU session on HPC Leipzig
#
# Usage:
#   ./run_interactive.sh              # fully interactive
# ==============================================================================

source "$(dirname "$0")/_output.sh"

# --- Prompt helpers (same as submit_training.sh) ---
# Usage: ask "Label" "default_value"  →  sets REPLY (or default if empty)
ask() {
    local label="$1"
    local default="$2"
    local input
    printf "   ${DIM}%-20s${RESET}${CYAN}[${default}]${RESET}: " "$label"
    read -r input
    REPLY="${input:-$default}"
}

banner "Interactive GPU Session" "$MAGENTA"

# ╔══════════════════════════════════════╗
# ║  1. Cluster resources                ║
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
NUM_GPUS="$REPLY"

ask "Memory" "16G"
MEM="$REPLY"

ask "CPUs" "4"
CPUS="$REPLY"

ask "Time (minutes)" "20"
MINUTES="$REPLY"

# ╔══════════════════════════════════════╗
# ║  2. Summary & confirm                ║
# ╚══════════════════════════════════════╝

# Format time for display and salloc
if [ "$MINUTES" -ge 60 ] 2>/dev/null; then
    HOURS=$(( MINUTES / 60 ))
    REMAINING=$(( MINUTES % 60 ))
    TIME_FMT=$(printf "%d:%02d:00" "$HOURS" "$REMAINING")
    TIME_DISPLAY="${HOURS}h${REMAINING}m"
else
    TIME_FMT=$(printf "%02d:00" "$MINUTES")
    TIME_DISPLAY="${MINUTES}min"
fi

section "Summary"
info "Partition" "$PARTITION"
info "GPU" "${GPU_TYPE} x${NUM_GPUS}"
info "Memory" "$MEM"
info "CPUs" "$CPUS"
info "Time limit" "$TIME_DISPLAY"

echo ""
printf "   ${BOLD}Request this allocation?${RESET} ${CYAN}[Y/n]${RESET}: "
read -r confirm
case "${confirm:-Y}" in
    [Nn]*) echo -e "\n   ${DIM}Cancelled.${RESET}\n"; exit 0 ;;
esac

# ╔══════════════════════════════════════╗
# ║  3. Allocate                         ║
# ╚══════════════════════════════════════╝
section "After allocation" "$SYM_GEAR"
echo ""
echo -e "   ${DIM}Load environment:${RESET}"
echo -e "   ${CYAN}source ~/genesis_v04/hpc/setup_env.sh --load${RESET}"
echo ""
echo -e "   ${DIM}Get an interactive shell:${RESET}"
echo -e "   ${CYAN}srun --pty bash${RESET}"
echo ""

section "Requesting allocation" "$SYM_ROCKET"
step "salloc --partition=$PARTITION --gpus=${GPU_TYPE}:${NUM_GPUS} --mem=$MEM --cpus-per-task=$CPUS --time=$TIME_FMT"
echo ""

salloc \
    --partition="$PARTITION" \
    --gpus="${GPU_TYPE}:${NUM_GPUS}" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --time="$TIME_FMT"
