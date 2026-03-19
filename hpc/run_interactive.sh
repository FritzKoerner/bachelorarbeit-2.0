#!/bin/bash
# ==============================================================================
# Request an interactive GPU session on HPC Leipzig
#
# Usage:
#   ./run_interactive.sh                                  # defaults
#   ./run_interactive.sh -p clara -g v100 -t 10           # partition, gpu, time
#   ./run_interactive.sh -p paula -g a30 --mem 16G -c 8   # more resources
#   ./run_interactive.sh -G 2 -N 1 --mem 32G              # 2 GPUs, 32GB RAM
#
# Options:
#   -p, --partition PART   Partition name          (default: clara)
#   -g, --gpu TYPE         GPU type                (default: rtx2080ti)
#   -t, --time MINUTES     Time limit in minutes   (default: 10)
#   -m, --mem SIZE         Memory (e.g. 8G, 16G)   (default: 8G)
#   -G, --num-gpus N       Number of GPUs           (default: 1)
#   -c, --cpus N           CPUs per task            (default: 4)
#   -N, --nodes N          Number of nodes          (default: 1)
# ==============================================================================

source "$(dirname "$0")/_output.sh"

# Defaults
PARTITION="clara"
GPU_TYPE="rtx2080ti"
MINUTES=10
MEM="8G"
NUM_GPUS=1
CPUS=4
NODES=1

show_help() {
    sed -n '2,16p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--partition) PARTITION="$2"; shift 2 ;;
        -g|--gpu)       GPU_TYPE="$2"; shift 2 ;;
        -t|--time)      MINUTES="$2"; shift 2 ;;
        -m|--mem)       MEM="$2"; shift 2 ;;
        -G|--num-gpus)  NUM_GPUS="$2"; shift 2 ;;
        -c|--cpus)      CPUS="$2"; shift 2 ;;
        -N|--nodes)     NODES="$2"; shift 2 ;;
        -h|--help)      show_help; exit 0 ;;
        *) echo "Unknown option: $1. Use -h for help."; exit 1 ;;
    esac
done

banner "Interactive GPU Session" "$MAGENTA"

section "Configuration"
info "Partition" "$PARTITION"
info "GPU" "${GPU_TYPE} x${NUM_GPUS}"
info "Memory" "$MEM"
info "CPUs" "$CPUS"
info "Nodes" "$NODES"
info "Time limit" "${MINUTES} min"

section "After allocation" "$SYM_GEAR"
echo ""
echo -e "   ${DIM}Run smoke tests on compute node:${RESET}"
echo -e "   ${CYAN}srun --pty ~/genesis_v04/hpc/smoke_test.sh${RESET}"
echo ""
echo -e "   ${DIM}Or get an interactive shell:${RESET}"
echo -e "   ${CYAN}srun --pty bash${RESET}"
echo ""

section "Requesting allocation" "$SYM_ROCKET"
step "salloc --partition=$PARTITION --gpus=${GPU_TYPE}:${NUM_GPUS} --mem=$MEM --cpus=$CPUS -N $NODES --time=00:${MINUTES}:00"
echo ""

salloc \
    --partition="$PARTITION" \
    --gpus="${GPU_TYPE}:${NUM_GPUS}" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --time="00:${MINUTES}:00" \
    -N "$NODES"
