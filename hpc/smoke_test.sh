#!/bin/bash
# ==============================================================================
# Smoke test: load env + run a short training on HPC
#
# Usage:
#   ./hpc/smoke_test.sh                        Global coordinate (default)
#   ./hpc/smoke_test.sh --obstacle             Obstacle avoidance
#   ./hpc/smoke_test.sh --both                 Run both sequentially
#   ./hpc/smoke_test.sh --setup                Full env setup first (not --load)
#   ./hpc/smoke_test.sh --obstacle --setup     Combine flags
# ==============================================================================

set -e
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/_output.sh"

GENESIS_DIR="$HOME/genesis_v04"

# --- Defaults ---
RUN_GLOBAL=false
RUN_OBSTACLE=false
FULL_SETUP=false

# --- Parse arguments ---
any_proto=false
for arg in "$@"; do
    case "$arg" in
        --obstacle)   RUN_OBSTACLE=true; any_proto=true ;;
        --global)     RUN_GLOBAL=true; any_proto=true ;;
        --both)       RUN_GLOBAL=true; RUN_OBSTACLE=true; any_proto=true ;;
        --setup)      FULL_SETUP=true ;;
        --help|-h)
            banner "Smoke Test" "$CYAN"
            echo -e "  ${BOLD}Usage:${RESET} ./hpc/smoke_test.sh [options]"
            echo ""
            echo -e "  ${BOLD}Prototypes:${RESET}"
            echo -e "    --global       Run global_coordinate (default)"
            echo -e "    --obstacle     Run obstacle_avoidance"
            echo -e "    --both         Run both sequentially"
            echo ""
            echo -e "  ${BOLD}Options:${RESET}"
            echo -e "    --setup        Full env setup instead of --load"
            echo -e "    -h, --help     Show this help"
            echo ""
            exit 0
            ;;
        *)
            fail "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Default to global_coordinate
if [ "$any_proto" = false ]; then
    RUN_GLOBAL=true
fi

banner "Smoke Test" "$MAGENTA"

# --- Environment setup ---
section "Environment" "$SYM_GEAR"

if [ "$FULL_SETUP" = true ]; then
    step "Running full environment setup..."
    echo ""
    source "$SCRIPT_DIR/setup_env.sh"
else
    step "Loading environment..."
    echo ""
    source "$SCRIPT_DIR/setup_env.sh" --load
fi

# --- Run smoke tests ---
run_smoke_test() {
    local proto="$1"
    local batch="$2"
    local proto_dir="${GENESIS_DIR}/prototyp_${proto}"

    section "prototyp_${proto}" "$SYM_ROCKET"

    if [ ! -d "$proto_dir" ]; then
        fail "Directory not found: $proto_dir"
        return 1
    fi

    info "Directory" "$proto_dir"
    # Prefer train_rl_wb.py if available
    local script="train_rl_wb.py"
    if [ ! -f "$proto_dir/$script" ]; then
        script="train_rl.py"
    fi

    info "Command" "python $script -B $batch --max_iterations 5"
    echo ""

    step "Running training (5 iterations, $batch envs)..."
    echo ""

    cd "$proto_dir"
    python "$script" -B "$batch" --max_iterations 5
    local rc=$?
    cd "$GENESIS_DIR"

    echo ""
    if [ $rc -eq 0 ]; then
        success "prototyp_${proto} passed"
    else
        fail "prototyp_${proto} failed (exit code $rc)"
    fi

    return $rc
}

FAILED=0

if [ "$RUN_GLOBAL" = true ]; then
    run_smoke_test "global_coordinate" 4 || FAILED=$((FAILED + 1))
fi

if [ "$RUN_OBSTACLE" = true ]; then
    run_smoke_test "obstacle_avoidance" 4 || FAILED=$((FAILED + 1))
fi

# --- Summary ---
echo ""
if [ $FAILED -eq 0 ]; then
    success "All smoke tests passed"
else
    fail "$FAILED smoke test(s) failed"
fi
