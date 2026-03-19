#!/bin/bash
# ==============================================================================
# Sync local code to HPC Leipzig cluster
#
# Usage:
#   ./hpc/sync_to_hpc.sh              Sync everything
#   ./hpc/sync_to_hpc.sh --dry-run    Preview what would be synced
# ==============================================================================

set -e
source "$(dirname "$0")/_output.sh"

LOCAL_DIR="/home/fritz-sfl/Bachelorarbeit/genesis_v04/"
REMOTE_DIR="hpc:/home/sc.uni-leipzig.de/fk67rahe/genesis_v04/"

# --- Parse arguments ---
DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN="--dry-run" ;;
        --help|-h)
            banner "Sync to HPC" "$CYAN"
            echo -e "  ${BOLD}Usage:${RESET} ./hpc/sync_to_hpc.sh [options]"
            echo ""
            echo -e "  ${BOLD}Options:${RESET}"
            echo -e "    --dry-run    Preview changes without transferring"
            echo -e "    -h, --help   Show this help"
            echo ""
            exit 0
            ;;
        *)
            fail "Unknown option: $arg"
            exit 1
            ;;
    esac
done

if [ -n "$DRY_RUN" ]; then
    banner "Sync to HPC  (dry run)" "$YELLOW"
    warn "No files will be transferred"
else
    banner "Sync to HPC" "$BLUE"
fi

section "Transfer" "$SYM_SYNC"
info "Local" "$LOCAL_DIR"
info "Remote" "$REMOTE_DIR"

# --- Excludes ---
EXCLUDES=(
    # Version control & IDE
    '.git'
    '.vscode/'
    'genesis.code-workspace'
    'CLAUDE.md'
    '.claude/'

    # Data & logs
    'wandb/'
    'logs/'
    'artifacts/'
    'eval_data/'
    'hpc_results/'

    # Build artifacts
    '__pycache__'
    '*.pyc'

    # Scraped data
    '.firecrawl/'

    # Inactive prototypes
    'prototyp_1/'
    'prototyp_2/'
    'prototyp_global_coordinate_rsl_2.2.4/'
)

EXCLUDE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=("--exclude=$pattern")
done

# --- Sync ---
section "Uploading" "$SYM_ROCKET"

# Count files first for progress tracking
file_count=$(rsync -ahz --dry-run --stats \
    "${EXCLUDE_ARGS[@]}" \
    "$LOCAL_DIR" "$REMOTE_DIR" 2>/dev/null \
    | grep -oP 'Number of regular files transferred: \K\d+' || echo "0")

if [ "$file_count" -eq 0 ]; then
    file_count=1  # avoid division by zero
fi

transferred=0
rsync -ahz --partial --out-format='%n' $DRY_RUN \
    "${EXCLUDE_ARGS[@]}" \
    "$LOCAL_DIR" "$REMOTE_DIR" 2>&1 \
    | while IFS= read -r line; do
        transferred=$(( transferred + 1 ))
        if [ $transferred -gt $file_count ]; then
            transferred=$file_count
        fi
        progress_bar "$transferred" "$file_count" "$line"
    done
rc=${PIPESTATUS[0]}

echo ""
if [ $rc -ne 0 ]; then
    fail "Sync failed (exit code $rc)"
    exit 1
fi

if [ -n "$DRY_RUN" ]; then
    success "Dry run complete"
    hint "Remove --dry-run to transfer files"
else
    success "Sync complete: local -> HPC"
fi
