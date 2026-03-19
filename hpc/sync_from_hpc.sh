#!/bin/bash
# ==============================================================================
# Sync training results from HPC Leipzig cluster back to local machine
#
# Usage:
#   ./sync_from_hpc.sh                          # sync all prototype logs
#   ./sync_from_hpc.sh global_coordinate        # sync specific prototype
#   ./sync_from_hpc.sh obstacle_avoidance       # sync specific prototype
#   ./sync_from_hpc.sh --dry-run                # preview only
# ==============================================================================

set -e
source "$(dirname "$0")/_output.sh"

LOCAL_DIR="/home/fritz-sfl/Bachelorarbeit/genesis_v04/hpc_results/"
REMOTE_BASE="hpc:/home/sc.uni-leipzig.de/fk67rahe/genesis_v04"

PROTOTYPE=""
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        *) PROTOTYPE="$arg" ;;
    esac
done

if [ -n "$DRY_RUN" ]; then
    banner "Sync from HPC  (dry run)" "$YELLOW"
    warn "No files will be transferred"
else
    banner "Sync from HPC" "$BLUE"
fi

section "Destination"
info "Local dir" "$LOCAL_DIR"
echo ""

mkdir -p "$LOCAL_DIR"

if [ -n "$PROTOTYPE" ]; then
    section "prototyp_${PROTOTYPE}" "$SYM_SYNC"
    mkdir -p "${LOCAL_DIR}prototyp_${PROTOTYPE}/"
    spin_start "Syncing prototyp_${PROTOTYPE}..."
    rsync -ahz --partial $DRY_RUN \
        "${REMOTE_BASE}/prototyp_${PROTOTYPE}/logs/" \
        "${LOCAL_DIR}prototyp_${PROTOTYPE}/"
    spin_stop $?
else
    for proto in global_coordinate obstacle_avoidance; do
        section "prototyp_${proto}" "$SYM_SYNC"
        mkdir -p "${LOCAL_DIR}prototyp_${proto}/"
        spin_start "Syncing prototyp_${proto}..."
        rsync -ahz --partial $DRY_RUN \
            "${REMOTE_BASE}/prototyp_${proto}/logs/" \
            "${LOCAL_DIR}prototyp_${proto}/" 2>/dev/null
        local rc=$?
        if [ $rc -eq 0 ]; then
            spin_stop 0
        else
            spin_stop 1
            warn "No logs yet for prototyp_${proto}"
        fi
    done
fi

# Also sync SLURM output logs
section "SLURM logs" "$SYM_SYNC"
mkdir -p "${LOCAL_DIR}slurm/"
spin_start "Syncing SLURM logs..."
rsync -ahz --partial $DRY_RUN \
    "${REMOTE_BASE}/logs/slurm-*.out" \
    "${REMOTE_BASE}/logs/slurm-*.err" \
    "${LOCAL_DIR}slurm/" 2>/dev/null
rc=$?
if [ $rc -eq 0 ]; then
    spin_stop 0
else
    spin_stop 1
    warn "No SLURM logs yet"
fi

done_banner "Sync complete: HPC -> local"
hint "$LOCAL_DIR"
