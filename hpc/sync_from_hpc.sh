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

# ---------------------------------------------------------------------------
# rsync_progress — run rsync showing a file-count progress bar
#   Usage: rsync_progress "label" rsync-args...
#   Does a dry-run first to count files, then transfers with a live bar.
#   Falls back to plain output in non-interactive shells or --dry-run mode.
# ---------------------------------------------------------------------------
rsync_progress() {
    local label="$1"
    shift

    # Non-interactive or --dry-run: plain rsync, no progress bar
    if [ ! -t 1 ] || [ -n "$DRY_RUN" ]; then
        rsync "$@" $DRY_RUN 2>/dev/null
        local rc=$?
        if [ $rc -eq 0 ]; then success "$label"; else warn "$label (no data)"; fi
        return $rc
    fi

    # 1) Count files needing transfer via dry-run
    printf "   ${DIM}Scanning...${RESET}" >&2
    local total
    total=$(rsync "$@" --dry-run --out-format='%n' 2>/dev/null | wc -l)
    total=${total:-0}
    printf "\r\033[K" >&2

    if [ "$total" -eq 0 ]; then
        printf "   ${SYM_CHECK} ${DIM}%s${RESET} ${GREEN}(up to date)${RESET}\n" "$label" >&2
        return 0
    fi

    # 2) Transfer with live progress bar
    progress_bar 0 "$total" "$label  0/${total}"

    local count=0
    local rc_file
    rc_file=$(mktemp)

    while IFS= read -r _; do
        count=$((count + 1))
        progress_bar "$count" "$total" "$label  ${count}/${total}"
    done < <(rsync "$@" --out-format='%n' 2>/dev/null; echo $? > "$rc_file")

    local rc
    rc=$(cat "$rc_file" 2>/dev/null)
    rm -f "$rc_file"

    if [ "${rc:-1}" -eq 0 ]; then
        printf "\r   ${SYM_CHECK} ${GREEN}%s${RESET} ${DIM}(%d files)${RESET}\033[K\n" "$label" "$total" >&2
    else
        printf "\r   ${SYM_CROSS} ${RED}%s (failed)${RESET}\033[K\n" "$label" >&2
    fi
    return "${rc:-1}"
}

# ---------------------------------------------------------------------------
# Build sync task list
# ---------------------------------------------------------------------------
PROTOS=()
if [ -n "$PROTOTYPE" ]; then
    PROTOS=("$PROTOTYPE")
else
    PROTOS=(global_coordinate obstacle_avoidance)
fi

TOTAL_STEPS=$(( ${#PROTOS[@]} + 1 ))   # +1 for SLURM logs
STEP=0

for proto in "${PROTOS[@]}"; do
    STEP=$((STEP + 1))
    section "prototyp_${proto}  (${STEP}/${TOTAL_STEPS})" "$SYM_SYNC"
    mkdir -p "${LOCAL_DIR}prototyp_${proto}/"
    rsync_progress "prototyp_${proto}" \
        -ahz --partial \
        "${REMOTE_BASE}/prototyp_${proto}/logs/" \
        "${LOCAL_DIR}prototyp_${proto}/" || true
done

# SLURM output logs
STEP=$((STEP + 1))
section "SLURM logs  (${STEP}/${TOTAL_STEPS})" "$SYM_SYNC"
mkdir -p "${LOCAL_DIR}slurm/"
rsync_progress "SLURM logs" \
    -ahz --partial \
    "${REMOTE_BASE}/logs/slurm-*.out" \
    "${REMOTE_BASE}/logs/slurm-*.err" \
    "${LOCAL_DIR}slurm/" || true

done_banner "Sync complete: HPC → local"
hint "$LOCAL_DIR"
