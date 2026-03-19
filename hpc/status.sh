#!/bin/bash
# ==============================================================================
# HPC Status Dashboard — view jobs, partitions, and GPU availability
#
# Usage:
#   ./hpc/status.sh                  Show all sections
#   ./hpc/status.sh --jobs           Jobs only
#   ./hpc/status.sh --partitions     Partitions only
#   ./hpc/status.sh --gpus           GPU availability only
#   ./hpc/status.sh -w [N]           Auto-refresh every N seconds (default: 30)
#   ./hpc/status.sh --help           Show usage
# ==============================================================================

set -e
source "$(dirname "$0")/_output.sh"

# --- Defaults ---
SHOW_JOBS=false
SHOW_PARTITIONS=false
SHOW_GPUS=false
WATCH_MODE=false
WATCH_INTERVAL=30

# --- Parse arguments ---
parse_args() {
    local any_section=false
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --jobs)        SHOW_JOBS=true; any_section=true; shift ;;
            --partitions)  SHOW_PARTITIONS=true; any_section=true; shift ;;
            --gpus)        SHOW_GPUS=true; any_section=true; shift ;;
            --watch|-w)
                WATCH_MODE=true
                shift
                if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
                    WATCH_INTERVAL="$1"
                    shift
                fi
                ;;
            --help|-h)
                banner "HPC Status" "$CYAN"
                echo -e "  ${BOLD}Usage:${RESET} ./hpc/status.sh [options]"
                echo ""
                echo -e "  ${BOLD}Sections:${RESET}"
                echo -e "    --jobs           Show active jobs only"
                echo -e "    --partitions     Show partition status only"
                echo -e "    --gpus           Show GPU availability only"
                echo ""
                echo -e "  ${BOLD}Options:${RESET}"
                echo -e "    -w, --watch [N]  Auto-refresh every N seconds (default: 30)"
                echo -e "    -h, --help       Show this help"
                echo ""
                exit 0
                ;;
            *)
                fail "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # No specific section selected -> show all
    if [ "$any_section" = false ]; then
        SHOW_JOBS=true
        SHOW_PARTITIONS=true
        SHOW_GPUS=true
    fi
}

# --- Collect data from HPC via single SSH ---
ssh_collect() {
    spin_start "Connecting to HPC cluster..."
    SSH_OUTPUT=$(ssh hpc bash -s 2>/dev/null <<'REMOTE_EOF'
echo "===JOBS==="
squeue -u fk67rahe -o "%i|%j|%T|%P|%M|%l|%D|%R" --noheader 2>/dev/null || echo "CMD_FAILED"

echo "===PENDING==="
squeue --start -u fk67rahe -o "%i|%j|%S|%P|%R" --noheader 2>/dev/null || echo "CMD_FAILED"

echo "===PARTITIONS==="
sinfo -p paula,clara,clara-long -o "%P|%a|%D|%t|%G" --noheader 2>/dev/null || echo "CMD_FAILED"

echo "===GPUS==="
sc_nodes_gap --gpu 2>/dev/null || echo "CMD_FAILED"

echo "===END==="
REMOTE_EOF
    )
    local rc=$?
    spin_stop $rc

    if [ $rc -ne 0 ] || [ -z "$SSH_OUTPUT" ]; then
        fail "Could not connect to HPC cluster"
        hint "Check your SSH config for host 'hpc' and network connectivity"
        exit 1
    fi
}

# --- Extract a section from SSH output ---
extract_section() {
    local name="$1"
    echo "$SSH_OUTPUT" | awk "/^===${name}===$/{found=1; next} /^===/{found=0} found"
}

# --- Display: My Jobs ---
display_jobs() {
    section "My Jobs" "$SYM_ROCKET"

    local jobs_data
    jobs_data=$(extract_section "JOBS")

    if [ -z "$jobs_data" ] || [ "$jobs_data" = "CMD_FAILED" ]; then
        if [ "$jobs_data" = "CMD_FAILED" ]; then
            warn "Could not query jobs (squeue failed)"
        else
            echo -e "   ${DIM}No active jobs${RESET}"
        fi
        return
    fi

    # Load pending start times into associative array
    declare -A start_times
    local pending_data
    pending_data=$(extract_section "PENDING")
    if [ -n "$pending_data" ] && [ "$pending_data" != "CMD_FAILED" ]; then
        while IFS='|' read -r pid pname pstart ppart preason; do
            [ -n "$pid" ] && start_times["$pid"]="$pstart"
        done <<< "$pending_data"
    fi

    # Display each job
    while IFS='|' read -r job_id job_name state partition elapsed limit nodes reason; do
        [ -z "$job_id" ] && continue

        # Color by state
        local state_color
        case "$state" in
            RUNNING)     state_color="${GREEN}" ;;
            PENDING)     state_color="${YELLOW}" ;;
            COMPLETING)  state_color="${CYAN}" ;;
            FAILED|CANCELLED|TIMEOUT|NODE_FAIL)
                         state_color="${RED}" ;;
            *)           state_color="${DIM}" ;;
        esac

        echo ""
        echo -e "   ${BOLD}${WHITE}#${job_id}${RESET}  ${job_name}"
        printf "   ${state_color}%-12s${RESET}" "$state"
        printf " ${DIM}partition:${RESET} %-12s" "$partition"
        printf " ${DIM}time:${RESET} %s / %s" "$elapsed" "$limit"
        echo ""

        if [ "$state" = "RUNNING" ]; then
            printf "   ${DIM}%-12s${RESET}" ""
            printf " ${DIM}node:${RESET} %s\n" "$reason"
        elif [ "$state" = "PENDING" ]; then
            local est="${start_times[$job_id]}"
            if [ -n "$est" ] && [ "$est" != "N/A" ]; then
                printf "   ${DIM}%-12s${RESET}" ""
                printf " ${DIM}est. start:${RESET} ${YELLOW}%s${RESET}\n" "$est"
            fi
            printf "   ${DIM}%-12s${RESET}" ""
            printf " ${DIM}reason:${RESET} %s\n" "$reason"
        fi
    done <<< "$jobs_data"
}

# --- Display: Partitions ---
display_partitions() {
    section "Partitions" "$SYM_GEAR"

    local part_data
    part_data=$(extract_section "PARTITIONS")

    if [ -z "$part_data" ] || [ "$part_data" = "CMD_FAILED" ]; then
        warn "Could not query partitions (sinfo failed)"
        return
    fi

    # GPU types per partition
    declare -A gpu_types
    gpu_types["paula"]="A30 24GB"
    gpu_types["clara"]="V100 32GB"
    gpu_types["clara-long"]="V100 32GB"

    # Time limits per partition
    declare -A time_limits
    time_limits["paula"]="2 days"
    time_limits["clara"]="2 days"
    time_limits["clara-long"]="10 days"

    # Aggregate node counts by partition and state
    declare -A part_idle
    declare -A part_total

    while IFS='|' read -r part avail count state gres; do
        [ -z "$part" ] && continue
        # Remove trailing * from partition name
        part="${part%\*}"

        local total="${part_total[$part]:-0}"
        part_total["$part"]=$(( total + count ))

        # State codes: idle, mix(ed), alloc(ated), drain, down, etc.
        case "$state" in
            idle|idle*)
                local idle="${part_idle[$part]:-0}"
                part_idle["$part"]=$(( idle + count ))
                ;;
            mix|mix*)
                # Mixed nodes have some resources free
                local idle="${part_idle[$part]:-0}"
                part_idle["$part"]=$(( idle + count ))
                ;;
        esac
    done <<< "$part_data"

    # Display each partition
    for part in paula clara clara-long; do
        local total="${part_total[$part]:-0}"
        local idle="${part_idle[$part]:-0}"
        local gpu="${gpu_types[$part]:-unknown}"
        local tlimit="${time_limits[$part]:-unknown}"

        # Color idle count
        local idle_color
        if [ "$idle" -gt 0 ]; then
            idle_color="${GREEN}"
        else
            idle_color="${RED}"
        fi

        echo ""
        printf "   ${BOLD}%-16s${RESET}" "$part"
        printf "${idle_color}%d${RESET}/${WHITE}%d${RESET} nodes avail   " "$idle" "$total"
        printf "${DIM}gpu:${RESET} %-14s" "$gpu"
        printf "${DIM}limit:${RESET} %s" "$tlimit"
        echo ""
    done
}

# --- Display: GPU Availability ---
display_gpus() {
    section "GPU Availability" "$SYM_SYNC"

    local gpu_data
    gpu_data=$(extract_section "GPUS")

    if [ -z "$gpu_data" ] || [ "$gpu_data" = "CMD_FAILED" ]; then
        warn "sc_nodes_gap --gpu not available on this cluster"
        hint "Use partition info above for GPU estimates"
        return
    fi

    # Display raw output indented
    while IFS= read -r line; do
        [ -n "$line" ] && echo -e "   ${DIM}${line}${RESET}"
    done <<< "$gpu_data"
}

# --- Main display ---
display_status() {
    ssh_collect

    banner "HPC Status" "$CYAN"

    [ "$SHOW_JOBS" = true ] && display_jobs
    [ "$SHOW_PARTITIONS" = true ] && display_partitions
    [ "$SHOW_GPUS" = true ] && display_gpus

    echo ""
    echo -e "   ${DIM}Updated: $(date '+%H:%M:%S')${RESET}"
    echo ""
}

# --- Entry point ---
parse_args "$@"

if [ "$WATCH_MODE" = true ]; then
    while true; do
        clear
        display_status
        hint "Refreshing every ${WATCH_INTERVAL}s  (Ctrl+C to stop)"
        sleep "$WATCH_INTERVAL"
    done
else
    display_status
fi
