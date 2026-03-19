#!/bin/bash
# Shared output helpers for HPC scripts
# Source this file: source "$(dirname "$0")/_output.sh"

# --- Colors & Styles ---
if [ -t 1 ]; then
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[1;37m'
    BG_BLUE='\033[44m'
    BG_GREEN='\033[42m'
    BG_RED='\033[41m'
    BG_YELLOW='\033[43m'
else
    BOLD='' DIM='' RESET=''
    RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' WHITE=''
    BG_BLUE='' BG_GREEN='' BG_RED='' BG_YELLOW=''
fi

# --- Symbols ---
SYM_CHECK="${GREEN}✔${RESET}"
SYM_CROSS="${RED}✘${RESET}"
SYM_ARROW="${CYAN}▶${RESET}"
SYM_GEAR="${YELLOW}⚙${RESET}"
SYM_ROCKET="${MAGENTA}»${RESET}"
SYM_SYNC="${BLUE}⇄${RESET}"
SYM_DOT="${DIM}·${RESET}"
SYM_WARN="${YELLOW}⚠${RESET}"

# --- Output Functions ---

banner() {
    local text="$1"
    local color="${2:-$CYAN}"
    local len=${#text}
    local pad=$(( len + 6 ))
    local line=$(printf '═%.0s' $(seq 1 $pad))
    echo ""
    echo -e "${color}╔${line}╗${RESET}"
    echo -e "${color}║${RESET}   ${BOLD}${color}${text}${RESET}   ${color}║${RESET}"
    echo -e "${color}╚${line}╝${RESET}"
    echo ""
}

section() {
    local text="$1"
    local sym="${2:-$SYM_ARROW}"
    echo -e "\n ${sym} ${BOLD}${text}${RESET}"
    echo -e " ${DIM}$(printf '─%.0s' $(seq 1 ${#text}))──${RESET}"
}

info() {
    local label="$1"
    local value="$2"
    printf "   ${DIM}%-14s${RESET} ${WHITE}%s${RESET}\n" "$label" "$value"
}

step() {
    echo -e "   ${SYM_GEAR} ${1}"
}

success() {
    echo -e "   ${SYM_CHECK} ${GREEN}${1}${RESET}"
}

warn() {
    echo -e "   ${SYM_WARN} ${YELLOW}${1}${RESET}"
}

fail() {
    echo -e "   ${SYM_CROSS} ${RED}${1}${RESET}"
}

done_banner() {
    local text="${1:-Done}"
    echo ""
    echo -e " ${BG_GREEN}${WHITE} ${SYM_CHECK} ${text} ${RESET}"
    echo ""
}

hint() {
    echo -e "   ${DIM}:: ${1}${RESET}"
}

# --- Spinner / Loading Animations ---

# Start a spinner in the background with a message.
# Usage: spin_start "Uploading files..."
#        long_running_command
#        spin_stop
SPIN_PID=""
SPIN_MSG=""

spin_start() {
    SPIN_MSG="$1"
    if [ ! -t 1 ]; then
        echo "   $SPIN_MSG"
        return
    fi
    (
        local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
        local i=0
        while true; do
            printf "\r   ${CYAN}${frames[$i]}${RESET} ${SPIN_MSG}" >&2
            i=$(( (i + 1) % ${#frames[@]} ))
            sleep 0.08
        done
    ) &
    SPIN_PID=$!
    disown "$SPIN_PID" 2>/dev/null
}

spin_stop() {
    local status="${1:-0}"
    if [ -n "$SPIN_PID" ]; then
        kill "$SPIN_PID" 2>/dev/null
        wait "$SPIN_PID" 2>/dev/null
        SPIN_PID=""
    fi
    if [ -t 1 ]; then
        if [ "$status" -eq 0 ]; then
            printf "\r   ${SYM_CHECK} ${GREEN}${SPIN_MSG}${RESET}\033[K\n" >&2
        else
            printf "\r   ${SYM_CROSS} ${RED}${SPIN_MSG} (failed)${RESET}\033[K\n" >&2
        fi
    fi
    SPIN_MSG=""
}

# Run a command with a spinner. Captures exit code.
# Usage: spin_run "Uploading files..." scp file remote:path
spin_run() {
    local msg="$1"
    shift
    spin_start "$msg"
    "$@"
    local rc=$?
    spin_stop $rc
    return $rc
}

# Run a command with a spinner, checking stderr for warnings/errors.
# Shows warning symbol if stderr contains WARNING/ERROR, even on exit 0.
# Usage: spin_run_check "Installing deps..." pip install -q foo
spin_run_check() {
    local msg="$1"
    shift
    spin_start "$msg"
    local errlog
    errlog=$(mktemp)
    "$@" 2>"$errlog"
    local rc=$?
    local has_warn=false
    if grep -qiE '(WARNING|ERROR)' "$errlog" 2>/dev/null; then
        has_warn=true
    fi
    if [ $rc -ne 0 ]; then
        spin_stop 1
        cat "$errlog" | while IFS= read -r line; do echo -e "     ${DIM}${line}${RESET}"; done
    elif [ "$has_warn" = true ]; then
        # Exit code 0 but warnings present
        if [ -n "$SPIN_PID" ]; then
            kill "$SPIN_PID" 2>/dev/null
            wait "$SPIN_PID" 2>/dev/null
            SPIN_PID=""
        fi
        if [ -t 1 ]; then
            printf "\r   ${SYM_WARN} ${YELLOW}${msg}${RESET}\033[K\n" >&2
        fi
        SPIN_MSG=""
        grep -iE '(WARNING|ERROR)' "$errlog" | while IFS= read -r line; do
            echo -e "     ${DIM}${line}${RESET}"
        done
    else
        spin_stop 0
    fi
    rm -f "$errlog"
    return $rc
}

# Progress bar for operations with known step count.
# Usage: progress_bar $current $total "label"
progress_bar() {
    local current="$1"
    local total="$2"
    local label="${3:-}"
    local width=30
    local pct=$(( current * 100 / total ))
    local filled=$(( current * width / total ))
    local empty=$(( width - filled ))
    local bar="${CYAN}$(printf '█%.0s' $(seq 1 $filled 2>/dev/null))${DIM}$(printf '░%.0s' $(seq 1 $empty 2>/dev/null))${RESET}"
    printf "\r   ${bar} ${WHITE}%3d%%${RESET} ${DIM}%s${RESET}\033[K" "$pct" "$label" >&2
    if [ "$current" -eq "$total" ]; then
        echo "" >&2
    fi
}
