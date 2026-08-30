#!/usr/bin/env bash

# Local Ubuntu prerequisite checker
# =================================
#
# Validate the paths and executables needed by the local Makefile.  This is a
# read-only check: it does not create directories, activate environments, or
# contact external services.  Missing optional tools are reported separately
# from required local infrastructure.
#
# Function             | description
# ---------------------|--------------------------------------------------------
# log                  | Print a timestamped informational message.
# fail                 | Report an error and terminate.
# usage                | Print command-line usage information.
# check_command        | Check one required command and record failures.
# check_directory      | Check one required directory and record failures.
# check_optional       | Report an optional command without failing the check.
# main                 | Parse options and run all local checks.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_OGS_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
readonly DEFAULT_WORKSPACE="${WORK_PATH:-${HOME:-$PWD}/AISeism/WORK}"

# log
# ---
# Print a timestamped informational message to standard error.
log() { # 6
    printf '[%s][%s] %s\n' \
        "$SCRIPT" \
        "$(date '+%Y-%m-%d %H:%M:%S%z')" \
        "$*" >&2
}

# fail
# ----
# Report an error message and terminate with STATUS.
fail() { # 7
    local -r status="$1"
    shift

    log "ERROR: $*"
    exit "$status"
}

# usage
# -----
# Print command-line usage information.
usage() { # 14
    cat <<EOF
Usage: $SCRIPT [OPTIONS]

Validate local Ubuntu prerequisites without changing the filesystem.

Options:
  --ogs-root DIRECTORY    OGS checkout (default: $DEFAULT_OGS_ROOT).
  --workspace DIRECTORY   Local workspace (default: $DEFAULT_WORKSPACE).
  --python COMMAND        Python executable (default: python3).
  --conda-prefix DIR      Optional Conda prefix to verify.
  -h, --help              Show this help message.
EOF
}

# check_command
# -------------
# Check COMMAND and increment the failure counter through the return status.
check_command() { # 10
    local -r command_name="$1"

    if command -v "$command_name" >/dev/null 2>&1; then
        printf 'OK  command: %s\n' "$command_name"
    else
        printf 'ERR command: %s (not found)\n' "$command_name" >&2
        return 1
    fi
}

# check_directory
# ---------------
# Check PATH and increment the failure counter through the return status.
check_directory() { # 10
    local -r path="$1"

    if [[ -d "$path" ]]; then
        printf 'OK  directory: %s\n' "$path"
    else
        printf 'ERR directory: %s (not found)\n' "$path" >&2
        return 1
    fi
}

# check_optional
# --------------
# Report an optional executable without treating it as a prerequisite.
check_optional() { # 9
    local -r command_name="$1"

    if command -v "$command_name" >/dev/null 2>&1; then
        printf 'OK  optional command: %s\n' "$command_name"
    else
        printf 'INFO optional command unavailable: %s\n' "$command_name"
    fi
}

# main
# ----
# Parse options and validate the local checkout, workspace, and tools.
main() { # 70
    local ogs_root="$DEFAULT_OGS_ROOT"
    local workspace="$DEFAULT_WORKSPACE"
    local python_command=python3
    local conda_prefix=
    local failures=0
    local directory
    local -a required_commands=(bash date find make)
    local -a required_directories=(config data src)

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --ogs-root)
                (( $# >= 2 )) || fail 2 "--ogs-root requires a directory argument"
                ogs_root="$2"
                shift 2
                ;;
            --workspace)
                (( $# >= 2 )) || fail 2 "--workspace requires a directory argument"
                workspace="$2"
                shift 2
                ;;
            --python)
                (( $# >= 2 )) || fail 2 "--python requires a command"
                python_command="$2"
                shift 2
                ;;
            --conda-prefix)
                (( $# >= 2 )) || fail 2 "--conda-prefix requires a directory argument"
                conda_prefix="$2"
                shift 2
                ;;
            -h|--help)
                usage
                return 0
                ;;
            --)
                shift
                (( $# == 0 )) || fail 2 "unexpected positional argument: $1"
                ;;
            *)
                fail 2 "unknown option: $1"
                ;;
        esac
    done

    for directory in "${required_commands[@]}"; do
        check_command "$directory" || (( failures += 1 ))
    done
    check_command "$python_command" || (( failures += 1 ))
    check_directory "$ogs_root" || (( failures += 1 ))
    for directory in "${required_directories[@]}"; do
        check_directory "$ogs_root/$directory" || (( failures += 1 ))
    done
    if [[ -e "$workspace" ]]; then
        check_directory "$workspace" || (( failures += 1 ))
    else
        printf 'INFO workspace will be created by init.sh: %s\n' "$workspace"
    fi
    if [[ -n "$conda_prefix" ]]; then
        check_command "$conda_prefix/bin/conda" || (( failures += 1 ))
    else
        check_optional conda
    fi

    if (( failures > 0 )); then
        fail 1 "local prerequisite check failed with $failures issue(s)"
    fi
    printf 'Local prerequisite check passed.\n'
}

main "$@"
