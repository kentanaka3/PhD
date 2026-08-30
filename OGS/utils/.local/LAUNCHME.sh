#!/usr/bin/env bash

# Local Ubuntu command runner
# ===========================
#
# Run one command with an explicit working directory and thread settings.
# Arguments are retained as an array, so paths and values containing spaces
# are not reparsed by a shell.  Dry-run mode prints the exact argv and returns
# without executing it.
#
# Function          | description
# ------------------|--------------------------------------------------------
# log               | Print a timestamped informational message.
# fail              | Report an error and terminate.
# usage             | Print command-line usage information.
# require_command   | Verify that a required executable is available.
# validate_integer  | Validate a positive integer option.
# validate_assignment| Validate an environment assignment.
# print_command     | Print an argv vector using shell-escaped values.
# main              | Parse options and execute or preview one command.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"

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
usage() { # 18
    cat <<EOF
Usage: $SCRIPT [OPTIONS] -- COMMAND [ARGUMENT ...]

Run COMMAND locally without reparsing its arguments.

Options:
  --dry-run             Print the action without executing COMMAND.
  --workdir DIRECTORY   Run COMMAND from DIRECTORY (default: current directory).
  --threads N           Set OMP_NUM_THREADS and NUMBA_NUM_THREADS to N.
  --env NAME=VALUE      Add an environment assignment (repeatable).
  -h, --help            Show this help message.

Examples:
  $SCRIPT --dry-run --threads 4 -- python3 script.py --input 'a file'
  $SCRIPT --workdir /path/to/work --env HYDRA_FULL_ERROR=1 -- command
EOF
}

# require_command
# ---------------
# Verify that COMMAND is available on PATH.
require_command() { # 6
    local -r command_name="$1"

    command -v "$command_name" >/dev/null 2>&1 ||
        fail 127 "required command not found: $command_name"
}

# validate_integer
# ----------------
# Validate that VALUE is a positive decimal integer.
validate_integer() { # 7
    local -r option_name="$1"
    local -r value="$2"

    [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
        fail 2 "$option_name must be a positive integer: $value"
}

# validate_assignment
# -------------------
# Validate the NAME=VALUE form accepted by the environment array.
validate_assignment() { # 10
    local -r assignment="$1"
    local name

    [[ "$assignment" == *=* ]] ||
        fail 2 "--env requires NAME=VALUE: $assignment"
    name="${assignment%%=*}"
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] ||
        fail 2 "invalid environment variable name: $name"
}

# print_command
# -------------
# Print an argv vector with shell-escaped arguments for reproducibility.
print_command() { # 9
    local argument

    printf 'Command:'
    for argument in "$@"; do
        printf ' %q' "$argument"
    done
    printf '\n'
}

# main
# ----
# Parse options and execute or preview the requested command.
main() { # 73
    local dry_run=false
    local workdir="$PWD"
    local threads=
    local option
    local -a environment=()
    local -a command=()

    while [[ $# -gt 0 ]]; do
        option="$1"
        case "$option" in
            --dry-run)
                dry_run=true
                shift
                ;;
            --workdir)
                (( $# >= 2 )) || fail 2 "--workdir requires a directory argument"
                workdir="$2"
                shift 2
                ;;
            --threads)
                (( $# >= 2 )) || fail 2 "--threads requires a value"
                validate_integer "--threads" "$2"
                threads="$2"
                shift 2
                ;;
            --env)
                (( $# >= 2 )) || fail 2 "--env requires NAME=VALUE"
                validate_assignment "$2"
                environment+=("$2")
                shift 2
                ;;
            -h|--help)
                usage
                return 0
                ;;
            --)
                shift
                command=("$@")
                break
                ;;
            *)
                fail 2 "unknown option: $option (use -- before COMMAND)"
                ;;
        esac
    done

    (( ${#command[@]} > 0 )) || {
        usage >&2
        fail 2 "a command is required after --"
    }
    if [[ ! -d "$workdir" && "$dry_run" != true ]]; then
        fail 1 "working directory not found: $workdir"
    fi
    if [[ -n "$threads" ]]; then
        environment+=("OMP_NUM_THREADS=$threads" "NUMBA_NUM_THREADS=$threads")
    fi

    log "Working directory: $workdir"
    [[ -d "$workdir" ]] || log "Dry run: working directory is not present yet"
    print_command "${command[@]}"
    if [[ "$dry_run" == true ]]; then
        log "Dry run: command was not executed"
        return 0
    fi
    require_command env
    require_command "${command[0]}"

    (
        cd -- "$workdir"
        env "${environment[@]}" "${command[@]}"
    )
}

main "$@"
