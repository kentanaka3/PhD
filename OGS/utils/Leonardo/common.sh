#!/usr/bin/env bash

# LLM Script Common Functions
# ===========================
#
# Provide shared diagnostic and dependency helpers for the LLM command scripts.
# This file is source-only and has no command-line interface or side effects.

# ---------------------------------------------------------------------------
# Logging and errors
# ---------------------------------------------------------------------------

# log
# ---
# Print a timestamped diagnostic message to standard error.
log() { # 6
    printf '[%s][%s] %s\n' \
        "$SCRIPT" \
        "$(date '+%Y-%m-%d %H:%M:%S%z')" \
        "$*" >&2
}

# fail
# ----
# Report an error message and terminate the script with a non-zero exit code.
fail() { # 7
    local -r status="$1"
    shift

    log "ERROR: $*"
    exit "$status"
}

# require_command
# ---------------
# Verify that a required executable is available.
require_command() { # 6
    local -r command_name="$1"

    command -v "$command_name" >/dev/null 2>&1 ||
        fail 127 "required command not found: $command_name"
}

# validate_positive_integer
# -------------------------
# Verify that an argument is a positive decimal integer.
validate_positive_integer() { # 7
    local -r value="$1"
    local -r name="$2"

    [[ "$value" =~ ^[1-9][0-9]*$ ]] ||
        fail 2 "$name must be a positive decimal integer: $value"
}

# validate_navigation_config
# --------------------------
# Check command-specific executable prerequisites.
validate_navigation_config() { # 15
    local -r command_name="$1"

    case "$command_name" in
        outline|get)
            require_command awk
            ;;
        slice)
            require_command sed
            ;;
        *)
            fail 2 "unsupported command: $command_name"
            ;;
    esac
}

# Parse navigation commands and dispatch to fixed script-owned handlers.
run_navigation_command() { # 34
    local -r usage_function="$1"
    local -r outline_function="$2"
    local -r get_function="$3"
    local -r slice_function="$4"
    shift 4

    local -r command_name="${1:-}"

    case "$command_name" in
        -h|--help)
            "$usage_function"
            return 0
            ;;
        outline|get|slice)
            shift
            ;;
        '')
            "$usage_function" >&2
            return 2
            ;;
        *)
            "$usage_function" >&2
            fail 2 "unknown command: $command_name"
            ;;
    esac

    validate_navigation_config "$command_name"
    case "$command_name" in
        outline) "$outline_function" "$@" ;;
        get) "$get_function" "$@" ;;
        slice) "$slice_function" "$@" ;;
    esac
}