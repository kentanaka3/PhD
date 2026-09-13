#!/usr/bin/env bash

# <Script name>
# =============
#
# <Short description of what this script does.>
#
# USAGE
#   bash <script>.sh [OPTIONS] [ARGUMENTS]
#   bash <script>.sh --help
#
# DESCRIPTION
#   <Document the workflow, important assumptions, side effects, and whether
#   the script is safe to run repeatedly.>
#
# OPTIONS
#   -h, --help    Show usage information and exit.
#
# Function          | description
# ------------------|--------------------------------------------------------
# usage             | Print command-line usage information.
# validate_config   | Validate arguments and configuration before side effects.
# run               | Perform the main operation.
# main              | Parse arguments, validate configuration, and dispatch.
#
# Function documentation format:
#   # Description of the function and its side effects.
#   <function_name>() { # <line count>
#       <function body>
#   }
#
# Keep the line count updated when changing a function. It is intended as a
# quick navigation aid, not as a substitute for source control line numbers.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# usage
# -----
# Print command-line usage information.
usage() { # 14
    cat <<EOF
Usage: $SCRIPT [OPTIONS] [ARGUMENTS]

Options:
  -h, --help    Show this help message and exit.

Environment:
  <NAME>        <Description and optional default value.>

Examples:
  bash $SCRIPT --help
EOF
}

# ---------------------------------------------------------------------------
# Configuration and operation
# ---------------------------------------------------------------------------

# validate_config
# ---------------
# Validate arguments and configuration before side effects.
validate_config() { # 9
    # Check every prerequisite before changing files, submitting jobs, or
    # contacting external services.
    require_command date
    require_command printf

    # Add argument and environment validation here. Prefer explicit checks
    # for absolute paths, allowed values, and positive integer parameters.
}

# run
# ---
# Perform the script's primary operation.
run() { # 11
    # Put the script's primary operation here. Quote every expansion and use
    # arrays when building commands with optional arguments.
    log "Starting operation"

    # Example:
    # local -a command=(some_command --input "$input_path")
    # "${command[@]}"

    log "Operation complete"
}

# ---------------------------------------------------------------------------
# Main lifecycle
# ---------------------------------------------------------------------------

# main
# ----
# Parse arguments, validate configuration, and dispatch the requested operation.
main() { # 24
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                return 0
                ;;
            --)
                shift
                break
                ;;
            -*)
                usage >&2
                fail 2 "unknown option: $1"
                ;;
            *)
                break
                ;;
        esac
    done

    validate_config "$@"
    run "$@"
}

main "$@"