#!/usr/bin/env bash

# Local Ubuntu workspace initializer
# ==================================
#
# Create a local AISeism workspace without scheduler, network, package, or
# credential side effects.  Existing matching links are reused; conflicting
# files and links are rejected rather than replaced.  Use --dry-run to preview
# every directory and link operation.
#
# Function             | description
# ---------------------|--------------------------------------------------------
# log                  | Print a timestamped informational message.
# fail                 | Report an error and terminate.
# usage                | Print command-line usage information.
# require_command      | Verify that a required executable is available.
# require_directory    | Verify a required directory exists.
# validate_workspace   | Reject unsafe or invalid workspace paths.
# link_directory       | Create or verify one non-destructive symlink.
# initialize           | Create local directories and OGS directory links.
# main                 | Parse options and run the initializer.

set -euo pipefail
umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly DEFAULT_OGS_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
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
usage() { # 15
    cat <<EOF
Usage: $SCRIPT [OPTIONS]

Create a local workspace and link OGS config/data/src directories.

Options:
  --ogs-root DIRECTORY    OGS checkout (default: $DEFAULT_OGS_ROOT).
  --workspace DIRECTORY   Workspace to create (default: $DEFAULT_WORKSPACE).
  --dry-run               Report operations without changing the filesystem.
  -h, --help              Show this help message.

The initializer never removes or replaces an existing path.
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

# require_directory
# -----------------
# Verify that PATH is an existing directory.
require_directory() { # 5
    local -r path="$1"

    [[ -d "$path" ]] || fail 1 "directory not found: $path"
}

# validate_workspace
# ------------------
# Reject empty, root, or OGS-identical workspace paths.
validate_workspace() { # 9
    local -r workspace="$1"
    local -r ogs_root="$2"

    [[ -n "$workspace" && "$workspace" != "/" ]] ||
        fail 2 "workspace must be a non-root path"
    [[ "$workspace" != "$ogs_root" ]] ||
        fail 2 "workspace must not be the OGS checkout itself"
}

# link_directory
# --------------
# Create or verify one symlink, refusing to overwrite any conflicting path.
link_directory() { # 20
    local -r source="$1"
    local -r target="$2"
    local -r dry_run="$3"

    if [[ -L "$target" ]]; then
        [[ "$(readlink -f -- "$target")" == "$source" ]] ||
            fail 1 "conflicting symlink exists: $target"
        log "Verified link: $target -> $source"
        return 0
    fi
    [[ ! -e "$target" ]] ||
        fail 1 "refusing to replace existing path: $target"
    if [[ "$dry_run" == true ]]; then
        log "Would link: $target -> $source"
    else
        ln -s -- "$source" "$target"
        log "Created link: $target -> $source"
    fi
}

# initialize
# ----------
# Create workspace directories and links to the OGS checkout.
initialize() { # 32
    local -r ogs_root="$1"
    local -r workspace="$2"
    local -r dry_run="$3"
    local directory name
    local -a directories=(catalogs waveform logs)
    local -a links=(config data src)

    require_directory "$ogs_root"
    for name in "${links[@]}"; do
        require_directory "$ogs_root/$name"
    done
    validate_workspace "$workspace" "$ogs_root"

    if [[ "$dry_run" == true ]]; then
        log "Would create directory: $workspace"
    else
        mkdir -p -- "$workspace"
        log "Created or verified directory: $workspace"
    fi
    for directory in "${directories[@]}"; do
        if [[ "$dry_run" == true ]]; then
            log "Would create directory: $workspace/$directory"
        else
            mkdir -p -- "$workspace/$directory"
        fi
    done
    for name in "${links[@]}"; do
        link_directory "$ogs_root/$name" "$workspace/$name" "$dry_run"
    done
    printf 'Local workspace ready: %s\n' "$workspace"
}

# main
# ----
# Parse options and invoke the non-destructive initializer.
main() { # 50
    local ogs_root="$DEFAULT_OGS_ROOT"
    local workspace="$DEFAULT_WORKSPACE"
    local dry_run=false

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
            --dry-run)
                dry_run=true
                shift
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

    require_command date
    require_command ln
    require_command mkdir
    require_command readlink
    ogs_root="$(cd -- "$ogs_root" 2>/dev/null && pwd -P)" ||
        fail 1 "could not resolve OGS root: $ogs_root"
    if [[ -e "$workspace" ]]; then
        workspace="$(cd -- "$workspace" 2>/dev/null && pwd -P)" ||
            fail 1 "could not resolve workspace: $workspace"
    else
        workspace="$(cd -- "$(dirname -- "$workspace")" 2>/dev/null && pwd -P)/$(basename -- "$workspace")" ||
            fail 1 "could not resolve workspace parent: $workspace"
    fi
    initialize "$ogs_root" "$workspace" "$dry_run"
}

main "$@"
