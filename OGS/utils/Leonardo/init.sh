#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# AISeism Leonardo workspace initializer
# =============================================================================
#
# OVERVIEW:
# Initialize the configured work directory and its external dependencies.
# Existing repositories, Conda installations, and Conda environments are
# verified and reused, making this script safe to run more than once.
#
# Function            | description
# --------------------|--------------------------------------------------------
# usage               | Print command-line usage information.
# require_variable    | Verify a required environment variable is set.
# require_directory   | Verify a required directory exists.
# clone_or_verify     | Verify or clone external repository dependencies.
# load_modules        | Load the compiler/CUDA modules required on Leonardo.
# install_conda       | Verify or install Miniconda.
# ensure_environment  | Verify or create the configured Conda environment.
# copy_directories    | Copy configured directories into the work directory.
# extract_directories | Extract configured directory contents into the work directory.
# link_directories    | Create the configured workspace symlinks.
# run_smoke_test      | Run the existing dummy pipeline test.
# main                | Validate configuration and perform initialization.
#
# Required configuration is supplied through environment variables by the
# Makefile. Override those variables at invocation time rather than editing
# this script.
#
# AUTHORS:
#   - 健
#   - Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)
#     Centro di Ricerche Sismologiche (CRS)
#   - Università degli Studi di Trieste (UniTS)
#     Dipartimento di Matematica, Informatica e Geoscienze (MIGe)
#     Applied Data Science and Artificial Intelligence (ADSAI)
#   - Terabit Network for Research and Academic Big Data in Italy (TeRABIT)
#     Consorzio Interuniversitario del Nord-Est per il Calcolo Automatico (CINECA)
#
# =============================================================================

umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# ---------------------------------------------------------------------------
# Usage and environment validation
# ---------------------------------------------------------------------------

# usage
# -----
# Print command-line usage information.
usage() { # 18
    cat <<EOF
Usage: $SCRIPT [OPTIONS]

AISeism Leonardo workspace initializer.
Initialize the configured work directory and its external dependencies.

Options:
  -h, --help    Show this help message and exit.

Required environment variables:
  WORK_PATH, OGS_PATH, NLL_PATH, DATASET_PATH, CONDA_ROOT,
  CONDA_ENV, SBC_PATH, CONDA_INSTALLER, SBC_RUN_BIN

Override variables at invocation time or via Makefile:
  make init
EOF
}

# require_variable
# ----------------
# Verify a required environment variable is set.
require_variable() { # 8
    local -r variable_name="$1"

    [[ -n "${!variable_name:-}" ]] ||
        fail 2 "$variable_name must be set"

    log "Verified required variable: $variable_name=${!variable_name}"
}

# require_directory
# -----------------
# Verify a required directory exists.
require_directory() { # 9
    local -r variable_name="$1"
    local -r directory_path="${!variable_name:-}"

    [[ -d "$directory_path" ]] ||
        fail 3 "required directory not found: $variable_name=$directory_path"

    log "Verified required directory: $variable_name=$directory_path"
}

# ---------------------------------------------------------------------------
# Dependency setup
# ---------------------------------------------------------------------------

# clone_or_verify
# ---------------
# Verify or clone external repository dependencies.
clone_or_verify() { # 33
    log "Verifying external dependencies"
    local -a dep_vars=()
    if [[ $# -ge 1 && "$1" == *\[@\] ]]; then
        dep_vars=("${!1}")
    else
        dep_vars=("$@")
    fi
    local -A dependency_urls=(
        [SBC_PATH]="https://github.com/<ORG>/ml_catalog_main.git"
        [NLL_PATH]="https://github.com/ut-beg-texnet/NonLinLoc.git"
        [DATASET_PATH]="https://zenodo.org/api/records/22106548/files-archive"
    )
    local dep_var dep_path dep_url
    for dep_var in "${dep_vars[@]}"; do
        [[ -n "$dep_var" ]] || continue
        dep_path="${!dep_var:-}"
        dep_url="${dependency_urls[$dep_var]:-}"
        if [[ -d "$dep_path" ]] && [[ -n "$(find "$dep_path" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
            log "Verified dependency: $dep_var=$dep_path"
        elif [[ -e "$dep_path" && ! -d "$dep_path" ]]; then
            fail 1 "expected directory for $dep_var, found non-directory path: $dep_path"
        elif ! command -v git >/dev/null 2>&1; then
            fail 127 "dependency $dep_var not found and git is not available: $dep_path"
        elif [[ -z "$dep_url" ]]; then
            fail 1 "repository $dep_var is not available and has no configured clone URL: $dep_path"
        else
            log "Cloning $dep_url into $dep_path"
            mkdir -p -- "$dep_path"
            git clone "$dep_url" "$dep_path"
        fi
    done
}

# load_modules
# ------------
# Load the compiler/CUDA modules required on Leonardo.
load_modules() { # 10
    # `module` is provided by Leonardo's login environment rather than a
    # standalone executable, so test it as a shell command.
    if ! type module >/dev/null 2>&1; then
        fail 127 "module command not available; run this script in a Leonardo environment"
    fi

    log "Loading Leonardo modules"
    module load nvhpc/ cuda/
}

# install_conda
# -------------
# Verify or install Miniconda.
install_conda() { # 13
    if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
        log "Verified existing Conda installation: $CONDA_ROOT"
        return
    fi

    log "Downloading Miniconda installer to $CONDA_INSTALLER"
    curl --fail --location --output "$CONDA_INSTALLER" \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    log "Installing Conda into $CONDA_ROOT"
    bash "$CONDA_INSTALLER" -b -p "$CONDA_ROOT"
}

# ensure_environment
# ------------------
# Verify or create the configured Conda environment.
ensure_environment() { # 13
    local -r conda="$CONDA_ROOT/bin/conda"

    if "$conda" env list | awk '{print $1}' | grep --fixed-strings --line-regexp --quiet "$CONDA_ENV"; then
        log "Verified existing Conda environment: $CONDA_ENV"
    else
        log "Creating Conda environment: $CONDA_ENV"
        "$conda" env create -f "$WORK_PATH/LEONARDO.yml" -n "$CONDA_ENV"

        log "Installing ml_catalog_main in editable mode"
        "$conda" run -n "$CONDA_ENV" python -m pip install --editable "$SBC_PATH"
    fi
}

# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

# copy_directories
# ----------------
# Copy configured directories into the work directory.
copy_directories() { # 22
    log "Copying directories"
    local -a dir_names=()
    if [[ $# -ge 1 && "$1" == *\[@\] ]]; then
        dir_names=("${!1}")
    else
        dir_names=("$@")
    fi
    local -r dest_path="$WORK_PATH"
    local dir_name source_path target_dest
    for dir_name in "${dir_names[@]}"; do
        [[ -n "$dir_name" ]] || continue
        source_path="$OGS_PATH/$dir_name"
        [[ -d "$source_path" ]] ||
            fail 1 "required directory to copy not found: $source_path"
        target_dest="$dest_path/$dir_name"
        if [[ -L "$target_dest" ]]; then
            rm -f -- "$target_dest"
        fi
        cp -a -- "$source_path" "$dest_path/"
    done
}

# extract_directories
# -------------------
# Extract configured directory contents into the work directory.
extract_directories() { # 17
    log "Extracting directories"
    local -a dir_names=()
    if [[ $# -ge 1 && "$1" == *\[@\] ]]; then
        dir_names=("${!1}")
    else
        dir_names=("$@")
    fi
    local dir_name source_path
    for dir_name in "${dir_names[@]}"; do
        [[ -n "$dir_name" ]] || continue
        source_path="$OGS_PATH/$dir_name"
        [[ -d "$source_path" ]] ||
            fail 1 "required directory to extract not found: $source_path"
        cp -a -- "$source_path/." "$WORK_PATH/"
    done
}

# link_directories
# ----------------
# Create the configured workspace symlinks.
link_directories() { # 19
    log "Linking directories"
    local -a link_names=()
    if [[ $# -ge 1 && "$1" == *\[@\] ]]; then
        link_names=("${!1}")
    else
        link_names=("$@")
    fi
    local link_name dest_path
    for link_name in "${link_names[@]}"; do
        [[ -n "$link_name" ]] || continue
        dest_path="$WORK_PATH/$link_name"
        if [[ -e "$dest_path" && ! -L "$dest_path" ]]; then
            log "WARNING: destination path exists and is not a symlink, skipping: $dest_path"
            continue
        fi
        ln -sfn -- "$OGS_PATH/$link_name" "$dest_path"
    done
}

# run_smoke_test
# --------------
# Run the existing dummy pipeline test.
run_smoke_test() { # 9
    log "Running the dummy pipeline smoke test"
    (
        cd "$WORK_PATH"
        OVERRIDE_CORES=1 CORE_COUNT=1 VERBOSE=1 \
        bash LAUNCHME.sh dummy 1 1 launchme_dummy \
            "$CONDA_ROOT/envs/$CONDA_ENV/bin/python" "$WORK_PATH/test/dummy.py"
        OVERRIDE_CORES=1 CORE_COUNT=1 VERBOSE=1 \
        bash LAUNCHME.sh dummy 1 1 sbc_help $SBC_RUN_BIN --help
    )
}

# ---------------------------------------------------------------------------
# Main lifecycle
# ---------------------------------------------------------------------------

# main
# ----
# Validate configuration and perform initialization.
main() { # 55
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                return 0
                ;;
            *)
                fail 2 "unknown option: $1"
                ;;
        esac
    done

    local variable_name
    local -a required_commands=(awk bash cp curl grep ln mkdir sbatch)
    local -a required_variables=(
        WORK_PATH OGS_PATH NLL_PATH DATASET_PATH CONDA_ROOT CONDA_ENV SBC_PATH
        CONDA_INSTALLER SBC_RUN_BIN
    )
    local -a extracted_directories=(
        utils/Leonardo
    )
    local -a copied_directories=(
        conf
    )
    local -a linked_directories=(
        data src
    )
    local -a external_dependencies=(
        SBC_PATH NLL_PATH DATASET_PATH
    )

    for variable_name in "${required_variables[@]}"; do
        require_variable "$variable_name"
    done
    for variable_name in "${required_commands[@]}"; do
        require_command "$variable_name"
    done

    require_directory WORK_PATH
    require_directory OGS_PATH

    copy_directories copied_directories[@]
    extract_directories extracted_directories[@]
    link_directories linked_directories[@]
    clone_or_verify external_dependencies[@]

    load_modules
    install_conda
    ensure_environment
    run_smoke_test

    printf '\nInitialization complete!\nWork directory: %s\nNext step:\n    cd %s && make help\n\n' \
        "$WORK_PATH" "$WORK_PATH"
}

main "$@"
