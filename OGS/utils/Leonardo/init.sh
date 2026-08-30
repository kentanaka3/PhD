#!/usr/bin/env bash

set -euo pipefail

# AISeism Leonardo workspace initializer
# ======================================
#
# Initialize the configured work directory and its external dependencies.
# Existing repositories, Conda installations, and Conda environments are
# verified and reused, making this script safe to run more than once.
#
# Function          | description
# ------------------|--------------------------------------------------------
# log               | Print a timestamped message.
# fail              | Report an error and terminate.
# require_command   | Verify a local executable is available.
# require_variable  | Verify a required environment variable is set.
# require_directory | Verify a required directory exists.
# clone_or_verify   | Verify or clone a Git repository.
# load_modules      | Load the compiler/CUDA modules required on Leonardo.
# install_conda     | Verify or install Miniconda.
# ensure_environment| Verify or create the configured Conda environment.
# copy_utilities    | Copy Leonardo utility scripts into the work directory.
# link_workspace    | Create the configured workspace symlinks.
# run_smoke_test    | Run the existing dummy pipeline test.
# main              | Validate configuration and perform initialization.
#
# Required configuration is supplied through environment variables by the
# Makefile. Override those variables at invocation time rather than editing
# this script.

umask 077

readonly SCRIPT="$(basename -- "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Logging and errors
# ---------------------------------------------------------------------------

log() {
    printf '[%s][%s] %s\n' \
        "$SCRIPT" \
        "$(date '+%Y-%m-%d %H:%M:%S%z')" \
        "$*"
}

fail() {
    local -r status="$1"
    shift

    log "ERROR: $*" >&2
    exit "$status"
}

require_command() {
    local -r command_name="$1"

    command -v "$command_name" >/dev/null 2>&1 ||
        fail 127 "required command not found: $command_name"
}

require_variable() {
    local -r variable_name="$1"

    [[ -n "${!variable_name:-}" ]] ||
        fail 2 "$variable_name must be set"
}

require_directory() {
    local -r variable_name="$1"
    local -r directory_path="${!variable_name:-}"

    [[ -d "$directory_path" ]] ||
        fail 3 "required directory not found: $variable_name=$directory_path"
}

# ---------------------------------------------------------------------------
# Dependency setup
# ---------------------------------------------------------------------------

clone_or_verify() {
    local -r repository_url="$1"
    local -r repository_path="$2"

    if [[ -d "$repository_path/.git" ]]; then
        log "Verified repository: $repository_path"
    elif [[ -e "$repository_path" ]]; then
        fail 1 "expected Git repository, found existing non-repository path: $repository_path"
    elif [[ -z "$repository_url" ]]; then
        fail 1 "repository is not available and has no configured clone URL: $repository_path"
    else
        log "Cloning $repository_url into $repository_path"
        git clone "$repository_url" "$repository_path"
    fi
}

load_modules() {
    # `module` is provided by Leonardo's login environment rather than a
    # standalone executable, so test it as a shell command.
    if ! type module >/dev/null 2>&1; then
        fail 127 "module command not available; run this script in a Leonardo environment"
    fi

    log "Loading Leonardo modules"
    module load nvhpc/ cuda/
}

install_conda() {
    if [[ -x "$CONDA_ROOT/bin/conda" ]]; then
        log "Verified existing Conda installation: $CONDA_ROOT"
        return
    fi

    require_command curl
    log "Downloading Miniconda installer to $CONDA_INSTALLER"
    curl --fail --location --output "$CONDA_INSTALLER" \
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

    log "Installing Conda into $CONDA_ROOT"
    bash "$CONDA_INSTALLER" -b -p "$CONDA_ROOT"
}

ensure_environment() {
    local -r conda="$CONDA_ROOT/bin/conda"

    if "$conda" env list | awk '{print $1}' | grep --fixed-strings --line-regexp --quiet "$CONDA_ENV"; then
        log "Verified existing Conda environment: $CONDA_ENV"
    else
        log "Creating Conda environment: $CONDA_ENV"
        "$conda" env create -f "$WORK_PATH/LEONARDO.yml" -n "$CONDA_ENV"
    fi

    log "Installing ml_catalog_main in editable mode"
    "$conda" run -n "$CONDA_ENV" python -m pip install --editable "$SBC_PATH"
}

# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

copy_utilities() {
    log "Copying Leonardo utility scripts to $WORK_PATH"
    cp -a "$OGS_PATH/utils/Leonardo/." "$WORK_PATH/"
}

link_workspace() {
    log "Linking directories"
    local -r link_names=("${!1:-}")
    for link_name in "${link_names[@]}"; do
        ln -sfn -- "$OGS_PATH/$link_name" "$WORK_PATH/$link_name"
    done
}

run_smoke_test() {
    log "Running the dummy pipeline smoke test"
    (
        cd "$WORK_PATH"
        CONDA_ROOT="$CONDA_ROOT" \
            NLL_PATH="$NLL_PATH" \
            DATASET_PATH="$DATASET_PATH" \
            bash LAUNCHME.sh dummy 1 1 launchme_dummy \
                python "$WORK_PATH/test/dummy.py"
    )
}

# ---------------------------------------------------------------------------
# Main lifecycle
# ---------------------------------------------------------------------------

main() {
    local variable_name
    local -a required_commands=(awk bash curl cp git grep ln mkdir)
    local -a required_variables=(
        WORK_PATH OGS_PATH NLL_PATH DATASET_PATH CONDA_ROOT CONDA_ENV
        SBC_PATH CONDA_INSTALLER
    )
    local -a required_directories=(
        WORK_PATH OGS_PATH NLL_PATH DATASET_PATH SBC_PATH
    )
    local -a required_links=(
        config data src
    )

    for variable_name in "${required_variables[@]}"; do
        require_variable "$variable_name"
    done
    for variable_name in "${required_commands[@]}"; do
        require_command "$variable_name"
    done
    for variable_name in "${required_directories[@]}"; do
        require_directory "$variable_name"
    done

    link_workspace required_links[@]

    mkdir -p -- "$WORK_PATH"
    copy_utilities

    log "Verifying external dependencies"
    clone_or_verify "" "$SBC_PATH"
    clone_or_verify "https://github.com/ut-beg-texnet/NonLinLoc.git" "$NLL_PATH"
    clone_or_verify "git@github.com:amagrin/bollettino_ogs_hypo71.git" "$DATASET_PATH"

    load_modules
    install_conda
    ensure_environment
    link_workspace
    run_smoke_test

    printf '\nInitialization complete!\nWork directory: %s\nNext step:\n    cd %s && make help\n\n' \
        "$WORK_PATH" "$WORK_PATH"
}

main "$@"
