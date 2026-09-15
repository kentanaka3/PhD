#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# LAUNCHME.sh - SLURM Job Submission Wrapper Script
# =============================================================================
#
# OVERVIEW:
# This script dynamically configures and submits SLURM batch jobs. It modifies
# a template job script by inserting the appropriate resource allocation
# parameters (nodes, tasks, cpus) and then submits it to the SLURM 
# scheduler.
#
# The script uses a "template-and-replace" pattern where placeholder values
# marked with '#' in the job script are replaced with actual values before
# submission, then restored to placeholders after submission for reuse.
#
# KEY FEATURES:
#   - Automatic MPI vs. serial mode selection based on task count
#   - Dynamic SLURM directive configuration (nodes, tasks, CPUs, GPUs)
#   - CPU count optimization (ensures tasks * cpus = $CORE_COUNT)
#   - Output file naming with resource configuration encoded
#   - Template restoration after submission for script reusability
#
# USAGE:
#   bash LAUNCHME.sh [OPTIONS] <jobFile> <nodes> <tasks> <jobName> <command> [args ...]
#   bash LAUNCHME.sh --help
#
# OPTIONS:
#   -v, --verbose  Enable verbose output
#   -h, --help     Show this help message and exit
#
# ARGUMENTS:
#   <jobFile>  - Base name of the SLURM job script (without .sh extension)
#   <nodes>    - Number of compute nodes to request
#   <tasks>    - Number of MPI tasks (or GPUs) per node
#   <jobName>  - Name to assign to the SLURM job
#   <command>  - The actual command/script to run
#
# RESOURCE ALLOCATION LOGIC:
#   The script ensures that (tasks * cpus) = $CORE_COUNT to fully utilize
#   node resources. For example:
#     - 1 task  → 32 cpus per task
#     - 2 tasks → 16 cpus per task
#     - 4 tasks →  8 cpus per task
#     - etc.
#
# EXAMPLES:
#   bash LAUNCHME.sh ktanakah 1 1 MyJob python script.py --arg value
#   bash LAUNCHME.sh ktanakah "1 2" "1 2" MyJob python script.py --arg value
#   bash LAUNCHME.sh -v ktanakah 1 1 MyJob python script.py --arg value
#
# Function            | description
# --------------------|--------------------------------------------------------
# usage               | Displays a usage message.
# set_execution_mode  | Selects the command form that the template will execute.
# configure_template  | Replaces placeholder markers in the SLURM template.
# restore_template    | Reverses substitutions and restores template placeholders.
# cleanup             | EXIT-trap callback to restore the template on termination.
# main                | Parses options/arguments and orchestrates job submissions.
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

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# MPI command prefix used in the template when a multi-task job is selected.
# The trailing space is intentional: it allows the script to append the MPI
# process count, producing commands such as `mpirun -np 4`.
readonly MPI_CMD="mpirun -np "

# Number of logical cores that the resource-selection loop tries to fill.
readonly CORE_COUNT=${CORE_COUNT:-32}

# Resolve paths from this script so it works regardless of the callers current
# directory. The submitted template is still executed by Slurm in its normal
# submission working directory.
readonly SCRIPT=$(basename -- "${BASH_SOURCE[0]}")
readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Verbosity flag (0: quiet, 1: verbose). Can be set via environment or -v/--verbose.
VERBOSE=${VERBOSE:-0}
case "${VERBOSE:-0}" in
  1|true|yes) VERBOSE=1 ;;
  *) VERBOSE=0 ;;
esac

# Keep cleanup state in named globals for recovery by the EXIT trap.
TEMPLATE_CONFIGURED=0
CURRENT_NODES=
CURRENT_TASKS=
CURRENT_CPUS=
OVERRIDE_CORES=${OVERRIDE_CORES:-}
TEMPLATE=
JOB_NAME=
JOB_NAME_SUFFIX=
NODES=
TASKS=

# USER is normally exported on the cluster, but id provides a safe fallback.
readonly CURRENT_USER=${USER:-$(id -un)}

# Extract first letter of username for output file naming convention
# e.g., "ktanakah" → "k"
readonly USER_INITIAL="${CURRENT_USER:0:1}"

# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# usage
# -----
# Displays a usage message.
usage() { # 14
  cat <<EOF
Usage: $SCRIPT [OPTIONS] <jobFile> <nodes> <tasks> <jobName> <command> [args ...]

Options:
  -v, --verbose  Enable verbose output.
  -h, --help     Show this help message and exit.

Environment:
  VERBOSE        Preset verbosity (0: quiet, 1: verbose).

Configure and submit ./<jobFile>.sh through SLURM.
EOF
}


# set_execution_mode TASKS
# -----------------------------
# Selects the command form that the template will execute.
#
# Arguments:
#   TASKS  Number of tasks per node; 1 selects serial Python, while values
#          greater than 1 select the MPI command.
#
# Side effects:
#   Edits $TEMPLATE in place. The template must contain a commented serial
#   Python command and a commented or active mpirun command in the expected
#   format.
set_execution_mode() { # 19
  # If only 1 task is requested, use serial Python execution
  # Otherwise, use MPI parallel execution
  if [ "$1" -eq 1 ]; then
    [[ $VERBOSE -ne 0 ]] && log "Setting execution mode to SERIAL (1 task)"
    # SERIAL MODE: Enable 'python' command, disable 'mpirun' command
    # - Uncomment lines starting with "# python"
    # - Comment out lines starting with "mpirun -np"
    sed -i -E -e "s/^# (python )/\1/g" \
              -e "s/^${MPI_CMD}/# ${MPI_CMD}/g" "$TEMPLATE"
  else
    [[ $VERBOSE -ne 0 ]] && log "Setting execution mode to PARALLEL ($1 tasks)"
    # PARALLEL MODE: Enable 'mpirun' command, disable 'python' command
    # - Comment out lines starting with "python"
    # - Uncomment lines starting with "# mpirun -np"
    sed -i -E -e "s/^(python )/# \1/g" \
              -e "s/^# ${MPI_CMD}/${MPI_CMD}/g" "$TEMPLATE"
  fi
}

# configure_template
# -----------------------------------------------------------------------
# Replaces the # markers in the SLURM template with one concrete resource
# configuration before submission. The textual, zero-padded values are used
# only in the stdout/stderr filenames.
#
# Side effects:
#   Modifies $TEMPLATE. Call restore_template after sbatch, or rely on the
#   EXIT trap if submission or a later command terminates unexpectedly.
configure_template() { # 13
  [[ $VERBOSE -ne 0 ]] && log "Configuring template for submission"
  sed -i -E -e "s/(#SBATCH --job-name=\"${CURRENT_USER} )#/\1${JOB_NAME}/g" \
            -e "s/(#SBATCH --nodes=)#/\1${CURRENT_NODES}/g" \
            -e "s/(#SBATCH --tasks-per-node=)#/\1${CURRENT_TASKS}/g" \
            -e "s/(#SBATCH --gres=gpu:)#/\1${CURRENT_TASKS}/g" \
            -e "s/(#SBATCH --cpus-per-task=)#/\1${CURRENT_CPUS}/g" \
            -e "s/(#SBATCH --error=${USER_INITIAL}_%j_)#.err/\1${JOB_NAME_SUFFIX}.err/g" \
            -e "s/(#SBATCH --output=${USER_INITIAL}_%j_)#.out/\1${JOB_NAME_SUFFIX}.out/g" \
            -e "s/(export NUMBA_NUM_THREADS=)#/\1${CURRENT_CPUS}/g" \
            -e "s/(export OMP_NUM_THREADS=)#/\1${CURRENT_CPUS}/g" \
            -e "s/${MPI_CMD}#/${MPI_CMD}${CURRENT_TASKS}/g" "$TEMPLATE"
}

# restore_template
# ----------------
# Reverses the substitutions made by configure_template and returns the
# template to its reusable placeholder state.
#
# State contract:
#   TEMPLATE_CONFIGURED must be 1, and the CURRENT_* variables must describe
#   the active configuration. If no configuration is active, the function is
#   a no-op. This makes it safe to call repeatedly from normal flow and from
#   the EXIT trap.
restore_template() { # 16
  [ "$TEMPLATE_CONFIGURED" -eq 1 ] || return 0
  [[ $VERBOSE -ne 0 ]] && log "Restoring template"
  sed -i -E \
    -e "s/(#SBATCH --job-name=\"${CURRENT_USER} )${JOB_NAME}/\1#/g" \
    -e "s/(#SBATCH --nodes=)${CURRENT_NODES}/\1#/g" \
    -e "s/(#SBATCH --tasks-per-node=)${CURRENT_TASKS}/\1#/g" \
    -e "s/(#SBATCH --gres=gpu:)${CURRENT_TASKS}/\1#/g" \
    -e "s/(#SBATCH --cpus-per-task=)${CURRENT_CPUS}/\1#/g" \
    -e "s/(#SBATCH --error=${USER_INITIAL}_%j_)${JOB_NAME_SUFFIX}.err/\1#.err/g" \
    -e "s/(#SBATCH --output=${USER_INITIAL}_%j_)${JOB_NAME_SUFFIX}.out/\1#.out/g" \
    -e "s/(export NUMBA_NUM_THREADS=)${CURRENT_CPUS}/\1#/g" \
    -e "s/(export OMP_NUM_THREADS=)${CURRENT_CPUS}/\1#/g" \
    -e "s/${MPI_CMD}${CURRENT_TASKS}/${MPI_CMD}#/g" "$TEMPLATE"
  TEMPLATE_CONFIGURED=0
}

# cleanup
# -------
# EXIT-trap callback. Cleanup errors are deliberately suppressed so that a
# failure during recovery does not hide the original submission error.
cleanup() { # 3
  restore_template || true
}

# main
# ----
# Parses options and arguments, validates configuration, and submits jobs.
main() { # 90
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        return 0
        ;;
      -v|--verbose)
        VERBOSE=1
        shift
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

  if [ "$#" -lt 5 ]; then
    usage >&2
    fail 2 "expected at least 5 arguments"
  fi

  TEMPLATE="${SCRIPT_DIR}/$1.sh"
  shift

  if [ ! -f "$TEMPLATE" ]; then
    fail 1 "template not found: $TEMPLATE"
  fi

  NODES=$1
  shift

  TASKS=$1
  shift

  JOB_NAME=$1
  shift

  for resource_value in $NODES $TASKS; do
    validate_positive_integer "$resource_value" "resource value"
  done

  if [ -z "$JOB_NAME" ]; then
    fail 2 "job name must not be empty"
  fi

  case "$JOB_NAME" in
    *[!A-Za-z0-9_.-]*)
      fail 2 "job name may contain only letters, numbers, _, ., and -"
      ;;
  esac

  trap cleanup EXIT

  local submission_status=0
  for CURRENT_NODES in ${NODES}; do
    for CURRENT_TASKS in ${TASKS}; do
      set_execution_mode "$CURRENT_TASKS"
      for CURRENT_CPUS in ${OVERRIDE_CORES:-64 32 16 8 4 2 1}; do
        if [ -n "$OVERRIDE_CORES" ] || [ "$(($CURRENT_TASKS * $CURRENT_CPUS))" -eq "$CORE_COUNT" ]; then
          log $(printf "Nodes: %02d, Tasks (MPI & GPU): %02d, CPUs (OpenMP): %03d" $CURRENT_NODES $CURRENT_TASKS $CURRENT_CPUS)
          JOB_NAME_SUFFIX=$(printf "%02d_%02d_%03d" $CURRENT_NODES $CURRENT_TASKS $CURRENT_CPUS)

          TEMPLATE_CONFIGURED=1
          configure_template

          if sbatch "$TEMPLATE" "$@"; then
            submission_status=0
            log "Submitted job for ${JOB_NAME}"
          else
            submission_status=$?
          fi

          restore_template
          if [ "$submission_status" -ne 0 ]; then
            fail "$submission_status" "sbatch submission failed"
          fi
        fi
      done
    done
  done
}

main "$@"
