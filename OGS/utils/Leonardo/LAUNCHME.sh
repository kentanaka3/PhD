#!/usr/bin/env bash
# =============================================================================
# LAUNCHME.sh - SLURM Job Submission Wrapper Script
# =============================================================================
#
# OVERVIEW:
# This script dynamically configures and submits SLURM batch jobs. It modifies
# a template job script by inserting the appropriate resource allocation
# parameters (nodes, tasks, threads) and then submits it to the SLURM 
# scheduler.
#
# The script uses a "template-and-replace" pattern where placeholder values
# marked with '#' in the job script are replaced with actual values before
# submission, then restored to placeholders after submission for reuse.
#
# KEY FEATURES:
#   - Automatic MPI vs. serial mode selection based on task count
#   - Dynamic SLURM directive configuration (nodes, tasks, CPUs, GPUs)
#   - Thread count optimization (ensures tasks * threads = 32)
#   - Output file naming with resource configuration encoded
#   - Template restoration after submission for script reusability
#
# USAGE:
#   bash LAUNCHME.sh <jobFile> <nodes> <tasks> <jobName> <script> [options]
#
# ARGUMENTS:
#   <jobFile>  - Base name of the SLURM job script (without .sh extension)
#   <nodes>    - Number of compute nodes to request
#   <tasks>    - Number of MPI tasks (or GPUs) per node
#   <jobName>  - Name to assign to the SLURM job
#   <script>   - The actual command/script to run
#   [options]  - Additional arguments passed to the script
#
# EXAMPLE:
#   bash LAUNCHME.sh ktanakah 1 1 MyJob python script.py --arg value
#   bash LAUNCHME.sh ktanakah "1 2" "1 2" MyJob python script.py --arg value
#
# RESOURCE ALLOCATION LOGIC:
#   The script ensures that (tasks * threads) = 32 to fully utilize
#   node resources. For example:
#     - 1 task  → 32 threads per task
#     - 2 tasks → 16 threads per task
#     - 4 tasks →  8 threads per task
#     - etc.
#
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Strict mode: exit on error, undefined variable, or pipe failure
set -euo pipefail

# MPI command prefix used in the template when a multi-task job is selected.
# The trailing space is intentional: it allows the script to append the MPI
# process count, producing commands such as `mpirun -np 4`.
readonly MPI_CMD="mpirun -np "

# Number of logical cores that the resource-selection loop tries to fill.
readonly CORE_COUNT=32

# Resolve paths from this script so it works regardless of the callers current
# directory. The submitted template is still executed by Slurm in its normal
# submission working directory.
readonly SCRIPT=$(basename -- "${BASH_SOURCE[0]}")
readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Keep cleanup state in named globals for recovery by the EXIT trap.
TEMPLATE_CONFIGURED=0
CURRENT_NODES=
CURRENT_TASKS=
CURRENT_CPUS=
CURRENT_THREADS=

# USER is normally exported on the cluster, but id provides a safe fallback.
readonly CURRENT_USER=${USER:-$(id -un)}

# log
# ---
# Prints a timestamped log message to stdout.
log() { printf "[%s][%s] %s\n" "$SCRIPT" "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*"; }
# fail
# ----
# Prints a timestamped error message and exits with STATUS.
fail() {
  local status=$1
  shift
  log "ERROR: $*" >&2
  exit "$status"
}

# usage
# -----
# Displays a usage message and exits with status 2.
usage() {
  cat <<EOF
Usage: $SCRIPT <jobFile> <nodes> <tasks> <jobName> <command> [args ...]

Configure and submit ./<jobFile>.sh through SLURM.
EOF
}

# Extract first letter of username for output file naming convention
# e.g., "ktanakah" → "k"
readonly USER_INITIAL="${CURRENT_USER:0:1}"

# -----------------------------------------------------------------------------
# ARGUMENT PARSING AND VALIDATION
# -----------------------------------------------------------------------------

if [ "$#" -lt 5 ]; then
  usage >&2
  fail 2 "expected at least 5 arguments"
fi

# Parse the job file name (the SLURM template script to modify).
TEMPLATE="${SCRIPT_DIR}/$1.sh"
shift

if [ ! -f "$TEMPLATE" ]; then
  fail 1 "template not found: $TEMPLATE"
fi

# Parse number of compute nodes to request
NODES=$1
shift

# Parse number of tasks (MPI processes or GPUs) per node
TASKS=$1
shift

# Parse the job name for SLURM identification
JOB_NAME=$1
shift
# Everything left in "$@" is the command and its arguments. It is forwarded
# to sbatch and becomes the positional arguments received by the template.

# NODES and TASKS may contain multiple space-separated values because the
# nested loops below support submitting several resource configurations in one
# invocation. Validate every value before modifying the template.
# NODES and TASKS intentionally accept space-separated value lists.
for resource_value in $NODES $TASKS; do
  case "$resource_value" in
    ""|*[!0-9]*)
      fail 2 "node/task counts must be positive integers: $resource_value"
      ;;
    0)
      fail 2 "node/task counts must be greater than zero"
      ;;
  esac
done

if [ -z "$JOB_NAME" ]; then
  fail 2 "job name must not be empty"
fi

case "$JOB_NAME" in
  *[!A-Za-z0-9_.-]*)
    fail 2 "job name may contain only letters, numbers, _, ., and -"
    ;;
esac


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
set_execution_mode() {
  # If only 1 task is requested, use serial Python execution
  # Otherwise, use MPI parallel execution
  if [ "$1" -eq 1 ]; then
    # SERIAL MODE: Enable 'python' command, disable 'mpirun' command
    # - Uncomment lines starting with "# python"
    # - Comment out lines starting with "mpirun -np"
    sed -i -E -e "s/^# (python )/\1/g" \
              -e "s/^${MPI_CMD}/# ${MPI_CMD}/g" "$TEMPLATE"
  else
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
configure_template() {
  sed -i -E -e "s/(#SBATCH --job-name=\"${CURRENT_USER} )#/\1${JOB_NAME}/g" \
            -e "s/(#SBATCH --nodes=)#/\1${CURRENT_NODES}/g" \
            -e "s/(#SBATCH --tasks-per-node=)#/\1${CURRENT_TASKS}/g" \
            -e "s/(#SBATCH --gres=gpu:)#/\1${CURRENT_TASKS}/g" \
            -e "s/(#SBATCH --cpus-per-task=)#/\1${CURRENT_CPUS}/g" \
            -e "s/(#SBATCH --error=${USER_INITIAL}_%j_)#.err/\1${JOB_NAME_SUFFIX}.err/g" \
            -e "s/(#SBATCH --output=${USER_INITIAL}_%j_)#.out/\1${JOB_NAME_SUFFIX}.out/g" \
            -e "s/(export NUMBA_NUM_THREADS=)#/\1${CURRENT_THREADS}/g" \
            -e "s/(export OMP_NUM_THREADS=)#/\1${CURRENT_THREADS}/g" \
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
restore_template() {
  [ "$TEMPLATE_CONFIGURED" -eq 1 ] || return 0
  sed -i -E -e "s/(#SBATCH --job-name=\"${CURRENT_USER} )${JOB_NAME}/\1#/g" \
            -e "s/(#SBATCH --nodes=)${CURRENT_NODES}/\1#/g" \
            -e "s/(#SBATCH --tasks-per-node=)${CURRENT_TASKS}/\1#/g" \
            -e "s/(#SBATCH --gres=gpu:)${CURRENT_TASKS}/\1#/g" \
            -e "s/(#SBATCH --cpus-per-task=)${CURRENT_CPUS}/\1#/g" \
            -e "s/(#SBATCH --error=${USER_INITIAL}_%j_)${JOB_NAME_SUFFIX}.err/\1#.err/g" \
            -e "s/(#SBATCH --output=${USER_INITIAL}_%j_)${JOB_NAME_SUFFIX}.out/\1#.out/g" \
            -e "s/(export NUMBA_NUM_THREADS=)${CURRENT_THREADS}/\1#/g" \
            -e "s/(export OMP_NUM_THREADS=)${CURRENT_THREADS}/\1#/g" \
            -e "s/${MPI_CMD}${CURRENT_TASKS}/${MPI_CMD}#/g" "$TEMPLATE"
  TEMPLATE_CONFIGURED=0
}

# cleanup
# -------
# EXIT-trap callback. Cleanup errors are deliberately suppressed so that a
# failure during recovery does not hide the original submission error.
cleanup() {
  restore_template || true
}

# -----------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------

# Iterate over requested node counts (supports space-separated list)
for CURRENT_NODES in ${NODES}; do

  # Iterate over requested task counts (supports space-separated list)
  for CURRENT_TASKS in ${TASKS}; do

    # -------------------------------------------------------------------------
    # SECTION 1: SELECT SERIAL OR MPI EXECUTION
    # -------------------------------------------------------------------------
    set_execution_mode "$CURRENT_TASKS"

    # -------------------------------------------------------------------------
    # SECTION 2: FIND A 32-CORE RESOURCE LAYOUT
    # -------------------------------------------------------------------------
    # Try different thread counts to find configuration where
    # tasks * threads = 32. This ensures full utilization of the 32 cores
    # typically available per node
    for CURRENT_CPUS in 32 16 8 4 2 1; do

      # Calculate total threads across all tasks on this node
      CURRENT_THREADS=$(($CURRENT_NODES * $CURRENT_CPUS))

      # Only proceed if this configuration fills 32 cores (tasks * threads = 32)
      if [ "$(($CURRENT_TASKS * $CURRENT_CPUS))" -eq "$CORE_COUNT" ]; then

        log $(printf "Nodes: %02d, Tasks (MPI and/or GPU): %02d, Threads (OpenMP): %02d" $CURRENT_NODES $CURRENT_TASKS $CURRENT_CPUS)  # e.g., "Nodes: 01, Tasks: 04, Threads: 08"
        JOB_NAME_SUFFIX=$(printf "%02d_%02d_%02d" $CURRENT_NODES $CURRENT_TASKS $CURRENT_CPUS)  # e.g., "01_04_08"

        # ---------------------------------------------------------------------
        # SECTION 3: CONFIGURE, SUBMIT, AND RESTORE THE TEMPLATE
        # ---------------------------------------------------------------------
        TEMPLATE_CONFIGURED=1
        trap cleanup EXIT
        configure_template

        # Preserve the submission exit status while always restoring the template.
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
