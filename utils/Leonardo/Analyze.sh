#!/bin/bash
# =============================================================================
# Analyze.sh - SLURM Job Submission Wrapper Script
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
#   bash Analyze.sh <jobFile> <nodes> <tasks> <jobName> <script> [options]
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
#   bash Analyze.sh ktanakah 1 1 MyJob python script.py --arg value
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

# MPI command prefix - used to launch parallel jobs
MPI_CMD="mpirun -np "

# Extract first letter of username for output file naming convention
# e.g., "ktanakah" → "k"
F="${USER:0:1}"

# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------

# Parse the job file name (the SLURM template script to modify)
FILE=$1
shift  # Remove first argument, shift remaining left

# Parse number of compute nodes to request
NODES=$1
shift

# Parse number of tasks (MPI processes or GPUs) per node
TASKS=$1
shift

# Parse the job name for SLURM identification
JOB_NAME=$1
shift
# Remaining arguments ($@) are passed to the submitted job

# -----------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -----------------------------------------------------------------------------

# Iterate over requested node counts (supports space-separated list)
for k in ${NODES}; do

  # Iterate over requested task counts (supports space-separated list)
  for j in ${TASKS}; do

    # -------------------------------------------------------------------------
    # MPI vs SERIAL MODE SELECTION
    # -------------------------------------------------------------------------
    # If only 1 task is requested, use serial Python execution
    # Otherwise, use MPI parallel execution
    if [ "$(($j))" -eq 1 ]; then
      # SERIAL MODE: Enable 'python' command, disable 'mpirun' command
      # - Uncomment lines starting with "# python"
      # - Comment out lines starting with "mpirun -np"
      sed -i -E -e "s/^# (python )/\1/g" \
                -e "s/^${MPI_CMD}/# ${MPI_CMD}/g" "./$FILE.sh"
    else
      # PARALLEL MODE: Enable 'mpirun' command, disable 'python' command
      # - Comment out lines starting with "python"
      # - Uncomment lines starting with "# mpirun -np"
      sed -i -E -e "s/^(python )/# \1/g" \
                -e "s/^# ${MPI_CMD}/${MPI_CMD}/g" "./$FILE.sh"
    fi

    # -------------------------------------------------------------------------
    # THREAD COUNT OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    # Try different thread counts to find configuration where
    # tasks * threads = 32. This ensures full utilization of the 32 cores
    # typically available per node
    for i in 32 16 8 4 2 1; do

      # Calculate total threads across all tasks on this node
      THREADS=$(($k * $i))

      # Only proceed if this configuration fills 32 cores (tasks * threads = 32)
      if [ "$(($j * $i))" -eq 32 ]; then

        # Format values with leading zeros for consistent output naming
        x=$(printf "%02d" $i)  # Threads per task (e.g., "08")
        y=$(printf "%02d" $j)  # Tasks per node   (e.g., "04")
        z=$(printf "%02d" $k)  # Number of nodes  (e.g., "01")

        # Display configuration summary
        echo "Nodes: $z, Tasks (MPI and/or GPU): $y, Threads (OpenMP): $x"

        # ---------------------------------------------------------------------
        # STEP 1: INJECT CONFIGURATION INTO TEMPLATE
        # ---------------------------------------------------------------------
        # Replace placeholder '#' values in SLURM directives with actual values
        # This modifies the job script with the computed resource configuration
        sed -i -E -e "s/(#SBATCH --job-name=\"${USER} )#/\1$JOB_NAME/g" \
                  -e "s/(#SBATCH --nodes=)#/\1${k}/g" \
                  -e "s/(#SBATCH --tasks-per-node=)#/\1${j}/g" \
                  -e "s/(#SBATCH --gres=gpu:)#/\1${j}/g" \
                  -e "s/(#SBATCH --cpus-per-task=)#/\1${i}/g" \
                  -e "s/(#SBATCH -e ${F}_%j_)#_#_#.err/\1${z}_${y}_${x}.err/g"\
                  -e "s/(#SBATCH -o ${F}_%j_)#_#_#.out/\1${z}_${y}_${x}.out/g"\
                  -e "s/(export NUMBA_NUM_THREADS=)#/\1${THREADS}/g" \
                  -e "s/(export OMP_NUM_THREADS=)#/\1${THREADS}/g" \
                  -e "s/${MPI_CMD}#/${MPI_CMD}${j}/g" "./$FILE.sh" && \

        # ---------------------------------------------------------------------
        # STEP 2: SUBMIT JOB TO SLURM
        # ---------------------------------------------------------------------
        # Submit the configured job script with any remaining arguments
        sbatch "./$FILE.sh" $@; \

        # ---------------------------------------------------------------------
        # STEP 3: RESTORE TEMPLATE PLACEHOLDERS
        # ---------------------------------------------------------------------
        # After submission, revert all values back to '#' placeholders
        # This allows the template to be reused for subsequent job submissions
	      sed -i -E -e "s/(#SBATCH --job-name=\"${USER} )$JOB_NAME/\1#/g" \
                  -e "s/(#SBATCH --nodes=)${k}/\1#/g" \
                  -e "s/(#SBATCH --tasks-per-node=)${j}/\1#/g" \
                  -e "s/(#SBATCH --gres=gpu:)${j}/\1#/g" \
                  -e "s/(#SBATCH --cpus-per-task=)${i}/\1#/g" \
                  -e "s/(#SBATCH -e ${F}_%j_)${z}_${y}_${x}.err/\1#_#_#.err/g"\
                  -e "s/(#SBATCH -o ${F}_%j_)${z}_${y}_${x}.out/\1#_#_#.out/g"\
                  -e "s/(export NUMBA_NUM_THREADS=)${THREADS}/\1#/g" \
                  -e "s/(export OMP_NUM_THREADS=)${THREADS}/\1#/g" \
                  -e "s/${MPI_CMD}${j}/${MPI_CMD}#/g" "./$FILE.sh"
      fi
    done
  done
done
