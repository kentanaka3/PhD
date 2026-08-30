#!/bin/bash
module load nvhpc/ cuda/  # Ensure the correct CUDA module is loaded
CONDA_ROOT="${CONDA_ROOT:-/leonardo_work/IscrC_AISeism/.miniconda3}"
CONDA_ENV="${CONDA_ENV:-SBC_3.12}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1
