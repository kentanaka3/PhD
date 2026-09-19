#!/bin/bash
module load nvhpc/ cuda/ intel-oneapi-compilers/2024.1.0
CONDA_ROOT="${CONDA_ROOT:-/leonardo_work/IscrC_AISeism/.miniconda3}"
CONDA_ENV="${CONDA_ENV:-SBC_3.12}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HYDRA_FULL_ERROR=1
