#!/bin/bash
module load nvhpc/ cuda/
source /leonardo_work/IscrC_AISeism/.miniconda3/etc/profile.d/conda.sh
conda activate SBC_3.12
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1