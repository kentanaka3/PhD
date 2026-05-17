#!/bin/sh

#SBATCH --job-name="ktanakah #"
#SBATCH --nodes=#
#SBATCH --tasks-per-node=#
#SBATCH --gres=gpu:#
#SBATCH --cpus-per-task=#
# #SBATCH  -A IscrC_AISeism
# #SBATCH  -A ICT24_MHPC
#SBATCH  -A OGS23_PRACE_IT_0
# #SBATCH --time 23:59:00
#SBATCH --time 00:30:00
#SBATCH --mem=490000MB
#SBATCH  -p boost_usr_prod
# #SBATCH --qos boost_qos_dbg
#SBATCH -e k_%j_#_#_#.err
#SBATCH -o k_%j_#_#_#.out

# srun -A OGS23_PRACE_IT_0 --mem=490000MB -p boost_usr_prod --qos boost_qos_dbg --nodes=1 --tasks-per-node=1 --gres=gpu:1 --cpus-per-task=32 --pty /bin/bash
# ml_catalog_run +libpath=. output_path=OGS22 data.starttime=2022-01-01 data.endtime=2022-01-02

# Reseting the number of Environment variables for specific use case
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1
module load nvhpc/ cuda/

date
cmd="/leonardo_work/IscrC_AISeism/.miniconda3/bin/conda run -n SBC_3.12 $@"
echo "Executing command: $cmd"
eval $cmd
date

