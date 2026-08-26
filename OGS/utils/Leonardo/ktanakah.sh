#!/bin/sh

#SBATCH --job-name="ktanakah #"
#SBATCH --nodes=#
#SBATCH --tasks-per-node=#
#SBATCH --gres=gpu:#
#SBATCH --cpus-per-task=#
# #SBATCH --account=IscrC_AISeism
# #SBATCH --account=ICT24_MHPC
#SBATCH --account=OGS23_PRACE_IT_0
# #SBATCH --time 23:59:00
#SBATCH --time 00:30:00
#SBATCH --mem=490000MB
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --error=k_%j_#_#_#.err
#SBATCH --output=k_%j_#_#_#.out

# srun -A OGS23_PRACE_IT_0 --mem=490000MB -p boost_usr_prod --qos boost_qos_dbg --nodes=1 --tasks-per-node=1 --gres=gpu:1 --cpus-per-task=32 --pty /bin/bash
# ml_catalog_run +libpath=. output_path=OGS22 data.starttime=2022-01-01 data.endtime=2022-01-02

# Reseting the number of Environment variables for specific use case
source ./ACTIVATEME.sh

CONDA_BIN="/leonardo_work/IscrC_AISeism/.miniconda3/bin/conda"
date
cmd="$CONDA_BIN run -n SBC_3.12 $@"
echo "Executing command: $cmd"
eval $cmd
date

