#!/bin/bash

#SBATCH --job-name="ktanakah #"
#SBATCH --nodes=#
#SBATCH --tasks-per-node=#
#SBATCH  -c 1
#SBATCH --account=IscrC_AI2Seism
#SBATCH --time 03:59:00
#SBATCH --partition=lrd_all_serial
#SBATCH --error=download_%j.err
#SBATCH --output=download_%j.out

# srun -A OGS23_PRACE_IT_0 -p lrd_all_serial --nodes=1 --tasks-per-node=1 --pty python OGS/src/ogsdownloader.py -D 220201 220215 -d /leonardo/home/userexternal/ktanakah/AI2Seism/WORK/waveform

# Reseting the number of Environment variables for specific use case
source ./ACTIVATEME.sh

CONDA_BIN="/leonardo_work/IscrC_AISeism/.miniconda3/bin/conda"
date
cmd="$CONDA_BIN run -n SBC_3.12 $@"
echo "Executing command: $cmd"
eval $cmd
date