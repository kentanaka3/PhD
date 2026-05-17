#!/bin/bash
# Manual run: python OGS/src/ogsdownloader.py -D 220201 220215 -d /leonardo/home/userexternal/ktanakah/AI2Seism/WORK/waveform

#SBATCH --job-name="ktanakah #"
#SBATCH --nodes=#
#SBATCH --tasks-per-node=#
#SBATCH  -c 1
#SBATCH  -A IscrC_AI2Seism
#SBATCH --time 03:59:00
#SBATCH  -p lrd_all_serial
#SBATCH  -e download_%j.err
#SBATCH  -o download_%j.out

# srun -A OGS23_PRACE_IT_0 -p lrd_all_serial --nodes=1 --tasks-per-node=1 --pty /usr/bin/rsync -avh /leonardo_store/DRES_EchoArch/amagrin0/Ken/2024 /leonardo_scratch/large/userexternal/ktanakah/

# Module activations and Environment variables initialization
date
cmd="/leonardo_work/IscrC_AISeism/.miniconda3/bin/conda run -n SBC_3.12 $@"
echo "Executing command: $cmd"
eval $cmd
date