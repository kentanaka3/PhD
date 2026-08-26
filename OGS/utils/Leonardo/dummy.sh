#!/bin/bash

#SBATCH --job-name="ktanakah #"
#SBATCH --nodes=#
#SBATCH --tasks-per-node=#
#SBATCH  -c 1
#SBATCH --gres=gpu:#
#SBATCH --account=IscrC_AISeism
#SBATCH --time 00:00:30
#SBATCH --mem=1024MB
#SBATCH --partition=boost_usr_prod
#SBATCH --output=k_%j_#.out
#SBATCH --error=k_%j_#.err

# Simple smoke-test payload for LAUNCHME.sh.
# LAUNCHME.sh dummy 1 1 launchme_dummy python AISeism/WORK/test/dummy.py

# Reseting the number of Environment variables for specific use case
source ./ACTIVATEME.sh

CONDA_BIN="/leonardo_work/IscrC_AISeism/.miniconda3/bin/conda"
date
printf 'Executing command: %s run -n SBC_3.12' "$CONDA_BIN"
printf ' %s' "$@"
printf '\n'
"$CONDA_BIN" run -n SBC_3.12 "$@"
date
