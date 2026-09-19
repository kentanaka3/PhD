#!/bin/bash

#SBATCH --job-name="ktanakah #"
#SBATCH  -N 1
#SBATCH  -n 1
#SBATCH --cpus-per-task=#
#SBATCH --account=OGS23_PRACE_IT_1
# #SBATCH --account=IscrC_AI2Seism
#SBATCH --time 00:30:00
#SBATCH --mem=490000MB
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=dcgp_qos_dbg
#SBATCH --error=parse_%j.err
#SBATCH --output=parse_%j.out

set -euo pipefail

# Reseting the number of Environment variables for specific use case
source ./ACTIVATEME.sh

date
cmd=("$@")
printf 'Executing command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
date
