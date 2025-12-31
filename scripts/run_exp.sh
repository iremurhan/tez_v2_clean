#!/bin/bash
#
# Wrapper script to submit training jobs easily.
# Usage: ./run_exp.sh "experiment_name"
#

# 1. Check if an experiment name is provided
if [ -z "$1" ]; then
    echo "Error: No experiment name provided."
    echo "Usage: ./run_exp.sh <experiment_name>"
    exit 1
fi

EXP_NAME="$1"

# 2. Print confirmation
echo "------------------------------------------------"
echo "Submitting Job for Experiment: '${EXP_NAME}'"
echo "------------------------------------------------"

# 3. Execute sbatch command
# Assigns the name to both the Slurm Job Name (--job-name) 
# and passes it as an argument to the script ($1 inside .slurm)
sbatch --job-name="${EXP_NAME}" scripts/train.slurm "${EXP_NAME}"