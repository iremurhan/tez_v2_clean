#!/bin/bash
#
# Wrapper script to submit mining job.
# Usage: ./scripts/run_mining.sh

echo "----------------------------------------------"
echo "Submitting Offline Mining Job..."
echo "----------------------------------------------"

JOB_ID=$(sbatch scripts/mining.slurm | awk '{print $4}')

echo "Job submitted! ID: $JOB_ID"
echo "You can verify logs later at: /users/beyza.urhan/experiments/runs/${JOB_ID}_mining.log"
echo "----------------------------------------------"