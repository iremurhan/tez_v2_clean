#!/bin/bash
#
# Usage: sbatch scripts/mining.slurm
#
#SBATCH --job-name=mining_targets
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j_mining.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j_mining_err.log
#SBATCH --time=07:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING (Mevcut train.slurm ile aynı mountlar)
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env

# W&B Key aktarımı (Login olmak için)
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

# ------------------------------------------------------------------
# JOB DIRECTORIES (train.slurm ile tutarlı)
# ------------------------------------------------------------------
JOB_DIR="/output/${SLURM_JOB_ID}"
LOGS_DIR="$JOB_DIR/logs"

mkdir -p "$JOB_DIR" "$LOGS_DIR"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Task:   Offline Hard-Negative Mining"
echo "Date:   $(date)"
echo "Job Dir: ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP
# ------------------------------------------------------------------
export WANDB_DIR="$LOGS_DIR"

if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
    echo "W&B API Key not found. Running offline."
else
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_mining"
fi

# ------------------------------------------------------------------
# RUN MINING
# ------------------------------------------------------------------
cd /workspace

echo "Starting mining process..."

python3 -u tools/mining_targets.py \
    --config config.yaml \
    2>&1 | tee "$LOGS_DIR/mining_output.log"

MINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $MINING_EXIT_CODE -eq 0 ]; then
    echo "Mining completed successfully!"
else
    echo "Mining failed with exit code: $MINING_EXIT_CODE"
fi
echo "=========================================="

exit $MINING_EXIT_CODE