#!/bin/bash
#
# Usage: sbatch --job-name="MY_EXP_NAME" scripts/train.slurm "MY_EXP_NAME"
#
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j/slurm.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j/err.log
#SBATCH --time=14-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env

# Pass W&B Env vars
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------

# 1. Capture Experiment Name from Argument
EXP_NAME="${1:-}"

# 2. Define Job Directory using ONLY Job ID to avoid duplicates
# Slurm automatically creates 'runs/%j' based on SBATCH --output header.
# We map this inside the container to '/output/%j'.
JOB_DIR="/output/${SLURM_JOB_ID}"

# Subdirectories
LOGS_DIR="$JOB_DIR/logs"
META_DIR="$JOB_DIR/metadata"
CKPT_DIR="$JOB_DIR/checkpoints"

# Create directories (mkdir -p is safe even if Slurm created the root)
mkdir -p "$JOB_DIR" "$LOGS_DIR" "$META_DIR" "$CKPT_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Experiment:    ${EXP_NAME:- (Unnamed)}"
echo "Job Directory: ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP
# ------------------------------------------------------------------
export WANDB_DIR="$LOGS_DIR"

# Load W&B Env file if mounted
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

# Configure W&B mode based on API key
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    # Use the Experiment Name in W&B if provided, otherwise Job ID
    export WANDB_NAME="${EXP_NAME:-${SLURM_JOB_ID}}"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

# ------------------------------------------------------------------
# METADATA SAVING
# ------------------------------------------------------------------
echo "Job ID: ${SLURM_JOB_ID}" > "$META_DIR/experiment_metadata.txt"
echo "Experiment: ${EXP_NAME:-}" >> "$META_DIR/experiment_metadata.txt"
echo "Date: $(date)" >> "$META_DIR/experiment_metadata.txt"

# Save a copy of the config for reproducibility
if [ -f /workspace/config.yaml ]; then
    cp /workspace/config.yaml "$META_DIR/config_saved.yaml"
fi

# ------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------
echo "Starting training..."
cd /workspace

# Run training
python3 -u train.py \
    --config config.yaml \
    --override \
        debug.debug_mode=false \
        logging.checkpoint_dir="$CKPT_DIR" \
    2>&1 | tee "$LOGS_DIR/training_output.log"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Results saved to: $JOB_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
