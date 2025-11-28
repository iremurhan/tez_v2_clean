#!/bin/bash
#
# Usage: sbatch scripts/train.slurm "EXPERIMENT_NAME"
#
#SBATCH --job-name=cmr-train
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j/slurm.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j/err.log
#SBATCH --time=14:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco/features_karpathy/resnet50_imagenet:/workspace/datasets/coco/features_karpathy,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env

# Pass W&B Env vars if they exist in the shell context
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
EXP_NAME="${1:-}"
JOB_DIR="/output/${SLURM_JOB_ID}${EXP_NAME:+_${EXP_NAME}}"

# Subdirectories
LOGS_DIR="$JOB_DIR/logs"
META_DIR="$JOB_DIR/metadata"
CKPT_DIR="$JOB_DIR/checkpoints"

mkdir -p "$JOB_DIR" "$LOGS_DIR" "$META_DIR" "$CKPT_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Experiment:    ${EXP_NAME:-}"
echo "Job Directory: ${JOB_DIR}"
echo "=========================================="

# ------------------------------------------------------------------
# W&B SETUP (ONLINE/OFFLINE AUTO-DETECT)
# ------------------------------------------------------------------
# Always save logs to our specific folder
export WANDB_DIR="$LOGS_DIR"

# Load W&B Env file if mounted
if [ -f /env/wandb.env ]; then
    echo "✓ Loading W&B credentials from /env/wandb.env"
    # Source the file and export variables
    set -a  # Automatically export all variables
    source /env/wandb.env
    set +a
fi

# Check if API key is available
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "✓ W&B API Key found."
    export WANDB_MODE=online
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

# ------------------------------------------------------------------
# METADATA & CONFIG
# ------------------------------------------------------------------
echo "Job ID: ${SLURM_JOB_ID}" > "$META_DIR/experiment_metadata.txt"
echo "Experiment: ${EXP_NAME:-(Unnamed)}" >> "$META_DIR/experiment_metadata.txt"
echo "Date: $(date)" >> "$META_DIR/experiment_metadata.txt"

if [ -f /workspace/config.yaml ]; then
    cp /workspace/config.yaml "$META_DIR/config_saved.yaml"
    echo "✓ Config backed up."
fi

# ------------------------------------------------------------------
# CHECKPOINT SETUP
# ------------------------------------------------------------------
rm -rf /workspace/checkpoints
ln -s "$CKPT_DIR" /workspace/checkpoints
echo "✓ Checkpoints linked to: $CKPT_DIR"

# ------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------
echo "Starting training..."
echo ""

cd /workspace

# PRODUCTION RUN (Debug Mode FALSE)
python3 -u train.py \
    --config config.yaml \
    --override debug.debug_mode=false \
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