#!/bin/bash
#
# Training Job Script running inside Docker
# Usage: sbatch scripts/train.slurm "EXPERIMENT_NAME"
# Example: sbatch scripts/train.slurm "baseline_v1"
#
#SBATCH --job-name=cmr-train
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j_%x/slurm.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j_%x/slurm_err.log
#SBATCH --time=14:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING (CRITICAL)
# ------------------------------------------------------------------
# We map server paths (Host) to the paths expected by config.yaml (Container).
#
# 1. Codebase: ~/tez_v2_clean -> /workspace
# 2. Features: ~/experiments/datasets/coco/features_karpathy/resnet50_scratch -> /workspace/datasets/coco/features_karpathy
# 3. Captions: ~/experiments/datasets/coco/caption_datasets -> /workspace/datasets/coco/caption_datasets
# 4. Outputs:  ~/experiments/runs -> /output
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco/features_karpathy/resnet50_scratch:/workspace/datasets/coco/features_karpathy,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output

# ------------------------------------------------------------------
# SCRIPT START
# ------------------------------------------------------------------
set -e  # Exit immediately if any command fails
set -u  # Exit if undefined variable is used
set -o pipefail  # Exit if any command in a pipeline fails

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
# Get experiment name from argument ($1), default to 'default_run' if not provided
EXP_NAME=${1:-}

# Create job directory (JobID_ExperimentName) if experiment name is provided
JOB_DIR="/output/${SLURM_JOB_ID}${EXP_NAME:+_${EXP_NAME}}"

echo "=========================================="
echo "Training Job Started"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Experiment Name: ${EXP_NAME}"
echo "Job Directory: ${JOB_DIR}"
echo "=========================================="
echo ""

# Create job directory and configure W&B
mkdir -p "$JOB_DIR"
export WANDB_MODE=offline
export WANDB_DIR="$JOB_DIR"

# ------------------------------------------------------------------
# PRE-FLIGHT CHECKS
# ------------------------------------------------------------------
echo "Running pre-flight checks..."

# Check if workspace exists
if [ ! -d /workspace ]; then
    echo "ERROR: /workspace directory not found"
    exit 1
fi

# Check if config exists
if [ ! -f /workspace/config.yaml ]; then
    echo "ERROR: /workspace/config.yaml not found"
    exit 1
fi

# Check if features directory exists
if [ ! -d /workspace/datasets/coco/features_karpathy ]; then
    echo "ERROR: Features directory not found at /workspace/datasets/coco/features_karpathy"
    exit 1
fi

# Check if captions JSON exists
if [ ! -f /workspace/datasets/coco/caption_datasets/dataset_coco.json ]; then
    echo "ERROR: Captions JSON not found at /workspace/datasets/coco/caption_datasets/dataset_coco.json"
    exit 1
fi

echo "✓ All pre-flight checks passed"
echo ""

# ------------------------------------------------------------------
# CONFIG & CHECKPOINT SETUP
# ------------------------------------------------------------------
# Backup config file
cp /workspace/config.yaml "$JOB_DIR/config_saved.yaml"
echo "✓ Config backed up to: $JOB_DIR/config_saved.yaml"

# Create symlink for checkpoints (safely)
if [ -L /workspace/checkpoints ]; then
    # It's a symlink, safe to remove
    rm /workspace/checkpoints
    echo "✓ Removed existing checkpoint symlink"
elif [ -d /workspace/checkpoints ]; then
    # It's a real directory
    echo "WARNING: /workspace/checkpoints is a real directory, not a symlink"
    echo "Moving it to ${JOB_DIR}/old_checkpoints for safety"
    mv /workspace/checkpoints "${JOB_DIR}/old_checkpoints"
elif [ -e /workspace/checkpoints ]; then
    # It exists but is neither directory nor symlink (file?)
    echo "WARNING: /workspace/checkpoints exists but is not a directory or symlink"
    echo "Removing it"
    rm -f /workspace/checkpoints
fi

ln -s "$JOB_DIR" /workspace/checkpoints
echo "✓ Checkpoint directory linked to: $JOB_DIR"
echo ""

# ------------------------------------------------------------------
# START TRAINING
# ------------------------------------------------------------------
# Note: debug.debug_mode=true for initial testing
#       Change to false once you're confident everything works
echo "Starting training..."
echo ""

# Change to workspace directory
cd /workspace

# Start training
python3 -u train.py \
    --config config.yaml \
    --override debug.debug_mode=true

TRAIN_EXIT_CODE=$?

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