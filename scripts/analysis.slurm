#!/bin/bash
#
# Usage: sbatch scripts/analysis.slurm [ABSOLUTE_PATH_TO_CHECKPOINT]
# Example: sbatch scripts/analysis.slurm /users/beyza.urhan/experiments/runs/20015_e2e_run/checkpoints/best_model.pth
#
#SBATCH --job-name=cmr-analysis
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j_analysis.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j_analysis_err.log
#SBATCH --time=14-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING
# ------------------------------------------------------------------
# 1. Codebase
# 2. Data (Images)
# 3. Runs (Outputs)
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/cocofeatures_karpathy/resnet50_imagenet:/workspace/datasets/coco/features_karpathy,/users/beyza.urhan/experiments/runs:/output

set -euo pipefail

# ------------------------------------------------------------------
# ARGUMENTS & PATH HANDLING
# ------------------------------------------------------------------
# Input: Absolute path from Host
HOST_CHECKPOINT_PATH="${1:-}"

if [ -z "$HOST_CHECKPOINT_PATH" ]; then
    echo "ERROR: Please provide the absolute path to the checkpoint file."
    echo "Usage: sbatch scripts/analysis.slurm /users/.../runs/JOBID/checkpoints/best_model.pth"
    exit 1
fi

# Path Translation: Host Path -> Container Path
# Replace the host runs directory with the container mount point (/output)
# Host: /users/beyza.urhan/experiments/runs/...
# Container: /output/...
HOST_RUNS_ROOT="/users/beyza.urhan/experiments/runs"
CONTAINER_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH/$HOST_RUNS_ROOT//output}"

echo "=========================================="
echo "Analysis Job Started ($SLURM_JOB_ID)"
echo "Host Checkpoint:      $HOST_CHECKPOINT_PATH"
echo "Container Checkpoint: $CONTAINER_CHECKPOINT_PATH"
echo "=========================================="

# ------------------------------------------------------------------
# DETERMINE OUTPUT DIRECTORY
# ------------------------------------------------------------------
# Checkpoint dosyasının bulunduğu klasörün iki üstü, o deneyin ana klasörüdür.
# Örn: /output/20015_run/checkpoints/best.pth -> /output/20015_run
JOB_DIR=$(dirname $(dirname "$CONTAINER_CHECKPOINT_PATH"))

echo "Target Run Directory: $JOB_DIR"

# Config dosyasını da aynı yerden alalım (O deneye ait config)
CONFIG_PATH="$JOB_DIR/config_saved.yaml"

# Eğer o deneyde config yedeklenmemişse, workspace'tekini kullan (Fallback)
if [ ! -f "$CONFIG_PATH" ]; then
    echo "WARNING: Saved config not found in run folder. Using current workspace config."
    CONFIG_PATH="/workspace/config.yaml"
fi

# ------------------------------------------------------------------
# START ANALYSIS
# ------------------------------------------------------------------
cd /workspace

python3 tools/export_hard_negatives.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CONTAINER_CHECKPOINT_PATH" \
    --output "$JOB_DIR/hard_negatives.json" \
    --top_k 50

echo ""
echo "=========================================="
echo "Analysis completed!"
echo "Results saved to: $JOB_DIR/hard_negatives.json"
echo "=========================================="