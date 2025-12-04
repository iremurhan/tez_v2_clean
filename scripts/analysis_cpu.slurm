#!/bin/bash
#
# Usage: sbatch scripts/analysis_cpu.slurm [ABSOLUTE_PATH_TO_CHECKPOINT]
#
# CPU-only version for when GPUs are busy.
# Runs hard negative analysis without GPU resources.
#
#SBATCH --job-name=cmr-analysis-cpu
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j_analysis_cpu.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j_analysis_cpu_err.log
#SBATCH --time=14-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/runs:/output

set -euo pipefail

# ------------------------------------------------------------------
# 1. PATH TRANSLATION (HOST -> CONTAINER)
# ------------------------------------------------------------------
HOST_CHECKPOINT_PATH="${1:-}"

if [ -z "$HOST_CHECKPOINT_PATH" ]; then
    echo "ERROR: Please provide the absolute path to the checkpoint file."
    echo "Usage: sbatch scripts/analysis_cpu.slurm /users/beyza.urhan/experiments/runs/.../best_model.pth"
    exit 1
fi

# Path roots
HOST_RUNS_ROOT="/users/beyza.urhan/experiments/runs"
CONTAINER_RUNS_ROOT="/output"

# Bash string replacement: Host path -> Container path
if [[ "$HOST_CHECKPOINT_PATH" == "$HOST_RUNS_ROOT"* ]]; then
    CONTAINER_CHECKPOINT_PATH="${HOST_CHECKPOINT_PATH/$HOST_RUNS_ROOT/$CONTAINER_RUNS_ROOT}"
else
    echo "ERROR: The checkpoint path must be inside '$HOST_RUNS_ROOT'"
    echo "Your input: $HOST_CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "Job ID:              $SLURM_JOB_ID"
echo "Host Path:           $HOST_CHECKPOINT_PATH"
echo "Container Path:      $CONTAINER_CHECKPOINT_PATH"
echo "Mode:                CPU ONLY"
echo "=========================================="

# ------------------------------------------------------------------
# 2. SETUP & VERIFICATION
# ------------------------------------------------------------------
# Determine Run Directory from checkpoint path
JOB_DIR=$(dirname $(dirname "$CONTAINER_CHECKPOINT_PATH"))
echo "Target Run Directory: $JOB_DIR"

# Determine Config Path (priority: saved config > workspace default)
if [ -f "$JOB_DIR/metadata/config_saved.yaml" ]; then
    CONFIG_PATH="$JOB_DIR/metadata/config_saved.yaml"
    echo "Using saved config: $CONFIG_PATH"
elif [ -f "$JOB_DIR/config_saved.yaml" ]; then
    CONFIG_PATH="$JOB_DIR/config_saved.yaml"
    echo "Using saved config: $CONFIG_PATH"
else
    CONFIG_PATH="/workspace/config.yaml"
    echo "Warning: Saved config not found, using default: $CONFIG_PATH"
fi

# Verify checkpoint exists
if [ ! -f "$CONTAINER_CHECKPOINT_PATH" ]; then
    echo "CRITICAL ERROR: Checkpoint file not found!"
    echo "Looking for: $CONTAINER_CHECKPOINT_PATH"
    echo "Check path mapping."
    exit 1
fi

# ------------------------------------------------------------------
# 3. RUN ANALYSIS (CPU MODE)
# ------------------------------------------------------------------
cd /workspace

echo "Starting analysis (CPU Mode)..."

# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""

python3 tools/export_hard_negatives.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CONTAINER_CHECKPOINT_PATH" \
    --output "$JOB_DIR/hard_negatives.json" \
    --top_k 50

echo ""
echo "=========================================="
echo "Analysis completed successfully!"
echo "Results saved to: $JOB_DIR/hard_negatives.json"
echo "=========================================="

