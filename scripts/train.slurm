#!/bin/bash
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
# 1. Codebase:
#    Host: /users/beyza.urhan/tez_v2_clean
#    Container: /workspace
#
# 2. Features:
#    Host: /users/beyza.urhan/experiments/features_karpathy/resnet50_scratch  <-- Point directly to the folder containing train/val/test
#    Container: /workspace/datasets/coco/features
#
# 3. Captions (JSON):
#    Host: /users/beyza.urhan/experiments/datasets/coco/caption_datasets
#    Container: /workspace/datasets/coco/caption_datasets
#
# 4. Outputs:
#    Host: /users/beyza.urhan/experiments/runs
#    Container: /output
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/features_karpathy/resnet50_scratch:/workspace/datasets/coco/features,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/runs:/output

export WANDB_MODE=offline
export WANDB_DIR=/output/${SLURM_JOB_ID}

JOB_DIR="/output/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"

echo "Job Started: $SLURM_JOB_ID"
echo "Experiment Name: $SLURM_JOB_NAME"
echo "Saving results to: $JOB_DIR"

# Çıktı klasörünü oluştur
mkdir -p /output/${SLURM_JOB_ID}

cp /workspace/configs/default.yaml "$JOB_DIR/config_saved.yaml"

rm -rf /workspace/checkpoints
ln -s "$JOB_DIR" /workspace/checkpoints

# Eğitimi Başlat
python3 train.py \
    --config configs/default.yaml \
    --resume ""