#!/bin/bash
#
# Extraction Job Script running inside Docker
# Usage: sbatch scripts/extract_features.slurm
#
#SBATCH --job-name=extract_feats
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j_extract.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j_extract_err.log
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# ------------------------------------------------------------------
# PATH MAPPING (MOUNTS)
# ------------------------------------------------------------------
# 1. Codebase: 
#    Host: ~/tez_v2_clean -> Container: /workspace
#
# 2. Raw Images & JSON Root: 
#    Host: ~/experiments/datasets/coco -> Container: /data
#    (Container will have /data/train2014 and /data/caption_datasets/dataset_coco.json)
#
# 3. Output Destination:
#    Host: ~/experiments/datasets/coco/features_karpathy -> Container: /output

#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/data,/users/beyza.urhan/experiments/datasets/coco/features_karpathy:/output

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
WEIGHTS="scratch"  # Options: 'scratch' (untrained) or 'imagenet' (pretrained)
BS=256
NW=8
JSON_PATH="/data/caption_datasets/dataset_coco.json"  # Container path

echo "Starting Feature Extraction inside Docker..."
echo "Images Root: /data"
echo "Output Root: /output"

# ------------------------------------------------------------------
# 1. TRAIN SPLIT (113k Images)
# ------------------------------------------------------------------
echo "--- Extracting TRAIN ---"
python3 tools/extract_resnet_features.py \
    --images-root /data \
    --split train \
    --karpathy-json-path "$JSON_PATH" \
    --out-dir "/output/resnet50_${WEIGHTS}/train" \
    --weights "$WEIGHTS" \
    --batch-size $BS \
    --num-workers $NW

# ------------------------------------------------------------------
# 2. VAL SPLIT (5k Images)
# ------------------------------------------------------------------
echo "--- Extracting VAL ---"
python3 tools/extract_resnet_features.py \
    --images-root /data \
    --split val \
    --karpathy-json-path "$JSON_PATH" \
    --out-dir "/output/resnet50_${WEIGHTS}/val" \
    --weights "$WEIGHTS" \
    --batch-size $BS \
    --num-workers $NW

# ------------------------------------------------------------------
# 3. TEST SPLIT (5k Images)
# ------------------------------------------------------------------
echo "--- Extracting TEST ---"
python3 tools/extract_resnet_features.py \
    --images-root /data \
    --split test \
    --karpathy-json-path "$JSON_PATH" \
    --out-dir "/output/resnet50_${WEIGHTS}/test" \
    --weights "$WEIGHTS" \
    --batch-size $BS \
    --num-workers $NW

echo "All extractions complete!"