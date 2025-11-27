#!/bin/bash
# scripts/download_coco.sh
#
# Downloads and extracts MS-COCO 2014 dataset (Train + Val)
# Required for feature extraction.
# Total download size: ~19 GB
# Unzipped size: ~20 GB

# Target directory
DATA_DIR="$HOME/experiments/datasets/coco"

echo "=============================================="
echo "Downloading MS-COCO 2014 to $DATA_DIR"
echo "=============================================="

# Create directories
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 1. Download Train2014 Images (13GB)
if [ ! -d "train2014" ]; then
    echo "[1/3] Downloading train2014.zip..."
    wget -c http://images.cocodataset.org/zips/train2014.zip
    
    echo "Unzipping train2014..."
    unzip -q train2014.zip
    rm train2014.zip
    echo "✓ Train2014 ready."
else
    echo "✓ Train2014 already exists. Skipping."
fi

# 2. Download Val2014 Images (6GB)
# Note: Karpathy Test split is a subset of this Val set.
if [ ! -d "val2014" ]; then
    echo "[2/3] Downloading val2014.zip..."
    wget -c http://images.cocodataset.org/zips/val2014.zip
    
    echo "Unzipping val2014..."
    unzip -q val2014.zip
    rm val2014.zip
    echo "✓ Val2014 ready."
else
    echo "✓ Val2014 already exists. Skipping."
fi

# 3. Download Captions/Annotations (Optional but good to have)
if [ ! -d "annotations" ]; then
    echo "[3/3] Downloading annotations..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    
    echo "Unzipping annotations..."
    unzip -q annotations_trainval2014.zip
    rm annotations_trainval2014.zip
    echo "✓ Annotations ready."
fi

echo "=============================================="
echo "Download Complete!"
echo "Structure:"
ls -F "$DATA_DIR"
echo "=============================================="