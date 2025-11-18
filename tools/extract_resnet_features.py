#!/usr/bin/env python3
"""
Extract image features with ResNet using Karpathy splits for reproducible evaluation.

Key goals:
- Uses Karpathy splits (train=113K, val=5K, test=5K) for standard benchmarking
- Deterministic ordering by COCO ID to avoid mapping mismatches
- Resumable: skip existing shards and continue where you left off
- Compact storage: fp16 by default (≈ 4 KB per image for 2048-D)
- Clear manifest: Parquet/CSV mapping image_id <-> filepath <-> shard/row

Usage examples:
  # Extract Karpathy train split (113K images)
  python tools/extract_resnet_features.py \
    --images-root /workspace/datasets/coco \
    --split train \
    --karpathy-json-path datasets/coco/caption_datasets/dataset_coco.json \
    --out-dir /experiments/features/resnet50_scratch/train \
    --weights scratch

  # Extract Karpathy val split (5K images) with ImageNet weights
  python tools/extract_resnet_features.py \
    --images-root /workspace/datasets/coco \
    --split val \
    --karpathy-json-path datasets/coco/caption_datasets/dataset_coco.json \
    --out-dir /experiments/features/resnet50_imagenet/val \
    --weights imagenet

  # Extract test split for final evaluation
  python tools/extract_resnet_features.py \
    --images-root /workspace/datasets/coco \
    --split test \
    --karpathy-json-path datasets/coco/caption_datasets/dataset_coco.json \
    --out-dir /experiments/features/resnet50_scratch/test \
    --weights scratch
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.hpc_device_utils import create_feature_extraction_device

try:
    import pyarrow as pa  # optional; only used for Parquet if available
    import pyarrow.parquet as pq

    HAS_PA = True
except Exception:
    HAS_PA = False


# ---------------------------
# Dataset & utilities
# ---------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BAD_PATHS = []


def list_images_karpathy_split(images_root: Path, split: str, karpathy_json_path: Path) -> List[Path]:
    """
    Collect images for the given split using Karpathy split JSON.
    
    This function uses the standard Karpathy splits for COCO evaluation,
    which ensures reproducibility and comparability with published benchmarks.
    
    Args:
        images_root: Root directory containing COCO images (e.g., /path/to/coco)
        split: Split name ('train', 'val', 'test', or 'restval')
        karpathy_json_path: Path to the Karpathy JSON file (e.g., dataset_coco.json)
        
    Returns:
        List of Paths to images in the specified split, sorted by image ID for determinism
    """
    if not karpathy_json_path.exists():
        raise FileNotFoundError(f"Karpathy JSON not found: {karpathy_json_path}")
    
    # Load JSON
    with open(karpathy_json_path, 'r') as f:
        data = json.load(f)
    
    if 'images' not in data:
        raise ValueError(f"Invalid Karpathy JSON: missing 'images' key")
    
    # Filter images by split
    split_images = [img for img in data['images'] if img.get('split') == split]
    
    if len(split_images) == 0:
        raise ValueError(f"No images found for split '{split}' in {karpathy_json_path}")
    
    # Build full paths: images_root / filepath / filename
    # Example: /coco / train2014 / COCO_train2014_000000123456.jpg
    image_paths = []
    for img in split_images:
        # filepath is like "train2014" or "val2014"
        # filename is like "COCO_train2014_000000123456.jpg"
        filepath = img.get('filepath', '')
        filename = img.get('filename', '')
        
        if not filename:
            continue
            
        # Construct full path
        full_path = images_root / filepath / filename
        
        if full_path.exists():
            image_paths.append((full_path, img.get('cocoid', img.get('imgid', 0))))
        else:
            BAD_PATHS.append(str(full_path))
    
    # Sort by COCO ID for deterministic ordering
    image_paths.sort(key=lambda x: x[1])
    
    # Return just the paths
    paths = [p[0] for p in image_paths]
    
    print(f"[Karpathy Split] Loaded {len(paths)} images for split '{split}'")
    if BAD_PATHS:
        print(f"[Warning] {len(BAD_PATHS)} images not found on disk (see bad_images.txt)")
    
    return paths


def coco_id_from_name(name: str) -> int:
    """
    Parse a COCO numeric id from standard filename patterns:
      COCO_train2014_000000123456.jpg
      COCO_val2014_000000123456.jpg
    """
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 12:
        return int(digits[-12:])
    # Fallback: hash-like, but we strongly prefer real ids
    return abs(hash(name)) % 10**9


class ImageList(Dataset):
    def __init__(self, paths: List[Path], img_size: int = 224):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        # Read with PIL and convert to RGB
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            BAD_PATHS.append(str(path))
            return None, None, str(path)
        img = self.transform(img)
        img_id = coco_id_from_name(path.name)
        return img, img_id, str(path)


def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    imgs, ids, paths = zip(*batch)
    return torch.stack(imgs, 0), torch.as_tensor(ids), list(paths)


def build_backbone(kind: str = "resnet50", weights: str = "scratch") -> nn.Module:
    """
    Build a ResNet backbone and strip the classification head.
    weights:
      - "scratch": random init (to test mapping issues without leakage)
      - "imagenet": torchvision's IMAGENET1K_V2 weights
    Output: 2048-D pooled features (avgpool).
    """
    if kind != "resnet50":
        raise ValueError("Only resnet50 is supported in this minimal script.")

    if weights == "imagenet":
        w = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=w)
    elif weights == "scratch":
        net = models.resnet50(weights=None)
    else:
        raise ValueError("weights must be one of: scratch, imagenet")

    # Remove FC head, keep up to avgpool
    modules = list(net.children())[:-1]  # drop the final FC
    backbone = nn.Sequential(*modules)
    backbone.eval()
    return backbone


# ---------------------------
# Extraction
# ---------------------------


def save_manifest(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)
    if HAS_PA:
        pq_path = out_dir / "manifest.parquet"
        table = pa.Table.from_pandas(df)
        pq.write_table(table, pq_path)


def extract_features(
    images_root: Path,
    split: str,
    out_dir: Path,
    karpathy_json_path: Optional[Path] = None,
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 224,
    dtype: str = "fp16",
    weights: str = "scratch",
    device: Optional[str] = None,
    shard_size: int = 10000,
    amp: bool = True,
):
    """
    Run feature extraction and write sharded .pt files to out_dir:
      out_dir/
        features_shard000.pt
        features_shard001.pt
        ...
        manifest.csv[.parquet]
    Each shard is a dict with:
      {"image_ids": LongTensor [N], "features": Tensor [N, 2048], "paths": list[str]}
    """
    # Use HPC-aware device initialization with SLURM support and validation
    if device is None:
        device = create_feature_extraction_device()
    else:
        device = torch.device(device)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover & sort files deterministically using Karpathy splits
    if karpathy_json_path is None:
        raise ValueError(
            "karpathy_json_path is required. Please provide --karpathy-json-path argument.\n"
            "Example: --karpathy-json-path datasets/coco/caption_datasets/dataset_coco.json"
        )
    
    img_paths = list_images_karpathy_split(images_root, split, karpathy_json_path)
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found for split={split} using Karpathy splits from {karpathy_json_path}")

    # Build dataset/loader
    ds = ImageList(img_paths, img_size=img_size)
    pin = device.type == "cuda"
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=safe_collate,
    )

    # Backbone
    backbone = build_backbone("resnet50", weights=weights).to(device)
    if dtype == "fp16":
        cast = torch.float16
    elif dtype == "fp32":
        cast = torch.float32
    else:
        raise ValueError("dtype must be fp16 or fp32")

    # Use new torch.amp API to suppress FutureWarning
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda")) if device.type == "cuda" else None

    # Resumable: find existing shards and skip them
    existing = sorted(out_dir.glob("features_shard*.pt"))
    start_shard = len(existing)
    total = len(ds)
    n_shards = (total + shard_size - 1) // shard_size

    # Manifest rows - load existing manifest if resuming
    rows: List[Dict[str, object]] = []
    manifest_path = out_dir / "manifest.csv"
    if start_shard > 0 and manifest_path.exists():
        # Load existing manifest entries
        existing_df = pd.read_csv(manifest_path)
        rows = existing_df.to_dict('records')
        print(f"[RESUME] Loaded {len(rows)} existing manifest entries from {start_shard} shards")

    with torch.no_grad():
        idx_global = 0
        shard_idx = 0
        buf_feats: List[torch.Tensor] = []
        buf_ids: List[torch.Tensor] = []
        buf_paths: List[str] = []

        pbar = tqdm(total=total, desc=f"Extract {split} ({weights},{dtype})", ncols=100)
        for batch in dl:
            imgs, ids, paths = batch  # imgs[B,3,H,W], ids[B], paths[B]
            
            # Handle empty batches from safe_collate
            if len(imgs) == 0:
                continue

            # Deterministic skip - skip entire batches in already-written shards
            batch_end = idx_global + len(imgs)
            
            # If this entire batch is in an already-written shard, skip it completely
            if shard_idx < start_shard and batch_end <= start_shard * shard_size:
                idx_global += len(imgs)
                pbar.update(len(imgs))
                continue  # Skip this batch entirely, move to next
            
            # If we've reached the first unwritten shard, advance shard_idx
            if shard_idx < start_shard:
                # Advance to the first incomplete shard and clear buffers
                shard_idx = start_shard
                buf_feats, buf_ids, buf_paths = [], [], []

            # Normal processing - extract features
            imgs = imgs.to(device, non_blocking=True)
            if amp and device.type == "cuda":
                # Use new torch.amp API to suppress FutureWarning
                with torch.amp.autocast("cuda"):
                    feat = backbone(imgs).squeeze(-1).squeeze(-1)  # [B,2048,1,1] -> [B,2048]
            else:
                feat = backbone(imgs).squeeze(-1).squeeze(-1)

            feat = feat.to(cast).cpu()
            ids = ids.long().cpu()

            buf_feats.append(feat)
            buf_ids.append(ids)
            buf_paths.extend(paths)

            B = imgs.size(0)
            idx_global += B
            pbar.update(B)

            # Flush shards as we cross boundaries
            # Keep flushing while we have enough data for complete shards
            while len(buf_paths) > 0 and (idx_global >= (shard_idx + 1) * shard_size or idx_global == total):
                # Determine how many samples go into this shard
                shard_start = shard_idx * shard_size
                shard_end = min((shard_idx + 1) * shard_size, total)
                n_samples_this_shard = shard_end - shard_start
                
                # Take only the samples that belong to this shard
                n_to_take = min(len(buf_paths), n_samples_this_shard)
                
                if n_to_take > 0 and buf_feats:
                    # Concatenate and slice
                    all_feats = torch.cat(buf_feats, dim=0)
                    all_ids = torch.cat(buf_ids, dim=0)
                    all_paths = buf_paths
                    
                    shard_feats = all_feats[:n_to_take]
                    shard_ids = all_ids[:n_to_take]
                    shard_paths = all_paths[:n_to_take]
                    
                    # Keep remainder in buffer
                    if n_to_take < len(all_paths):
                        buf_feats = [all_feats[n_to_take:]]
                        buf_ids = [all_ids[n_to_take:]]
                        buf_paths = all_paths[n_to_take:]
                    else:
                        buf_feats, buf_ids, buf_paths = [], [], []
                else:
                    shard_feats = torch.empty(0, 2048, dtype=cast)
                    shard_ids = torch.empty(0, dtype=torch.long)
                    shard_paths = []
                    buf_feats, buf_ids, buf_paths = [], [], []

                shard_path = out_dir / f"features_shard{shard_idx:03d}.pt"
                
                # Save shard if it doesn't exist or we're past the resume point
                if not (shard_path.exists() and shard_idx < start_shard):
                    torch.save(
                        {"image_ids": shard_ids, "features": shard_feats, "paths": shard_paths},
                        shard_path,
                    )

                # Add to manifest
                for i, p in enumerate(shard_paths):
                    rows.append(
                        {"image_id": int(shard_ids[i]), "path": p, "shard": shard_idx, "row": i}
                    )

                # Move to next shard
                shard_idx += 1
                
                # If we've processed all data, exit
                if idx_global == total and len(buf_paths) == 0:
                    break

        pbar.close()

    # Save manifest
    df = pd.DataFrame(rows, columns=["image_id", "path", "shard", "row"])
    save_manifest(df, out_dir)

    # Basic sanity: unique ids == #rows?
    n_unique = df["image_id"].nunique()
    if n_unique != len(df):
        print(f"[WARN] Non-unique image_id detected: unique={n_unique} rows={len(df)}")

    # Save bad images list if any
    if BAD_PATHS:
        with open(out_dir / "bad_images.txt", "w") as f:
            f.write("\n".join(BAD_PATHS))
        print(f"[WARN] Skipped {len(BAD_PATHS)} bad images. See {out_dir/'bad_images.txt'}")
    
    # Final statistics (using actual manifest length for accuracy)
    print("\n" + "=" * 70)
    print(f"EXTRACTION COMPLETE: {split.upper()} SPLIT")
    print("=" * 70)
    print(f"  Features written: {len(df):,} images")
    print(f"  Unique image IDs: {n_unique:,}")
    print(f"  Shards created:   {shard_idx}")
    print(f"  Output directory: {out_dir}")
    print(f"  Manifest:         {out_dir / 'manifest.csv'}")
    if BAD_PATHS:
        print(f"  Skipped images:   {len(BAD_PATHS):,}")
    print("=" * 70)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract ResNet50 features using Karpathy splits for reproducible evaluation"
    )
    ap.add_argument(
        "--images-root", 
        type=Path, 
        required=True, 
        help="Root directory containing COCO images (e.g., /path/to/coco with train2014/ and val2014/ subdirs)"
    )
    ap.add_argument(
        "--split", 
        type=str, 
        required=True, 
        choices=["train", "val", "test", "restval"],
        help="Karpathy split name (train=113K, val=5K, test=5K, restval=30K)"
    )
    ap.add_argument(
        "--karpathy-json-path",
        type=Path,
        required=True,
        help="Path to Karpathy split JSON file (e.g., datasets/coco/caption_datasets/dataset_coco.json)"
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for features")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for extraction")
    ap.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers")
    ap.add_argument("--img-size", type=int, default=224, help="Image size (224 for standard ResNet)")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Feature dtype")
    ap.add_argument("--weights", type=str, default="scratch", choices=["scratch", "imagenet"], help="ResNet weights")
    ap.add_argument("--device", type=str, default=None, help="Device: cuda|cpu|mps (auto-detect if None)")
    ap.add_argument("--shard-size", type=int, default=10000, help="Number of images per shard file")
    ap.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    return ap.parse_args()


def main():
    args = parse_args()
    extract_features(
        images_root=args.images_root,
        split=args.split,
        out_dir=args.out_dir,
        karpathy_json_path=args.karpathy_json_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dtype=args.dtype,
        weights=args.weights,
        device=args.device,
        shard_size=args.shard_size,
        amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()


