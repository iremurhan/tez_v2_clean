#!/usr/bin/env python3
"""
Extract image features with ResNet using Karpathy splits.
Optimized for the tez_v2_clean project structure.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PA = True
except ImportError:
    HAS_PA = False

# ---------------------------
# Dataset & Utilities
# ---------------------------

def list_images_karpathy_split(images_root: Path, split: str, karpathy_json_path: Path) -> List[Path]:
    """Collect images for the given split using Karpathy split JSON."""
    if not karpathy_json_path.exists():
        raise FileNotFoundError(f"Karpathy JSON not found: {karpathy_json_path}")
    
    with open(karpathy_json_path, 'r') as f:
        data = json.load(f)
    
    # Filter images by split
    # Note: 'restval' images are technically in the train set for some definitions, 
    # but Karpathy defines specific splits.
    split_images = [img for img in data['images'] if img.get('split') == split]
    
    if not split_images:
        raise ValueError(f"No images found for split '{split}' in {karpathy_json_path}")
    
    image_paths = []
    missing_count = 0
    
    for img in split_images:
        filepath = img.get('filepath', '')
        filename = img.get('filename', '')
        if not filename: continue
            
        full_path = images_root / filepath / filename
        
        if full_path.exists():
            # Store tuple (path, cocoid) for sorting
            image_paths.append((full_path, img.get('cocoid', img.get('imgid', 0))))
        else:
            missing_count += 1
    
    # Sort by COCO ID for deterministic ordering
    image_paths.sort(key=lambda x: x[1])
    
    # Return just the paths
    paths = [p[0] for p in image_paths]
    
    print(f"[Karpathy Split] Loaded {len(paths)} images for split '{split}'")
    if missing_count > 0:
        print(f"[Warning] {missing_count} images defined in JSON were not found on disk.")
    
    return paths

def coco_id_from_path(path: Path) -> int:
    """Parse COCO numeric id from standard filename patterns."""
    name = path.name
    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 12:
        return int(digits[-12:])
    return abs(hash(name)) % 10**9

class ImageList(Dataset):
    def __init__(self, paths: List[Path], img_size: int = 224):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            img_id = coco_id_from_path(path)
            return img, img_id, str(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None, str(path)

def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.empty(0), torch.empty(0), []
    imgs, ids, paths = zip(*batch)
    return torch.stack(imgs, 0), torch.as_tensor(ids), list(paths)

def build_backbone(weights: str = "scratch") -> nn.Module:
    """Build ResNet50 backbone (pooled features)."""
    if weights == "imagenet":
        w = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=w)
    else:
        net = models.resnet50(weights=None) # Scratch

    # Remove FC head, keep up to avgpool (output: 2048)
    modules = list(net.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.eval()
    return backbone

def save_manifest(rows: List[Dict], out_dir: Path):
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "manifest.csv", index=False)
    if HAS_PA:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_dir / "manifest.parquet")

def extract_features(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get Image List
    img_paths = list_images_karpathy_split(args.images_root, args.split, args.karpathy_json_path)
    if not img_paths:
        raise RuntimeError(f"No images found for split {args.split}")

    # 2. Setup Loader
    ds = ImageList(img_paths, img_size=args.img_size)
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate
    )

    # 3. Setup Model
    model = build_backbone(weights=args.weights).to(device)
    
    # 4. Extraction Loop
    rows = []
    shard_idx = 0
    buffer_feats, buffer_ids, buffer_paths = [], [], []
    
    # Resume logic could go here, but for clean regeneration, we start fresh usually.
    # Simple shard counter
    current_shard_size = 0

    print(f"Starting extraction for {args.split}...")
    
    # FP16 casting
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    cast_dtype = dtype_map[args.dtype]

    with torch.no_grad():
        for imgs, ids, paths in tqdm(dl, desc="Extracting"):
            if len(imgs) == 0: continue
            
            imgs = imgs.to(device)
            
            # Forward pass (with AMP if needed)
            if device.type == "cuda" and args.dtype == "fp16":
                with torch.amp.autocast('cuda'):
                    feats = model(imgs).squeeze(-1).squeeze(-1)
            else:
                feats = model(imgs).squeeze(-1).squeeze(-1)
            
            # Cast and CPU
            feats = feats.to(cast_dtype).cpu()
            
            buffer_feats.append(feats)
            buffer_ids.append(ids)
            buffer_paths.extend(paths)
            current_shard_size += len(imgs)
            
            # Write Shard if buffer is full
            if current_shard_size >= args.shard_size:
                # Concatenate buffer
                all_feats = torch.cat(buffer_feats, dim=0)
                all_ids = torch.cat(buffer_ids, dim=0)
                
                # We might have slightly more than shard_size, that's fine for simple logic,
                # or we can slice exactly. Let's write everything in buffer to keep it simple and fast.
                
                save_path = args.out_dir / f"features_shard{shard_idx:03d}.pt"
                torch.save({
                    "image_ids": all_ids,
                    "features": all_feats,
                    "paths": buffer_paths
                }, save_path)
                
                # Update manifest rows
                for i, p in enumerate(buffer_paths):
                    rows.append({
                        "image_id": int(all_ids[i]),
                        "path": p,
                        "shard": shard_idx,
                        "row": i
                    })
                
                # Reset
                shard_idx += 1
                buffer_feats, buffer_ids, buffer_paths = [], [], []
                current_shard_size = 0

        # Flush remaining
        if buffer_feats:
            all_feats = torch.cat(buffer_feats, dim=0)
            all_ids = torch.cat(buffer_ids, dim=0)
            save_path = args.out_dir / f"features_shard{shard_idx:03d}.pt"
            torch.save({
                "image_ids": all_ids,
                "features": all_feats,
                "paths": buffer_paths
            }, save_path)
            
            for i, p in enumerate(buffer_paths):
                rows.append({
                    "image_id": int(all_ids[i]),
                    "path": p,
                    "shard": shard_idx,
                    "row": i
                })

    save_manifest(rows, args.out_dir)
    print("Extraction Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--karpathy-json-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--weights", type=str, default="scratch", choices=["scratch", "imagenet"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=10000)
    args = parser.parse_args()
    
    extract_features(args)