#!/usr/bin/env python3
"""
data.py
-------

Data loading and preprocessing for cross-modal retrieval.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configure logger
logger = logging.getLogger(__name__)

class CocoFeatureDataset(Dataset):
    def __init__(self, features_dir, captions_path, tokenizer, max_length=50, split='train', noise_std=0.0):
        """
        Args:
            features_dir (str): Directory containing split folders (train/val/test) with sharded .pt files.
            captions_path (str): Path to the Karpathy JSON file.
            tokenizer: HuggingFace tokenizer instance.
            max_length (int): Maximum token sequence length.
            split (str): 'train', 'val', or 'test'.
            noise_std (float): Standard deviation for Gaussian noise (feature jitter). 
                               Used for intra-modal positive view generation.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.noise_std = noise_std
        
        # ---------------------------------------------------------
        # 1. Load Captions (Karpathy JSON)
        # ---------------------------------------------------------
        logger.info(f"Loading captions from {captions_path} for split: {split}")
        with open(captions_path, 'r') as f:
            data = json.load(f)
        
        # Filter images belonging to the requested split
        self.samples = []
        valid_image_ids = set()
        
        # Karpathy JSON structure: images list contains items with 'split' and 'sentences'
        for img in data['images']:
            # Karpathy uses 'val' and 'test', but sometimes 'restval' is used for training
            # Adjust logic if you want to include 'restval' in training
            current_split = img['split']
            if current_split == 'restval' and split == 'train':
                current_split = 'train'

            if current_split == split:
                # Get COCO ID (prefer 'cocoid', fallback to 'imgid')
                img_id = int(img.get('cocoid', img.get('imgid')))
                valid_image_ids.add(img_id)
                
                # Add all 5 captions for this image
                for sent in img['sentences']:
                    self.samples.append({
                        'image_id': img_id,
                        'caption': sent['raw'],
                        'image_path': img.get('filepath', '') + '/' + img.get('filename', '')
                    })
                    
        logger.info(f"Found {len(self.samples)} captions for {len(valid_image_ids)} unique images in split '{split}'.")

        # ---------------------------------------------------------
        # 2. Load Features (In-Memory Caching)
        # ---------------------------------------------------------
        # Construct path to the specific split folder (e.g., datasets/coco/features/train)
        split_dir = Path(features_dir) / split
        manifest_path = split_dir / "manifest.csv"
        
        if not manifest_path.exists():
            # Fallback: maybe features_dir is already the split dir?
            if (Path(features_dir) / "manifest.csv").exists():
                split_dir = Path(features_dir)
                manifest_path = split_dir / "manifest.csv"
            else:
                raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        logger.info(f"Loading feature manifest from {manifest_path}")
        
        # Use dictionary for O(1) access: image_id -> feature_tensor
        self.image_features = {} 
        
        # Find all shard files
        shard_files = sorted(list(split_dir.glob("features_shard*.pt")))
        
        if not shard_files:
             raise FileNotFoundError(f"No .pt shard files found in {split_dir}")

        logger.info(f"Found {len(shard_files)} shards. Loading into memory...")
        
        loaded_count = 0
        for shard_file in tqdm(shard_files, desc=f"Loading {split} features"):
            try:
                # Load shard: {'image_ids': [...], 'features': [...]}
                shard_data = torch.load(shard_file, map_location='cpu')
                
                ids = shard_data['image_ids']
                feats = shard_data['features']
                
                # Store features only if the image is in our caption set
                # This filters out images from other splits if mixed, or enables partial loading for debug
                for i, img_id in enumerate(ids):
                    img_id = int(img_id)
                    if img_id in valid_image_ids:
                        # Clone to avoid keeping the whole shard graph in memory and ensure float32
                        self.image_features[img_id] = feats[i].float().clone()
                        loaded_count += 1
                        
            except Exception as e:
                logger.error(f"Error loading shard {shard_file}: {e}")
                
        logger.info(f"Successfully loaded {loaded_count} unique image features.")
        
        # ---------------------------------------------------------
        # 3. Consistency Check
        # ---------------------------------------------------------
        # It is possible that we have captions for images we haven't loaded features for
        # (e.g., during local debug with only shard000). We must prune those captions.
        available_ids = set(self.image_features.keys())
        initial_len = len(self.samples)
        
        self.samples = [s for s in self.samples if s['image_id'] in available_ids]
        
        if len(self.samples) < initial_len:
            logger.warning(f"Pruned dataset from {initial_len} to {len(self.samples)} based on available features.")
            logger.warning("This is expected during local debug with partial data.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']
        
        # 1. Retrieve Image Feature
        # Shape: [2048]
        image_feat = self.image_features[image_id] 
        
        # 2. Feature Jitter (Augmentation for Intra-Modal Learning)
        # Only apply noise if noise_std is set (usually > 0 during training)
        if self.noise_std > 0:
            noise = torch.randn_like(image_feat) * self.noise_std
            image_feat = image_feat + noise
            
        # 3. Tokenize Text
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image_feat,       
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'index': idx,
            'image_id': image_id
        }

def get_dataloader(config, tokenizer, split='train'):
    """
    Factory function to create dataloaders.
    """
    # Determine noise level based on config and split
    # Only apply noise during training and if feature jitter is enabled
    noise = 0.0
    shuffle = False
    
    if split == 'train':
        shuffle = True
        # Check if augmentation is enabled in config structure
        # Fallback to 0.0 if keys are missing (safety)
        try:
            if config.get('augment', {}).get('image', {}).get('feature_jitter', {}).get('enabled', False):
                noise = config['augment']['image']['feature_jitter'].get('std', 0.02)
        except:
            noise = 0.0
    
    # Path handling
    # The config points to the root features directory
    features_root = config['data']['features_path']
    
    dataset = CocoFeatureDataset(
        features_dir=features_root,
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer,
        max_length=config['data'].get('max_length', 50),
        split=split,
        noise_std=noise
    )

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train') # Drop incomplete batch only during training
    )
    
    return loader