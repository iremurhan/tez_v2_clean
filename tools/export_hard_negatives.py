#!/usr/bin/env python3
"""
export_hard_negatives.py
------------------------

Diagnostic script to export hard negative mining results to JSON.
Finds the hardest retrieval failures AND best successes for offline analysis.

Supports bi-directional retrieval analysis:
- T2I (Text-to-Image): Query caption → Find image
- I2T (Image-to-Text): Query image → Find caption

Uses robust ID-based ground truth matching (no index assumptions).

Usage:
    python tools/export_hard_negatives.py --config config.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import yaml
import torch
import json
import os
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DualEncoder
from src.data import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_embeddings(model, dataloader, device):
    """
    Compute embeddings for all samples in the dataloader.
    Returns: img_embeds, txt_embeds, metadata (image_ids, captions, filenames)
    """
    model.eval()
    
    img_embeds_list = []
    txt_embeds_list = []
    metadata_list = []
    
    logger.info("Computing embeddings...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            img_emb, txt_emb = model(images, input_ids, attention_mask)
            
            img_embeds_list.append(img_emb.cpu())
            txt_embeds_list.append(txt_emb.cpu())
            
            # Collect metadata for each sample in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # Global index of the sample in the dataset
                global_idx = batch_idx * dataloader.batch_size + i
                # Check if the sample is within the dataset
                if global_idx < len(dataloader.dataset.samples):
                    sample = dataloader.dataset.samples[global_idx]
                    metadata_list.append({
                        'image_id': batch['image_id'][i].item() if torch.is_tensor(batch['image_id'][i]) else batch['image_id'][i],
                        'caption': sample['caption'],
                        'filepath': sample['filepath'],
                        'filename': sample['filename']
                    })
    
    img_embeds = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    
    # Truncate metadata to match actual embeddings (in case of drop_last)
    metadata_list = metadata_list[:len(img_embeds)]
    
    logger.info(f"Computed embeddings: {img_embeds.shape}, {txt_embeds.shape}")
    logger.info(f"Collected metadata for {len(metadata_list)} samples")
    
    return img_embeds, txt_embeds, metadata_list


def build_mappings(metadata_list):
    """
    Build efficient lookup mappings from metadata.
    
    Returns:
        unique_image_ids: List of unique image IDs (in order of first appearance)
        image_id_to_unique_idx: Dict mapping image_id -> index in unique list
        image_id_to_caption_indices: Dict mapping image_id -> list of caption indices
        unique_idx_to_sample_idx: Dict mapping unique image idx -> first sample idx (for metadata)
    """
    unique_image_ids = []
    image_id_to_unique_idx = {}
    image_id_to_caption_indices = defaultdict(list)
    unique_idx_to_sample_idx = {}
    
    for idx, meta in enumerate(metadata_list):
        img_id = meta['image_id']
        
        # Track caption indices for this image
        image_id_to_caption_indices[img_id].append(idx)
        
        # Track unique images
        if img_id not in image_id_to_unique_idx:
            unique_idx = len(unique_image_ids)
            image_id_to_unique_idx[img_id] = unique_idx
            unique_image_ids.append(img_id)
            unique_idx_to_sample_idx[unique_idx] = idx
    
    return unique_image_ids, image_id_to_unique_idx, image_id_to_caption_indices, unique_idx_to_sample_idx


def mine_t2i(similarity, metadata_list, unique_image_ids, image_id_to_unique_idx, unique_idx_to_sample_idx, top_k=50):
    """
    Text-to-Image Mining: Query caption → Find image
    Uses ID-based ground truth matching.
    
    Returns:
        failures: Top-K hardest failures (wrong image retrieved)
        successes: Top-K best successes (correct image with highest confidence)
    """
    logger.info("Mining T2I (Text → Image) results...")
    failures = []
    successes = []
    
    n_captions = similarity.shape[0]
    
    for caption_idx in tqdm(range(n_captions), desc="T2I Mining"):
        meta = metadata_list[caption_idx]
        query_image_id = meta['image_id']
        
        # Ground truth unique image index
        gt_unique_idx = image_id_to_unique_idx[query_image_id]
        
        # Similarities for this caption query
        scores = similarity[caption_idx]
        
        # Sort images by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Find rank of correct image
        rank = (sorted_indices == gt_unique_idx).nonzero(as_tuple=True)[0].item()
        
        # Top-1 retrieved image
        top1_unique_idx = sorted_indices[0].item()
        top1_image_id = unique_image_ids[top1_unique_idx]
        top1_score = scores[top1_unique_idx].item()
        
        # Correct image score
        correct_score = scores[gt_unique_idx].item()
        
        # Build result dict
        result = {
            'query_caption': meta['caption'],
            'query_image_id': query_image_id,
            'correct_image_id': query_image_id,
            'correct_image_filename': os.path.join(meta['filepath'], meta['filename']),
            'correct_score': round(correct_score, 4),
            'retrieved_image_id': top1_image_id,
            'retrieved_score': round(top1_score, 4),
            'rank': rank
        }
        
        # Get retrieved image filename
        retrieved_sample_idx = unique_idx_to_sample_idx[top1_unique_idx]
        retrieved_meta = metadata_list[retrieved_sample_idx]
        result['retrieved_image_filename'] = os.path.join(retrieved_meta['filepath'], retrieved_meta['filename'])
        
        if rank == 0:
            # Success: correct image was top-1
            result['confidence'] = round(correct_score, 4)
            successes.append(result)
        else:
            # Failure: wrong image was top-1
            result['score_gap'] = round(top1_score - correct_score, 4)
            failures.append(result)
    
    logger.info(f"T2I: {len(successes)} successes, {len(failures)} failures")
    
    # Sort failures by score gap (hardest first)
    failures.sort(key=lambda x: x['score_gap'], reverse=True)
    
    # Sort successes by confidence (best first)
    successes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return failures[:top_k], successes[:top_k]


def mine_i2t(similarity, metadata_list, unique_image_ids, image_id_to_unique_idx, image_id_to_caption_indices, unique_idx_to_sample_idx, top_k=50):
    """
    Image-to-Text Mining: Query image → Find caption
    Uses ID-based ground truth matching.
    
    Returns:
        failures: Top-K hardest failures (wrong caption retrieved)
        successes: Top-K best successes (correct caption with highest confidence)
    """
    logger.info("Mining I2T (Image → Text) results...")
    failures = []
    successes = []
    
    # Transpose for I2T: [N_unique_images, N_captions]
    similarity_i2t = similarity.t()
    
    n_unique_images = len(unique_image_ids)
    
    for unique_idx in tqdm(range(n_unique_images), desc="I2T Mining"):
        query_image_id = unique_image_ids[unique_idx]
        
        # Ground truth caption indices for this image
        gt_caption_indices = set(image_id_to_caption_indices[query_image_id])
        
        # Similarities for this image query
        scores = similarity_i2t[unique_idx]
        
        # Sort captions by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        # Top-1 retrieved caption
        top1_caption_idx = sorted_indices[0].item()
        top1_score = scores[top1_caption_idx].item()
        top1_meta = metadata_list[top1_caption_idx]
        
        # Check if top-1 is among ground truth captions
        is_success = top1_caption_idx in gt_caption_indices
        
        # Find rank of first correct caption
        rank = -1
        first_correct_score = 0.0
        first_correct_caption = ""
        for r, idx in enumerate(sorted_indices.tolist()):
            if idx in gt_caption_indices:
                rank = r
                first_correct_score = scores[idx].item()
                first_correct_caption = metadata_list[idx]['caption']
                break
        
        if rank == -1:
            rank = len(sorted_indices)  # Should not happen
        
        # Query image metadata
        query_sample_idx = unique_idx_to_sample_idx[unique_idx]
        query_meta = metadata_list[query_sample_idx]
        
        # Build result dict
        result = {
            'query_image_id': query_image_id,
            'query_image_filename': os.path.join(query_meta['filepath'], query_meta['filename']),
            'correct_caption': first_correct_caption,
            'correct_score': round(first_correct_score, 4),
            'retrieved_caption': top1_meta['caption'],
            'retrieved_caption_image_id': top1_meta['image_id'],
            'retrieved_score': round(top1_score, 4),
            'rank': rank
        }
        
        if is_success:
            # Success: a correct caption was top-1
            result['confidence'] = round(top1_score, 4)
            successes.append(result)
        else:
            # Failure: wrong caption was top-1
            result['score_gap'] = round(top1_score - first_correct_score, 4)
            failures.append(result)
    
    logger.info(f"I2T: {len(successes)} successes, {len(failures)} failures")
    
    # Sort failures by score gap (hardest first)
    failures.sort(key=lambda x: x['score_gap'], reverse=True)
    
    # Sort successes by confidence (best first)
    successes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return failures[:top_k], successes[:top_k]


def find_hard_negatives(img_embeds, txt_embeds, metadata_list, top_k=50):
    """
    Find the hardest negative examples AND best successes for both T2I and I2T.
    
    Returns:
        Dictionary with t2i_failures, t2i_successes, i2t_failures, i2t_successes
    """
    logger.info("Building ID mappings...")
    unique_image_ids, image_id_to_unique_idx, image_id_to_caption_indices, unique_idx_to_sample_idx = build_mappings(metadata_list)
    
    logger.info(f"Found {len(unique_image_ids)} unique images from {len(metadata_list)} captions")
    
    logger.info("Computing similarity matrix...")
    
    # Get unique image embeddings (first occurrence of each image)
    unique_indices = [unique_idx_to_sample_idx[i] for i in range(len(unique_image_ids))]
    img_embeds_unique = img_embeds[unique_indices]
    
    # Similarity matrix: [N_captions, N_unique_images]
    similarity = torch.matmul(txt_embeds, img_embeds_unique.t())
    
    logger.info(f"Similarity matrix shape: {similarity.shape}")
    
    # Mine both directions
    t2i_failures, t2i_successes = mine_t2i(
        similarity, metadata_list, unique_image_ids, 
        image_id_to_unique_idx, unique_idx_to_sample_idx, top_k
    )
    
    i2t_failures, i2t_successes = mine_i2t(
        similarity, metadata_list, unique_image_ids,
        image_id_to_unique_idx, image_id_to_caption_indices, unique_idx_to_sample_idx, top_k
    )
    
    return {
        't2i_failures': t2i_failures,
        't2i_successes': t2i_successes,
        'i2t_failures': i2t_failures,
        'i2t_successes': i2t_successes
    }


def main():
    parser = argparse.ArgumentParser(description="Export bi-directional hard negative mining results")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='outputs/hard_negatives.json', help='Output JSON path')
    parser.add_argument('--top_k', type=int, default=50, help='Number of results per category')
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
    # Load validation dataloader
    logger.info("Loading validation dataloader...")
    val_loader = get_dataloader(config, tokenizer, split='val')
    
    # Load model
    logger.info("Loading model...")
    model = DualEncoder(config).to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("Model loaded successfully")
    
    # Compute embeddings with metadata
    img_embeds, txt_embeds, metadata_list = compute_embeddings(model, val_loader, device)
    
    # Find hard negatives and successes
    results = find_hard_negatives(img_embeds, txt_embeds, metadata_list, top_k=args.top_k)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    logger.info(f"Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"T2I Failures: {len(results['t2i_failures'])}")
    logger.info(f"T2I Successes: {len(results['t2i_successes'])}")
    logger.info(f"I2T Failures: {len(results['i2t_failures'])}")
    logger.info(f"I2T Successes: {len(results['i2t_successes'])}")
    
    # Print T2I examples
    if results['t2i_failures']:
        logger.info("\n--- Top 3 Hardest T2I Failures ---")
        for i, case in enumerate(results['t2i_failures'][:3], 1):
            logger.info(f"{i}. \"{case['query_caption'][:50]}...\"")
            logger.info(f"   Rank: {case['rank']} | Gap: {case['score_gap']:.4f}")
    
    if results['t2i_successes']:
        logger.info("\n--- Top 3 Best T2I Successes ---")
        for i, case in enumerate(results['t2i_successes'][:3], 1):
            logger.info(f"{i}. \"{case['query_caption'][:50]}...\"")
            logger.info(f"   Confidence: {case['confidence']:.4f}")
    
    # Print I2T examples
    if results['i2t_failures']:
        logger.info("\n--- Top 3 Hardest I2T Failures ---")
        for i, case in enumerate(results['i2t_failures'][:3], 1):
            logger.info(f"{i}. Image: {case['query_image_id']}")
            logger.info(f"   Rank: {case['rank']} | Gap: {case['score_gap']:.4f}")
            logger.info(f"   Retrieved: \"{case['retrieved_caption'][:40]}...\"")
    
    if results['i2t_successes']:
        logger.info("\n--- Top 3 Best I2T Successes ---")
        for i, case in enumerate(results['i2t_successes'][:3], 1):
            logger.info(f"{i}. Image: {case['query_image_id']}")
            logger.info(f"   Confidence: {case['confidence']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
