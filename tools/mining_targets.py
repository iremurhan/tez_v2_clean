#!/usr/bin/env python3
"""
mining_targets.py
-----------------

Offline Hard-Negative Mining using Knowledge Distillation logic.

This script computes text-to-text similarity for all captions in the training set
and finds the Top-K most similar captions for each caption. These can be used as:
- Hard negatives: similar captions from different images
- Soft positives: semantically similar captions for soft-label distillation

Pipeline:
1. Load CLIP text encoder and compute embeddings for all ~565k captions
2. Use chunked cosine similarity to find top-K neighbors for each caption
3. Save indices and scores to datasets/coco/mining_targets.pt

Configuration:
    top_k is read from config['distillation']['top_k']
    output path is read from config['distillation']['mining_targets_path']

Usage:
    python tools/mining_targets.py --config config.yaml
    python tools/mining_targets.py --config config.yaml --query_chunk_size 2000
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import os
import sys
import logging
import wandb
from datetime import datetime
from pathlib import Path
from transformers import CLIPModel, CLIPTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CocoImageDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_clip_text_encoder(model_name: str, device: torch.device):
    """
    Load CLIP model for text encoding only.
    Returns the model and tokenizer.
    """
    logger.info(f"Loading CLIP model: {model_name}")
    
    # Use safetensors for security
    clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    # Move to device and set to eval mode
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Freeze all parameters (inference only)
    for param in clip_model.parameters():
        param.requires_grad = False
    
    logger.info(f"CLIP projection dim: {clip_model.config.projection_dim}")
    
    return clip_model, tokenizer


def compute_all_text_embeddings(
    clip_model: CLIPModel,
    dataset: CocoImageDataset,
    tokenizer: CLIPTokenizer,
    batch_size: int,
    device: torch.device,
    max_length: int
) -> torch.Tensor:
    """
    Phase 1: Compute normalized text embeddings for all captions.
    
    Stores embeddings on GPU for fast similarity computation.
    Logs progress to wandb every 50 batches.
    
    Args:
        clip_model: CLIP model for text encoding
        dataset: CocoImageDataset with .samples containing captions
        tokenizer: CLIP tokenizer
        batch_size: Batch size for processing
        device: torch device (should be CUDA)
        max_length: Maximum token length
    
    Returns:
        all_embeddings: Tensor[N, embed_dim] on GPU - L2 normalized, FP16
    """
    n_samples = len(dataset.samples)
    embed_dim = clip_model.config.projection_dim
    
    logger.info(f"Phase 1: Computing text embeddings for {n_samples:,} captions...")
    logger.info(f"Batch size: {batch_size}, Embed dim: {embed_dim}, Max length: {max_length}")
    logger.info(f"Estimated GPU memory: {n_samples * embed_dim * 2 / 1024**3:.2f} GB (FP16)")
    
    # Pre-allocate output tensor on GPU in FP16 (~1.7GB for 565k x 768)
    all_embeddings = torch.zeros(n_samples, embed_dim, dtype=torch.float16, device=device)
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            # Get captions for this batch
            captions = [dataset.samples[i]['caption'] for i in range(start_idx, end_idx)]
            
            # Tokenize
            tokenized = tokenizer(
                captions,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            # Compute embeddings with FP16 autocast
            with torch.amp.autocast(device_type='cuda'):
                text_embeds = clip_model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # L2 normalize (keep in FP16)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
            
            # Store directly in FP16 tensor
            all_embeddings[start_idx:end_idx] = text_embeds.half()
            
            # Log progress to wandb every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                progress = (batch_idx + 1) / n_batches * 100
                wandb.log({
                    "extraction_progress": progress,
                    "extraction_batch": batch_idx + 1,
                    "extraction_samples_processed": end_idx
                })
                logger.info(f"Extraction progress: {progress:.1f}% ({end_idx:,}/{n_samples:,})")
    
    logger.info(f"Phase 1 complete. Embeddings shape: {all_embeddings.shape}")
    logger.info(f"GPU memory used: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    return all_embeddings


def mine_topk_neighbors(
    all_embeddings: torch.Tensor,
    top_k: int,
    query_chunk_size: int = 1000,
    device: torch.device = None
) -> tuple:
    """
    Phase 2: Find Top-K most similar embeddings using chunked computation.
    
    For each chunk of queries:
    - Compute similarity: chunk @ all_embeddings.T
    - Extract top-k scores and indices
    - Store results on CPU
    
    Note on Self-Similarity Masking:
        We ONLY mask the exact same index (the query caption itself) by setting
        its similarity to -inf. We intentionally DO NOT mask other captions 
        belonging to the same image. This is because captions from the same image
        are "Semantic Positives" - they describe the same visual content and serve
        as valid soft targets for Knowledge Distillation. Including them in the
        top-K neighbors helps the model learn that semantically equivalent 
        descriptions should have similar embeddings.
    
    Args:
        all_embeddings: Tensor[N, D] on GPU - L2 normalized, FP16
        top_k: Number of neighbors to find (from config['distillation']['top_k'])
        query_chunk_size: Number of queries to process at once
        device: torch device
    
    Returns:
        indices: Tensor[N, K] on CPU - Top-K neighbor indices
        scores: Tensor[N, K] on CPU - Top-K similarity scores
    """
    n_samples = all_embeddings.shape[0]
    
    logger.info(f"Phase 2: Mining Top-{top_k} neighbors for {n_samples:,} samples...")
    logger.info(f"Query chunk size: {query_chunk_size}")
    
    # Lists to store results (will concatenate at the end)
    all_indices = []
    all_scores = []
    
    n_chunks = (n_samples + query_chunk_size - 1) // query_chunk_size
    
    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * query_chunk_size
            end_idx = min(start_idx + query_chunk_size, n_samples)
            
            # Get query chunk
            query_chunk = all_embeddings[start_idx:end_idx]  # [chunk_size, D]
            
            # Compute cosine similarity against all embeddings
            # Since embeddings are L2 normalized, matmul = cosine similarity
            sims = query_chunk @ all_embeddings.T  # [chunk_size, N]
            
            # Mask out ONLY self-similarity (exact same index) by setting to -inf.
            # We keep other captions from the same image as they are valid 
            # "Positive Pairs" for distillation (semantic positives).
            chunk_size = end_idx - start_idx
            for i in range(chunk_size):
                sims[i, start_idx + i] = float('-inf')
            
            # Get top-k scores and indices
            top_k_scores, top_k_indices = sims.topk(k=top_k, dim=1)
            
            # Store on CPU (convert to FP32 for scores)
            all_indices.append(top_k_indices.cpu())
            all_scores.append(top_k_scores.float().cpu())
            
            # Log progress to wandb every 50 chunks
            if (chunk_idx + 1) % 50 == 0 or chunk_idx == n_chunks - 1:
                progress = (chunk_idx + 1) / n_chunks * 100
                wandb.log({
                    "mining_progress": progress,
                    "mining_chunk": chunk_idx + 1,
                    "mining_samples_processed": end_idx
                })
                logger.info(f"Mining progress: {progress:.1f}% ({end_idx:,}/{n_samples:,})")
    
    # Concatenate all results
    indices = torch.cat(all_indices, dim=0)  # [N, K]
    scores = torch.cat(all_scores, dim=0)    # [N, K]
    
    logger.info(f"Phase 2 complete. Output shapes: indices={indices.shape}, scores={scores.shape}")
    
    return indices, scores


def main():
    parser = argparse.ArgumentParser(
        description="Offline Hard-Negative Mining for Knowledge Distillation"
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=512,
        help='Batch size for text embedding computation (default: 512)'
    )
    parser.add_argument(
        '--query_chunk_size', type=int, default=1000,
        help='Query chunk size for similarity computation (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Device setup (requires CUDA for efficient computation)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficient mining. Please run on a GPU.")
    
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get distillation config (with safety fallback)
    distillation_config = config.get('distillation', {})
    if not distillation_config:
        logger.warning("Config is missing 'distillation' section. Using defaults.")
    
    top_k = distillation_config.get('top_k', 30)
    output_path = distillation_config.get('mining_targets_path', 'datasets/coco/mining_targets.pt')
    
    logger.info(f"Distillation config: top_k={top_k}, output={output_path}")
    
    # Get CLIP model name from config
    clip_model_name = config['model']['image_model_name']
    if not clip_model_name:
        raise ValueError("config['model']['image_model_name'] must be specified")
    
    # Load CLIP model and tokenizer first to get model_max_length
    clip_model, tokenizer = load_clip_text_encoder(clip_model_name, device)
    
    # Determine max_length: prefer tokenizer.model_max_length, fallback to config, then 77
    if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None:
        # Some tokenizers report very large max_length (e.g., 1e30), cap it
        tokenizer_max = tokenizer.model_max_length
        if tokenizer_max > 1000:
            # Fallback to config or CLIP default
            max_length = config['data'].get('max_length', 77)
            logger.info(f"Tokenizer max_length ({tokenizer_max}) too large, using config: {max_length}")
        else:
            max_length = tokenizer_max
            logger.info(f"Using tokenizer.model_max_length: {max_length}")
    else:
        max_length = config['data'].get('max_length', 77)
        logger.info(f"Using config max_length: {max_length}")
    
    # Initialize wandb (use project from config if available)
    logging_config = config.get('logging', {})
    wandb_project = logging_config.get('wandb_project', 'coco-distillation')
    
    wandb.init(
        project=wandb_project,
        job_type="mining",
        config={
            "batch_size": args.batch_size,
            "query_chunk_size": args.query_chunk_size,
            "top_k": top_k,
            "max_length": max_length,
            "output_path": output_path,
            "clip_model": clip_model_name
        }
    )
    logger.info(f"Initialized wandb (project: {wandb_project})")
    
    # Load dataset (train split)
    logger.info("Loading COCO training dataset...")
    
    dataset = CocoImageDataset(
        images_root_path=config['data']['images_path'],
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer,
        max_length=max_length,
        split='train',
        transform=None,
        intra_modal_aug=False
    )
    
    # Respect debug_mode from config (consistent with get_dataloader behavior)
    debug_config = config.get('debug', {})
    if debug_config.get('debug_mode', False):
        debug_limit = debug_config.get('debug_samples', 100)
        if len(dataset.samples) > debug_limit:
            logger.warning(f"DEBUG MODE: Truncating dataset from {len(dataset.samples):,} to {debug_limit} samples.")
            dataset.samples = dataset.samples[:debug_limit]
    
    n_samples = len(dataset.samples)
    logger.info(f"Loaded {n_samples:,} captions from training split")
    
    wandb.log({"total_captions": n_samples})
    
    # Phase 1: Compute all text embeddings
    all_embeddings = compute_all_text_embeddings(
        clip_model=clip_model,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        max_length=max_length
    )
    
    # Free CLIP model memory before mining
    del clip_model
    torch.cuda.empty_cache()
    logger.info(f"Freed CLIP model. GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    # Phase 2: Mine top-K neighbors
    indices, scores = mine_topk_neighbors(
        all_embeddings=all_embeddings,
        top_k=top_k,
        query_chunk_size=args.query_chunk_size,
        device=device
    )
    
    # Prepare output
    output_data = {
        'indices': indices,  # Tensor[N, top_k]
        'scores': scores     # Tensor[N, top_k]
    }
    
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    torch.save(output_data, output_path)
    
    # Log final metrics to wandb
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    wandb.log({
        "output_file_size_mb": file_size_mb,
        "score_min": scores.min().item(),
        "score_max": scores.max().item(),
        "score_mean": scores.mean().item()
    })
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total captions: {n_samples:,}")
    logger.info(f"Top-K neighbors: {top_k}")
    logger.info(f"Output shape: indices={indices.shape}, scores={scores.shape}")
    logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    logger.info(f"Mean score: {scores.mean():.4f}")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    logger.info("=" * 60)
    
    # Print example neighbors
    logger.info("\n--- Example: Top-3 neighbors for first 3 captions ---")
    for i in range(min(3, n_samples)):
        caption = dataset.samples[i]['caption']
        logger.info(f"\nQuery [{i}]: \"{caption[:60]}...\"")
        for k in range(min(3, top_k)):
            neighbor_idx = indices[i, k].item()
            neighbor_score = scores[i, k].item()
            neighbor_caption = dataset.samples[neighbor_idx]['caption']
            logger.info(f"  [{k+1}] (score={neighbor_score:.4f}) \"{neighbor_caption[:50]}...\"")
    
    # Finish wandb run
    wandb.finish()
    logger.info("Wandb run finished.")


if __name__ == "__main__":
    main()
