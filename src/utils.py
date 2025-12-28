import torch
import numpy as np
import random
import os
import logging

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed=42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def compute_recall_at_k(img_embeds, txt_embeds, image_ids, unique_image_ids, k_values=[1, 5, 10]):
    """
    Compute Recall@K for Image-Text Retrieval metrics.
    
    Uses image_id matching from JSON data for accurate ground truth.
    
    Args:
        img_embeds (Tensor): Shape [N_images, Dim] -> Normalized image features (unique images)
        txt_embeds (Tensor): Shape [N_captions, Dim] -> Normalized text features
        image_ids (Tensor): Shape [N_captions] -> Image ID for each caption (from dataset) - REQUIRED
        unique_image_ids (Tensor): Shape [N_images] -> Unique image IDs corresponding to img_embeds - REQUIRED
        k_values (list): Recall thresholds (e.g., R@1, R@5, R@10)
        
    Returns:
        r_t2i (dict): Text-to-Image Recall scores
        r_i2t (dict): Image-to-Text Recall scores
    """
    # Validate required parameters
    if image_ids is None or unique_image_ids is None:
        raise ValueError(
            "image_ids and unique_image_ids are required for accurate ground truth matching. "
            "These should be extracted from the dataset JSON file."
        )
    
    # 1. Similarity Matrix Computation
    # [N, D] x [M, D]^T -> [N, M]
    # Rows: Images, Columns: Captions
    sims = torch.matmul(img_embeds, txt_embeds.t())
    
    n_imgs = img_embeds.shape[0]
    n_txts = txt_embeds.shape[0]
    
    # 2. Build mapping from image_id to unique image index
    # Create mapping: image_id -> index in unique_image_ids
    image_id_to_idx = {img_id.item(): idx for idx, img_id in enumerate(unique_image_ids)}
    
    # Build ground truth: for each caption, which unique image index does it belong to?
    caption_to_image_idx = []
    for caption_idx in range(n_txts):
        caption_image_id = image_ids[caption_idx].item()
        if caption_image_id not in image_id_to_idx:
            raise ValueError(
                f"Caption {caption_idx} has image_id {caption_image_id} which is not found in unique_image_ids. "
                "This indicates a mismatch between caption and image data."
            )
        unique_img_idx = image_id_to_idx[caption_image_id]
        caption_to_image_idx.append(unique_img_idx)
    caption_to_image_idx = torch.tensor(caption_to_image_idx, dtype=torch.long)
    
    # Build reverse mapping: for each unique image, which caption indices belong to it?
    image_to_caption_indices = {}
    for img_idx in range(n_imgs):
        img_id = unique_image_ids[img_idx].item()
        # Find all caption indices that have this image_id
        matching_captions = (image_ids == img_id).nonzero(as_tuple=True)[0]
        image_to_caption_indices[img_idx] = set(matching_captions.tolist())

    # ----------------------------------------------------------------------
    # A. Text-to-Image Retrieval (T2I)
    # Query: Caption (Size M), Database: Images (Size N)
    # Goal: For a given caption, find its 1 correct image.
    # ----------------------------------------------------------------------
    # Transpose similarity matrix to shape [M, N] -> [Captions, Images]
    sims_t2i = sims.t()
    
    scores_t2i = {k: 0.0 for k in k_values}
    
    for i in range(n_txts):
        # Ground Truth: Get the correct unique image index for this caption
        gt_img_idx = caption_to_image_idx[i].item()
        
        # Sort scores for this query (descending)
        # We only care about the rank of the correct image
        # argsort gives indices of images sorted by similarity
        sorted_indices = sims_t2i[i].argsort(descending=True)
        
        # Find where the GT image is in the sorted list
        # (nonzero returns a tuple, we take the first item)
        rank = (sorted_indices == gt_img_idx).nonzero(as_tuple=True)[0][0].item()
        
        for k in k_values:
            if rank < k: # rank is 0-indexed (0 means R@1 success)
                scores_t2i[k] += 1.0

    # Normalize (percentage)
    for k in k_values:
        scores_t2i[k] = (scores_t2i[k] / n_txts) * 100.0

    # ----------------------------------------------------------------------
    # B. Image-to-Text Retrieval (I2T)
    # Query: Image (Size N), Database: Captions (Size M)
    # Goal: For a given image, find ANY of its correct captions.
    # ----------------------------------------------------------------------
    scores_i2t = {k: 0.0 for k in k_values}
    
    for i in range(n_imgs):
        # Ground Truth: Get all caption indices that belong to this image
        gt_cap_indices = image_to_caption_indices[i]
        
        # Sort captions for this image
        sorted_indices = sims[i].argsort(descending=True)
        
        # We need to check if ANY of the ground truth captions appeared in top K
        # Optimization: Just convert top-K to list and check intersection
        top_indices = sorted_indices.tolist()
        
        for k in k_values:
            current_top_k = set(top_indices[:k])
            
            # Intersection: Do we have any overlap?
            if not gt_cap_indices.isdisjoint(current_top_k):
                scores_i2t[k] += 1.0

    # Normalize (percentage)
    for k in k_values:
        scores_i2t[k] = (scores_i2t[k] / n_imgs) * 100.0
        
    return scores_t2i, scores_i2t