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
def compute_recall_at_k(img_embeds, txt_embeds, k_values=[1, 5, 10]):
    """
    Compute Recall@K for Image-Text Retrieval metrics.
    
    Logic for MS-COCO:
    - There are N unique images.
    - There are 5*N captions.
    - Captions are ordered sequentially: [Img1_C1, Img1_C2 ... Img1_C5, Img2_C1 ...]
    
    Args:
        img_embeds (Tensor): Shape [N_images, Dim] -> Normalized image features
        txt_embeds (Tensor): Shape [N_captions, Dim] -> Normalized text features
                             (Where N_captions should be 5 * N_images)
        k_values (list): Recall thresholds (e.g., R@1, R@5, R@10)
        
    Returns:
        r_t2i (dict): Text-to-Image Recall scores
        r_i2t (dict): Image-to-Text Recall scores
    """
    # 1. Similarity Matrix Computation
    # [N, D] x [M, D]^T -> [N, M]
    # Rows: Images, Columns: Captions
    sims = torch.matmul(img_embeds, txt_embeds.t())
    
    n_imgs = img_embeds.shape[0]
    n_txts = txt_embeds.shape[0]
    
    # 2. Safety Check
    if n_txts != 5 * n_imgs:
        logger.warning(
            f"Metric Warning: Num captions ({n_txts}) is not 5x Num images ({n_imgs}). "
            "Recall calculation assumes standard COCO 1-to-5 mapping. "
            "If using a custom subset, ignore this."
        )

    # ----------------------------------------------------------------------
    # A. Text-to-Image Retrieval (T2I)
    # Query: Caption (Size M), Database: Images (Size N)
    # Goal: For a given caption, find its 1 correct image.
    # ----------------------------------------------------------------------
    # Transpose similarity matrix to shape [M, N] -> [Captions, Images]
    sims_t2i = sims.t()
    
    scores_t2i = {k: 0.0 for k in k_values}
    
    for i in range(n_txts):
        # Ground Truth: For caption index 'i', the correct image index is 'i // 5'
        # Example: Caption 0,1,2,3,4 -> Image 0
        gt_img_idx = i // 5 
        
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
    # Goal: For a given image, find ANY of its 5 correct captions.
    # ----------------------------------------------------------------------
    scores_i2t = {k: 0.0 for k in k_values}
    
    for i in range(n_imgs):
        # Ground Truth: Image 'i' corresponds to caption indices [5*i ... 5*i+4]
        gt_cap_indices = set(range(i * 5, (i + 1) * 5))
        
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