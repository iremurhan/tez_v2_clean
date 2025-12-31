import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config['loss']['temperature']
        # Separate weights for Intra-Modal Image and Text consistency
        self.w_img = config['loss'].get('intra_img_weight', 0.0)
        self.w_txt = config['loss'].get('intra_txt_weight', 0.0)
        self.criterion = nn.CrossEntropyLoss()

    def compute_contrastive_loss(self, embed_a, embed_b):
        """
        Computes InfoNCE loss between batch A and batch B.
        Output: Scalar loss
        """
        # Similarity Matrix: [Batch, Batch]
        logits = torch.matmul(embed_a, embed_b.t()) / self.temperature
        
        # Labels: diagonal (0, 1, 2...) implies index i in A matches index i in B
        batch_size = embed_a.shape[0]
        labels = torch.arange(batch_size).to(embed_a.device)
        
        loss_a2b = self.criterion(logits, labels)
        loss_b2a = self.criterion(logits.t(), labels)
        return (loss_a2b + loss_b2a) / 2

    def forward(self, img_embeds, txt_embeds, img_aug_embeds=None, txt_aug_embeds=None):
        """
        Calculates full composite loss.
        """
        # 1. Inter-Modal (Image <-> Text) - MAIN OBJECTIVE
        loss_inter = self.compute_contrastive_loss(img_embeds, txt_embeds)
        
        # 2. Intra-Modal (Structure Preserving) - CONDITIONAL
        loss_img = 0.0
        if self.w_img > 0 and img_aug_embeds is not None:
             loss_img = self.compute_contrastive_loss(img_embeds, img_aug_embeds)

        loss_txt = 0.0
        if self.w_txt > 0 and txt_aug_embeds is not None:
             loss_txt = self.compute_contrastive_loss(txt_embeds, txt_aug_embeds)

        # Total Loss = Inter + (w_img * Image_Intra) + (w_txt * Text_Intra)
        return loss_inter + (self.w_img * loss_img) + (self.w_txt * loss_txt)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for Cross-Modal Retrieval.
    
    Uses pre-mined soft targets (teacher similarities) to guide the student model.
    The teacher provides a soft probability distribution over similar texts,
    and the student learns to match this distribution.
    
    Key Insight:
        - Teacher indices are GLOBAL (0 to N_dataset)
        - We need to map them to LOCAL batch indices (0 to BatchSize)
        - Only neighbors that appear in the current batch contribute to the loss
        - Self is always the strongest match (diagonal = 1.0)
    """
    
    def __init__(self, alpha: float = 0.5, temperature: float = 1.0):
        """
        Args:
            alpha: Weight for distillation loss (final_loss = (1-alpha)*clip_loss + alpha*distill_loss)
            temperature: Softmax temperature for softening distributions
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        student_logits: torch.Tensor,
        batch_indices: torch.Tensor,
        teacher_indices: torch.Tensor,
        teacher_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between student predictions and teacher soft targets.
        
        Args:
            student_logits: [B, B] - Student's similarity matrix (text-to-text)
            batch_indices: [B] - Global indices of samples in current batch
            teacher_indices: [B, K] - Global indices of top-K neighbors (from mining)
            teacher_scores: [B, K] - Teacher similarity scores for top-K neighbors
        
        Returns:
            loss: Scalar KL divergence loss (scaled by T^2)
        """
        batch_size = student_logits.shape[0]
        device = student_logits.device
        K = teacher_indices.shape[1]
        
        # ---------------------------------------------------------
        # Step 1: Build mapping from Global Index -> Local Batch Position
        # ---------------------------------------------------------
        # Create a lookup: global_idx -> batch_pos (or -1 if not in batch)
        # For efficiency, we use a tensor-based approach
        
        # Get max global index we might encounter
        max_global_idx = max(batch_indices.max().item(), teacher_indices.max().item()) + 1
        
        # Create mapping tensor: -1 means "not in batch"
        global_to_local = torch.full((max_global_idx,), -1, dtype=torch.long, device=device)
        
        # Fill in the mapping for samples that ARE in this batch
        local_positions = torch.arange(batch_size, device=device)
        global_to_local[batch_indices] = local_positions
        
        # ---------------------------------------------------------
        # Step 2: Build target probability matrix
        # ---------------------------------------------------------
        target_probs = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(batch_size):
            # Get this sample's teacher neighbors (global indices)
            neighbor_global_indices = teacher_indices[i]  # [K]
            neighbor_scores = teacher_scores[i]            # [K]
            
            # Map global indices to local batch positions
            # Clamp to valid range for indexing (out-of-range will map to -1 anyway)
            clamped_indices = neighbor_global_indices.clamp(0, max_global_idx - 1)
            local_positions = global_to_local[clamped_indices]  # [K]
            
            # Find which neighbors are actually in this batch (local_pos >= 0)
            valid_mask = local_positions >= 0
            
            if valid_mask.any():
                valid_local_pos = local_positions[valid_mask]
                valid_scores = neighbor_scores[valid_mask]
                
                # Assign teacher scores to valid positions
                target_probs[i, valid_local_pos] = valid_scores
            
            # Self-loop: Self is ALWAYS the best match (score = 1.0)
            # This ensures the model learns that identical text -> identical text
            target_probs[i, i] = 1.0
        
        # ---------------------------------------------------------
        # Step 3: Normalize target probabilities (row-wise sum to 1)
        # ---------------------------------------------------------
        # Use L1 normalization (sum normalization)
        row_sums = target_probs.sum(dim=1, keepdim=True)
        row_sums = row_sums.clamp(min=1e-8)  # Avoid division by zero
        target_probs = target_probs / row_sums
        
        # ---------------------------------------------------------
        # Step 4: Compute KL Divergence Loss
        # ---------------------------------------------------------
        # Student: log_softmax(logits / T)
        # Teacher: target_probs (already normalized)
        
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL(P || Q) where P=target_probs, Q=student
        # PyTorch KLDivLoss expects (log_probs, probs) -> KL(probs || exp(log_probs))
        loss = self.kl_loss(student_log_probs, target_probs)
        
        # Scale by T^2 to preserve gradient magnitude (standard KD practice)
        loss = loss * (self.temperature ** 2)
        
        return loss
