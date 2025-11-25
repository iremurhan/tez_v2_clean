import torch
import torch.nn as nn

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
