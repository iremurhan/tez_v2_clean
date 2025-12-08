import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Image Encoder using ResNet50 backbone with a specific freezing strategy.
    
    Freezing Strategy (hard-coded for experiment):
    - Freeze: Stem (conv1, bn1, relu, maxpool) and Layer1
    - Partially Unfreeze Layer2: First 2 blocks frozen, last 2 blocks unfrozen
    - Unfreeze: Layer3 and Layer4 completely
    """
    
    # ResNet50 outputs 2048-dimensional features
    OUTPUT_DIM = 2048
    
    def __init__(self):
        super().__init__()
        
        print("Loading Image Encoder: ResNet50 (Pretrained)...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final FC layer - keep everything up to avgpool
        # ResNet50 children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # Apply freezing strategy
        self._apply_freezing_strategy()
    
    def _apply_freezing_strategy(self):
        """
        Hard-coded freezing strategy for Layer2-partial experiment:
        - Freeze: Stem (indices 0-3) and Layer1 (index 4)
        - Layer2 (index 5): Freeze blocks 0,1 | Unfreeze blocks 2,3
        - Unfreeze: Layer3 (index 6) and Layer4 (index 7)
        """
        # First, freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get layer2 (index 5 in Sequential: conv1, bn1, relu, maxpool, layer1, layer2, ...)
        layer2 = self.backbone[5]
        
        # Unfreeze last 2 blocks of Layer2 (blocks 2 and 3)
        # ResNet50 Layer2 has 4 Bottleneck blocks (indices 0, 1, 2, 3)
        for block_idx in [2, 3]:
            for param in layer2[block_idx].parameters():
                param.requires_grad = True
        
        # Unfreeze Layer3 completely (index 6)
        layer3 = self.backbone[6]
        for param in layer3.parameters():
            param.requires_grad = True
        
        # Unfreeze Layer4 completely (index 7)
        layer4 = self.backbone[7]
        for param in layer4.parameters():
            param.requires_grad = True
        
        # Print summary
        self._print_freezing_summary()
    
    def _print_freezing_summary(self):
        """Print a summary of frozen/unfrozen parameters."""
        total_params = 0
        trainable_params = 0
        
        for param in self.backbone.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        frozen_params = total_params - trainable_params
        print(f"  ImageEncoder: {trainable_params:,} trainable / {total_params:,} total params")
        print(f"  Frozen: Stem, Layer1, Layer2[0:2] | Unfrozen: Layer2[2:4], Layer3, Layer4")
    
    def forward(self, images):
        """
        Args:
            images: [Batch, 3, 224, 224]
        Returns:
            features: [Batch, 2048, 1, 1]
        """
        return self.backbone(images)


class TextEncoder(nn.Module):
    """
    Text Encoder using DistilBERT backbone.
    Returns the [CLS] token representation.
    """
    def __init__(self, model_name: str):
        super().__init__()
        
        print(f"Loading Text Encoder: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [Batch, SeqLen]
            attention_mask: [Batch, SeqLen]
        Returns:
            cls_embedding: [Batch, 768]
        """
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Return [CLS] token (first token)
        return output.last_hidden_state[:, 0, :]


class DualEncoder(nn.Module):
    """
    Dual Encoder architecture for End-to-End Image-Text Retrieval.
    Assembles ImageEncoder, TextEncoder, and Projection Heads.
    """
    def __init__(self, config):
        super().__init__()
        
        self.text_dim = config['model']['text_dim']     # e.g., 768
        self.embed_dim = config['model']['embed_dim']   # e.g., 256
        self.dropout_p = config['model'].get('dropout', 0.1)
        
        # ---------------------------------------------------------
        # 1. Encoders
        # ---------------------------------------------------------
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(config['model']['text_model_name'])
        
        # Get image_dim from encoder (safer than config)
        self.image_dim = ImageEncoder.OUTPUT_DIM
        
        # ---------------------------------------------------------
        # 2. Projection Heads
        # ---------------------------------------------------------
        self.image_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_dim, self.text_dim),  # 2048 -> 768
            nn.BatchNorm1d(self.text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.text_dim, self.embed_dim)   # 768 -> 256
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.text_dim, self.embed_dim)
        )
        
        # Initialize projection head weights (don't touch backbones)
        self._init_head_weights()

    def _init_head_weights(self):
        """Initialize projection heads with Xavier uniform."""
        for proj in [self.image_proj, self.text_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: [Batch, 3, 224, 224]
            input_ids: [Batch, SeqLen]
            attention_mask: [Batch, SeqLen]
        Returns:
            img_embeds: [Batch, embed_dim] - L2 normalized
            txt_embeds: [Batch, embed_dim] - L2 normalized
        """
        # --- Image Pathway ---
        img_features = self.image_encoder(images)  # [Batch, 2048, 1, 1]
        img_embeds = self.image_proj(img_features) # [Batch, 256]
        
        # --- Text Pathway ---
        txt_cls = self.text_encoder(input_ids, attention_mask)  # [Batch, 768]
        txt_embeds = self.text_proj(txt_cls)  # [Batch, 256]
        
        # --- L2 Normalization ---
        img_embeds = F.normalize(img_embeds, p=2, dim=1)
        txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
        
        return img_embeds, txt_embeds
