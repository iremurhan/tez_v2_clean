import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torchvision.models as models

class DualEncoder(nn.Module):
    def __init__(self, config):
        """
        Dual Encoder architecture for End-to-End Image-Text Retrieval.
        Integrates a trainable ResNet backbone and a DistilBERT backbone.
        """
        super().__init__()
        
        self.image_dim = config['model']['image_dim']   # e.g., 2048
        self.text_dim = config['model']['text_dim']     # e.g., 768
        self.embed_dim = config['model']['embed_dim']   # e.g., 256
        self.dropout_p = config['model'].get('dropout', 0.1)
        
        # ---------------------------------------------------------
        # 1. Image Encoder (ResNet50 Backbone)
        # ---------------------------------------------------------
        print("Loading Image Encoder: ResNet50 (Pretrained)...")
        # Load with ImageNet weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final FC layer (classification head)
        # ResNet feature extractor: (conv1 -> bn1 -> ... -> layer4 -> avgpool)
        modules = list(resnet.children())[:-1]
        self.image_backbone = nn.Sequential(*modules)
        
        # --- FREEZING STRATEGY (Critical) ---
        # Freeze early layers (low-level features), train only final blocks
        for name, param in self.image_backbone.named_parameters():
            # Freeze everything except "layer4" and "layer3"
            if "layer4" not in name and "layer3" not in name:
                param.requires_grad = False
        
        # Image Projection Head
        self.image_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_dim, self.text_dim), # 2048 -> 768
            nn.BatchNorm1d(self.text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.text_dim, self.embed_dim)  # 768 -> 256
        )

        # ---------------------------------------------------------
        # 2. Text Encoder (DistilBERT)
        # ---------------------------------------------------------
        model_name = config['model']['text_model_name']
        print(f"Loading Text Encoder: {model_name}...")
        self.text_backbone = AutoModel.from_pretrained(model_name)
        
        # Text Projection Head
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
        for m in self.image_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        for m in self.text_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, images, input_ids, attention_mask):
        # --- Image Pathway ---
        # Input: [Batch, 3, 224, 224] (Raw Image)
        img_features = self.image_backbone(images) # Output: [Batch, 2048, 1, 1]
        img_embeds = self.image_proj(img_features) # Output: [Batch, 256]
        
        # --- Text Pathway ---
        txt_output = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_output.last_hidden_state[:, 0, :] 
        txt_embeds = self.text_proj(txt_cls)
        
        # --- Normalization ---
        img_embeds = F.normalize(img_embeds, p=2, dim=1)
        txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
        
        return img_embeds, txt_embeds
