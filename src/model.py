import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTModel


class ImageEncoder(nn.Module):
    """
    Image Encoder using Vision Transformer (ViT).
    Uses HuggingFace 'ViTModel' to extract global image features.
    """
    def __init__(self, model_name: str):
        super().__init__()
        
        print(f"Loading Image Encoder: {model_name} (ViT)...")
        # Load Pretrained ViT (e.g., 'google/vit-base-patch16-224-in21k')
        self.backbone = ViTModel.from_pretrained(model_name)
        
        # Dynamic Feature Dimension (Usually 768 for ViT-Base)
        self.feature_dim = self.backbone.config.hidden_size
        
        # Freezing Strategy (Optional - Start with full fine-tuning or freeze lower layers)
        # For now, let's keep it fully trainable or freeze just embeddings
        # self._freeze_lower_layers() 

    def _freeze_lower_layers(self):
        # Example: Freeze embeddings and first 6 layers
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for layer in self.backbone.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Args:
            images: [Batch, 3, 224, 224] - Standard normalized tensors
        Returns:
            cls_token: [Batch, feature_dim]
        """
        # HuggingFace ViT expects 'pixel_values' argument
        outputs = self.backbone(pixel_values=images)
        
        # Take the [CLS] token (first token of the last hidden state)
        # Shape: [Batch, SeqLen, Dim] -> [Batch, Dim]
        return outputs.last_hidden_state[:, 0, :]


class TextEncoder(nn.Module):
    """
    Text Encoder using DistilBERT (or any BERT-like model).
    """
    def __init__(self, model_name: str):
        super().__init__()
        
        print(f"Loading Text Encoder: {model_name}...")
        self.backbone = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.hidden_size
    
    def forward(self, input_ids, attention_mask):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Return [CLS] token
        return output.last_hidden_state[:, 0, :]


class DualEncoder(nn.Module):
    """
    Dual Encoder architecture with ViT + DistilBERT.
    """
    def __init__(self, config):
        super().__init__()
        
        self.embed_dim = config['model']['embed_dim']
        self.dropout_p = config['model'].get('dropout', 0.1)
        
        # 1. Initialize Encoders
        # Get model names from config
        img_model_name = config['model'].get('image_model_name', 'google/vit-base-patch16-224-in21k')
        txt_model_name = config['model']['text_model_name']
        
        self.image_encoder = ImageEncoder(img_model_name)
        self.text_encoder = TextEncoder(txt_model_name)
        
        # 2. Get Dynamic Dimensions
        image_in_dim = self.image_encoder.feature_dim
        text_in_dim = self.text_encoder.feature_dim
        
        print(f"Feature Dimensions -> Image: {image_in_dim}, Text: {text_in_dim}")
        
        # 3. Projection Heads
        # ViT output is already a flat vector [Batch, 768], so no Flatten needed unlike ResNet
        self.image_proj = nn.Sequential(
            nn.Linear(image_in_dim, text_in_dim), 
            nn.BatchNorm1d(text_in_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(text_in_dim, self.embed_dim)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_in_dim, text_in_dim),
            nn.BatchNorm1d(text_in_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(text_in_dim, self.embed_dim)
        )
        
        self._init_head_weights()

    def _init_head_weights(self):
        for proj in [self.image_proj, self.text_proj]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, images, input_ids, attention_mask):
        # Image Pathway
        img_feat = self.image_encoder(images)
        img_emb = self.image_proj(img_feat)
        
        # Text Pathway
        txt_cls = self.text_encoder(input_ids, attention_mask)
        txt_emb = self.text_proj(txt_cls)
        
        # Normalization
        return F.normalize(img_emb, p=2, dim=1), F.normalize(txt_emb, p=2, dim=1)
