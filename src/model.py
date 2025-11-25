import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DualEncoder(nn.Module):
    def __init__(self, config):
        """
        Dual Encoder architecture for Cross-Modal Retrieval.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super().__init__()
        
        self.image_dim = config['model']['image_dim']   # e.g., 2048
        self.text_dim = config['model']['text_dim']     # e.g., 768
        self.embed_dim = config['model']['embed_dim']   # e.g., 256
        self.dropout_p = config['model'].get('dropout', 0.1)
        
        model_name = config['model']['text_model_name'] # e.g., "distilbert-base-uncased"

        # --- Text Encoder ---
        print(f"Loading text encoder: {model_name}...")
        self.text_backbone = AutoModel.from_pretrained(model_name)
        
        # Text Projection Head
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.text_dim, self.embed_dim)
        )

        # --- Image Encoder ---
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_dim, self.text_dim),
            nn.BatchNorm1d(self.text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.text_dim, self.embed_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, input_ids, attention_mask):
        # --- Image Pathway ---
        img_feat = self.image_proj(images)
        
        # --- Text Pathway ---
        txt_output = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        txt_cls = txt_output.last_hidden_state[:, 0, :] # [CLS] token
        txt_feat = self.text_proj(txt_cls)
        
        # --- Normalization ---
        img_embeds = F.normalize(img_feat, p=2, dim=1)
        txt_embeds = F.normalize(txt_feat, p=2, dim=1)
        
        return img_embeds, txt_embeds