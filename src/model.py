import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class DualEncoder(nn.Module):
    """
    Dual Encoder using OpenAI CLIP (clip-vit-base-patch32).
    
    CLIP is pre-trained on 400M image-text pairs from the internet,
    providing extremely strong visual-semantic representations.
    
    Strategy:
    - Freeze CLIP backbone (vision + text encoders)
    - Train only projection layers for domain adaptation
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config  # Store for use in freezing strategy
        self.embed_dim = config['model']['embed_dim']
        self.dropout_p = config['model'].get('dropout', 0.1)
        
        # 1. Load CLIP Model
        clip_model_name = config['model'].get('image_model_name', 'openai/clip-vit-base-patch32')
        print(f"Loading CLIP Model: {clip_model_name}...")
        # Use safetensors to avoid torch.load security vulnerability (CVE-2025-32434)
        self.clip = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        
        # Get CLIP's projection dimension (usually 512 for base models)
        self.clip_embed_dim = self.clip.config.projection_dim
        print(f"CLIP Projection Dimension: {self.clip_embed_dim}")
        
        # 2. Freezing Strategy - Freeze backbone, train projections
        self._apply_freezing_strategy()
        
        # 3. Optional: Additional projection head to match our embed_dim
        # If CLIP's output (512) != our target embed_dim (256), add a small MLP
        if self.clip_embed_dim != self.embed_dim:
            print(f"Adding projection: {self.clip_embed_dim} -> {self.embed_dim}")
            self.image_proj = nn.Sequential(
                nn.Linear(self.clip_embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.clip_embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )
            self._init_head_weights()
        else:
            self.image_proj = None
            self.text_proj = None
    
    def _apply_freezing_strategy(self):
        """
        Freeze CLIP backbone, selectively unfreeze layers for fine-tuning.
        
        CLIP ViT-B/32 architecture:
        - vision_model.encoder.layers.0-11  (12 transformer blocks)
        - text_model.encoder.layers.0-11    (12 transformer blocks)
        - visual_projection: Linear layer
        - text_projection: Linear layer
        
        Strategy: Unfreeze projections + last N vision transformer blocks
        """
        # First, freeze everything
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 1. Unfreeze CLIP's projection layers (always)
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True
        
        # 2. Unfreeze last N transformer blocks of Vision Encoder
        # ViT-B/32 has 12 blocks (0-11)
        # Config: unfreeze_vision_layers = 2 means unfreeze blocks [10, 11]
        num_vision_layers = self.config.get('model', {}).get('unfreeze_vision_layers', 0)
        unfreeze_strategy = self.config.get('model', {}).get('unfreeze_strategy', 'full')
        
        if num_vision_layers > 0:
            # Calculate which blocks to unfreeze (from the end)
            total_blocks = 12  # ViT-B/32 has 12 blocks
            unfreeze_vision_blocks = list(range(total_blocks - num_vision_layers, total_blocks))
            print(f"  Unfreezing Vision Blocks: {unfreeze_vision_blocks} (strategy: {unfreeze_strategy})")
            
            for name, param in self.clip.vision_model.named_parameters():
                for block_idx in unfreeze_vision_blocks:
                    if f"encoder.layers.{block_idx}." in name:
                        # Apply partial unfreezing strategy
                        should_unfreeze = self._should_unfreeze_param(name, unfreeze_strategy)
                        if should_unfreeze:
                            param.requires_grad = True
                        break
        
        # 3. Unfreeze vision model's post_layernorm (only if vision blocks are unfrozen)
        if num_vision_layers > 0 and unfreeze_strategy in ['full', 'layernorm']:
            for param in self.clip.vision_model.post_layernorm.parameters():
                param.requires_grad = True
        
        # 4. Unfreeze CLIP's learnable temperature (logit_scale)
        # This is critical for proper contrastive learning!
        if hasattr(self.clip, 'logit_scale'):
            self.clip.logit_scale.requires_grad = True
        
        # Print summary
        self._print_freezing_summary()
    
    def _should_unfreeze_param(self, name: str, strategy: str) -> bool:
        """
        Determine if a parameter should be unfrozen based on strategy.
        
        Strategies:
        - "full": Unfreeze entire block (all parameters)
        - "attention": Only Q, K, V, and output projections in self-attention
        - "mlp": Only MLP/FFN layers (fc1, fc2)
        - "layernorm": Only LayerNorm parameters (very lightweight)
        - "bias": Only bias terms (BitFit style, extremely lightweight)
        
        ViT layer naming convention:
        - encoder.layers.X.self_attn.q_proj, k_proj, v_proj, out_proj
        - encoder.layers.X.mlp.fc1, fc2
        - encoder.layers.X.layer_norm1, layer_norm2
        """
        if strategy == "full":
            return True
        
        elif strategy == "attention":
            # Unfreeze: q_proj, k_proj, v_proj, out_proj
            attention_keywords = ['self_attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
            return any(kw in name for kw in attention_keywords)
        
        elif strategy == "mlp":
            # Unfreeze: fc1, fc2 (MLP layers)
            mlp_keywords = ['mlp', 'fc1', 'fc2']
            return any(kw in name for kw in mlp_keywords)
        
        elif strategy == "layernorm":
            # Unfreeze: layer_norm1, layer_norm2, LayerNorm
            return 'layer_norm' in name.lower() or 'layernorm' in name.lower()
        
        elif strategy == "bias":
            # Unfreeze: Only bias parameters (BitFit)
            return 'bias' in name
        
        else:
            print(f"  Warning: Unknown unfreeze strategy '{strategy}', defaulting to 'full'")
            return True
    
    def _print_freezing_summary(self):
        """Print trainable parameter summary."""
        total_params = sum(p.numel() for p in self.clip.parameters())
        trainable_params = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        
        print(f"  CLIP: {trainable_params:,} trainable / {total_params:,} total params")
        print(f"  Frozen: Vision Encoder (partial), Text Encoder | Unfrozen: Projections + Selected")
    
    def _init_head_weights(self):
        """Initialize additional projection heads."""
        for proj in [self.image_proj, self.text_proj]:
            if proj is not None:
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass returning L2-normalized embeddings.
        
        Args:
            images: [Batch, 3, 224, 224] - Pixel values
            input_ids: [Batch, SeqLen] - Tokenized text
            attention_mask: [Batch, SeqLen] - Attention mask
            
        Returns:
            img_embeds: [Batch, embed_dim] - L2 normalized image embeddings
            txt_embeds: [Batch, embed_dim] - L2 normalized text embeddings
        """
        # Get CLIP embeddings (already projected)
        # Note: CLIP expects 'pixel_values' for images
        image_embeds = self.clip.get_image_features(pixel_values=images)
        text_embeds = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        # Optional: Additional projection to match target embed_dim
        if self.image_proj is not None:
            image_embeds = self.image_proj(image_embeds)
            text_embeds = self.text_proj(text_embeds)
        
        # L2 Normalization (critical for contrastive learning)
        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        
        return image_embeds, text_embeds
    
    def forward_with_clip_loss(self, images, input_ids, attention_mask):
        """
        Alternative forward that returns CLIP's built-in contrastive loss.
        
        Use this if you want to bypass custom loss function.
        Note: This ignores our additional projection heads.
        
        Returns:
            loss: Scalar contrastive loss computed by CLIP
            logits_per_image: [Batch, Batch] similarity matrix
            logits_per_text: [Batch, Batch] similarity matrix (transposed)
        """
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
            return_loss=True
        )
        return outputs.loss, outputs.logits_per_image, outputs.logits_per_text
