#!/usr/bin/env python3
"""
data.py
-------

Data loading and preprocessing for cross-modal retrieval.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
from PIL import Image
from torchvision import transforms

# Configure logger
logger = logging.getLogger(__name__)

class CocoImageDataset(Dataset):
    def __init__(self, images_root_path, captions_path, tokenizer, max_length=50, split='train', transform=None, intra_modal_aug=False):
        """
        Args:
            images_root_path (str): Root directory containing image folders (train2014/val2014).
            captions_path (str): Path to the Karpathy JSON file.
            tokenizer: HuggingFace tokenizer instance.
            max_length (int): Maximum token sequence length.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Base transform to be applied to the image.
            intra_modal_aug (bool): If True, generates a second augmented view of the image for intra-modal loss.
        """
        self.images_root_path = images_root_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.intra_modal_aug = intra_modal_aug
        
        # Define Transforms
        if transform is None:
            # The values are from the ImageNet dataset
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            if split == 'train':
                self.transform = transforms.Compose([
                    # The goal is to obtain scale invariance
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            self.transform = transform

        # ---------------------------------------------------------
        # 1. Load Captions (Karpathy JSON)
        # ---------------------------------------------------------
        logger.info(f"Loading captions from {captions_path} for split: {split}")
        with open(captions_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        
        # Karpathy JSON structure: images list contains items with 'split' and 'sentences'
        for img in data['images']:
            current_split = img['split']
            if current_split == 'restval' and split == 'train':
                current_split = 'train'

            if current_split == split:
                if 'cocoid' not in img:
                    if 'id' in img:
                        img_id = int(img['id'])
                    else:
                        raise ValueError(f"Image entry missing 'cocoid' or 'id': {img}")     
                else:
                    img_id = int(img['cocoid'])
                
                # Add all 5 captions for this image
                for sent in img['sentences']:
                    self.samples.append({
                        'image_id': img_id,
                        'caption': sent['raw'],
                        'filepath': img.get('filepath', ''),
                        'filename': img.get('filename', '')
                    })
                    
        logger.info(f"Found {len(self.samples)} captions for split '{split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']
        
        # 1. Load Image
        image_path = os.path.join(self.images_root_path, sample['filepath'], sample['filename'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except (OSError, FileNotFoundError) as e:
            # Handle missing images gracefully (though dataset should be clean)
            logger.error(f"Failed to load image {image_path}: {e}. Using black image.")
            image = Image.new('RGB', (224, 224))

        # 2. Apply Transforms
        # View 1: Standard View
        img_tensor = self.transform(image)
        
        # View 2: Augmented View (For Intra-Modal Learning)
        # Only generated if explicitly requested (usually only for training)
        if self.intra_modal_aug:
            # Re-apply train transform to get a different view
            img_aug_tensor = self.transform(image)
        else:
            # If not augmenting, just return a copy or zero tensor
            # Returning a clone ensures valid tensor shape
            img_aug_tensor = img_tensor.clone()

        # 3. Tokenize Text
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': img_tensor,       
            'image_aug': img_aug_tensor,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'index': idx,
            'image_id': image_id
        }

def get_dataloader(config, tokenizer, split='train'):
    """
    Factory function to create dataloaders.
    Strict Mode: All configurations must be present in the YAML file.
    """
    shuffle = (split == 'train')
    
    intra_modal_aug = False
    
    if split == 'train':
        intra_modal_aug = config['augment']['image']['enabled']
    
    images_root = config['data']['images_path']
    
    dataset = CocoImageDataset(
        images_root_path=images_root,
        captions_path=config['data']['captions_path'],
        tokenizer=tokenizer,
        max_length=config['data'].get('max_length', 50),
        split=split,
        transform=None, # Sınıf içinde default transformları kullanacak (Train/Val ayrımı orada var)
        intra_modal_aug=intra_modal_aug
    )

    # Debug Truncation
    if config['debug']['debug_mode']:
        debug_limit = config['debug']['debug_samples']
        
        if len(dataset.samples) > debug_limit:
            logger.warning(f"DEBUG MODE: Truncating dataset from {len(dataset.samples)} to {debug_limit} samples.")
            dataset.samples = dataset.samples[:debug_limit]

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return loader