import torch
import logging
import wandb
import os
from tqdm import tqdm
from .utils import AverageMeter, compute_recall_at_k

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.use_wandb = config['logging']['use_wandb'] and not config['debug']['disable_wandb_sync']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        self.best_r1 = 0.0
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=100)
        
        for step, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            images_aug = batch['image_aug'].to(self.device) # For Intra-modal Image consistency
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 1. Forward (Original Views)
            img_embeds, txt_embeds = self.model(images, input_ids, attention_mask)
            
            # 2. Conditional Forward (Augmented Views for Intra-Modal Loss)
            img_aug_embeds = None
            txt_aug_embeds = None

            # A. Image Intra-Modal (Img <-> Img_Aug)
            if self.config['loss'].get('intra_img_weight', 0.0) > 0:
                # Pass augmented images through the image encoder
                img_aug_embeds, _ = self.model(images_aug, input_ids, attention_mask)

            # B. Text Intra-Modal (Text <-> Text_Aug)
            # SimCSE style: Pass same text again (dropout acts as augmentation)
            if self.config['loss'].get('intra_txt_weight', 0.0) > 0:
                _, txt_aug_embeds = self.model(images, input_ids, attention_mask)
            
            # 3. Loss Calculation
            loss = self.criterion(img_embeds, txt_embeds, img_aug_embeds, txt_aug_embeds)
            
            # 4. Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            pbar.set_postfix({"Loss": f"{losses.avg:.4f}"})
            
            if self.use_wandb and step % self.config['logging']['log_freq'] == 0:
                wandb.log({"train/loss": losses.val, "train/epoch": epoch})
        
        self.scheduler.step()
        return losses.avg

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        img_embeds_list = []
        txt_embeds_list = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            img_emb, txt_emb = self.model(images, input_ids, attention_mask)
            img_embeds_list.append(img_emb.cpu())
            txt_embeds_list.append(txt_emb.cpu())
            
        img_embeds = torch.cat(img_embeds_list, dim=0)
        txt_embeds = torch.cat(txt_embeds_list, dim=0)
        
        # COCO structure: 5 captions per image
        # Unique images are at indices 0, 5, 10...
        img_embeds_unique = img_embeds[::5]
        
        r_t2i, r_i2t = compute_recall_at_k(img_embeds_unique, txt_embeds)
        
        # Log
        logger.info(f"Epoch {epoch} Results: T2I R@1: {r_t2i[1]:.2f} | I2T R@1: {r_i2t[1]:.2f}")
        
        if self.use_wandb:
            wandb.log({"val/t2i_r1": r_t2i[1], "val/i2t_r1": r_i2t[1], "epoch": epoch})
            
        return r_t2i[1]

    def fit(self):
        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch(epoch)
            # Save checkpoint based on config
            if (epoch + 1) % self.config['logging']['save_freq'] == 0:
                score = self.evaluate(epoch)
                if score > self.best_r1:
                    self.best_r1 = score
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pth"))
