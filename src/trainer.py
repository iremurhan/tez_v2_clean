import torch
import logging
import wandb
import os
from .utils import AverageMeter, compute_recall_at_k

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, use_wandb=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.use_wandb = use_wandb
        self.log_freq = config['logging']['log_freq']
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        self.best_r1 = 0.0
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        """
        Loads full training state from a checkpoint file.
        Returns start_epoch.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 1. Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 2. Load State Info (STRICT)
        # Eğer bu anahtarlar yoksa kod burada KeyError verip duracak.
        # Bu sayede "bozuk checkpoint" ile yanlışlıkla en baştan başlamazsın.
        if 'best_r1' not in checkpoint or 'epoch' not in checkpoint:
            logger.error(f"Checkpoint file {checkpoint_path} is missing critical keys ('epoch' or 'best_r1').")
            logger.error("Cannot resume training safely. Aborting.")
            raise KeyError("Invalid checkpoint format for resuming.")

        self.best_r1 = checkpoint['best_r1']
        start_epoch = checkpoint['epoch']
        
        logger.info(f"Resuming successfully from epoch {start_epoch} with Best R@1: {self.best_r1:.2f}")
        
        return start_epoch

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        
        num_batches = len(self.train_loader)
        
        # Check if we should use CLIP's native loss (with learnable temperature)
        use_clip_loss = self.config['loss'].get('use_clip_loss', False)
        
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # ============================================================
            # Option A: Use CLIP's Native Loss (Learnable Temperature)
            # ============================================================
            if use_clip_loss:
                loss, _, _ = self.model.forward_with_clip_loss(images, input_ids, attention_mask)
            
            # ============================================================
            # Option B: Use Custom Loss (Fixed Temperature + Intra-Modal)
            # ============================================================
            else:
                images_aug = batch['image_aug'].to(self.device)
                
                # Forward (Original Views)
                img_embeds, txt_embeds = self.model(images, input_ids, attention_mask)
                
                # Conditional Forward (Augmented Views for Intra-Modal Loss)
                img_aug_embeds = None
                txt_aug_embeds = None

                # A. Image Intra-Modal (Img <-> Img_Aug)
                if self.config['loss'].get('intra_img_weight', 0.0) > 0:
                    img_aug_embeds, _ = self.model(images_aug, input_ids, attention_mask)

                # B. Text Intra-Modal (Text <-> Text_Aug)
                # SimCSE style: Pass same text again (dropout acts as augmentation)
                if self.config['loss'].get('intra_txt_weight', 0.0) > 0:
                    _, txt_aug_embeds = self.model(images, input_ids, attention_mask)
                
                # Loss Calculation
                loss = self.criterion(img_embeds, txt_embeds, img_aug_embeds, txt_aug_embeds)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            losses.update(loss.item(), images.size(0))
            

            if step % self.log_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1} [{step}/{num_batches}] Loss: {losses.avg:.4f} | LR: {current_lr:.6f}")
                
                if self.use_wandb:
                    try:
                        wandb.log({
                            "train/loss": losses.val, 
                            "train/epoch": epoch,
                            "train/lr": current_lr
                        })
                    except Exception as e:
                        logger.warning(f"Failed to log to W&B: {e}")
        
        return losses.avg

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        img_embeds_list = []
        txt_embeds_list = []
        
        # Handle "TEST" epoch logging string
        epoch_log = epoch if isinstance(epoch, str) else epoch
        logger.info(f"Starting Evaluation ({epoch_log})...")
        
        for batch in self.val_loader:
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
        logger.info(
            f"Epoch {epoch_log} Results:\n"
            f"  T2I: R@1: {r_t2i[1]:.2f} | R@5: {r_t2i[5]:.2f} | R@10: {r_t2i[10]:.2f}\n"
            f"  I2T: R@1: {r_i2t[1]:.2f} | R@5: {r_i2t[5]:.2f} | R@10: {r_i2t[10]:.2f}"
        )
        
        if self.use_wandb:
            if isinstance(epoch, int):
                try:
                    wandb.log({
                        "val/t2i_r1": r_t2i[1], "val/t2i_r5": r_t2i[5], "val/t2i_r10": r_t2i[10],
                        "val/i2t_r1": r_i2t[1], "val/i2t_r5": r_i2t[5], "val/i2t_r10": r_i2t[10],
                        "epoch": epoch
                    })
                except Exception as e:
                    logger.warning(f"Failed to log to W&B: {e}")
            
        return r_t2i[1]

    def fit(self, start_epoch=0):
        eval_frequency = self.config['logging']['eval_freq'] 
        save_frequency = self.config['logging']['save_freq']

        # If start_epoch is 0 (i.e., not resuming), run initial evaluation.
        if start_epoch == 0:
            logger.info("Running initial evaluation at Epoch 0...")
            score = self.evaluate(epoch=0)
            
            # At Epoch 0, only saving Last Model makes sense, not Best.
            # However, to simplify: If score > 0, let this be the first record.
            if score > self.best_r1:
                self.best_r1 = score
                
            logger.info(f"Initial state saved. Starting training loop from Epoch {start_epoch}.")

        # --- MAIN TRAINING LOOP ---
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # 1. Train one epoch
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
            
            # Determine when to evaluate and save
            is_eval_time = ((epoch + 1) % eval_frequency == 0) or ((epoch + 1) == self.config['training']['epochs'])
            is_save_time = ((epoch + 1) % save_frequency == 0) or ((epoch + 1) == self.config['training']['epochs'])

            # Ensure checkpoint directory exists before any save operation
            os.makedirs(self.checkpoint_dir, exist_ok=True)

            # 2. EVALUATION & BEST MODEL CHECK
            if is_eval_time:
                score = self.evaluate(epoch)
                
                # Check if this is a new best model
                if score > self.best_r1:
                    self.best_r1 = score
                    logger.info(f"New Best R@1: {score:.2f} found at Epoch {epoch+1}!")
                    
                    # Save best model immediately (independent of save_frequency)
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_r1': self.best_r1,
                        'config': self.config
                    }
                    
                    # Overwrite best_model.pth (no epoch/score in filename to save disk)
                    best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                    torch.save(checkpoint, best_path)
                    logger.info(f"Saved best model to {best_path}")

            # 3. PERIODIC CHECKPOINT SAVE (Last Model)
            if is_save_time:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_r1': self.best_r1,
                    'config': self.config
                }
                
                last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
                torch.save(checkpoint, last_path)
                logger.info(f"Checkpoint saved: last_model.pth")
