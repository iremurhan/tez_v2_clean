import argparse
import yaml
import torch
import logging
import wandb
import sys
import os
from transformers import CLIPTokenizer, get_cosine_schedule_with_warmup

from src.data import get_dataloader
from src.model import DualEncoder
from src.loss import RetrievalLoss
from src.trainer import Trainer
from src.utils import setup_seed

def setup_logging(save_dir):
    """
    Configures logging to split standard info and errors.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Root logger
    logger = logging.getLogger()
    # Clear existing handlers if any to avoid duplicate logs during restarts or interactive sessions
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Train Log - Everything)
    file_handler = logging.FileHandler(os.path.join(save_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. Error Handler (Errors Only - Delay creation)
    err_handler = logging.FileHandler(os.path.join(save_dir, "errors.log"), delay=True)
    err_handler.setLevel(logging.WARNING)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)
    
    return logger

def get_optimizer(model, config):
    """
    Factory function to create optimizer with parameter groups for CLIP.
    
    CLIP Parameter Groups:
    1. Backbone (vision_model, text_model) - Very low LR or frozen (0)
    2. CLIP Projections (visual_projection, text_projection) - Low LR
    3. Custom Projection Heads (image_proj, text_proj) - Higher LR
    """
    opt_name = config['training']['optimizer'].lower()
    wd = float(config['training']['weight_decay'])
    
    # Learning Rates from Config
    lr_clip_proj = float(config['training'].get('clip_projection_lr', 1e-5))
    lr_head = float(config['training'].get('head_lr', 1e-3))
    # Optional: If backbone is unfrozen later
    lr_backbone = float(config['training'].get('backbone_lr', 1e-6))
    
    # Get only trainable (requires_grad=True) parameters
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    
    # Define Groups
    clip_proj_params = []   # CLIP's original projections
    custom_head_params = [] # Your custom Linear layers (if any)
    backbone_params = []    # If backbone is unfrozen, falls here
    
    for n, p in trainable_params:
        # 1. CLIP Projection Layers (visual_projection, text_projection)
        if 'visual_projection' in n or 'text_projection' in n:
            clip_proj_params.append(p)
        # 2. Custom Heads (image_proj, text_proj - defined in model.py)
        elif 'image_proj' in n or 'text_proj' in n:
            custom_head_params.append(p)
        # 3. Backbone (everything else)
        else:
            backbone_params.append(p)
            
    # Build Optimizer Groups
    optimizer_grouped_parameters = [
        {'params': clip_proj_params, 'lr': lr_clip_proj},    # Group 1: CLIP Projections
        {'params': custom_head_params, 'lr': lr_head},       # Group 2: Custom Heads
        {'params': backbone_params, 'lr': lr_backbone}       # Group 3: Backbone (if unfrozen)
    ]
    
    # Log info (useful for debugging)
    print(f"Optimizer Groups Created:")
    print(f"  - CLIP Projections: {len(clip_proj_params)} tensors (LR: {lr_clip_proj})")
    print(f"  - Custom Heads:     {len(custom_head_params)} tensors (LR: {lr_head})")
    print(f"  - Backbone/Other:   {len(backbone_params)} tensors (LR: {lr_backbone})")

    if opt_name == 'adamw':
        return torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=wd)
    elif opt_name == 'adam':
        return torch.optim.Adam(optimizer_grouped_parameters, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def get_scheduler(optimizer, config, num_training_steps):
    """
    Factory function to create scheduler based on config.
    Now supports Linear Warmup + Cosine Decay from Transformers.
    """
    sched_name = config['training']['scheduler'].lower()
    epochs = config['training']['epochs']
    
    if sched_name == 'cosine':
        # Standard Transformer Scheduler: Linear Warmup + Cosine Decay
        # Warmup is usually 10% of training steps
        num_warmup_steps = int(0.1 * num_training_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
    else:
        # Default or fallback
        logging.warning(f"Scheduler {sched_name} not explicitly handled, defaulting to StepLR or None")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=0.1) 
        
    return scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    
    # Config override arguments
    # Allows overriding config values using dot notation (e.g., --override logging.checkpoint_dir=/tmp)
    parser.add_argument('--override', nargs='+', help='Override config params (key=value)', default=[])
    
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override Logic
    for item in args.override:
        if '=' not in item:
            logging.warning(f"Skipping invalid override format (missing '='): {item}")
            continue
        key, value = item.split('=', 1)
        
        # Parse nested keys (e.g., training.lr) and update config
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        
        # Attempt to preserve value type (int, float, bool)
        original_value = value
        try:
            # Simple type conversion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except (ValueError, AttributeError):
            # Keep as string if conversion fails
            value = original_value
            
        current[keys[-1]] = value
        logging.info(f"Config Override: {key} -> {value}")
    
    # 2. Setup Logging
    log_dir = config['logging']['checkpoint_dir']
    logger = setup_logging(log_dir)
    
    # 3. Debug Mode Check
    debug_mode = config['debug']['debug_mode']
    if debug_mode:
        logger.info("DEBUG MODE ENABLED: Disabling W&B sync, using subset.")
    
    # 4. Setup Seed & Device
    setup_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 5. W&B Init
    use_wandb = config['logging']['use_wandb'] and not debug_mode
    if use_wandb:
        try:
            wandb.init(project=config['logging']['wandb_project'], config=config)
            logger.info(f"W&B initialized with project: {config['logging']['wandb_project']}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. Continuing without W&B.")
            use_wandb = False

    # 5. Data Loaders
    logger.info("Initializing Data Loaders...")
    
    # CLIP Tokenizer (different from DistilBERT!)
    clip_model_name = config['model']['image_model_name']
    if not clip_model_name:
        raise ValueError("config['model']['image_model_name'] must be specified in config file.")
    logger.info(f"Loading CLIP Tokenizer from: {clip_model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    train_loader = get_dataloader(config, tokenizer, split='train')
    
    # Strict Validation Logic
    if debug_mode:
        logger.warning("Debug mode: Using train set as validation set.")
        val_loader = train_loader
    else:
        # In production/full run, failure to load validation is critical
        logger.info("Loading Validation Set...")
        val_loader = get_dataloader(config, tokenizer, split='val')

    # 6. Model & Loss
    logger.info("Building Model...")
    model = DualEncoder(config).to(device)
    criterion = RetrievalLoss(config).to(device)

    # 7. Optimizer & Scheduler (Factory)
    optimizer = get_optimizer(model, config)
    
    # Calculate training steps for scheduler
    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_scheduler(optimizer, config, num_training_steps)

    # 8. Trainer
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, use_wandb=use_wandb
    )
    
    # 9. Resume Logic
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume)
        else:
            logger.error(f"Checkpoint not found at {args.resume}")
            raise FileNotFoundError(f"Checkpoint {args.resume} does not exist")
    
    # 10. Start Training
    logger.info(f"Starting training from epoch {start_epoch}...")
    try:
        trainer.fit(start_epoch=start_epoch)
        
        # --- 11. FINAL TEST EVALUATION ---
        logger.info("Training finished. Loading best model for Test evaluation...")
        
        # Load Best Model
        best_model_path = os.path.join(config['logging']['checkpoint_dir'], "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Best model loaded successfully.")
            
            # Prepare Test Loader
            try:
                test_loader = get_dataloader(config, tokenizer, split='test')
                trainer.val_loader = test_loader
                logger.info("Running evaluation on TEST set...")
                trainer.evaluate(epoch="TEST_FINAL")
            except Exception as e:
                logger.warning(f"Could not run test evaluation (Test data missing?): {e}")
        else:
            logger.warning("Best model checkpoint not found, skipping test evaluation.")

    except KeyboardInterrupt:
        logger.info("Training interrupted manually.")
        if use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B: {e}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise e
    finally:
        if use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B: {e}")

if __name__ == "__main__":
    main()
