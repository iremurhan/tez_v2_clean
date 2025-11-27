import argparse
import yaml
import torch
import logging
import wandb
import sys
import os
from transformers import AutoTokenizer

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
    Factory function to create optimizer based on config.
    """
    opt_name = config['training']['optimizer'].lower()
    lr_backbone = float(config['training']['text_encoder_lr'])
    lr_head = float(config['training']['head_lr'])
    weight_decay = float(config['training']['weight_decay'])
    
    # Separate parameters
    param_optimizer = list(model.named_parameters())
    
    # Filter out frozen parameters if any (though usually we train all)
    optimizer_grouped_parameters = [
        # Text Backbone (Low LR)
        {'params': [p for n, p in param_optimizer if 'text_backbone' in n], 
         'lr': lr_backbone},
        # Heads & Image Proj (High LR)
        {'params': [p for n, p in param_optimizer if 'text_backbone' not in n], 
         'lr': lr_head}
    ]
    
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
        
    return optimizer

def get_scheduler(optimizer, config):
    """
    Factory function to create scheduler based on config.
    """
    sched_name = config['training']['scheduler'].lower()
    epochs = config['training']['epochs']
    
    if sched_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
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
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_name'])
    
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
    scheduler = get_scheduler(optimizer, config)

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
