import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Tuple, List

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

# Imports from your environment
from mmfe_utils.data_utils import create_datasets
# from dataloading.unified_dataset import UnifiedDataset # Uncomment if implicit registration is needed

from roma.roma_pl_module import RoMaFineTuner

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

def create_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing
    # Using .get() with defaults to ensure robustness if config keys miss strict typing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging.output_dir, "checkpoints"),
        filename=f"{cfg.logging.experiment_name}-{{val_loss:.4f}}-{{epoch:02d}}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.logging.get("save_top_k", 3),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping (Optional based on your sample)
    if cfg.train.get("early_stopping", False):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.train.get("patience", 100),
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    if cfg.logging.get("wandb", False):
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    return callbacks


def create_loggers(cfg: DictConfig) -> List[pl.loggers.Logger]:
    """Create training loggers."""
    loggers = []
    
    if cfg.logging.get("wandb", False):
        try:
            wandb_logger = WandbLogger(
                project=cfg.logging.project_name,
                name=cfg.logging.experiment_name,
                log_model=True,
                save_dir=cfg.logging.output_dir,
            )
            loggers.append(wandb_logger)
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
    
    return loggers


@hydra.main(config_path="../../configs", config_name="train_romav2", version_base="1.3")
def main(cfg: DictConfig):
    """Main training function (Hydra)."""
    # Pretty-print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.train.seed)
    
    # Create output directory
    os.makedirs(cfg.logging.output_dir, exist_ok=True)
    
    # Create datasets
    print("Creating datasets...")
    # Using your custom dataset creator
    train_dataset, val_dataset = create_datasets(cfg)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Create Lightning module (RoMaFineTuner)
    print("Creating model...")
    # Mapping optimizer config from your unified config structure to the module
    model = RoMaFineTuner(
        matcher_name=cfg.model.matcher_name,
        use_pretrained=cfg.model.use_pretrained,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        backbone_kwargs = cfg.model.backbone
    )
    
    model.to(cfg.train.device)
    
    # Create callbacks
    callbacks = create_callbacks(cfg)
    
    # Create loggers
    loggers = create_loggers(cfg)

    # Update config in W&B (must be after wandb_logger is initialized)
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    # GPU handling
    gpus = cfg.train.gpus if torch.cuda.is_available() else 0
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=gpus if isinstance(gpus, int) else 'auto', 
        precision="16-mixed" if int(cfg.train.get("precision", 32)) == 16 else "32-true",
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=loggers,
        deterministic=cfg.train.get("deterministic", False),
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        # Default strategy usually works, but DDP is standard for multi-gpu
        strategy="auto" 
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, 'best_model_path'):
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()