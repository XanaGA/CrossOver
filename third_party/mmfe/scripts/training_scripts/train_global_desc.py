#!/usr/bin/env python3
"""
Training script for global descriptor models (e.g., NetVLAD).

This script trains a global descriptor model on top of a backbone using
contrastive learning with InfoNCELoss1D.

Usage (Hydra):
    python scripts/training_scripts/train_global_desc.py
    
Override examples:
    python scripts/training_scripts/train_global_desc.py backbone.name=resnet50 backbone.freeze=true
"""

import os
import sys

from global_descriptors.backbones import GlobalDescriptorBackbone, load_dino_backbone, load_vgg16_backbone
from mmfe_utils.models_utils import load_salad

# Must be set before importing libraries that use libpng
os.environ["PNG_IGNORE_WARNINGS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Tuple, Dict, Any, Optional
import wandb
import subprocess

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import (
    PairRandomAffine, PairToTensor, PairResize, 
    PairGrayscale, PairNormalize, PairRandomRotation, PairToPIL
)

from global_descriptors.global_descriptor_modules import GlobalDescriptorLightningModule
from global_descriptors.global_descriptor_models import create_global_descriptor_model

from mmfe_utils.data_utils import create_datasets

from dotenv import load_dotenv


def create_callbacks(cfg: DictConfig) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging.output_dir, "checkpoints"),
        filename=f"{cfg.logging.experiment_name}-{{val_loss:.4f}}-{{epoch:02d}}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.logging.save_top_k,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if cfg.train.get("early_stopping_patience", 0) > 0:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.train.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping)
    
    if cfg.logging.wandb:
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    
    return callbacks

def create_loggers(cfg: DictConfig) -> list:
    """Create training loggers."""
    loggers = []
    
    if cfg.logging.wandb:
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


@hydra.main(config_path="../../configs", config_name="train_global_desc", version_base="1.3")
def main(cfg: DictConfig):
    """Main training function."""
    # Load environment variables
    load_dotenv()
    # Pretty-print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.train.seed)
    
    # Create output directory
    os.makedirs(cfg.logging.output_dir, exist_ok=True)
    
    # Create datasets
    print("Creating datasets...")
    if cfg.descriptor.type.lower() == "salad":
        if "dinov2" in cfg.backbone.name:
            cfg.data.image_size = (224, 224)
        # elif "dinov3" in cfg.backbone.name:
        #     cfg.data.image_size = (224, 224)#(224, 224)
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
    
    # Create model
    print("Creating model...")
    if cfg.backbone.name.startswith("mmfe") and cfg.descriptor.type.lower() == "salad":
        OmegaConf.set_struct(cfg.backbone.kwargs, False)
        cfg.backbone.kwargs.return_cls = True
    model = create_global_descriptor_model(cfg)
    
    # Create Lightning module
    scheduler_config = None
    if cfg.scheduler.use:
        scheduler_config = {
            "T_max": cfg.train.epochs,
            "eta_min": cfg.scheduler.eta_min,
        }
    
    lightning_module = GlobalDescriptorLightningModule(
        model=model,
        learning_rate=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        loss_config=cfg.loss,
        scheduler_config=scheduler_config,
    )
    
    # Create callbacks
    callbacks = create_callbacks(cfg)
    
    # Create loggers
    loggers = create_loggers(cfg)
    
    # Update config in W&B (must be after wandb_logger is initialized)
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        devices=cfg.train.gpus if torch.cuda.is_available() else 0,
        precision="16-mixed" if int(cfg.train.precision) == 16 else "32-true",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        deterministic=cfg.train.deterministic,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(lightning_module, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

