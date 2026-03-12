#!/usr/bin/env python3
"""
Main training script for contrastive learning with dual modalities.

This script trains a contrastive learning model using PyTorch Lightning
on the unified dataset (CubiCasa5k + Structured3D).

Usage (Hydra):
    python scripts/train_contrastive.py
    
Override examples:
    python scripts/train_contrastive.py train.batch_size=64 model.backbone=resnet50 loss.type=infonce2d
"""

import os

from torch._inductor.ir import NoneAsConstantBuffer
# Must be set before importing libraries that use libpng (e.g., Pillow, matplotlib, OpenCV)
os.environ["PNG_IGNORE_WARNINGS"] = "1"

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Tuple

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

from dataloading.unified_dataset import UnifiedDataset
from training.lightning_module import create_lightning_module
from dataloading.dual_transforms import  PairRandomAffine, PairToTensor, PairResize, PairGrayscale, PairNormalize, PairRandomRotation, PairToPIL
from mmfe_utils.data_utils import create_datasets



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
    
    # # Early stopping
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     patience=100,
    #     verbose=True,
    # )
    # callbacks.append(early_stopping)
    
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


# @hydra.main(config_path="../../configs", config_name="train_contrastive_baselines", version_base="1.3")
@hydra.main(config_path="../../configs", config_name="train_contrastive_dino", version_base="1.3")
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

    if cfg.model.backbone.startswith("dino"):
        assert cfg.model.backbone_kwargs.get("dino_weights_path") is not None, "DINO weights path is required"
        cfg.model.backbone_kwargs["dino_weights_path"] = to_absolute_path(cfg.model.backbone_kwargs["dino_weights_path"])

    model_config = {
        "model_type": "dual_modality",
        "backbone_name": cfg.model.backbone,
        "projection_dim": cfg.model.projection_dim,
        "pretrained": True,
        "freeze_backbone": cfg.model.freeze_backbone,
        "projection_head_type": cfg.model.projection_head,
        "projection_spatial": tuple(cfg.model.projection_spatial),
        "backbone_kwargs": cfg.model.backbone_kwargs or {},
    }
    
    # Loss configuration
    loss_config = {
        "loss_type": cfg.loss.type,
        "temperature": cfg.loss.temperature,
        "margin": cfg.loss.margin,
        "reduction": "mean",
        "block_size": cfg.loss.block_size,
        "neg_sampling": cfg.loss.neg_sampling,
        "neg_k": cfg.loss.neg_k,
        "random_per_row": cfg.loss.random_per_row,
    }
    
    # Optimizer configuration
    optimizer_config = {
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
        "betas": (0.9, 0.999),
    }
    
    # Scheduler configuration
    scheduler_config = None
    if cfg.scheduler.use:
        scheduler_config = {
            "T_max": cfg.train.epochs,
            "eta_min": 1e-6,
        }
    
    # Create Lightning module
    print("Creating model...")
    model = create_lightning_module(
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        transform_in_val=True,
        train_equivariant=cfg.train.train_equivariant,
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
        num_nodes=cfg.train.gpus if torch.cuda.is_available() else 0,
        precision="16-mixed" if int(cfg.train.precision) == 16 else "32-true",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        deterministic=cfg.train.deterministic,
        log_every_n_steps=10,
        val_check_interval=0.5 if cfg.loss.type == "infonce2d" else 1.,
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
