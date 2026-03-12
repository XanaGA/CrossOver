#!/usr/bin/env python3
"""
Training script for FPV image encoders.

This script trains a CNN encoder on FPV images so that its features align
with a frozen floorplan encoder trained with contrastive learning.

Usage (Hydra):
    python scripts/training_scripts/train_fpv_img.py

You will need a Hydra config (e.g. configs/train_fpv_img.yaml) that defines:
  - data.aria_synthenv.{path, train, val, image_size, n_fpv_images (number of FPV images to sample)}
  - logging.{output_dir, experiment_name, project_name, save_top_k, wandb}
  - train.{batch_size, epochs, num_workers, seed, precision, accumulate_grad_batches, deterministic, gpus}
  - optim.{lr, weight_decay}
  - fpv.floorplan_checkpoint
  - fpv.image_encoder.{backbone, out_channels, pretrained, freeze_backbone}
"""

import os
from typing import List

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path

from dataloading.aria_se_data import AriaSynthEenvDataset
from dataloading.dual_transforms import PairToPIL, PairResize, PairToTensor, PairNormalize
from fpv.fpv_modules import create_fpv_lightning_module
import torchvision.transforms as T


# Avoid noisy libpng warnings
os.environ["PNG_IGNORE_WARNINGS"] = "1"


def create_callbacks(cfg: DictConfig) -> List:
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.logging.output_dir, "checkpoints"),
        filename=f"{cfg.logging.experiment_name}-{{val_loss:.4f}}-{{epoch:02d}}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.logging.save_top_k,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    if cfg.logging.wandb:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    return callbacks


def create_loggers(cfg: DictConfig) -> List:
    loggers: List = []
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


def _build_aria_dataset(cfg: DictConfig, split: str) -> AriaSynthEenvDataset:
    """
    Build an AriaSynthEenvDataset with FPV images enabled.

    Expects:
      cfg.data.aria_synthenv.path
      cfg.data.aria_synthenv.{train,val}  (text file with scene ids)
      cfg.data.image_size
      cfg.data.aria_synthenv.n_fpv_images
    """
    assert split in ("train", "val")

    root_dir = to_absolute_path(cfg.data.aria_synthenv.path)
    ids_file = getattr(cfg.data.aria_synthenv, split, None)
    ids_file = to_absolute_path(ids_file) if ids_file is not None else None

    # Floorplan / map transforms (shared for both modalities)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    dual_transform = [
        PairToPIL(),
        PairResize(tuple(cfg.data.image_size)),
        PairToTensor(),
        PairNormalize(mean=mean, std=std),
    ]

    # FPV image transforms: convert list of HxWxC uint8 images to
    # (n_fpv, 3, H, W) normalized to ImageNet stats.
    imagenet_mean = mean.view(1, 3, 1, 1)
    imagenet_std = std.view(1, 3, 1, 1)

    def fpv_to_tensor(fpv_images: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for img in fpv_images:
            # img: H x W x C, uint8
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        if len(tensors) == 0:
            return torch.empty(0, 3, 0, 0)
        stacked = torch.stack(tensors, dim=0)  # (N, 3, H, W)
        return (stacked - imagenet_mean) / imagenet_std

    n_fpv = int(getattr(cfg.data.aria_synthenv, "n_fpv_images", 1))

    dataset = AriaSynthEenvDataset(
        root_dir=root_dir,
        scene_ids_file=ids_file,
        image_size=tuple(cfg.data.image_size),
        dual_transform=dual_transform,
        n_fpv_images=n_fpv,
        fpv_transforms=[fpv_to_tensor, T.Normalize(mean=mean, std=std)],
        load_depth= cfg.data.aria_synthenv.use_depth
    )
    return dataset


@hydra.main(config_path="../../configs", config_name="train_fpv_img", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main FPV training entry point (Hydra)."""
    print(OmegaConf.to_yaml(cfg))

    # Seed & output directory
    pl.seed_everything(cfg.train.seed)
    os.makedirs(cfg.logging.output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Datasets & dataloaders
    # ----------------------------------------------------------------------
    print("Creating Aria FPV datasets...")
    train_dataset = _build_aria_dataset(cfg, split="train")
    val_dataset = _build_aria_dataset(cfg, split="val")

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

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    floorplan_ckpt = to_absolute_path(cfg.fpv.floorplan_checkpoint)

    image_encoder_config = {
        "backbone_name": cfg.fpv.image_encoder.backbone,
        "out_channels": cfg.fpv.image_encoder.out_channels,
        "pretrained": cfg.fpv.image_encoder.get("pretrained", True),
        "freeze_backbone": cfg.fpv.image_encoder.get("freeze_backbone", False),
        "backbone_kwargs": cfg.fpv.image_encoder.get("backbone_kwargs", {}) or {},
    }

    depth_pred_config = {
        "n_depth_planes": cfg.fpv.depth_pred.n_depth_planes,
        "depth_min": cfg.fpv.depth_pred.depth_min,
        "depth_max": cfg.fpv.depth_pred.depth_max,
    }

    train_config = {
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
        "epochs": cfg.train.epochs,
        "n_neg_poses": cfg.train.n_neg_poses,
    }

    loss_config = {
        "loss_type": cfg.loss.type,
    }

    print("Creating FPV Lightning module...")
    model = create_fpv_lightning_module(
        floorplan_checkpoint=floorplan_ckpt,
        image_encoder_config=image_encoder_config,
        depth_pred_config=depth_pred_config,
        train_config=train_config,
        loss_config=loss_config,
    )

    # ----------------------------------------------------------------------
    # Trainer / callbacks / loggers
    # ----------------------------------------------------------------------
    callbacks = create_callbacks(cfg)
    loggers = create_loggers(cfg)

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        num_nodes=cfg.train.gpus if torch.cuda.is_available() else 0,
        precision="16-mixed" if int(cfg.train.precision) == 16 else "32-true",
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        deterministic=cfg.train.deterministic,
        log_every_n_steps=10,
        check_val_every_n_epoch=10
    )

    print("Starting FPV training...")
    trainer.fit(model, train_loader, val_loader)
    print("FPV training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()

