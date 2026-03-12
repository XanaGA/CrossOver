"""
PyTorch Lightning module for steerer training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


from steerers.steerers import DiscreteSteerer
from steerers.descriptor_loss import DescriptorLoss
from steerers.dummy_detector import DummyDetector
from training.lightning_module import ContrastiveLearningModule
from steerers.steerers_utils import dict_to_device


class DescriptorAdapter(torch.nn.Module):
    """Adapter to use contrastive learning module as descriptor for steerer training."""
    
    def __init__(self, descriptor_module: ContrastiveLearningModule):
        super().__init__()
        self.descriptor = descriptor_module
        self.descriptor.eval()
        for p in self.descriptor.parameters():
            p.requires_grad = False

    def forward(self, batch: dict):
        # Expect batch to contain modality_0 and modality_1
        modality_0 = batch["modality_0"]
        modality_1 = batch["modality_1"]
        with torch.no_grad():
            emb0, emb1 = self.descriptor(modality_0, modality_1)
        # Pack into the structure expected by DescriptorLoss
        # Concatenate along batch dim -> [2B, C, H, W]
        description_grid = torch.cat([emb0, emb1], dim=0)
        return {"description_grid": description_grid}


class SteererLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for steerer training.
    
    Args:
        descriptor_checkpoint: Path to the trained descriptor checkpoint
        steerer_config: Configuration for the steerer
        train_config: Configuration for training
    """
    
    def __init__(
        self,
        descriptor_checkpoint: str,
        steerer_config: Dict[str, Any],
        train_config: Dict[str, Any],
    ):
        super().__init__()
        
        # Save configurations
        self.descriptor_checkpoint = descriptor_checkpoint
        self.steerer_config = steerer_config
        self.train_config = train_config
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Load descriptor model
        self.descriptor = ContrastiveLearningModule.load_from_checkpoint(
            checkpoint_path=descriptor_checkpoint,
            map_location="cpu",
            load_dino_weights=False,
        )
        self.descriptor_adapter = DescriptorAdapter(self.descriptor)
        
        # Create steerer
        NUM_PROTOTYPES = 32  # == descriptor size
        generator = torch.nn.Linear(
            in_features=NUM_PROTOTYPES,
            out_features=NUM_PROTOTYPES,
            bias=False
        )
        self.steerer = DiscreteSteerer(generator.weight.data)
        
        # Create detector and loss
        self.detector = DummyDetector()
        self.loss_fn = DescriptorLoss(
            detector=self.detector,
            normalize_descriptions=False,
            inv_temp=20,
            num_keypoints=train_config.get("num_keypoints", 1000)
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, batch: dict):
        """Forward pass through the descriptor adapter."""
        return self.descriptor_adapter(batch)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with rotation augmentation."""
        batch = dict_to_device(batch)
        
        # Apply rotation augmentation if generator_rot is specified
        generator_rot = self.train_config.get("generator_rot")
        if generator_rot is not None:
            batch = self._apply_rotation_augmentation(batch, generator_rot)
        
        if self.train_config.get("debug"):
            # Show image A and Image B with the rotation augmentation side by side using matplotlib
            im_A = batch["im_A"][0].permute(1, 2, 0).cpu().numpy()
            im_B = batch["im_B"][0].permute(1, 2, 0).cpu().numpy()
            top = np.concatenate([im_A, im_B], axis=1)
            plt.figure(figsize=(8, 4))
            plt.imshow(top)
            plt.title(f"Image A {batch['rot_deg_A']} | Image B {batch['rot_deg_B']}")
            plt.axis('off')
            plt.show()
        
        # Forward pass
        out = self.forward(batch)
        
        # Compute loss
        if generator_rot is not None:
            # Extract rotation info from batch
            nbr_rot_A = batch.get("nbr_rot_A", 0)
            nbr_rot_B = batch.get("nbr_rot_B", 0)
            loss = self.loss_fn(out, batch, nbr_rot_A, nbr_rot_B, steerer=self.steerer, debug=self.train_config.get("debug"))
        else:
            loss = self.loss_fn(out, batch)
        
        # Log metrics
        self.train_losses.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        batch = dict_to_device(batch)
        
        # Forward pass
        out = self.forward(batch)
        
        # Compute loss
        loss = self.loss_fn(out, batch)
        
        # Log metrics
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_losses:
            train_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("train_loss_epoch", train_loss, on_epoch=True)
            self.train_losses = []
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_losses:
            val_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("val_loss_epoch", val_loss, on_epoch=True)
            self.val_losses = []
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Only optimize steerer parameters
        params = [
            {"params": self.steerer.parameters(), "lr": self.train_config.get("lr", 1e-3)},
        ]
        
        optimizer = AdamW(params, weight_decay=0)
        
        # Create scheduler
        epochs = self.train_config.get("epochs", 100)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
    
    def _apply_rotation_augmentation(self, batch: dict, generator_rot: int) -> dict:
        """Apply rotation augmentation to the batch."""
        import torchvision.transforms.functional as TTF
        from dataloading.inversible_tf import make_valid_mask
        from mmfe_utils.tensor_utils import torch_erode
        
        # Generate random rotations
        nbr_rot_A = random.randint(0, (360 // generator_rot) - 1)
        nbr_rot_B = random.randint(0, (360 // generator_rot) - 1)
        while nbr_rot_A == nbr_rot_B:
            nbr_rot_B = random.randint(0, (360 // generator_rot) - 1)
        
        rot_deg_A = (nbr_rot_A * generator_rot) % 360
        rot_deg_B = (nbr_rot_B * generator_rot) % 360
        
        # Apply rotations
        batch["im_A"] = TTF.rotate(
            batch["modality_0"],
            rot_deg_A,
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        batch["im_B"] = TTF.rotate(
            batch["modality_1"],
            rot_deg_B,
            interpolation=TTF.InterpolationMode.BILINEAR,
        )
        
        # Create valid masks
        valid_mask_A = make_valid_mask(
            {"angle": rot_deg_A, "translate": [0, 0], "scale": 1, "shear": 0, 
             "image_size": batch["im_A"].shape[-2:]}, 
            device=batch["im_A"].device, 
            dtype=batch["im_A"].dtype
        )
        valid_mask_B = make_valid_mask(
            {"angle": rot_deg_B, "translate": [0, 0], "scale": 1, "shear": 0, 
             "image_size": batch["im_B"].shape[-2:]}, 
            device=batch["im_B"].device, 
            dtype=batch["im_B"].dtype
        )
        
        # Erode masks
        valid_mask_A = torch_erode(valid_mask_A, kernel_size=3, iterations=1)
        valid_mask_B = torch_erode(valid_mask_B, kernel_size=3, iterations=1)
        
        # Apply masks
        batch["im_A"] = torch.where(~valid_mask_A.bool(), 1, batch["im_A"])
        batch["im_B"] = torch.where(~valid_mask_B.bool(), 1, batch["im_B"])
        
        # Store rotation info
        batch["rot_deg_A"] = rot_deg_A
        batch["rot_deg_B"] = rot_deg_B
        batch["rot_deg_A_to_B"] = rot_deg_B - rot_deg_A
        batch["generator_rot"] = generator_rot
        batch["nbr_rot_A"] = nbr_rot_A
        batch["nbr_rot_B"] = nbr_rot_B
        
        return batch


def create_steerer_lightning_module(
    descriptor_checkpoint: str,
    steerer_config: Dict[str, Any],
    train_config: Dict[str, Any],
) -> SteererLightningModule:
    """
    Factory function to create a steerer Lightning module.
    
    Args:
        descriptor_checkpoint: Path to the trained descriptor checkpoint
        steerer_config: Configuration for the steerer
        train_config: Configuration for training
        
    Returns:
        Configured Lightning module
    """
    return SteererLightningModule(
        descriptor_checkpoint=descriptor_checkpoint,
        steerer_config=steerer_config,
        train_config=train_config,
    )
