"""
Lightning modules for FPV image training.

This module wires together:
  - a frozen floorplan encoder (ContrastiveLearningModule) loaded from
    a contrastive-learning checkpoint, and
  - a trainable FPV image encoder that maps RGB images to a feature grid.

For now, the training / validation steps implement a simple placeholder
objective that encourages FPV features to match floorplan features.
You can replace the loss with a more task-specific one later.
"""

from typing import Dict, Any
import os

from numpy import ones_like
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from aria_mmfe.code_snippets.plotters import change_params_resolution
from fpv.fpv_losses import FrustumRegressionLoss, SNAPContrastiveLoss, SimplifiedSNAPLoss
from third_parties.MoGe.modules import MoGe_1_Head
from training.lightning_module import (
    ContrastiveLearningModule,
    load_contrastive_model_from_checkpoint,
)
from fpv.fpv_models import FPVImageEncoder
from fpv.fpv_3D_utils import (FrustumConfig, compute_projected_frustum, expand_neg_fustrums, 
                                transform_fustrums_to_floorplan, sample_random_poses, precompute_frustum_grid)


class FPVLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for FPV image encoder training.

    Args:
        floorplan_checkpoint: Path to a trained contrastive-learning checkpoint
                              (used as frozen floorplan encoder).
        image_encoder_config: Dict with FPVImageEncoder kwargs
                              (e.g. backbone_name, out_channels, pretrained).
        train_config: Dict with training hyperparameters
                      (e.g. lr, weight_decay, epochs).
    """

    def __init__(
        self,
        floorplan_checkpoint: str = None,
        image_encoder_config: Dict[str, Any] = None,
        depth_pred_config: Dict[str, Any] = None,
        train_config: Dict[str, Any] = None,
        loss_config: Dict[str, Any] = None,
        floorplan_module: ContrastiveLearningModule = None,
        floorplan_creation_config: Dict[str, Any] = None,
    ) -> None:
        super().__init__()

        self.floorplan_checkpoint = floorplan_checkpoint
        self.image_encoder_config = image_encoder_config or {}
        self.depth_pred_config = depth_pred_config or {}
        self.train_config = train_config or {}
        self.n_neg_poses = self.train_config.get("n_neg_poses", 10)
        self.loss_config = loss_config or {}
        
        # ------------------------------------------------------------------
        # 1) Load and freeze floorplan encoder
        # ------------------------------------------------------------------
        if floorplan_module is not None:
            # Use provided floorplan module (for loading from checkpoint)
            floorplan_module.eval()
            for p in floorplan_module.parameters():
                p.requires_grad = False
            self.floorplan_module = floorplan_module
            self.floorplan_encoder = floorplan_module.model.encoder
            # Use provided creation config or create from module
            if floorplan_creation_config is None:
                floorplan_creation_config = {
                    "model_config": floorplan_module.model_config,
                    "loss_config": floorplan_module.loss_config,
                    "optimizer_config": floorplan_module.optimizer_config,
                }
        elif floorplan_checkpoint is not None:
            # Load floorplan module from checkpoint
            floorplan_module, floorplan_creation_config = load_contrastive_model_from_checkpoint(
                checkpoint_path=floorplan_checkpoint,
                load_weights=True,
                return_cls=False,
            )
            assert isinstance(
                floorplan_module, ContrastiveLearningModule
            ), "Expected a ContrastiveLearningModule from checkpoint"

            floorplan_module.eval()
            for p in floorplan_module.parameters():
                p.requires_grad = False

            self.floorplan_module = floorplan_module
            self.floorplan_encoder = floorplan_module.model.encoder
        else:
            raise ValueError("Either floorplan_checkpoint or floorplan_module must be provided")
        
        # Store creation information for loading from checkpoint
        # Include floorplan_creation_config so we can recreate the floorplan model
        # without needing the checkpoint path
        self.info_for_loading = {
            "floorplan_checkpoint": floorplan_checkpoint,
            "floorplan_creation_config": floorplan_creation_config,
            "image_encoder_config": self.image_encoder_config,
            "depth_pred_config": self.depth_pred_config,
            "train_config": self.train_config,
            "loss_config": self.loss_config,
            "model_creation_config": {
                "floorplan_checkpoint": floorplan_checkpoint,
                "floorplan_creation_config": floorplan_creation_config,
                "image_encoder_config": self.image_encoder_config,
                "depth_pred_config": self.depth_pred_config,
                "train_config": self.train_config,
                "loss_config": self.loss_config,
            }
        }
        
        # Save hyperparameters for checkpointing (after loading floorplan to include its config)
        self.save_hyperparameters({
            "floorplan_checkpoint": floorplan_checkpoint,
            "image_encoder_config": self.image_encoder_config,
            "depth_pred_config": self.depth_pred_config,
            "train_config": self.train_config,
            "loss_config": self.loss_config,
            "info_for_loading": self.info_for_loading,
        })

        # ------------------------------------------------------------------
        # 2) Create trainable FPV image encoder
        # ------------------------------------------------------------------
        self.image_backbone = self.floorplan_encoder.backbone
        self.image_head = MoGe_1_Head(
                                    num_features=self.floorplan_encoder.n_layers, 
                                    dim_in=self.floorplan_encoder.backbone.blocks[0].attn.qkv.in_features, 
                                    dim_out=[self.floorplan_encoder.projection_dim], 
                                    dim_proj=512,
                                    dim_upsample=[256, 128, 64],
                                    dim_times_res_block_hidden=2,
                                    num_res_blocks=2,
                                    res_block_norm="group_norm",
                                    last_res_blocks=0,
                                    last_conv_channels=2*self.floorplan_encoder.projection_dim,
                                    last_conv_size=1,
                                    projection_spatial=image_encoder_config.get("projection_spatial", [704, 704])
                                )

        self.depth_head = MoGe_1_Head(
                                    num_features=self.floorplan_encoder.n_layers, 
                                    dim_in=self.floorplan_encoder.backbone.blocks[0].attn.qkv.in_features, 
                                    dim_out=[self.depth_pred_config["n_depth_planes"]], # Output channels is the number of depth bins 
                                    dim_proj=512,
                                    dim_upsample=[256, 128, 64],
                                    dim_times_res_block_hidden=2,
                                    num_res_blocks=2,
                                    res_block_norm="group_norm",
                                    last_res_blocks=0,
                                    last_conv_channels=2*self.floorplan_encoder.projection_dim,
                                    last_conv_size=1,
                                    projection_spatial=image_encoder_config.get("projection_spatial", [704, 704])
                                )

        # hidden_dim = 256
        # self.depth_head = nn.Sequential(
        #         nn.Conv2d(self.floorplan_encoder.projection_dim, hidden_dim, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(hidden_dim),
        #         nn.ReLU(inplace=True),
        #         # Output depth-bin logits
        #         nn.Conv2d(hidden_dim, self.depth_pred_config["n_depth_planes"], kernel_size=1),
        #     )

        # Precompute Frustum Coordinates
        self.frustum_coordinates = precompute_frustum_grid(FrustumConfig(), self.device)

        if self.loss_config["loss_type"] == "frustum_regression":
            self.criterion = FrustumRegressionLoss()
        elif self.loss_config["loss_type"] == "frustum_contrastive":
            if self.n_neg_poses > 0:
                self.criterion = SNAPContrastiveLoss()
            else:
                self.criterion = SimplifiedSNAPLoss()

        # Bookkeeping for simple epoch-wise averages
        self.train_losses = []
        self.val_losses = []

        # self.frustum_mlp = nn.Sequential(
        #     nn.Linear(self.image_encoder_config["out_channels"] + 1, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, self.train_config.get("frustum_feature_dim", 128)),
        # )

    def forward(self, fpv_images: torch.Tensor) -> torch.Tensor:
        """
        Encode FPV RGB images into a feature grid.

        Args:
            fpv_images: (B, 3, H, W) tensor.
        """
        raise NotImplementedError("Not implemented")
        x = torch.nn.functional.interpolate(fpv_images, (fpv_images.shape[-2] // 16 * 16, fpv_images.shape[-1] // 16 * 16), mode="bilinear", align_corners=False, antialias=True)
        features = self.image_backbone.get_intermediate_layers(x, n=range(self.floorplan_encoder.n_layers), return_class_token=True)
        embeddings = self.image_head(features, x)[0]
        return embeddings

    def encode_fpv_image(self, image: torch.Tensor, moge_head: str = "image") -> torch.Tensor:
        """
        Encode an image into a feature grid.

        Args:
            image: (B, 3, H, W) tensor.
        """
        x = torch.nn.functional.interpolate(image, (image.shape[-2] // 16 * 16, image.shape[-1] // 16 * 16), mode="bilinear", align_corners=False, antialias=True)
        features = self.image_backbone.get_intermediate_layers(x, n=range(self.floorplan_encoder.n_layers), return_class_token=True)
        embeddings = self.image_head(features, x)[0] if moge_head == "image" else self.depth_head(features, x)[0]
        return embeddings

    # ------------------------------
    # Inference helpers for FPV
    # ------------------------------
    def forward_fpv_images(self, fpv_images: torch.Tensor, gt_depth: torch.Tensor = None):
        """
        Compute projected frustums for a batch of FPV images.
        Mirrors the logic in training/validation steps.

        Args:
            fpv_images: (B, N, 3, H, W) tensor
        Returns:
            List[FrustumData] length B
        """
        B, N, C, H, W = fpv_images.shape
        fpv_images_flat = fpv_images.reshape(B * N, C, H, W)

        # Encode RGB features and depth logits
        fpv_feats_flat = self.encode_fpv_image(fpv_images_flat)
        fpv_depth_flat = self.encode_fpv_image(fpv_images_flat, moge_head="depth")

        fpv_feats = fpv_feats_flat.reshape(B, N, -1, H, W)
        fpv_depth = fpv_depth_flat.reshape(B, N, -1, H, W)

        # Compute frustums
        frustum_data = compute_projected_frustum(
            fpv_depth,
            fpv_feats,
            self.frustum_coordinates,
            FrustumConfig(),
            gt_depth=gt_depth,
        )
        return frustum_data

    def forward_floorplan(self, floorplan: torch.Tensor) -> torch.Tensor:
        """
        Encode floorplan image(s) with the frozen floorplan encoder.

        Args:
            floorplan: (B, 3, H, W)
        Returns:
            (B, C, H', W') feature map
        """
        return self.floorplan_encoder(floorplan)
    # ----------------------------------------------------------------------
    # Training / validation
    # ----------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        floorplan = batch["modality_0"]
        fpv_images = batch["fpv_dict"]["images"].to(self.device)
        image_H, image_W = floorplan.shape[2], floorplan.shape[3]
        fpv_params = batch["fpv_dict"]["params"]

        # 1. Floorplan features (frozen, same as validation)
        with torch.no_grad():
            floorplan_feats = self.floorplan_encoder(floorplan)
            floorplan_feats = floorplan_feats.detach()
            # floorplan_feats = F.interpolate(floorplan_feats, (image_H, image_W), mode="bilinear", align_corners=False)

        fpv_params = change_params_resolution(fpv_params, (floorplan_feats.shape[2], floorplan_feats.shape[3]))

        # 2. Prepare FPV images for batch processing
        B, N, C, H, W = fpv_images.shape
        fpv_images_flat = fpv_images.reshape(B * N, C, H, W)

        # 3. Encode FPV RGB + depth
        fpv_feats_flat = self.encode_fpv_image(fpv_images_flat)
        fpv_depth_flat = self.encode_fpv_image(
            fpv_images_flat, moge_head="depth"
        )

        fpv_feats = fpv_feats_flat.reshape(B, N, -1, H, W)
        fpv_depth = fpv_depth_flat.reshape(B, N, -1, H, W)

        # 4. Compute projected frustums
        frustum_data = compute_projected_frustum(
            fpv_depth,
            fpv_feats,
            self.frustum_coordinates,
            FrustumConfig(),
            gt_depth=batch["fpv_dict"]["depths"] if "depths" in batch["fpv_dict"] else None,
        )

        # 5. Positive poses (ground truth)
        gt_poses = batch["fpv_dict"]["pose_2D_world"]
        gt_poses = torch.cat(
            [gt_poses["xy"], gt_poses["theta"][..., None]], dim=-1
        )  # (B, N, 3)

        pos_fustrums = transform_fustrums_to_floorplan(
            frustum_data, gt_poses, fpv_params
        )

        if self.n_neg_poses > 0:
            # 6. Negative poses (random sampling)
            neg_poses = sample_random_poses(self.n_neg_poses, fpv_params, self.device)
            neg_poses = neg_poses.unsqueeze(1).repeat(1, N, 1, 1)

            frustum_data = expand_neg_fustrums(
                frustum_data, self.n_neg_poses
            )

            neg_fustrums = transform_fustrums_to_floorplan(
                frustum_data, neg_poses, fpv_params
            )

            # 7. Loss
            loss = self.criterion(floorplan_feats, pos_fustrums, neg_fustrums)
        else:
            loss = self.criterion(floorplan_feats, pos_fustrums)

        # 8. Logging
        self.train_losses.append(loss.item())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        floorplan = batch["modality_0"]
        fpv_images = batch["fpv_dict"]["images"].to(self.device)
        image_H, image_W = floorplan.shape[2], floorplan.shape[3]
        fpv_params = batch["fpv_dict"]["params"]

        with torch.no_grad():
            floorplan_feats = self.floorplan_encoder(floorplan)
            # floorplan_feats = F.interpolate(floorplan_feats, (image_H, image_W), mode="bilinear", align_corners=False)

        fpv_params = change_params_resolution(fpv_params, (floorplan_feats.shape[2], floorplan_feats.shape[3]))

        # Prepare the images for batch processing
        B, N, C, H, W = fpv_images.shape
        fpv_images_flat = fpv_images.reshape(B * N, C, H, W)

        fpv_feats_flat = self.encode_fpv_image(fpv_images_flat) 
        fpv_depth_flat = self.encode_fpv_image(fpv_images_flat, moge_head="depth")   

        fpv_feats = fpv_feats_flat.reshape(B, N, -1, H, W)
        fpv_depth = fpv_depth_flat.reshape(B, N, -1, H, W)

        # 2. Compute Frustum with Z-UP & Config
        frustum_data = compute_projected_frustum(fpv_depth, fpv_feats, self.frustum_coordinates, FrustumConfig(), 
                            gt_depth=batch["fpv_dict"]["depths"] if "depths" in batch["fpv_dict"] else None)

        # 3. Transform (Poses are pixels)
        gt_poses = batch["fpv_dict"]["pose_2D_world"]  # Dict
        gt_poses = torch.cat(
            [gt_poses["xy"], gt_poses["theta"][..., None]], dim=-1
        )  # (B, N, 3)

        pos_fustrums = transform_fustrums_to_floorplan(
            frustum_data, gt_poses, fpv_params
        )

        if self.n_neg_poses > 0:
            # 6. Negative poses (random sampling)
            neg_poses = sample_random_poses(self.n_neg_poses, fpv_params, self.device)
            neg_poses = neg_poses.unsqueeze(1).repeat(1, N, 1, 1)

            frustum_data = expand_neg_fustrums(
                frustum_data, self.n_neg_poses
            )

            neg_fustrums = transform_fustrums_to_floorplan(
                frustum_data, neg_poses, fpv_params
            )

            # 7. Loss
            loss = self.criterion(floorplan_feats, pos_fustrums, neg_fustrums)
        else:
            loss = self.criterion(floorplan_feats, pos_fustrums)

        self.val_losses.append(loss.item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # sample_id = batch["sample_id"]
        # print(f"Validating sample: {sample_id} | Loss: {loss}")

        return loss

    def on_train_epoch_end(self) -> None:
        if self.train_losses:
            mean_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("train_loss_epoch", mean_loss, prog_bar=True)
            self.train_losses = []

    def on_validation_epoch_end(self) -> None:
        if self.val_losses:
            mean_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("val_loss_epoch", mean_loss, prog_bar=True)
            self.val_losses = []

    # ----------------------------------------------------------------------
    # Optimizer / scheduler
    # ----------------------------------------------------------------------
    def configure_optimizers(self):
        """
        Only optimize the FPV image encoder parameters.
        """
        params = [p for p in self.image_head.parameters() if p.requires_grad]

        lr = self.train_config.get("lr", 1e-4)
        weight_decay = self.train_config.get("weight_decay", 1e-4)
        epochs = self.train_config.get("epochs", 100)

        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


def create_fpv_lightning_module(
    floorplan_checkpoint: str,
    image_encoder_config: Dict[str, Any],
    depth_pred_config: Dict[str, Any],
    train_config: Dict[str, Any],   
    loss_config: Dict[str, Any],
) -> FPVLightningModule:
    """
    Factory function to create an FPVLightningModule.
    """
    return FPVLightningModule(
        floorplan_checkpoint=floorplan_checkpoint,
        image_encoder_config=image_encoder_config,
        depth_pred_config=depth_pred_config,
        train_config=train_config,
        loss_config=loss_config,
    )


def load_fpv_model_from_checkpoint(
    checkpoint_path: str,
    load_weights: bool = True,
) -> FPVLightningModule:
    """
    Load an FPV model from checkpoint.
    
    This function creates the model architecture first using the stored creation
    configuration, then loads the weights. This avoids issues with hardcoded paths
    in the checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        load_weights: Whether to load weights from the checkpoint (default: True)
        
    Returns:
        Loaded FPVLightningModule in eval mode
    """
    from omegaconf import OmegaConf
    from hydra.utils import to_absolute_path
    
    print(f"Loading FPV model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    state_dict = checkpoint['state_dict']
    
    # Try to get model creation config from checkpoint
    model_creation_config = checkpoint["hyper_parameters"].get('info_for_loading', {}).get('model_creation_config', None)
    
    if model_creation_config is not None:
        print("Found model creation config in checkpoint")
        # Use the stored creation config to recreate the model
        floorplan_checkpoint = model_creation_config.get("floorplan_checkpoint")
        floorplan_creation_config = model_creation_config.get("floorplan_creation_config")
        image_encoder_config = model_creation_config.get("image_encoder_config")
        depth_pred_config = model_creation_config.get("depth_pred_config")
        train_config = model_creation_config.get("train_config")
        loss_config = model_creation_config.get("loss_config")
        
        # Create floorplan module from creation config if available (avoids needing checkpoint path)
        floorplan_module = None
        if floorplan_creation_config is not None:
            print("Creating floorplan module from stored creation config (no checkpoint path needed)")
            floorplan_module = ContrastiveLearningModule(
                model_config=floorplan_creation_config.get("model_config", {}),
                loss_config=floorplan_creation_config.get("loss_config", {}),
                optimizer_config=floorplan_creation_config.get("optimizer_config", {}),
                load_dino_weights=False,  # We'll load weights from FPV checkpoint
            )
        elif floorplan_checkpoint:
            # Fallback: try to load from checkpoint path if creation config not available
            print("WARNING: floorplan_creation_config not found, attempting to load from checkpoint path")
            # Try to resolve floorplan checkpoint path if it doesn't exist
            if not os.path.exists(floorplan_checkpoint):
                # Try to resolve using to_absolute_path if it's a relative path
                try:
                    floorplan_checkpoint = to_absolute_path(floorplan_checkpoint)
                except:
                    pass  # If it still doesn't exist, let it fail during model creation
            
            floorplan_module, _ = load_contrastive_model_from_checkpoint(
                checkpoint_path=floorplan_checkpoint,
                load_weights=True,
                return_cls=False,
            )
        
        if floorplan_module is None:
            raise ValueError("Could not create floorplan module. Need either floorplan_creation_config or valid floorplan_checkpoint.")
        
        # Create the FPV model using the stored config and pre-created floorplan module
        model = FPVLightningModule(
            floorplan_checkpoint=floorplan_checkpoint,
            image_encoder_config=image_encoder_config,
            depth_pred_config=depth_pred_config,
            train_config=train_config,
            loss_config=loss_config,
            floorplan_module=floorplan_module,
            floorplan_creation_config=floorplan_creation_config,
        )
    else:
        print("WARNING: Model creation config not found in checkpoint. Resorting to legacy loading")
        # Fallback to extracting from hyper_parameters directly
        hparams = checkpoint["hyper_parameters"]
        floorplan_checkpoint = hparams.get("floorplan_checkpoint")
        image_encoder_config = hparams.get("image_encoder_config")
        depth_pred_config = hparams.get("depth_pred_config")
        train_config = hparams.get("train_config")
        loss_config = hparams.get("loss_config")
        
        if floorplan_checkpoint is None:
            raise ValueError("Could not find floorplan_checkpoint in checkpoint. Cannot recreate model.")
        
        # Try to resolve floorplan checkpoint path if it doesn't exist
        if not os.path.exists(floorplan_checkpoint):
            # Try to resolve using to_absolute_path if it's a relative path
            try:
                floorplan_checkpoint = to_absolute_path(floorplan_checkpoint)
            except:
                pass  # If it still doesn't exist, let it fail during model creation
        
        # Create the model
        model = FPVLightningModule(
            floorplan_checkpoint=floorplan_checkpoint,
            image_encoder_config=image_encoder_config or {},
            depth_pred_config=depth_pred_config or {},
            train_config=train_config or {},
            loss_config=loss_config or {},
        )
    
    # Load state dict if requested
    if load_weights:
        load_result = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)
    
    # Set to eval mode
    model.eval()
    
    print("✓ FPV model loaded successfully")
    return model


__all__ = ["FPVLightningModule", "create_fpv_lightning_module", "load_fpv_model_from_checkpoint"]

