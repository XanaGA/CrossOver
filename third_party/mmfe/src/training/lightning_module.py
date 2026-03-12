"""
PyTorch Lightning module for contrastive learning training.
"""

from ast import List
import random
import omegaconf
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple
import wandb
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataloading.dual_transforms import PairNormalize, PairRandomAffine
from dataloading.inversible_tf import warp_feature_map, warp_feature_map_batch

from .models import create_model
from .losses import create_loss
from mmfe_utils.tensor_utils import tensor_to_numpy, tensor_to_numpy_image
from mmfe_utils.viz_utils import get_color_map, show_equivariance_debug


class ContrastiveLearningModule(pl.LightningModule):
    """
    PyTorch Lightning module for contrastive learning.
    
    Args:
        model_config: Configuration for the model
        loss_config: Configuration for the loss function
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the learning rate scheduler
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        transform_in_val: Optional[bool] = False,
        train_equivariant: Optional[bool] = False,
        load_dino_weights: bool = True,
    ):
        super().__init__()
        
        # Save configurations
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config or {}
        self.transform_in_val = transform_in_val
        self.train_equivariant = train_equivariant
        self.load_dino_weights = load_dino_weights
        
        # Create model
        self.model = create_model(**model_config, load_dino_weights=self.load_dino_weights)
        
        # Create loss function
        self.loss_fn = create_loss(**loss_config)
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Training metrics - use simple averaging instead of deprecated pl.metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        if self.loss_config["loss_type"] == "infonce2d":
            self.val_accuracies_self = []
            self.val_accuracies_others = []
        
        # Store validation batch for logging (only first batch of each epoch)
        self.val_batch_for_logging = None
        
    def forward(self, modality_0: torch.Tensor, modality_1: torch.Tensor = None, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(modality_0, modality_1, normalize=normalize)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract data
        modality_0 = batch["modality_0"]
        modality_1 = batch["modality_1"]
        
        # Forward pass
        embeddings_0, embeddings_1 = self.forward(modality_0, modality_1)
        loss = self.loss_fn(embeddings_0, embeddings_1)
        
        # Log metrics
        self.train_losses.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.train_equivariant:

            # Compute & log cross-equivariant loss
            modality_0_noise = batch["modality_0_noise"]
            modality_1_noise = batch["modality_1_noise"]
            embeddings_0_noise, embeddings_1_noise = self.forward(modality_0_noise, modality_1_noise)
            wrapped_embeddings_0_to_og, warp_mask_0 = warp_feature_map_batch(
                    embeddings_0_noise, batch["noise_params"], image_size=batch["noise_params"]['image_size'], 
                    align_corners=False, 
                    og_valid_mask=batch["noise_params"]['valid_mask']*batch["transform_params"]['valid_mask'], 
                    return_mask=True
                )
            wrapped_embeddings_1_to_og, warp_mask_1 = warp_feature_map_batch(
                    embeddings_1_noise, batch["noise_params"], image_size=batch["noise_params"]['image_size'], 
                    align_corners=False, 
                    og_valid_mask=batch["noise_params"]['valid_mask']*batch["transform_params"]['valid_mask'], 
                    return_mask=True
                )

            # # DEBUG SHOW (uncomment to visualize equivariance)
            # show_equivariance_debug(
            #     modality_0=modality_0,
            #     modality_1=modality_1,
            #     modality_0_noise=modality_0_noise,
            #     modality_1_noise=modality_1_noise,
            #     warp_mask_0=warp_mask_0,
            #     warp_mask_1=warp_mask_1,
            #     noise_params=batch["noise_params"],
            #     align_corners=False,
            #     wait_key=0,
            # )

            warp_mask_0 = F.resize(warp_mask_0, size=wrapped_embeddings_0_to_og.shape[-2:], interpolation=F.InterpolationMode.NEAREST)
            warp_mask_1 = F.resize(warp_mask_1, size=embeddings_1_noise.shape[-2:], interpolation=F.InterpolationMode.NEAREST)
                
            loss0 = self.loss_fn(embeddings_0, wrapped_embeddings_0_to_og, warp_mask_0)
            loss1 = self.loss_fn(embeddings_1, wrapped_embeddings_1_to_og, warp_mask_1)
            loss2 = self.loss_fn(embeddings_0, wrapped_embeddings_1_to_og, warp_mask_1)
            loss3 = self.loss_fn(embeddings_1, wrapped_embeddings_0_to_og, warp_mask_0)
            loss = loss0 + loss1 + loss2 + loss3
            self.train_losses.append(loss.item())
            self.log("train_equivariant_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Extract data
        modality_0 = batch["modality_0"]
        modality_1 = batch["modality_1"]
        
        # Forward pass
        modality_0_mod = modality_0
        modality_1_mod = modality_1
        modality_0_mod_noise = batch["modality_0_noise"]
        modality_1_mod_noise = batch["modality_1_noise"]
        params_noise = batch["noise_params"]

        embeddings_0_mod, embeddings_1_mod = self.forward(modality_0_mod, modality_1_mod)
        embeddings_0_mod_noise, embeddings_1_mod_noise = self.forward(modality_0_mod_noise, modality_1_mod_noise)

        # Compute & log self-equivariant loss
        wrapped_embeddings_0_noise, warp_mask_0 = warp_feature_map_batch(
                embeddings_0_mod_noise, params_noise, image_size=params_noise["image_size"], 
                align_corners=False, 
                og_valid_mask=params_noise["valid_mask"]*batch["transform_params"]['valid_mask'], 
                return_mask=True
            )
        wrapped_embeddings_1_noise, warp_mask_1 = warp_feature_map_batch(
                embeddings_1_mod_noise, params_noise, image_size=params_noise["image_size"], 
                align_corners=False, 
                og_valid_mask=params_noise["valid_mask"]*batch["transform_params"]['valid_mask'], 
                return_mask=True
            )

        warp_mask_0 = F.resize(warp_mask_0, size=embeddings_0_mod.shape[-2:], interpolation=F.InterpolationMode.NEAREST)
        warp_mask_1 = F.resize(warp_mask_1, size=embeddings_1_mod.shape[-2:], interpolation=F.InterpolationMode.NEAREST)
        # warp_mask_0 = warp_mask_0.flatten()
        # warp_mask_1 = warp_mask_1.flatten()

        loss0 = self.loss_fn(embeddings_0_mod, wrapped_embeddings_0_noise, warp_mask_0)
        loss1 = self.loss_fn(embeddings_1_mod, wrapped_embeddings_1_noise, warp_mask_1)
        loss = loss0 + loss1
        self.val_losses.append(loss.item())
        self.log("val_self_equivariant_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute & log cross-equivariant loss
        loss0 = self.loss_fn(embeddings_0_mod, wrapped_embeddings_1_noise, warp_mask_1)
        loss1 = self.loss_fn(embeddings_1_mod, wrapped_embeddings_0_noise, warp_mask_0)
        loss = loss0 + loss1
        self.val_losses.append(loss.item())
        self.log("val_cross_equivariant_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


        # Forward pass
        embeddings_0, embeddings_1 = self.forward(modality_0, modality_1)
        
        # Compute & log loss
        loss = self.loss_fn(embeddings_0, embeddings_1)
        self.val_losses.append(loss.item())
        self.log("val_loss_canonical", loss, on_step=False, on_epoch=True, prog_bar=True)

        loss = self.loss_fn(embeddings_0_mod, embeddings_1_mod)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute & log retrieval accuracy
        mask_0_resized = F.resize(
            batch["transform_params"]['valid_mask'],
            size=embeddings_1_mod_noise.shape[-2:],
            interpolation=F.InterpolationMode.NEAREST,
        )

        # Make the distance threshold resolution-independent: keep a constant
        # distance in input-image pixels (3px for 256x256), and convert it to
        # feature-space units based on the current embedding resolution.
        _, _, img_h, img_w = modality_0.shape
        _, _, feat_h, feat_w = embeddings_0_mod.shape
        # Use horizontal scaling (assumes approximately square images/features)
        pixels_per_feat = img_w / float(feat_w) if feat_w > 0 else 1.0
        base_image_pixel_threshold = 8 * 3  # number of pixels in the original image
        distance_th_feat = base_image_pixel_threshold / pixels_per_feat

        accuracy = self.compute_retrieval_accuracy2D(
            embeddings_0_mod,
            embeddings_1_mod,
            sample_percentage=0.3,
            topk=3,
            distance_th=distance_th_feat,
            valid_mask=mask_0_resized.bool(),
        )

        self.val_accuracies.append(accuracy["acc"].item())
        self.val_accuracies_self.append(accuracy["acc_self"].item())
        self.val_accuracies_others.append(accuracy["acc_others"].item())
        self.log("val_accuracy_all", accuracy["acc"].item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy_self", accuracy["acc_self"].item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy_others", accuracy["acc_others"].item(), on_step=False, on_epoch=True, prog_bar=True)

        # Store first batch for logging (only log once per epoch)
        if batch_idx == 0:
            self.val_batch_for_logging = batch
            self.val_embeddings = (embeddings_0, embeddings_1)
            self.val_embeddings_mod = (embeddings_0_mod, embeddings_1_mod)
            self.val_embeddings_noise = (embeddings_0_mod_noise, embeddings_1_mod_noise)
            self.val_embeddings_wrapped_noise = (wrapped_embeddings_0_noise, wrapped_embeddings_1_noise)

        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log epoch metrics
        if self.train_losses:
            train_loss = sum(self.train_losses) / len(self.train_losses)
            self.log("train_loss_epoch", train_loss, on_epoch=True)
            self.train_losses = []
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Log epoch metrics
        if self.val_losses:
            val_loss = sum(self.val_losses) / len(self.val_losses)
            self.log("val_loss_epoch", val_loss, on_epoch=True)
            self.val_losses = []
        
        if self.val_accuracies:
            val_accuracy = sum(self.val_accuracies) / len(self.val_accuracies)
            self.log("val_accuracy_epoch", val_accuracy, on_epoch=True)
            self.val_accuracies = []

        if hasattr(self, 'val_accuracies_self') and self.val_accuracies_self:
            val_accuracy_self = sum(self.val_accuracies_self) / len(self.val_accuracies_self)
            self.log("val_accuracy_self_epoch", val_accuracy_self, on_epoch=True)
            self.val_accuracies_self = []

        if hasattr(self, 'val_accuracies_others') and self.val_accuracies_others:
            val_accuracy_others = sum(self.val_accuracies_others) / len(self.val_accuracies_others)
            self.log("val_accuracy_others_epoch", val_accuracy_others, on_epoch=True)
            self.val_accuracies_others = []
        
        # Log validation examples if we have stored batch data
        if (self.val_batch_for_logging is not None and 
            hasattr(self, 'val_embeddings_mod')):
            
            embeddings_0, embeddings_1 = self.val_embeddings
            embeddings_0_mod, embeddings_1_mod = self.val_embeddings_mod
            embeddings_0_noise, embeddings_1_noise = self.val_embeddings_noise
            # wrapped_embeddings_0_noise, wrapped_embeddings_1_noise = self.val_embeddings_wrapped_noise
            modality_0_imgs = self.val_batch_for_logging["original_modality_0"]
            modality_1_imgs = self.val_batch_for_logging["original_modality_1"]
            modality_0_imgs_mod = self.val_batch_for_logging["modality_0"]
            modality_1_imgs_mod = self.val_batch_for_logging["modality_1"]
            modality_0_imgs_mod_noise = self.val_batch_for_logging["modality_0_noise"]
            modality_1_imgs_mod_noise = self.val_batch_for_logging["modality_1_noise"]

            # Log examples to wandb
            n=2
            indices = torch.randperm(modality_0_imgs_mod.shape[0])[:n].tolist()
            self.log_validation_examples2D(modality_0_imgs_mod, modality_1_imgs_mod, embeddings_0_mod, embeddings_1_mod, 
                                            mode="all_to_all", indices=indices)
            self.log_validation_examples2D(modality_0_imgs_mod, modality_1_imgs_mod, embeddings_0_mod, embeddings_1_mod, 
                                            mode="one_to_all", indices=indices)

            self.log_validation_examples2D(modality_0_imgs_mod, modality_0_imgs_mod_noise, embeddings_0_mod, embeddings_0_noise, 
                                            mode="one_to_all", indices=indices, prefix="equivariant_")
            self.log_validation_examples2D(modality_0_imgs_mod, modality_1_imgs_mod_noise, embeddings_0_mod, embeddings_1_noise, 
                                            mode="one_to_all", indices=indices, prefix="equivariant_hard_")
            self.log_validation_examples2D(modality_0_imgs, modality_1_imgs_mod_noise, embeddings_0, embeddings_1_noise, 
                                            mode="one_to_all", indices=indices, prefix="equivariant_hard_")
            
            # Clear stored data
            self.val_batch_for_logging = None
            self.val_embeddings_mod = None
            self.val_embeddings_noise = None
            self.val_embeddings_wrapped_noise = None
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_config.get("lr", 1e-4),
            weight_decay=self.optimizer_config.get("weight_decay", 1e-4),
            betas=self.optimizer_config.get("betas", (0.9, 0.999)),
        )
        
        # Create scheduler if specified
        if self.scheduler_config:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get("T_max", 100),
                eta_min=self.scheduler_config.get("eta_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        
        return optimizer
    
    def get_embeddings(self, modality_0: torch.Tensor, modality_1: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for inference (no gradients)."""
        with torch.no_grad():
            return self.forward(modality_0, modality_1)
    
    def compute_similarity_matrix(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between two sets of embeddings."""
        # Normalize embeddings
        embeddings_0 = nn.functional.normalize(embeddings_0, p=2, dim=1)
        embeddings_1 = nn.functional.normalize(embeddings_1, p=2, dim=1)
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(embeddings_0, embeddings_1.T)
        
        return similarity_matrix

    @torch.no_grad()
    def compute_retrieval_accuracy(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval accuracy (cross-modal, top-1).
        Each sample in modality 0 should retrieve its paired sample in modality 1,
        and vice versa.
        """
        batch_size = embeddings_0.shape[0]
        z = torch.cat([embeddings_0, embeddings_1], dim=0)  # (2B, D)

        # normalize embeddings
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        # similarity matrix (2B x 2B)
        sim = torch.matmul(z, z.T)

        # mask self-similarities
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(diag_mask, float('-inf'))

        correct = 0

        # modality 0 → modality 1
        for i in range(2 * batch_size):
            closest = (sim[i] == sim[i].max()).nonzero(as_tuple=True)[0]
            is_correct = int((i + batch_size) % (2 * batch_size) in closest)
            correct += is_correct/len(closest)

        # accuracy over 2B queries
        acc = correct / (2 * batch_size)
        return torch.tensor(acc, device=embeddings_0.device, dtype=torch.float32)

    @torch.no_grad()
    def compute_retrieval_accuracy2D(self, embeddings_0_2d: torch.Tensor, embeddings_1_2d: torch.Tensor,
                                    sample_percentage: float = 0.3, topk: int = 1, distance_th: float = None, 
                                    valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute retrieval accuracy (cross-modal, top-1).
        Each sample in modality 0 should retrieve its paired sample in modality 1,
        and vice versa.
        
        Args:
            embeddings_0_2d: First modality 2D embeddings (B, C, H, W)
            embeddings_1_2d: Second modality 2D embeddings (B, C, H, W)
            sample_percentage: Percentage of samples to use for evaluation (default: 0.3)
            topk: Top-k accuracy to compute (default: 1)
            distance_th: Distance threshold for Euclidean distance metrics (default: None)
            valid_mask: Boolean mask indicating valid indices with shape (B, H, W) (default: None, all indices valid)
        """
        batch_size, channels, height, width = embeddings_0_2d.shape
        separation = batch_size * height * width
        embeddings_0 = embeddings_0_2d.permute(0, 2, 3, 1) 
        embeddings_0 = embeddings_0.reshape(separation, channels )
        embeddings_1 = embeddings_1_2d.permute(0, 2, 3, 1) 
        embeddings_1 = embeddings_1.reshape(separation, channels )
        z = torch.cat([embeddings_0, embeddings_1], dim=0)  # (2BHW, D)
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        # Initialize valid mask if not provided
        if valid_mask is None:
            valid_mask = torch.ones_like(z[:, 0], dtype=torch.bool, device=z.device)
        else:
            # Reshape mask from (B, H, W) to (2BHW,) to match z
            # The mask applies to both modalities, so we duplicate it
            valid_mask_2d = valid_mask.reshape(separation)  # (BHW,)
            valid_mask = torch.cat([valid_mask_2d, valid_mask_2d], dim=0)  # (2BHW,)

        num_selected = int(len(z) * sample_percentage)
        # Filter indices based on valid mask, similar to losses.py implementation
        if sample_percentage > 0.0:
            indices = torch.arange(len(z), device=z.device)
            permutation = torch.randperm(len(indices))
            perm_mask = valid_mask.bool()[permutation]
            valid_indices = indices[permutation][perm_mask]
            selected_indices = valid_indices[:num_selected] if len(valid_indices) >= num_selected else valid_indices
        else:
            selected_indices = torch.arange(len(z), device=z.device)[valid_mask.bool()]
        block_size = self.loss_fn.block_size
        correct_all = 0
        correct_self = 0
        correct_others = 0
        correct_no_self = 0
        correct_topk_all = 0
        total_euclidean_distance = 0.0
        total_distance_below_threshold = 0
        total = 0
        
        # Track incorrect indices for visualization
        incorrect_all_indices = []
        incorrect_self_indices = []
        incorrect_others_indices = []
        incorrect_candidates_all_indices = []
        incorrect_candidates_self_indices = []
        incorrect_candidates_others_indices = []

        # # Visualization of valid mask and selected indices
        # self._visualize_valid_masks_and_indices(valid_mask, selected_indices, batch_size, height, width)

        for block_idx, i in enumerate(range(0, len(selected_indices), block_size)):
            # print(f"Processing block {block_idx} of {len(selected_indices) // block_size}")
            block_indices = selected_indices[i:i+block_size]
            z_block = z[block_indices]

            # similarity matrix (2BHW x 2BHW)
            sim = torch.matmul(z_block, z.T)

            # mask out self-similarities on the diagonal
            # Use z_block.shape[0] instead of block_size for the last block
            diag_mask = torch.zeros_like(sim, dtype=torch.bool, device=sim.device)
            diag_mask[list(range(z_block.shape[0])), block_indices] = True
            sim = sim.masked_fill(diag_mask, float('-inf'))  # or a very large negative number
            
            # # # Apply valid mask to similarity matrix (similar to losses.py)
            # if valid_mask is not None:
            #     sim = sim.masked_fill(~valid_mask.repeat(z_block.shape[0], 1), float('-inf'))

            # modality 0 → modality 1
            for i in range(z_block.shape[0]):
                index_in_all = block_indices[i]

                sim_row = sim[i]
                sim_row_no_self = sim_row.clone()
                sim_row_no_self[index_in_all] = float('-inf')

                closest_all = (sim_row == sim_row.max()).nonzero(as_tuple=True)[0] # closest in the whole row (batch)

                index_in_latent = index_in_all % (height * width)
                index_latent_in_2xbatch = index_in_all // (height * width)

                indices_in_batch = list(range(index_in_latent, 2*separation, height * width))
                first_crossmodal_self = (index_latent_in_2xbatch * height * width + separation)% (2 * separation)
                indices_in_crossmodal_self = list(range(first_crossmodal_self, first_crossmodal_self + height * width))

                first_self = index_latent_in_2xbatch * height * width
                indices_in_self = list(range(first_self, first_self + height * width))

                sim_row_no_self = sim_row.clone()
                sim_row_no_self[indices_in_self] = float('-inf')

                closest_no_self = (sim_row_no_self == sim_row_no_self.max()).nonzero(as_tuple=True)[0] # closest in the whole row (batch) but not self
                closest_crossmodal_self = (sim_row[indices_in_crossmodal_self] == sim_row[indices_in_crossmodal_self].max()).nonzero(as_tuple=True)[0] # closest in the same 2D latent
                closest_others = (sim_row[indices_in_batch] == sim_row[indices_in_batch].max()).nonzero(as_tuple=True)[0] # closest in the same cell of other latents

                is_correct_no_self = int(((index_in_all + separation) % (2 * separation)) in closest_no_self) 
                is_correct_self = index_in_latent in closest_crossmodal_self # Needs to be the same as index_in_latent
                is_correct_others = int(((index_latent_in_2xbatch + batch_size) % (2 * batch_size)) in closest_others)
                is_correct_all = int(((index_in_all + separation) % (2 * separation)) in closest_all)
                
                correct_all += is_correct_all/len(closest_all)
                correct_self += is_correct_self/len(closest_crossmodal_self)
                correct_others += is_correct_others/len(closest_others)
                correct_no_self += is_correct_no_self/len(closest_no_self)

                # Track incorrect indices for visualization
                if not is_correct_all:
                    incorrect_all_indices.append(index_in_all.item())
                    incorrect_candidates_all_indices.append(closest_all)
                if not is_correct_self:
                    incorrect_self_indices.append(index_in_all.item())
                    incorrect_candidates_self_indices.append(((index_latent_in_2xbatch + batch_size) % (2 * batch_size)) + closest_crossmodal_self)
                if not is_correct_others:
                    incorrect_others_indices.append(index_in_all.item())
                    incorrect_candidates_others_indices.append(closest_others* (height * width) + index_in_latent)

                # Compute Euclidean distance if distance_th is provided
                if distance_th is not None:
                    # Get the closest predicted index from closest_all
                    predicted_idx_1d = closest_crossmodal_self
                    ground_truth_idx_1d = index_in_latent
                    
                    # Convert 1D indices to 2D indices
                    # predicted_idx_1d -> (pred_h, pred_w)
                    pred_h = predicted_idx_1d // width
                    pred_w = predicted_idx_1d % width
                    
                    # ground_truth_idx_1d -> (gt_h, gt_w)  
                    gt_h = (ground_truth_idx_1d % (height * width)) // width
                    gt_w = (ground_truth_idx_1d % (height * width)) % width
                    
                    # Calculate Euclidean distance in 2D
                    euclidean_dist = torch.sqrt((pred_h - gt_h)**2 + (pred_w - gt_w)**2)
                    total_euclidean_distance += euclidean_dist.mean().item()
                    
                    # Count instances below threshold
                    total_distance_below_threshold += (euclidean_dist < distance_th).sum().item() / len(euclidean_dist)

                # Compute top-k accuracy if topk > 1
                if topk > 1:
                    # Get top-k indices for each category
                    topk_indices_all = torch.topk(sim[i], k=min(topk, len(sim[i])), largest=True).indices
                    
                    # Check if target is in top-k
                    is_correct_topk_all = int(((index_in_all + separation) % (2 * separation)) in topk_indices_all)
                    
                    correct_topk_all += is_correct_topk_all

                total += 1

        # accuracy over 2B queries
        acc = correct_all / total
        acc_self = correct_self / total
        acc_others = correct_others / total
        acc_no_self = correct_no_self / total
        
        result = {
            "acc": torch.tensor(acc, device=embeddings_0.device, dtype=torch.float32),
            "acc_self": torch.tensor(acc_self, device=embeddings_0.device, dtype=torch.float32),
            "acc_others": torch.tensor(acc_others, device=embeddings_0.device, dtype=torch.float32),
            "acc_no_self": torch.tensor(acc_no_self, device=embeddings_0.device, dtype=torch.float32),
        }
        
        # Add Euclidean distance if distance_th is provided
        if distance_th is not None:
            mean_euclidean_distance = total_euclidean_distance / total
            result["AEPE"] = torch.tensor(mean_euclidean_distance, device=embeddings_0.device, dtype=torch.float32)
            
            # Add percentage of instances below distance threshold
            percentage_below_threshold = total_distance_below_threshold / total
            result[f"PCK@{distance_th}"] = torch.tensor(percentage_below_threshold, device=embeddings_0.device, dtype=torch.float32)
        
        # Add top-k accuracy if topk > 1
        if topk > 1:
            acc_topk_all = correct_topk_all / total
            
            result[f"acc_top{topk}_all"] = torch.tensor(acc_topk_all, device=embeddings_0.device, dtype=torch.float32)
        
        # # # Visualize incorrect indices for each metric
        # self._visualize_incorrect_indices(valid_mask, incorrect_all_indices, batch_size, height, width, 
        #                                 "correct_all", max_examples=10, incorrect_candidates_indices=incorrect_candidates_all_indices)
        # self._visualize_incorrect_indices(valid_mask, incorrect_self_indices, batch_size, height, width, 
        #                                 "correct_self", max_examples=10, incorrect_candidates_indices=incorrect_candidates_self_indices)
        # self._visualize_incorrect_indices(valid_mask, incorrect_others_indices, batch_size, height, width, 
        #                                 "correct_others", max_examples=10, incorrect_candidates_indices=incorrect_candidates_others_indices)
            
        return result
    
    @torch.no_grad()
    def analyze_similarities_for_logging(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze similarities to find best and worst examples for logging.
        
        Returns:
            Dictionary with indices and similarity scores for:
            - best_pairs: 2 highest self-similarities (correct pairs)
            - worst_pairs: 2 lowest self-similarities (correct pairs) 
            - worst_mistakes: 2 highest cross-similarities (incorrect pairs)
        """
        batch_size = embeddings_0.shape[0]
        z = torch.cat([embeddings_0, embeddings_1], dim=0)  # (2B, D)
        
        # Normalize embeddings
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        
        # Similarity matrix (2B x 2B)
        sim = torch.matmul(z, z.T)
        
        # Get self-similarities (correct pairs)
        self_similarities = []
        for i in range(batch_size):
            self_sim = sim[i, i + batch_size].item()  # modality_0[i] with modality_1[i]
            self_similarities.append((i, self_sim))
        
        # Get cross-similarities (incorrect pairs) - excluding self
        cross_similarities = []
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:  # Exclude self
                    cross_sim = sim[i, j + batch_size].item()  # modality_0[i] with modality_1[j]
                    cross_similarities.append((i, j, cross_sim))
        
        # Sort and get top examples
        self_similarities.sort(key=lambda x: x[1], reverse=True)  # Highest first
        cross_similarities.sort(key=lambda x: x[2], reverse=True)  # Highest first
        
        return {
            "best_pairs": self_similarities[:2],  # 2 highest self-similarities
            "worst_pairs": self_similarities[-2:],  # 2 lowest self-similarities
            "worst_mistakes": cross_similarities[:2],  # 2 highest cross-similarities
        }
    
    @torch.no_grad()
    def display_grid_locally(self, grid: np.ndarray, step: int, save_path: str = "/tmp"):
        """
        Display the validation grid locally using OpenCV.
        
        Args:
            grid: The grid image as numpy array
            step: Current global step number
            save_path: Directory to save the image
        """
        # Convert from [0, 1] to [0, 255] for OpenCV display
        display_grid = (grid * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        display_grid = cv2.cvtColor(display_grid, cv2.COLOR_RGB2BGR)
        
        # Add text labels to the grid
        h, w = display_grid.shape[:2]
        cell_h, cell_w = h // 2, w // 6
        
        # Column labels
        column_labels = ["best_pair1", "best_pair2", "worst_pair1", "worst_pair2", "mistake1", "mistake2"]
        
        # Add column labels at the top
        for i, label in enumerate(column_labels):
            x = i * cell_w + 10
            y = 25
            cv2.putText(display_grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add row labels
        cv2.putText(display_grid, "modality_0", (10, cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_grid, "modality_0", (10, cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        cv2.putText(display_grid, "modality_1", (10, cell_h + cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_grid, "modality_1", (10, cell_h + cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add step info
        step_text = f"Step {step}"
        cv2.putText(display_grid, step_text, (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_grid, step_text, (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # # Save the image
        # os.makedirs(save_path, exist_ok=True)
        # filename = os.path.join(save_path, f"val_examples_epoch_{epoch:03d}.png")
        # cv2.imwrite(filename, display_grid)
        # print(f"Saved validation grid to: {filename}")
        
        # Display the image (this will open a window)
        cv2.imshow("Validation Examples Grid", display_grid)
        cv2.waitKey(0)  # Display for 1 second
        cv2.destroyAllWindows()
    
    @torch.no_grad()
    def log_validation_examples(self, batch: Dict[str, Any], similarity_analysis: Dict[str, Any], step: int):
        """
        Log validation examples to wandb as a grid.
        
        Args:
            batch: Validation batch containing images and metadata
            similarity_analysis: Results from analyze_similarities_for_logging
            step: Current global step number
        """
        # Check if wandb logger is available
        if (self.logger is None or 
            not hasattr(self.logger, 'experiment') or 
            self.logger.experiment is None or
            not hasattr(self.logger.experiment, 'log')):
            return
            
        modality_0 = batch["modality_0"]  # [B, C, H, W]
        modality_1 = batch["modality_1"]  # [B, C, H, W]
        m0_type = batch["m0_type"]
        m1_type = batch["m1_type"]
        sample_ids = batch["sample_id"]
    
        
        # Create grid: 2 rows x 6 columns
        # Row 0: modality_0 images
        # Row 1: modality_1 images
        # Columns: best_pair1, best_pair2, worst_pair1, worst_pair2, mistake1, mistake2
        
        # Get image dimensions from first image
        sample_img = tensor_to_numpy_image(modality_0[0])
        h, w, c = sample_img.shape
        
        # Create grid canvas
        grid_h = h * 2  # 2 rows
        grid_w = w * 6  # 6 columns
        grid = np.zeros((grid_h, grid_w, c), dtype=np.float32)
        
        
        # Fill grid with images
        col_idx = 0
        
        # Best pairs (columns 0-1)
        for i, (idx, sim_score) in enumerate(similarity_analysis["best_pairs"]):
            if col_idx < 6:
                # Row 0: modality_0
                grid[0:h, col_idx*w:(col_idx+1)*w] = tensor_to_numpy_image(modality_0[idx])
                # Row 1: modality_1
                grid[h:2*h, col_idx*w:(col_idx+1)*w] = tensor_to_numpy_image(modality_1[idx])
                col_idx += 1
        
        # Worst pairs (columns 2-3)
        for i, (idx, sim_score) in enumerate(similarity_analysis["worst_pairs"]):
            if col_idx < 6:
                # Row 0: modality_0
                grid[0:h, col_idx*w:(col_idx+1)*w] = tensor_to_numpy_image(modality_0[idx])
                # Row 1: modality_1
                grid[h:2*h, col_idx*w:(col_idx+1)*w] = tensor_to_numpy_image(modality_1[idx])
                col_idx += 1
        
        # Worst mistakes (columns 4-5)
        for i, (idx_0, idx_1, sim_score) in enumerate(similarity_analysis["worst_mistakes"]):
            if col_idx < 6:
                # Row 0: modality_0 from idx_0
                grid[0:h, col_idx*w:(col_idx+1)*w] = tensor_to_numpy_image(modality_0[idx_0])
                # Row 1: modality_1 from idx_1
                grid[h:2*h, col_idx*w:(col_idx+1)*w] = tensor_to_numpy_image(modality_1[idx_1])
                col_idx += 1
        
        # Create caption with similarity scores and sample IDs
        caption_parts = []
        
        # Add best pairs info
        for i, (idx, sim_score) in enumerate(similarity_analysis["best_pairs"]):
            caption_parts.append(f"Best {i+1}: sim={sim_score:.3f}, id={sample_ids[idx]}")
        
        # Add worst pairs info
        for i, (idx, sim_score) in enumerate(similarity_analysis["worst_pairs"]):
            caption_parts.append(f"Worst {i+1}: sim={sim_score:.3f}, id={sample_ids[idx]}")
        
        # Add worst mistakes info
        for i, (idx_0, idx_1, sim_score) in enumerate(similarity_analysis["worst_mistakes"]):
            caption_parts.append(f"Mistake {i+1}: sim={sim_score:.3f}, ids={sample_ids[idx_0]},{sample_ids[idx_1]}")
        
        caption = " | ".join(caption_parts)
        
        # # Display grid locally first
        # self.display_grid_locally(grid, epoch, save_path="/tmp/validation_grids")
        
        # Create wandb image
        wandb_image = wandb.Image(grid, caption=caption)
        
        # Log the grid
        self.logger.experiment.log({
            "val_examples_grid": wandb_image,
            "val_examples_step": step,
        }, step=step)

    @torch.no_grad()
    def log_validation_examples2D(self, modality_0_imgs: torch.Tensor, modality_1_imgs: torch.Tensor, 
                                        embeddings_0: torch.Tensor, embeddings_1: torch.Tensor, 
                                        n_examples: int = 2, mode: str = "all_to_all", prefix: str = "",
                                        indices: torch.Tensor = None):
        """
        Create a similarity image of n_examples with the other modality and other latents.
        Then log the image to wandb.
        
        Args:
            embeddings_0: Embeddings from modality 0
            embeddings_1: Embeddings from modality 1
            n_examples: Number of examples to log
        """
        # Require stored batch to fetch original images
        if self.val_batch_for_logging is None:
            return

        # Determine batch size and sample indices
        batch_size = embeddings_0.shape[0]
        n = min(n_examples, batch_size)
        indices = torch.randperm(batch_size)[:n].tolist() if indices is None else indices

        # Ensure embeddings have shape [B,C,H,W]
        assert embeddings_0.dim() == 4 and embeddings_1.dim() == 4, "2D embeddings expected [B,C,H,W]"

        # Prepare rows and determine target image shape
        # Use original modality_0 image size for grid cells
        sample_img_np = tensor_to_numpy_image(modality_0_imgs[0])
        img_h, img_w, _ = sample_img_np.shape

        row_images = []
        for idx in indices:
            # Original images
            img0 = tensor_to_numpy_image(modality_0_imgs[idx])  # [H,W,3]
            img1 = tensor_to_numpy_image(modality_1_imgs[idx])  # [H,W,3]

            # Negative examples
            random_increment = random.randint(1, batch_size)
            neg_idx = (idx+random_increment)%batch_size
            img_neg = tensor_to_numpy_image(modality_0_imgs[neg_idx])  # [H,W,3]

            if mode == "one_to_all":
                i = random.randint(0, embeddings_0[idx].shape[1]-1)
                j = random.randint(0, embeddings_0[idx].shape[2]-1)
                e0 = embeddings_0[idx][:, i, j].detach().cpu()

                # Mark the query point as a 8x8 square
                embedding_to_img_ratio = img0.shape[1] / embeddings_0[idx].shape[1]
                i = np.clip(int(i * embedding_to_img_ratio), 0, img0.shape[0]-1)
                j = np.clip(int(j * embedding_to_img_ratio), 0, img0.shape[1]-1)
                i_min = max(0, i-3)
                i_max = min(img0.shape[0], i+3)
                j_min = max(0, j-3)
                j_max = min(img0.shape[1], j+3)
                img0[i_min:i_max, j_min:j_max] = [1, 0, 0]
                if not prefix.startswith("equivariant"):
                    img1[i_min:i_max, j_min:j_max] = [1, 0, 0]
                img_neg[i_min:i_max, j_min:j_max] = [1, 0, 0]
            elif mode == "all_to_all":
                e0 = embeddings_0[idx].detach().cpu()
            else:
                raise ValueError(f"Invalid mode: {mode}")
                
            color_map = get_color_map(e0, embeddings_1[idx].detach().cpu())
            color_map_neg = get_color_map(e0, embeddings_0[neg_idx].detach().cpu())

            color_map = cv2.resize(color_map, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            color_map_neg = cv2.resize(color_map_neg, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

            separator = np.ones((img_h, 10, 3), dtype=np.float32)
            
            # Concatenate the three columns horizontally for this row
            row = np.concatenate([img0, img1, separator, color_map, separator, img_neg, separator, color_map_neg], axis=1)
            row_images.append(row)

        # Stack rows vertically
        grid = np.concatenate(row_images, axis=0) if len(row_images) > 0 else None
        if grid is None:
            return

        if hasattr(self, 'logger') and self.logger is not None:
            # Log to wandb
            caption = f"Random {n} examples with cosine similarity maps (blue=low, red=high)"
            wandb_image = wandb.Image(grid, caption=caption)
            self.logger.experiment.log({
                f"{prefix}val_examples_grid_{mode}": wandb_image,
                "val_examples_step": self.global_step,
            }, step=self.global_step)
        else:
            # Display grid locally first
            self.display_grid_locally(grid, self.global_step)
    
    def _visualize_valid_masks_and_indices(self, valid_mask: torch.Tensor, selected_indices: torch.Tensor, 
                                         batch_size: int, height: int, width: int):
        """
        Visualize valid masks and sampled indices using matplotlib.
        
        Args:
            valid_mask: 1D boolean mask of shape (2BHW,) indicating valid indices
            selected_indices: 1D tensor of selected indices
            batch_size: Batch size B
            height: Height of the 2D feature maps
            width: Width of the 2D feature maps
        """
        # Convert 1D valid mask to 2D for visualization
        # The mask is (2BHW,) - first BHW are modality 0, second BHW are modality 1
        separation = batch_size * height * width
        
        # Split mask into two modalities
        valid_mask_0 = valid_mask[:separation].reshape(batch_size, height, width)
        valid_mask_1 = valid_mask[separation:].reshape(batch_size, height, width)
        
        # Convert selected indices to 2D coordinates
        # For each selected index, determine which modality and 2D position it corresponds to
        selected_coords_0 = []
        selected_coords_1 = []
        
        for idx in selected_indices:
            if idx < separation:
                # Index belongs to modality 0
                batch_idx = idx // (height * width)
                spatial_idx = idx % (height * width)
                h_idx = spatial_idx // width
                w_idx = spatial_idx % width
                selected_coords_0.append((batch_idx, h_idx, w_idx))
            else:
                # Index belongs to modality 1
                idx_mod1 = idx - separation
                batch_idx = idx_mod1 // (height * width)
                spatial_idx = idx_mod1 % (height * width)
                h_idx = spatial_idx // width
                w_idx = spatial_idx % width
                selected_coords_1.append((batch_idx, h_idx, w_idx))
        
        # Create visualization for B iterations, showing 2B images total
        num_iterations = min(batch_size, 4)  # Limit to 4 iterations for readability
        
        fig, axes = plt.subplots(num_iterations, 2, figsize=(12, 3 * num_iterations))
        if num_iterations == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_iterations):
            # Left plot: Modality 0 (batch i)
            ax_left = axes[i, 0]
            ax_left.imshow(valid_mask_0[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_left.set_title(f'Modality 0 - Batch {i}')
            ax_left.set_xlabel('Width')
            ax_left.set_ylabel('Height')
            
            # Plot selected indices as green points for modality 0
            selected_coords_0 = selected_coords_0
            for batch_idx, h_idx, w_idx in selected_coords_0:
                if batch_idx == i:
                    ax_left.scatter(w_idx.cpu().item(), h_idx.cpu().item(), c='green', s=20, marker='o', alpha=0.8)
            
            # Right plot: Modality 1 (batch i)
            ax_right = axes[i, 1]
            ax_right.imshow(valid_mask_1[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_right.set_title(f'Modality 1 - Batch {i}')
            ax_right.set_xlabel('Width')
            ax_right.set_ylabel('Height')
            
            # Plot selected indices as green points for modality 1
            for batch_idx, h_idx, w_idx in selected_coords_1:
                if batch_idx == i:
                    ax_right.scatter(w_idx.cpu().item(), h_idx.cpu().item(), c='green', s=20, marker='o', alpha=0.8)
        
        plt.tight_layout()
        plt.suptitle('Valid Masks (gray=invalid, white=valid) and Selected Indices (green points)', 
                     y=1.02, fontsize=14)
        plt.show()
    
    def _visualize_incorrect_indices(self, valid_mask: torch.Tensor, incorrect_indices: list, 
                                   batch_size: int, height: int, width: int, 
                                   metric_name: str, max_examples: int = 5, incorrect_candidates_indices: list = None):
        """
        Visualize valid masks and incorrect indices using matplotlib.
        
        Args:
            valid_mask: 1D boolean mask of shape (2BHW,) indicating valid indices
            incorrect_indices: List of incorrect indices to visualize
            batch_size: Batch size B
            height: Height of the 2D feature maps
            width: Width of the 2D feature maps
            metric_name: Name of the metric (e.g., 'correct_all', 'correct_self', 'correct_others')
            max_examples: Maximum number of examples to show
        """
        if len(incorrect_indices) == 0:
            print(f"No incorrect indices found for {metric_name}")
            return
            
        # Limit the number of examples
        num_examples = min(batch_size, max_examples)
        
        # Convert 1D valid mask to 2D for visualization
        separation = batch_size * height * width
        
        # Split mask into two modalities
        all_valid_mask = valid_mask.reshape(2*batch_size, height, width)
        
        # Convert incorrect indices to 2D coordinates
        incorrect_coords_0 = []
        incorrect_candidates = []
        incorrect_count = 0
        plotting_info_list = []
        incorrect_list_idx = 0
        while incorrect_count < num_examples and incorrect_list_idx < len(incorrect_indices):
            first_px_idx_global = incorrect_indices[incorrect_list_idx]
            # Skip if modality 1
            if first_px_idx_global >= separation:
                incorrect_list_idx += 1
                continue

            # Get the batch index of modality 0
            current_batch_idx_m0 = first_px_idx_global // (height * width)

            if metric_name == "correct_self":
                current_batch_idx_m1 = current_batch_idx_m0 + batch_size
            elif metric_name == "correct_all" or metric_name == "correct_others":
                current_batch_idx_m1 = incorrect_candidates_indices[incorrect_list_idx].squeeze() // (height * width)
            else:
                raise ValueError(f"Invalid metric name: {metric_name}")

            info_dict = {"batch_idx_modality_0": current_batch_idx_m0, 
                         "batch_idx_modality_1": current_batch_idx_m1,
                         "incorrect_coords_0": [],
                         "incorrect_candidates": []}
            current_px_idx_global = first_px_idx_global
            while (current_px_idx_global // (height * width) == current_batch_idx_m0) and (incorrect_list_idx < len(incorrect_indices)):
                # Coordinates of current_px_idx_global
                spatial_idx = current_px_idx_global % (height * width)
                h_idx = spatial_idx // width
                w_idx = spatial_idx % width

                # Coordinates of incorrect_candidates_indices[incorrect_list_idx]
                spatial_idx_m1 = incorrect_candidates_indices[incorrect_list_idx].squeeze() % (height * width)
                h_idx_m1 = spatial_idx_m1 // width
                w_idx_m1 = spatial_idx_m1 % width

                # Add to info dict
                info_dict["incorrect_coords_0"].append((h_idx, w_idx))
                info_dict["incorrect_candidates"].append((h_idx_m1.cpu().item(), w_idx_m1.cpu().item()))

                incorrect_list_idx += 1
                if incorrect_list_idx >= len(incorrect_indices):
                    break
                current_px_idx_global = incorrect_indices[incorrect_list_idx]
            
            plotting_info_list.append(info_dict)
            incorrect_count += 1
            
        # Create visualization
        num_examples = min(num_examples, len(plotting_info_list))
        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_examples):
            plotting_info = plotting_info_list[i]
            batch_idx_m0 = plotting_info["batch_idx_modality_0"]
            batch_idx_m1 = plotting_info["batch_idx_modality_1"]
            incorrect_coords_0 = plotting_info["incorrect_coords_0"]
            incorrect_candidates = plotting_info["incorrect_candidates"]
            
            # Left plot: Modality 0 (batch batch_idx)
            ax_left = axes[i, 0]
            ax_left.imshow(all_valid_mask[batch_idx_m0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_left.set_title(f'Modality 0 - Batch {batch_idx_m0}')
            ax_left.set_xlabel('Width')
            ax_left.set_ylabel('Height')
            
            # Plot incorrect indices as red points for modality 0
            for h_idx, w_idx in incorrect_coords_0:
                ax_left.scatter(w_idx, h_idx, c='red', s=10, marker='o', alpha=0.9)
            
            # Right plot: Modality 1 (batch batch_idx)
            ax_right = axes[i, 1]
            ax_right.imshow(all_valid_mask[batch_idx_m1].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_right.set_title(f'Modality 1 - Batch {batch_idx_m1}')
            ax_right.set_xlabel('Width')
            ax_right.set_ylabel('Height')
            
            # Plot incorrect indices as red points for modality 1
            for h_idx, w_idx in incorrect_candidates:
                    ax_right.scatter(w_idx, h_idx, c='red', s=10, marker='x', alpha=0.9)
        
        plt.tight_layout()
        plt.suptitle(f'{metric_name}: Valid Masks (gray=invalid, white=valid) and Incorrect Indices (red X)', 
                     y=1.02, fontsize=14)
        plt.show()
        
def create_lightning_module(
    model_config: Dict[str, Any],
    loss_config: Dict[str, Any],
    optimizer_config: Dict[str, Any],
    scheduler_config: Optional[Dict[str, Any]] = None,
    transform_in_val: Optional[bool] = False,
    train_equivariant: Optional[bool] = False,
) -> ContrastiveLearningModule:
    """
    Factory function to create a contrastive learning Lightning module.
    
    Args:
        model_config: Configuration for the model
        loss_config: Configuration for the loss function
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the learning rate scheduler
        
    Returns:
        Configured Lightning module
    """
    return ContrastiveLearningModule(
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        transform_in_val=transform_in_val,
        train_equivariant=train_equivariant,
    )

def load_contrastive_model_from_checkpoint(checkpoint_path: str, load_weights: bool = True, return_cls: bool = False) -> ContrastiveLearningModule:
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = checkpoint['hyper_parameters']

    model_config = hparams.get("model_config", {})
    # Safely access model_config
    backbone_kwargs = model_config.get("backbone_kwargs", {})

    # Remove dino_weights_path if present
    
    with omegaconf.open_dict(backbone_kwargs):  # allows modifications
        if "dino_weights_path" in backbone_kwargs:
            print("Removing dino_weights_path from backbone_kwargs")
            del backbone_kwargs["dino_weights_path"]

        if return_cls:
            backbone_kwargs["return_cls"] = return_cls

    loss_config = hparams.get("loss_config", {})
    optimizer_config = hparams.get("optimizer_config", {})

    creation_config = {
        "model_config": model_config,
        "loss_config": loss_config,
        "optimizer_config": optimizer_config,
    }

    model = ContrastiveLearningModule(**creation_config)
    if load_weights:
        load_result = model.load_state_dict(checkpoint['state_dict'])
        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)
    model.eval()
    return model, creation_config