import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import pytorch_lightning as pl
from mmfe_utils.models_utils import load_salad
from mmfe_utils.tensor_utils import tensor_to_numpy_image
from third_parties.salad.utils.losses import get_loss, get_miner
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint
from training.losses import InfoNCELoss1D

from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
import os
from hydra.utils import to_absolute_path
from global_descriptors.backbones import GlobalDescriptorBackbone
import subprocess
import sys
from global_descriptors.global_descriptor_models import( GlobalDescriptorModel, 
                                                            create_no_train_agg_model, 
                                                            create_global_descriptor_model)

class GlobalDescriptorLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training global descriptor models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_config: Dict[str, Any] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        # Model name
        if isinstance(model, GlobalDescriptorModel):
            self.info_for_loading = {"model_name": model.model_name,
                                      "descriptor_type": model.descriptor_type,
                                      "descriptor_kwargs": model.descriptor_kwargs,
                                      "output_dim": model.output_dim,
                                      "model_creation_config": model.creation_config
                                      }
        else:
            self.info_for_loading = {"model_name": "salad"}
        
        # Save hyperparameters
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss_config": loss_config or {},
            "scheduler_config": scheduler_config or {},
            "info_for_loading": self.info_for_loading
        }, ignore=['model'])

        # Loss function (only needed during training)
        if loss_config is not None:
            if loss_config["type"] == "infonce1d":
                self.loss_fn = InfoNCELoss1D(temperature=loss_config["kwargs"]["temperature"], reduction='mean', epsilon=loss_config["kwargs"].get("epsilon", None), max_negatives=loss_config["kwargs"].get("max_negatives", None))
            else:
                self.loss_fn = self.loss_with_miner
                self.loss_function = get_loss(loss_config["type"])
                self.miner = get_miner(loss_config.get("kwargs", {}).get("miner_name", None))
        else:
            self.loss_fn = None
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Store validation batch for logging
        self.val_batch_for_logging = None
        self.val_embeddings_for_logging = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def setup_crossover(
        self,
        base_dir: str,
        image_size=(256, 256),
        floorplan_img_name: str = 'mmfe_floorplan.png',
        point_source: str = 'density',
        density_name: str = 'density.png',
        scene_modalities=None,
        debug_dir: Optional[str] = None,
        point_size: int = 1,
    ):
        """Configure the model for CrossOver evaluation.

        Args:
            base_dir: Root directory of the ScanNet data.
            image_size: Target image size for preprocessing.
            floorplan_img_name: Filename of the floorplan image
                (e.g. ``floor+obj.png`` or ``mmfe_floorplan.png``).
            point_source: ``'density'`` to load pre-rendered density images,
                ``'coordinates'`` to render on-the-fly from the raw coordinate
                tensor, ``'pcl_sparse'`` to render from the
                ``ME.SparseTensor`` already built by the dataloader.
            density_name: Filename of the density image when
                *point_source* is ``'density'``.
            scene_modalities: List of modalities to process
                (default ``['point', 'floorplan']``).
            debug_dir: If set, rendered / loaded images are saved here for
                visual inspection.
            point_size: Brush size (in pixels) used when rendering density
                maps on-the-fly (``'coordinates'`` / ``'pcl_sparse'``).
        """
        import torchvision.transforms as T

        self._crossover_base_dir = base_dir
        self._crossover_image_size = tuple(image_size) if not isinstance(image_size, tuple) else image_size
        self._crossover_floorplan_img_name = floorplan_img_name
        self._crossover_point_source = point_source
        self._crossover_density_name = density_name
        self._crossover_modalities = scene_modalities or ['point', 'floorplan']
        self._crossover_debug_dir = debug_dir
        self._crossover_point_size = point_size
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
        self._crossover_transform = T.Compose([
            T.Resize(self._crossover_image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _debug_save(self, pil_img, scan_id: str, modality: str):
        """Save a PIL image to the debug directory (if configured)."""
        if self._crossover_debug_dir is None:
            return
        safe_id = scan_id.replace("/", "_")
        out_path = os.path.join(
            self._crossover_debug_dir, f"{safe_id}_{modality}.png",
        )
        pil_img.save(out_path)

    def _render_points_to_pil(self, pts_np: np.ndarray):
        """Render an (N, 3) float numpy point cloud to a PIL RGB image."""
        from PIL import Image
        from third_parties.CrossOver.render_utils import render_pointcloud_density

        density_img = render_pointcloud_density(pts_np, point_size=self._crossover_point_size)
        return Image.fromarray(density_img).convert('RGB')

    @torch.no_grad()
    def crossover_forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass that bridges CrossOver's data format to MMFE.

        Loads/renders floorplan and point-cloud images from *data_dict*,
        encodes them with ``self.model``, and writes the resulting
        embeddings back into *data_dict* so that the CrossOver evaluation
        pipeline can consume them.

        Supports three ``point_source`` modes:

        * ``'density'``      – load a pre-rendered density PNG from disk.
        * ``'coordinates'``  – render on-the-fly from the raw ``(N, 4)``
          coordinate tensor (col-0 = batch id, cols 1-3 = XYZ).
        * ``'pcl_sparse'``   – render on-the-fly from the
          ``ME.SparseTensor`` that CrossOver already builds.  Per-batch
          coordinates are recovered via
          ``SparseTensor.decomposed_coordinates``.
        """
        from PIL import Image

        device = next(self.parameters()).device
        scan_ids = data_dict['scan_id']
        batch_size = len(scan_ids)

        data_dict['embeddings'] = {}

        # ---- floorplan ------------------------------------------------
        if 'floorplan' in self._crossover_modalities:
            floorplan_images = []
            for scan_id in scan_ids:
                img_path = os.path.join(
                    self._crossover_base_dir, scan_id,
                    self._crossover_floorplan_img_name,
                )
                img = Image.open(img_path).convert('RGB')
                self._debug_save(img, scan_id, 'floorplan')
                floorplan_images.append(self._crossover_transform(img))
            floorplan_batch = torch.stack(floorplan_images).to(device)
            data_dict['embeddings']['floorplan'] = self.model(floorplan_batch)

        # ---- point cloud ----------------------------------------------
        if 'point' in self._crossover_modalities:
            if self._crossover_point_source == 'pcl_sparse':
                pcl_sparse = data_dict['pcl_sparse']
                per_batch_coords = pcl_sparse.decomposed_coordinates
                point_images = []
                for i in range(batch_size):
                    pts = per_batch_coords[i].float().cpu().numpy()
                    img = self._render_points_to_pil(pts)
                    self._debug_save(img, scan_ids[i], 'point')
                    point_images.append(self._crossover_transform(img))
                point_batch = torch.stack(point_images).to(device)

            elif self._crossover_point_source == 'coordinates':
                coordinates = data_dict['coordinates']
                point_images = []
                for i in range(batch_size):
                    mask = coordinates[:, 0] == i
                    pts = coordinates[mask, 1:].float().cpu().numpy()
                    img = self._render_points_to_pil(pts)
                    self._debug_save(img, scan_ids[i], 'point')
                    point_images.append(self._crossover_transform(img))
                point_batch = torch.stack(point_images).to(device)

            else:  # density – load pre-rendered image from disk
                point_images = []
                for scan_id in scan_ids:
                    img_path = os.path.join(
                        self._crossover_base_dir, scan_id,
                        self._crossover_density_name,
                    )
                    img = Image.open(img_path).convert('RGB')
                    self._debug_save(img, scan_id, 'point')
                    point_images.append(self._crossover_transform(img))
                point_batch = torch.stack(point_images).to(device)

            data_dict['embeddings']['point'] = self.model(point_batch)

        # ---- masks ----------------------------------------------------
        if 'scene_masks' not in data_dict:
            data_dict['scene_masks'] = {}
        for mod in self._crossover_modalities:
            data_dict['scene_masks'][mod] = torch.ones(batch_size, device=device)

        return data_dict

    def loss_with_miner(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> torch.Tensor:
        # we mine the pairs/triplets if there is an online mining strategy

        descriptors, labels = self.embeddings_to_salad_format(embeddings_0, embeddings_1)

        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_function(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        return loss

    def embeddings_to_salad_format(self, embeddings_0: torch.Tensor, embeddings_1: torch.Tensor) -> tuple:
        # (B, D), (B, D)
        B = embeddings_0.size(0)

        # concat into a single descriptor matrix (2B, D)
        descriptors = torch.cat([embeddings_0, embeddings_1], dim=0)

        # labels: 0..B-1 for modality A, 0..B-1 for modality B
        labels = torch.arange(B, device=embeddings_0.device)
        labels = torch.cat([labels, labels], dim=0)  # (2B,)

        return descriptors, labels
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Extract data
        modality_0 = batch["modality_0"]
        modality_1 = batch["modality_1"]
        original_modality_0 = batch["original_modality_0"]
        original_modality_1 = batch["original_modality_1"]
        
        # Forward pass
        embeddings_0 = self.forward(modality_0)
        embeddings_1 = self.forward(modality_1)
        embeddings_0_original = self.forward(original_modality_0)
        embeddings_1_original = self.forward(original_modality_1)
        # Compute loss
        loss_up_right = self.loss_fn(embeddings_0, embeddings_1)

        loss_self_equivariant = self.loss_fn(embeddings_0, embeddings_0_original) + self.loss_fn(embeddings_1, embeddings_1_original) 
        loss_cross_equivariant = self.loss_fn(embeddings_0, embeddings_1_original) + self.loss_fn(embeddings_0_original, embeddings_1)
        total_loss = loss_up_right + loss_self_equivariant + loss_cross_equivariant
        
        # Log metrics
        self.train_losses.append(total_loss.item())
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_up_right", loss_up_right, on_step=True, on_epoch=True)
        self.log("train_loss_self_equivariant", loss_self_equivariant, on_step=True, on_epoch=True)
        self.log("train_loss_cross_equivariant", loss_cross_equivariant, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Extract data
        modality_0 = batch["modality_0"]
        modality_1 = batch["modality_1"]
        original_modality_0 = batch["original_modality_0"]
        original_modality_1 = batch["original_modality_1"]
        
        # Forward pass
        embeddings_0 = self.forward(modality_0)
        embeddings_1 = self.forward(modality_1)
        embeddings_0_original = self.forward(original_modality_0)
        embeddings_1_original = self.forward(original_modality_1)
        # Compute loss
        loss_up_right = self.loss_fn(embeddings_0, embeddings_1)
        loss_self_equivariant = self.loss_fn(embeddings_0, embeddings_0_original) + self.loss_fn(embeddings_1, embeddings_1_original) 
        loss_cross_equivariant = self.loss_fn(embeddings_0, embeddings_1_original) + self.loss_fn(embeddings_0_original, embeddings_1)
        total_loss = loss_up_right + loss_self_equivariant + loss_cross_equivariant
        
        # Compute retrieval accuracy
        accuracy = self.compute_retrieval_accuracy(embeddings_0, embeddings_1_original)
        
        # Log metrics
        self.val_losses.append(total_loss.item())
        self.val_accuracies.append(accuracy.item())
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_up_right", loss_up_right, on_step=False, on_epoch=True)
        self.log("val_loss_self_equivariant", loss_self_equivariant, on_step=False, on_epoch=True)
        self.log("val_loss_cross_equivariant", loss_cross_equivariant, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store first batch for logging (only once per epoch)
        if batch_idx == 0:
            self.val_batch_for_logging = batch
            self.val_embeddings_for_logging = (embeddings_0, embeddings_1_original)
        
        return total_loss
    
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
            val_accuracy = sum(self.val_accuracies) / len(self.val_accuracies)
            self.log("val_loss_epoch", val_loss, on_epoch=True)
            self.log("val_accuracy_epoch", val_accuracy, on_epoch=True)
            self.val_losses = []
            self.val_accuracies = []
        
        # Log retrieval visualization every 5 epochs
        current_epoch = self.current_epoch
        if current_epoch % 5 == 0  and self.val_batch_for_logging is not None:
            self.log_retrieval_examples()
            # Clear stored data
            self.val_batch_for_logging = None
            self.val_embeddings_for_logging = None
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.scheduler_config:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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
    
    @torch.no_grad()
    def compute_retrieval_accuracy(
        self, 
        embeddings_0: torch.Tensor, 
        embeddings_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute retrieval accuracy (cross-modal, top-1).
        Each sample in modality 0 should retrieve its paired sample in modality 1.
        """
        batch_size = embeddings_0.shape[0]
        z = torch.cat([embeddings_0, embeddings_1], dim=0)  # (2B, D)

        # Normalize embeddings
        z = torch.nn.functional.normalize(z, p=2, dim=1)

        # Similarity matrix (2B x 2B)
        sim = torch.matmul(z, z.T)

        # Mask self-similarities
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(diag_mask, float('-inf'))

        correct = 0

        # Check retrieval accuracy
        for i in range(2 * batch_size):
            closest = (sim[i] == sim[i].max()).nonzero(as_tuple=True)[0]
            target_idx = (i + batch_size) % (2 * batch_size)
            is_correct = int(target_idx in closest)
            correct += is_correct / len(closest)

        # Accuracy over 2B queries
        acc = correct / (2 * batch_size)
        return torch.tensor(acc, device=embeddings_0.device, dtype=torch.float32)
    
    @torch.no_grad()
    def log_retrieval_examples(self, num_examples: int = 4):
        """
        Log retrieval visualization: 2x4 grid showing modality 0 images
        and their top-1 retrieved modality 1 images.
        
        Args:
            num_examples: Number of examples to show (default: 4)
        """
        # Check if wandb logger is available
        if self.logger is None or not hasattr(self.logger, 'experiment'):
            return
        
        if self.val_batch_for_logging is None or self.val_embeddings_for_logging is None:
            return
        
        batch = self.val_batch_for_logging
        embeddings_0, embeddings_1 = self.val_embeddings_for_logging
        
        # Get images
        modality_0_imgs = batch["modality_0"]  # (B, 3, H, W)
        modality_1_imgs = batch["original_modality_1"]  # (B, 3, H, W)
        
        batch_size = min(embeddings_0.shape[0], num_examples)
        
        # Normalize embeddings
        embeddings_0_norm = torch.nn.functional.normalize(embeddings_0, p=2, dim=1)
        embeddings_1_norm = torch.nn.functional.normalize(embeddings_1, p=2, dim=1)
        
        # Compute similarity matrix (B x B)
        similarity = torch.matmul(embeddings_0_norm, embeddings_1_norm.T)  # (B, B)
        
        # For each image in modality 0, find the top-1 match in modality 1
        retrieved_indices = similarity.argmax(dim=1)  # (B,)
        
        # Select first num_examples
        query_indices = list(range(min(batch_size, num_examples)))
        
        # Create 2x4 grid
        grid_rows = []
        
        # Top row: modality 0 images (queries)
        row_0_images = []
        for idx in query_indices:
            img = tensor_to_numpy_image(modality_0_imgs[idx])
            row_0_images.append(img)
        row_0 = np.concatenate(row_0_images, axis=1)  # Concatenate horizontally
        grid_rows.append(row_0)
        
        # Bottom row: retrieved modality 1 images
        row_1_images = []
        for idx in query_indices:
            retrieved_idx = retrieved_indices[idx].item()
            img = tensor_to_numpy_image(modality_1_imgs[retrieved_idx])
            row_1_images.append(img)
        row_1 = np.concatenate(row_1_images, axis=1)  # Concatenate horizontally
        grid_rows.append(row_1)
        
        # Combine rows vertically
        grid = np.concatenate(grid_rows, axis=0)
        
        # Create caption with retrieval information
        caption_parts = []
        for i, idx in enumerate(query_indices):
            retrieved_idx = retrieved_indices[idx].item()
            sim_score = similarity[idx, retrieved_idx].item()
            is_correct = (idx == retrieved_idx)
            caption_parts.append(
                f"Q{i}: Retrieved={retrieved_idx}, Sim={sim_score:.3f}, "
                f"{'✓' if is_correct else '✗'}"
            )
        caption = " | ".join(caption_parts)
        
        # Log to wandb
        wandb_image = wandb.Image(grid, caption=caption)
        self.logger.experiment.log({
            "val_retrieval_examples": wandb_image,
            "epoch": self.current_epoch,
        }, step=self.global_step)

def create_salad_og_model() -> nn.Module:
    salad_config = {'backbone': {'name': 'dinov2_vitb14', 'pretrained': True, 'freeze': 'salad', 'kwargs': None}, 
                    'descriptor': {'type': 'salad', 'output_dim': 512, 'kwargs': {'pretrained': True}}, 
                            }

    salad_config = OmegaConf.create(salad_config)
    
    model = create_global_descriptor_model(salad_config)
    lightning_module = GlobalDescriptorLightningModule(
        model=model,
        learning_rate=None,
        weight_decay=None,
        loss_config=None,
        scheduler_config=None,
    )
    return lightning_module