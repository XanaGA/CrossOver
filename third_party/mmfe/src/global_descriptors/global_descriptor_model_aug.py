import os
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from global_descriptors.global_descriptor_models import load_global_descriptor_model_from_checkpoint
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint

from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
import os
from hydra.utils import to_absolute_path
from global_descriptors.backbones import GlobalDescriptorBackbone
from global_descriptors.cls_poolers import GeMPool, AttentionPool, MHSA_Pooler, power_mean_pooling
from mmfe_utils.models_utils import load_salad
from global_descriptors.netvlad.netvlad import generate_netvlad_clusters
from torchvision.transforms import functional as TF

class GlobalDescriptorModelAug(nn.Module):
    """
    Model combining a backbone with a global descriptor aggregation module.
    
    Args:
        backbone: Pre-configured backbone model
        descriptor_type: Type of global descriptor ('netvlad', 'gem', 'avg', 'max')
        descriptor_kwargs: Additional kwargs for the descriptor module
        output_dim: Final output dimension (default: 512)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        backbone_channels: int,
        n_augs: int = 8,
        descriptor_type: str = "netvlad",
        descriptor_kwargs: Optional[Dict[str, Any]] = None,
        output_dim: int = "same",
        model_name: str = "global_descriptor",
        creation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.descriptor_type = descriptor_type.lower()
        self.descriptor_kwargs = descriptor_kwargs or {}
        self.model_name = model_name
        self.output_dim = output_dim
        self.creation_config = creation_config or {}
        self.n_augs = n_augs
        self.aug_step = 360 / n_augs
        self._internal_forward = self._forward_speed 

        # Create global descriptor module
        if self.descriptor_type == "netvlad":
            # Import NetVLAD locally to avoid circular imports
            from global_descriptors.netvlad.netvlad import NetVLAD
            
            num_clusters = self.descriptor_kwargs.get('num_clusters', 64)
            vladv2 = self.descriptor_kwargs.get('vladv2', False)
            cluster_path = self.descriptor_kwargs.get('cluster_path', None)
            
            self.descriptor = NetVLAD(
                num_clusters=num_clusters,
                dim=backbone_channels,
                normalize_input=True,
                vladv2=vladv2
            )
            
            # Initialize clusters if path provided
            if cluster_path is not None and os.path.exists(cluster_path):
                print(f"Loading NetVLAD clusters from: {cluster_path}")
                cluster_data = torch.load(cluster_path, map_location='cpu', weights_only=False)
                centroids = cluster_data['centroids']
                descriptors = cluster_data.get('descriptors', centroids)
                self.descriptor.init_params(centroids, descriptors)
                print(f"✓ Loaded {len(centroids)} cluster centers")
            else:
                if cluster_path is not None:
                    print(f"Warning: Cluster file not found: {cluster_path}")
                print("NetVLAD will be initialized randomly. Consider running init_netvlad_clusters.py first.")
            
            # NetVLAD output is num_clusters * dim
            descriptor_output_dim = num_clusters * backbone_channels

        elif self.descriptor_type == "gem":
            # GeM (Generalized Mean) pooling
            p = self.descriptor_kwargs.get('p', 3.0)
            self.descriptor = nn.Sequential(
                GeMPool(p=p),
                nn.Flatten(),
            )
            descriptor_output_dim = backbone_channels

        elif self.descriptor_type == "avg":
            # Global average pooling
            self.descriptor = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            descriptor_output_dim = backbone_channels
            
        elif self.descriptor_type == "max":
            # Global max pooling
            self.descriptor = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(),
            )
            descriptor_output_dim = backbone_channels
            
        elif self.descriptor_type == "global_desc":
            self.descriptor = None
            descriptor_output_dim = None
            self._internal_forward = self._forward_global_desc
        else:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}")
        
        # Final projection to output dimension
        if self.output_dim == "same":
            self.output_dim = descriptor_output_dim
            self.final_projection = nn.Identity()
        else:
            self.final_projection = nn.Sequential(
                nn.Linear(descriptor_output_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
            )

    def _create_x_augmented_list(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_augmented_list = []
        for i in range(self.n_augs):
            angle = i * self.aug_step
            # Rotate. Interpolation bilinear is safe for images and features.
            x_rot = TF.affine(
                x, angle=angle, translate=[0,0], scale=1.0, shear=0.0, 
                interpolation=TF.InterpolationMode.BILINEAR
            )
            x_augmented_list.append(x_rot)
        return x_augmented_list

    def _forward_speed(self, x: List[torch.Tensor]) -> torch.Tensor:

        # 1. Create Batch of Rotations: (B * n_augs, C, H, W)
        # This is faster than looping the backbone 8 times
        x_augmented_list = self._create_x_augmented_list(x)

        # Stack into a large batch
        x_large_batch = torch.cat(x_augmented_list, dim=0) # (B*8, C, H, W)

        # 2. Run Backbone ONCE on large batch
        # We assume backbone maps (B*8, C, H, W) -> (B*8, C_out, H_out, W_out)
        with torch.no_grad():
            feats_large = self.backbone(x_large_batch) # (B*8, C_out, H_out, W_out)

        # 3. Run Descriptor
        # Output: (B*8, desc_dim)
        descs_large = self.descriptor(feats_large)
        return descs_large

    def _forward_memory(self, x: List[torch.Tensor]) -> torch.Tensor:

        descs_large = []
        for i in range(self.n_augs):
            angle = i * self.aug_step
            # Rotate. Interpolation bilinear is safe for images and features.
            x_rot = TF.affine(
                x, angle=angle, translate=[0,0], scale=1.0, shear=0.0, 
                interpolation=TF.InterpolationMode.BILINEAR
            )
            with torch.no_grad():
                feats_large = self.backbone(x_rot)

            descs = self.descriptor(feats_large)
            descs_large.append(descs)

        return torch.stack(descs_large)

    def _forward_global_desc(self, x: List[torch.Tensor]) -> torch.Tensor:

        # 1. Create Batch of Rotations: (B * n_augs, C, H, W)
        # This is faster than looping the backbone 8 times
        x_augmented_list = self._create_x_augmented_list(x)

        # Stack into a large batch
        x_large_batch = torch.cat(x_augmented_list, dim=0) # (B*8, C, H, W)

        # 2. Run Backbone ONCE on large batch
        # We assume backbone maps (B*8, C, H, W) -> (B*8, C_out, H_out, W_out)
        with torch.no_grad():
            descs_large = self.backbone(x_large_batch) # (B*8, C_out, H_out, W_out)

        return descs_large

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) - Can be image or dense feature map
        """
        B, C, H, W = x.shape

        if self.n_augs <= 1:
            feats = self.backbone(x)
            global_desc = self.descriptor(feats)
        else:
            # Inference Mode with TTA (Test Time Augmentation)
            
            descs_large = self._internal_forward(x)

            # 4. Reshape to separate views
            # (B*8, D) -> (8, B, D) -> (B, 8, D)
            descs_views = descs_large.view(self.n_augs, B, -1).permute(1, 0, 2)

            # 5. Multi-View Pooling (The crucial step)
            # Max pooling creates the strongest rotation invariance
            # global_desc = torch.max(descs_views, dim=1)[0] # (B, D)

            global_desc = power_mean_pooling(descs_views, p=3.0)

            # descs_views = nn.functional.normalize(descs_views, p=2, dim=-1)
            # global_desc = descs_views.mean(dim=1)
            # global_desc = nn.functional.normalize(global_desc, p=2, dim=-1)

            # scores = torch.norm(descs_views, dim=-1)        # (B, N)
            # weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            # global_desc = (descs_views * weights).sum(dim=1)


        # Final Projection
        output = self.final_projection(global_desc)
        return nn.functional.normalize(output, p=2, dim=1)

    def get_embeddings(self, *inputs: torch.Tensor) -> tuple:
        """
        (Compatibility with DualModalityContrastiveModel)
        Get embeddings from the backbone for each provided input.
        Returns a tuple containing the forward outputs for each input.
        """
        return tuple(self.forward(x) for x in inputs)

def create_aug_global_descriptor_model(backbone_chkp_path: str,
                                        descriptor_type: str, 
                                        descriptor_kwargs: Dict[str, Any] = None, 
                                        output_dim: int = "same", 
                                        n_augs: int = 8, 
                                        device: str = "cpu") -> nn.Module:
    if descriptor_type == "global_desc":
        backbone = load_global_descriptor_model_from_checkpoint(
            checkpoint_path=backbone_chkp_path,
        )
        num_channels = None
    else:
        backbone = ContrastiveLearningModule.load_from_checkpoint(
                checkpoint_path=backbone_chkp_path,
                map_location=device,
                load_dino_weights=False,
                weights_only=False
            )
        num_channels = backbone.model_config["projection_dim"]
    

    model = GlobalDescriptorModelAug(
        backbone=backbone,
        backbone_channels=num_channels,
        descriptor_type=descriptor_type,
        descriptor_kwargs=descriptor_kwargs,
        output_dim=output_dim,
        n_augs=n_augs
    )
    model.backbone.eval()
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.to(device)
    return model
