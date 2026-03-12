import os
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint

from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
import os
from hydra.utils import to_absolute_path
from global_descriptors.backbones import GlobalDescriptorBackbone
from global_descriptors.cls_poolers import GeMPool, AttentionPool, MHSA_Pooler
from mmfe_utils.models_utils import load_salad
from global_descriptors.netvlad.netvlad import generate_netvlad_clusters

from torchvision.transforms import functional as TF

class GlobalDescriptorModel(nn.Module):
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
        descriptor_type: str = "netvlad",
        descriptor_kwargs: Optional[Dict[str, Any]] = None,
        output_dim: int = 512,
        model_name: str = "global_descriptor",
        creation_config: Optional[Dict[str, Any]] = None,
        mmfe_to_salad_adapter: Optional[nn.Module] = None,
        n_augs: int = 0,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.descriptor_type = descriptor_type.lower()
        self.descriptor_kwargs = descriptor_kwargs or {}
        self.model_name = model_name
        self.output_dim = output_dim
        self.creation_config = creation_config or {}
        self.cls_adapter = None
        self.cls_aggregator = None
        self.mmfe_to_salad_adapter = mmfe_to_salad_adapter
        self.n_augs = n_augs
        self.aug_step = 360 / n_augs if n_augs > 0 else 0

        if self.n_augs > 0:
            self.forward_backbone = self._augmented_backbone_forward
        else:
            self.forward_backbone = self._single_backbone_forward
        
        if (creation_config is not None and
         creation_config.backbone.get("kwargs", None) is not None and
         creation_config.backbone.kwargs.get("cls_token_type", None) is not None):
            self.cls_token_type = creation_config.backbone.kwargs.get("cls_token_type")
        else:
            self.cls_token_type = "max_pool"

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

        elif self.descriptor_type == "salad":
            descriptor_output_dim = backbone_channels
            self.descriptor = self.descriptor_kwargs["salad_aggregator"]
            
            mmfe_salad = (isinstance(self.backbone, GlobalDescriptorBackbone) and # self.backbone is a GlobalDescriptorBackbone
                            isinstance(self.backbone.backbone, ContrastiveLearningModule) # self.backbone.backbone is a ContrastiveLearningModule
                           ) 

            if mmfe_salad:
                if self.cls_token_type == "dino":
                    assert self.backbone.backbone.model.encoder.return_cls, (
                        f"DINO backbone does not return CLS token for cls type dino"
                    )
                    self.cls_adapter = nn.Sequential(
                        nn.Linear(768, 512),
                        nn.ReLU(),
                        nn.Linear(512, self.backbone.backbone.model_config["projection_dim"])
                    )

                else:
                    projection_dim = 768
                    hidden_dim = projection_dim // 2

                    if self.cls_token_type == "max_pool":
                        self.cls_aggregator = nn.Sequential(
                            nn.AdaptiveMaxPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(projection_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, projection_dim),
                        )

                    elif self.cls_token_type == "mean_pool":
                        self.cls_aggregator = nn.Sequential(
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(projection_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, projection_dim),
                        )

                    elif self.cls_token_type == "gem":
                        self.cls_aggregator = nn.Sequential(
                            GeMPool(),                          # outputs [B, C]
                            nn.Linear(projection_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, projection_dim),
                        )

                    elif self.cls_token_type == "attention_pool":
                        self.cls_aggregator = nn.Sequential(
                            AttentionPool(projection_dim),      # outputs [B, C]
                            nn.Linear(projection_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, projection_dim),
                        )

                    elif self.cls_token_type == "mhsa_pool":
                        self.cls_aggregator = nn.Sequential(
                            MHSA_Pooler(projection_dim, num_heads=4),  # outputs [B, C]
                            nn.Linear(projection_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, projection_dim),
                        )

                    else:
                        raise ValueError(
                            f"Not supported backbone aggregator for CLS type {self.cls_token_type}"
                        )
            

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
            
        else:
            raise ValueError(f"Unsupported descriptor type: {descriptor_type}")
        
        # Final projection to output dimension
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

    def _single_backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _augmented_backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_augmented_list = self._create_x_augmented_list(x)
        x_large_batch = torch.cat(x_augmented_list, dim=0) # (B*8, C, H, W)
        features_2d = self.backbone(x_large_batch) # (B*8, C, H, W)
        views_features = features_2d = features_2d.view(B, self.n_augs, *features_2d.shape[1:])
        single_feats = torch.max(views_features, dim=0)[0]
        return single_feats

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone and global descriptor.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Global descriptors (B, output_dim)
        """
        # Get 2D feature maps from backbone
        features_2d = self.forward_backbone(x)

        if self.mmfe_to_salad_adapter is not None:
            og_features = features_2d[0] if (not isinstance(features_2d, torch.Tensor) and len(features_2d) == 2) else features_2d  # (B, C, H, W)
            adapted_features = self.mmfe_to_salad_adapter(og_features)
            features_2d = (adapted_features, features_2d[1]) if (not isinstance(features_2d, torch.Tensor) and len(features_2d) == 2) else adapted_features

        if hasattr(self, "cls_adapter") and self.cls_adapter is not None:
            new_cls = self.cls_adapter(features_2d[-1])
            features_2d = [features_2d[0], new_cls] # Replace the last CLS token with the new one
        elif getattr(self, "cls_aggregator", None) is not None:
            cls_token = self.cls_aggregator(features_2d)
            features_2d = (features_2d, cls_token)

        # Aggregate to 1D global descriptor
        global_desc = self.descriptor(features_2d)  # (B, descriptor_dim)
        
        # Project to final output dimension
        output = self.final_projection(global_desc)  # (B, output_dim)
        
        # L2 normalize for contrastive learning
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        
        return output

    def get_embeddings(self, *inputs: torch.Tensor) -> tuple:
        """
        (Compatibility with DualModalityContrastiveModel)
        Get embeddings from the backbone for each provided input.
        Returns a tuple containing the forward outputs for each input.
        """
        return tuple(self.forward(x) for x in inputs)

def create_global_descriptor_model(cfg: DictConfig, load_weights: bool = True) -> nn.Module:
    """Create global descriptor model."""
    if cfg.descriptor.type.lower() != "salad":
        backbone = GlobalDescriptorBackbone(backbone_configs=cfg.backbone)
    else:
        backbone = None

    # Create global descriptor model
    descriptor_kwargs = OmegaConf.to_container(cfg.descriptor.kwargs, resolve=True) if cfg.descriptor.get("kwargs") else {}

    mmfe_to_salad_adapter = None
    
    # Check if using NetVLAD and if clusters need to be generated
    if cfg.descriptor.type.lower() == "netvlad":
        cluster_path = descriptor_kwargs.get("cluster_path")
        
        if cluster_path is not None and load_weights:
            # Cluster path is specified
            cluster_path = to_absolute_path(cluster_path)
            if not os.path.exists(cluster_path):
                cluster_path = generate_netvlad_clusters(cfg)
                descriptor_kwargs["cluster_path"] = cluster_path

        elif cluster_path is None:
            print(f"Warning: Specified cluster file not found: {cluster_path}")
            print("NetVLAD will use random initialization. Set precomputed_clusters=true to precompute clusters.")  

        if cfg.backbone.name.startswith("vgg"):
            backbone_channels = 512
        elif cfg.backbone.name.startswith("moge_dino"):
            backbone_channels = 32
        elif cfg.backbone.name.startswith("dino"):
            backbone_channels = 768
        elif cfg.backbone.name == "mmfe":
            backbone_channels = 32
        else:
            raise ValueError(f"Unsupported backbone name: {cfg.backbone.name}")


    elif cfg.descriptor.type.lower() == "salad":
        if cfg.descriptor.kwargs.pretrained:
            if cfg.backbone.name == "dinov2_vitb14":
                DINOV2_LOCAL_PATH = os.getenv("DINOV2_LOCAL_PATH")
                backbone_config = {"loading_config": {"local_path": to_absolute_path(DINOV2_LOCAL_PATH)}, 
                                    'return_token': True, 'norm_layer': True,"num_trainable_blocks": 4}
                agg_config = {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256}
                backbone_channels = 128*64 + 256

                SALAD_LOCAL_PATH = os.getenv("SALAD_LOCAL_PATH")
                SALAD_WEIGHTS_PATH = os.getenv("SALAD_WEIGHTS_PATH")
                
                model = load_salad(backbone_name=cfg.backbone.name,
                                    local_path_salad=to_absolute_path(SALAD_LOCAL_PATH), 
                                    salad_weights_path=to_absolute_path(SALAD_WEIGHTS_PATH),
                                    agg_config=agg_config,
                                    backbone_config=backbone_config)

            elif cfg.backbone.name == "dinov3_vitb16":
                if cfg.backbone.get("kwargs", None) is not None:
                    dino_local_path = cfg.backbone.kwargs.get("local_path", None)
                    dino_weights_path = cfg.backbone.kwargs.get("dino_weights_path", None)
                else:
                    dino_local_path = None
                    dino_weights_path = None
                    
                DINOV3_LOCAL_PATH = os.getenv("DINOV3_LOCAL_PATH", dino_local_path)
                DINOV3_WEIGHTS_PATH = os.getenv("DINOV3_WEIGHTS_PATH", dino_weights_path)
                backbone_config = {"local_path": to_absolute_path(DINOV3_LOCAL_PATH) if DINOV3_LOCAL_PATH is not None else None, 
                                    "dino_weights_path": to_absolute_path(DINOV3_WEIGHTS_PATH) if DINOV3_WEIGHTS_PATH is not None else None,
                                    "pretrained": True,
                                    'return_token': True, 'norm_layer': True,"num_trainable_blocks": 4}
                agg_config = {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256}
                backbone_channels = 128*64 + 256

                SALAD_LOCAL_PATH = os.getenv("SALAD_LOCAL_PATH")
                
                model = load_salad(backbone_name=cfg.backbone.name,
                                    local_path_salad=to_absolute_path(SALAD_LOCAL_PATH), 
                                    salad_weights_path=None,
                                    agg_config=agg_config,
                                    backbone_config=backbone_config)
            
            elif cfg.backbone.name == "mmfe":
                # backbone_model, backbone_creation_config = load_contrastive_model_from_checkpoint(checkpoint_path=to_absolute_path(cfg.backbone.kwargs.checkpoint_path),
                #                                                         return_cls=True)
                backbone_model = GlobalDescriptorBackbone(backbone_configs=cfg.backbone)
                if cfg.backbone.get("kwargs", {}).get("return_cls", False) and cfg.backbone.get("kwargs", {}).get("cls_token", None) == "dino":
                    backbone_model.backbone.model.encoder.return_cls = True
                backbone_creation_config = cfg.backbone
                #agg_config = {"num_channels": 32, "num_clusters": 64, "cluster_dim": 128, "token_dim": 32}
                agg_config = {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256}
                backbone_channels = 128*64 + 256

                mmfe_to_salad_adapter = nn.Sequential(
                                        nn.Conv2d(
                                            in_channels=32,     # mmfe feature channels
                                            out_channels=768,   # what SALAD expects
                                            kernel_size=1,      # preserve H,W
                                            stride=1,
                                            padding=0,
                                            bias=False
                                        ),
                                        nn.BatchNorm2d(768),    # optional but recommended for stability
                                        nn.ReLU(inplace=True)   # or remove if SALAD expects raw features
                                    )

                SALAD_LOCAL_PATH = os.getenv("SALAD_LOCAL_PATH")
                SALAD_WEIGHTS_PATH = os.getenv("SALAD_WEIGHTS_PATH")
                
                model = load_salad(backbone_name=backbone_model,
                                    local_path_salad=to_absolute_path(SALAD_LOCAL_PATH), 
                                    salad_weights_path=to_absolute_path(SALAD_WEIGHTS_PATH) if SALAD_WEIGHTS_PATH is not None else None,
                                    agg_config=agg_config,
                                    backbone_config=backbone_creation_config)

                            
        else:
            raise ValueError(f"Unsupported pretrained mode for salad: {cfg.descriptor.pretrained}")
            if cfg.backbone.name == "dinov2_vitb14":
                DINOV2_LOCAL_PATH = os.getenv("DINOV2_LOCAL_PATH")
                backbone_config = {"local_path": to_absolute_path(DINOV2_LOCAL_PATH), 
                                    'return_token': True, 'norm_layer': True,"num_trainable_blocks": 4}
                agg_config = {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256}
                backbone_channels = 128*64 + 256

            elif cfg.backbone.name == "dinov3_vitb16":
                raise ValueError(f"Non implemented for dinov3_vitb16")
                local_path_backbone = to_absolute_path(cfg.backbone.kwargs.local_path_backbone) if cfg.backbone.kwargs.local_path_backbone is not None else None
                DINOV3_LOCAL_PATH = os.getenv("DINOV3_LOCAL_PATH", local_path_backbone)
                DINOV3_WEIGHTS_PATH = os.getenv("DINOV3_WEIGHTS_PATH", cfg.backbone.kwargs.dino_weights_path)
                backbone_config = {"local_path": to_absolute_path(DINOV3_LOCAL_PATH), 
                                    "dino_weights_path": to_absolute_path(DINOV3_WEIGHTS_PATH),
                                    'return_token': True, 'norm_layer': True,"num_trainable_blocks": 4}
                agg_config = {"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256}
                backbone_channels = 128*64 + 256
            else:
                raise ValueError(f"Unsupported backbone name for salad: {cfg.backbone.name}")

        # if cfg.backbone.freeze == "dino":
        #     for param in model.backbone.parameters():
        #         param.requires_grad = False

        # elif cfg.backbone.freeze == "salad":
        #     pass

        # elif cfg.backbone.freeze == "all":
        #     for param in model.backbone.parameters():
        #         param.requires_grad = False
        # else:
        #     raise ValueError(f"Unsupported freeze mode for s: {cfg.backbone.freeze}")

        salad_aggregator = model.aggregator
        backbone = model.backbone
        descriptor_kwargs = {"salad_aggregator": salad_aggregator}
    
    
    model = GlobalDescriptorModel(
        backbone=backbone,
        backbone_channels=backbone_channels,
        descriptor_type=cfg.descriptor.type,
        descriptor_kwargs=descriptor_kwargs,
        output_dim=cfg.descriptor.output_dim,
        model_name=cfg.backbone.name,
        creation_config=cfg,
        mmfe_to_salad_adapter=mmfe_to_salad_adapter,
    )
    
    return model

def create_no_train_agg_model(cfg: DictConfig) -> nn.Module:
    mmfe_model = ContrastiveLearningModule.load_from_checkpoint(
            checkpoint_path=cfg.model.checkpoint,
            map_location=cfg.device,
            load_dino_weights=False
        )
    model = GlobalDescriptorModel(
        backbone=mmfe_model,
        backbone_channels=mmfe_model.backbone.backbone.model_config["projection_dim"],
        descriptor_type=cfg.descriptor.type,
        output_dim=cfg.descriptor.output_dim,
        model_name=cfg.backbone.name,
        creation_config=cfg,
    )
    return model

def load_global_descriptor_model_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    backbone: Optional[nn.Module] = None,
    backbone_channels: Optional[int] = None,
    descriptor_type: Optional[str] = None,
    descriptor_kwargs: Optional[Dict[str, Any]] = None,
    output_dim: Optional[int] = None,
) -> GlobalDescriptorModel:
    """
    Load a global descriptor model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        backbone: Optional backbone model. If None, will load from checkpoint.
        backbone_channels: Optional number of backbone output channels. If None, will infer.
        descriptor_type: Optional descriptor type ('netvlad', 'gem', 'avg', 'max'). If None, will infer.
        descriptor_kwargs: Optional descriptor kwargs. If None, will infer.
        output_dim: Optional final output dimension. If None, will infer.
        
    Returns:
        Loaded GlobalDescriptorModel in eval mode
        
    Note:
        The function will automatically load the backbone from the checkpoint and infer
        most configuration parameters from the state dict.
    """

    # if "salad" in checkpoint_path:
    #     local_path = os.getenv("DINOV2_LOCAL_PATH")
    #     salad_weights_path = os.getenv("SALAD_WEIGHTS_PATH")
    #     return load_salad(backbone_name="dinov2_salad", 
    #                         local_path_salad=to_absolute_path(local_path), 
    #                         salad_weights_path=to_absolute_path(salad_weights_path))

    print(f"Loading global descriptor model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    state_dict = checkpoint['state_dict']
    
    model_creation_config = checkpoint["hyper_parameters"].get('info_for_loading', {}).get('model_creation_config', None)

    if model_creation_config is not None:
        
        model = create_global_descriptor_model(model_creation_config, load_weights=False)
        model_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('model.'):
                # Remove "model." prefix
                model_key = key[6:]
                # if model_key.startswith("backbone."):
                #     model_key = model_key[9:]
                model_state_dict[model_key] = value
    else:
        print("WARNING: Model creation config not found in checkpoint. Resorting to legacy loading")
        # Extract model name from checkpoint
        load_info = checkpoint["hyper_parameters"].get('info_for_loading', None)

        if load_info is not None:
            model_name = load_info.get('model_name', None)
            backbone_channels = load_info.get('backbone_channels', None)
            descriptor_type = load_info.get('descriptor_type', None)
            descriptor_kwargs = load_info.get('descriptor_kwargs', None)
            output_dim = load_info.get('output_dim', None)
        else:
            raise ValueError("Load info not found in checkpoint")
        
        # The model state dict keys start with "model."
        # We need to extract the model-specific keys
        model_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('model.'):
                # Remove "model." prefix
                model_key = key[6:]
                # if model_key.startswith("backbone."):
                #     model_key = model_key[9:]
                model_state_dict[model_key] = value
        
        # Load backbone from checkpoint if not provided
        if backbone is None:
            print("Loading backbone from checkpoint...")
            
            if model_name.startswith("dinov3"):
                from global_descriptors.backbones import GlobalDescriptorBackbone
                backbone_configs = {"name": model_name, "pretrained": True, "freeze": "all"}
                backbone = GlobalDescriptorBackbone(backbone_configs=backbone_configs)
            elif model_name.startswith("vgg16"):
                from global_descriptors.backbones import GlobalDescriptorBackbone
                backbone_configs = {"name": model_name, "pretrained": True, "freeze": "all"}
                backbone = GlobalDescriptorBackbone(backbone_configs=backbone_configs)
            elif model_name.startswith("dinov2"):
                from global_descriptors.backbones import GlobalDescriptorBackbone
                backbone_configs = {"name": model_name, "pretrained": True, "freeze": "all"}
                backbone = GlobalDescriptorBackbone(backbone_configs=backbone_configs)
            elif model_name.startswith("mmfe"):
                from global_descriptors.backbones import GlobalDescriptorBackbone
                backbone_configs = {"name": model_name, "pretrained": True, "freeze": "all"}
                backbone = GlobalDescriptorBackbone(backbone_configs=backbone_configs)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

        # Infer model configuration from state dict if not provided
        if output_dim is None:
            # Try to infer output_dim from final_projection weights
            if 'final_projection.0.weight' in model_state_dict:
                output_dim = model_state_dict['final_projection.0.weight'].shape[0]
                print(f"Inferred output_dim: {output_dim}")
        
        if descriptor_type is None:
            # Try to infer descriptor type from state dict keys
            if any('descriptor.conv' in key for key in model_state_dict.keys()):
                descriptor_type = 'netvlad'
                print(f"Inferred descriptor_type: {descriptor_type}")
            elif any('descriptor.0.p' in key for key in model_state_dict.keys()):
                descriptor_type = 'gem'
                print(f"Inferred descriptor_type: {descriptor_type}")
            else:
                # Could be avg or max pooling
                print("Warning: Could not infer descriptor type. Defaulting to 'avg'")
                descriptor_type = 'avg'
        
        if descriptor_kwargs is None:
            descriptor_kwargs = {}
            if descriptor_type == 'netvlad':
                # Try to infer num_clusters from state dict
                if 'descriptor.conv.weight' in model_state_dict:
                    num_clusters = model_state_dict['descriptor.conv.weight'].shape[0]
                    descriptor_kwargs['num_clusters'] = num_clusters
                    print(f"Inferred num_clusters: {num_clusters}")
                # NetVLAD clusters will be loaded from state dict
                descriptor_kwargs['cluster_path'] = None
            elif descriptor_type == 'gem':
                # Try to infer p from state dict
                if 'descriptor.0.p' in model_state_dict:
                    p = model_state_dict['descriptor.0.p'].item()
                    descriptor_kwargs['p'] = p
                    print(f"Inferred GeM p: {p}")
        
        if backbone_channels is None:
            # Try to infer backbone channels from descriptor weights
            if descriptor_type == 'netvlad' and 'descriptor.conv.weight' in model_state_dict:
                backbone_channels = model_state_dict['descriptor.conv.weight'].shape[1]
                print(f"Inferred backbone_channels: {backbone_channels}")
            elif descriptor_type in ['avg', 'max'] and 'final_projection.0.weight' in model_state_dict:
                backbone_channels = model_state_dict['final_projection.0.weight'].shape[1]
                print(f"Inferred backbone_channels: {backbone_channels}")
            elif descriptor_type == 'gem' and 'final_projection.0.weight' in model_state_dict:
                backbone_channels = model_state_dict['final_projection.0.weight'].shape[1]
                print(f"Inferred backbone_channels: {backbone_channels}")
            else:
                raise ValueError("Could not infer backbone_channels. Please provide it explicitly.")
    
        # Create the model
        model = GlobalDescriptorModel(
            backbone=backbone,
            backbone_channels=backbone_channels,
            descriptor_type=descriptor_type,
            descriptor_kwargs=descriptor_kwargs,
            output_dim=output_dim,
        )
    
    # Load state dict
    load_result = model.load_state_dict(model_state_dict, strict=False)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)
    
    # Set to eval mode
    model.eval()
    
    print("✓ Model loaded successfully")
    return model
