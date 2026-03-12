"""
Flexible model architecture for contrastive learning.
Supports different backbones and projection heads.
"""

import itertools
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any, List
from mmfe_utils.dino_utils import load_dino, MODEL_TO_NUM_LAYERS
from third_parties.MoGe.modules import MoGe_1_Head, MoGe_2_Head

class ContrastiveModel(nn.Module):
    """
    Flexible contrastive learning model with configurable backbone and projection head.
    
    Args:
        backbone_name: Name of the backbone architecture (e.g., 'resnet18', 'resnet50')
        projection_dim: Dimension of the projection head output (default: 128)
        pretrained: Whether to use pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone parameters (default: False)
        projection_head_type: Type of projection head ('mlp', 'linear', 'cnn', 'none')
        backbone_kwargs: Additional kwargs for backbone initialization
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        projection_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        projection_spatial: Optional[tuple] = None,
        projection_head_type: str = "mlp",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        load_dino_weights: bool = True,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.projection_head_type = projection_head_type
        self.projection_spatial = projection_spatial
        self.load_dino_weights = load_dino_weights

        if backbone_name.startswith("dinov3"):
            self.return_cls = backbone_kwargs.get("return_cls", False)
        
        # Initialize backbone
        self.backbone = self._create_backbone(backbone_name, pretrained, backbone_kwargs or {})

        backbone_dim: Optional[int] = None
        # Only derive backbone_dim / strip classifiers when a projection head needs it
        if projection_head_type != "none":
            if hasattr(self.backbone, 'fc'):
                # ResNet
                backbone_dim = self.backbone.fc.in_features
                # Remove the final classification layer
                # self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            elif hasattr(self.backbone, 'classifier'):
                # Other architectures
                backbone_dim = self.backbone.classifier.in_features
            elif self.backbone_name.startswith("dinov3"):
                # DINOv3
                backbone_dim = self.backbone.patch_embed.proj.out_channels
            else:
                raise ValueError(f"Unsupported Backbone: {backbone_name}")

        if backbone_name.startswith("dinov3"):

            self.n_layers = MODEL_TO_NUM_LAYERS[backbone_name]

            # min_tokens, max_tokens = [1200, 2500]
            # resolution_level = 0
            # self.num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
            self.num_tokens = 1024
            
        else:
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create projection head
        if projection_head_type == "mlp":
            self.projection_head = nn.Sequential(
                nn.Linear(backbone_dim, backbone_dim),
                nn.ReLU(inplace=True),
                nn.Linear(backbone_dim, projection_dim),
            )

        elif projection_head_type == "linear":
            self.projection_head = nn.Linear(backbone_dim, projection_dim)

        elif projection_head_type == "cnn":
            # Calculate the spatial dimensions for the 2D latent
            # Assuming backbone outputs features of size (B, backbone_dim, 1, 1) after global pooling
            # Allow explicit control of spatial resolution (H x W). Defaults to square (projection_dim x projection_dim).
            if self.projection_spatial is None:
                out_h, out_w = 32,32
            else:
                if not (isinstance(self.projection_spatial, (list, tuple)) and len(self.projection_spatial) == 2):
                    raise ValueError("projection_spatial must be a (H, W) tuple if provided")
                out_h, out_w = int(self.projection_spatial[0]), int(self.projection_spatial[1])
            self.projection_head = self._create_cnn_projection_head(backbone_dim, projection_dim, out_h, out_w)

        elif projection_head_type == "none":
            self.projection_head = nn.Identity()

        elif projection_head_type == "MoGe1":
            self.projection_head = MoGe_1_Head(
                                    num_features=self.n_layers, 
                                    dim_in=self.backbone.blocks[0].attn.qkv.in_features, 
                                    dim_out=[self.projection_dim], 
                                    dim_proj=512,
                                    dim_upsample=[256, 128, 64],
                                    dim_times_res_block_hidden=2,
                                    num_res_blocks=2,
                                    res_block_norm="group_norm",
                                    last_res_blocks=0,
                                    last_conv_channels=2*self.projection_dim,
                                    last_conv_size=1,
                                    projection_spatial=self.projection_spatial
                                )

        elif projection_head_type == "MoGe2":
            raise NotImplementedError("MoGe2 is not implemented")
            self.projection_head = MoGeProjectionHead(backbone_dim, projection_dim)

        else:
            raise ValueError(f"Unsupported projection head type: {projection_head_type}")
        
        # Normalization layer for contrastive learning
        self.normalize = nn.functional.normalize
    
    def _create_backbone(self, name: str, pretrained: bool, kwargs: Dict[str, Any]) -> nn.Module:
        """Create backbone model by name."""
        name = name.lower()
        
        if name.startswith("resnet"):
            if name == "resnet18":
                return models.resnet18(pretrained=pretrained, **kwargs)
            elif name == "resnet34":
                return models.resnet34(pretrained=pretrained, **kwargs)
            elif name == "resnet50":
                return models.resnet50(pretrained=pretrained, **kwargs)
            elif name == "resnet101":
                return models.resnet101(pretrained=pretrained, **kwargs)
            elif name == "resnet152":
                return models.resnet152(pretrained=pretrained, **kwargs)
            else:
                raise ValueError(f"Unsupported ResNet variant: {name}")
        elif name.startswith("efficientnet"):
            if name == "efficientnet_b0":
                return models.efficientnet_b0(pretrained=pretrained, **kwargs)
            elif name == "efficientnet_b1":
                return models.efficientnet_b1(pretrained=pretrained, **kwargs)
            elif name == "efficientnet_b2":
                return models.efficientnet_b2(pretrained=pretrained, **kwargs)
            elif name == "efficientnet_b3":
                return models.efficientnet_b3(pretrained=pretrained, **kwargs)
            elif name == "efficientnet_b4":
                return models.efficientnet_b4(pretrained=pretrained, **kwargs)
            else:
                raise ValueError(f"Unsupported EfficientNet variant: {name}")
        elif name.startswith("dino"):
            return load_dino(name, load_dino_weights=self.load_dino_weights, **kwargs)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
    
    def _create_cnn_projection_head(self, backbone_dim: int, out_channels: int, out_height: int, out_width: Optional[int] = None) -> nn.Module:
        """
        Create CNN projection head using transposed convolutions.
        
        Args:
            backbone_dim: Input dimension from backbone
            out_channels: Number of output channels for the projected feature map
            out_height: Target output height (H)
            out_width: Target output width (W). If None, uses out_height (square output)
            
        Returns:
            CNN projection head module
        """
        # Normalize width argument
        if out_width is None:
            out_width = out_height

        def next_dim_div(dim: int) -> int:
            return dim // 2 

        def next_dim_mul(dim: int) -> int:
            return dim * 2

        def find_convtranspose2d_params(H_in, H_out, 
                                        stride_options=(1,2,3,4),
                                        kernel_options=(2,3,4,5,6),
                                        padding_options=(0,1,2,3),
                                        output_padding_options=(0,1)):
            """
            Find ConvTranspose2d parameter sets that map H_in -> H_out.
            Works for both height and width since the formula is the same.
            """
            if 2 * H_in == H_out:
                return {
                    "stride": 2,
                    "kernel_size": 4,
                    "padding": 1,
                    "output_padding": 0,
                }
            else:
                for stride, kernel, padding, out_pad in itertools.product(
                    stride_options, kernel_options, padding_options, output_padding_options
                ):
                    if out_pad >= stride:
                        continue
                    H_calc = (H_in - 1) * stride - 2 * padding + kernel + out_pad
                    if H_calc == H_out:
                        return {
                            "stride": stride,
                            "kernel_size": kernel,
                            "padding": padding,
                            "output_padding": out_pad,
                        }

        next_dim = next_dim_div if backbone_dim > out_channels else next_dim_mul

        # Calculate intermediate dimensions for transposed convolutions
        # Start from 1x1 and upscale to out_height x out_width
        layers = []
        
        # First layer: expand from 1x1 to 2x2
        initial_channels = max(next_dim(backbone_dim), out_channels)
        layers.append(nn.ConvTranspose2d(backbone_dim, initial_channels, kernel_size=2, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(initial_channels))
        layers.append(nn.ReLU(inplace=True))
        
        current_h = 2 if not self.backbone_name.startswith("dinov3") else 16
        current_w = 2 if not self.backbone_name.startswith("dinov3") else 16
        current_channels = initial_channels
        
        # Progressively upscale until we reach the target size
        while (current_h <= out_height) or (current_w <= out_width):
            next_h = min(current_h * 2, out_height)
            next_w = min(current_w * 2, out_width)

            convtranspose2d_params = find_convtranspose2d_params(current_h, next_h)
            kernel_h = convtranspose2d_params["kernel_size"]
            kernel_w = convtranspose2d_params["kernel_size"]
            stride_h = convtranspose2d_params["stride"]
            stride_w = convtranspose2d_params["stride"]
            padding_h = convtranspose2d_params["padding"]
            padding_w = convtranspose2d_params["padding"]
            output_padding_h = convtranspose2d_params["output_padding"]
            output_padding_w = convtranspose2d_params["output_padding"]
            
            # Reduce channels as we increase spatial size
            next_channels = max(next_dim(current_channels), out_channels)
            
            layers.append(nn.ConvTranspose2d(
                current_channels, 
                next_channels, 
                kernel_size=(kernel_h, kernel_w), 
                stride=(stride_h, stride_w), 
                padding=(padding_h, padding_w),
                output_padding=(output_padding_h, output_padding_w)
            ))
            layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.ReLU(inplace=True))
            
            current_channels = next_channels

            if current_h == out_height and current_w == out_width:
                break
            current_h = next_h
            current_w = next_w
        
        # Final layer to set the desired number of channels, preserve HxW
        layers.append(nn.ConvTranspose2d(
            current_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Normalized embeddings of shape (B, projection_dim) for linear/mlp heads
            or (B, C, H, W) for cnn head where C=projection_dim and (H,W)=projection_spatial or (projection_dim, projection_dim) by default
        """
        if self.backbone_name.startswith("dinov3"):
            if self.projection_head_type == "cnn":
                if self.return_cls:
                    features = self.backbone.get_intermediate_layers(x, n=range(self.n_layers), reshape=True, norm=True, return_class_token=True)
                    cls_token = features[-1][-1] # Last CLS token
                    feats = features[-1][0] # Last features
                else:
                    features = self.backbone.get_intermediate_layers(x, n=range(self.n_layers), reshape=True, norm=True)
                    feats = features[-1]
                embeddings = self.projection_head(feats)

            elif self.projection_head_type == "MoGe1":

                # Resize to expected resolution defined by num_tokens
                if self.projection_spatial is None:
                    original_height, original_width = x.shape[-2:]
                    num_tokens = self.num_tokens
                    resize_factor = ((num_tokens * 16 ** 2) / (original_height * original_width)) ** 0.5
                    resized_width, resized_height = int(original_width * resize_factor), int(original_height * resize_factor)
                    image = torch.nn.functional.interpolate(x, (resized_height, resized_width), mode="bicubic", align_corners=False, antialias=True)
                    # Apply image transformation for DINOv2
                    image_16 = torch.nn.functional.interpolate(image, (resized_height // 16 * 16, resized_width // 16 * 16), mode="bilinear", align_corners=False, antialias=True)
                    features = self.backbone.get_intermediate_layers(image_16, n=range(self.n_layers), return_class_token=True)
                    if self.return_cls:
                        cls_token = features[-1][-1] # Last CLS token
                    embeddings = self.projection_head(features, image)[0]
                else:
                    # We skip resizing because we don't want to get per pixel features
                    x = torch.nn.functional.interpolate(x, (x.shape[-2] // 16 * 16, x.shape[-1] // 16 * 16), mode="bilinear", align_corners=False, antialias=True)
                    features = self.backbone.get_intermediate_layers(x, n=range(self.n_layers), return_class_token=True)
                    if self.return_cls:
                        cls_token = features[-1][-1] # Last CLS token
                    embeddings = self.projection_head(features, x)[0]

            elif self.projection_head_type == "MoGe2":
                raise NotImplementedError("MoGe2 is not implemented")

            elif self.projection_head_type == "none":
                if self.return_cls:
                    features, cls_token = self.backbone.get_intermediate_layers(x.cuda(), n=range(self.n_layers), reshape=True, norm=True, return_class_token=True)
                else:
                    features = self.backbone.get_intermediate_layers(x.cuda(), n=range(self.n_layers), reshape=True, norm=True)
                embeddings = features[-1]

            else:
                raise ValueError(f"Unsupported projection head type: {self.backbone_name}")


        else:
            # Extract features from backbone
            features = self.backbone(x)
            
            # Handle different projection head types
            if self.projection_head_type == "cnn":
                # For CNN head, features should be (B, backbone_dim, 1, 1) from global pooling
                # Reshape to ensure proper spatial dimensions
                if features.dim() == 2:
                    # If features are flattened, reshape to (B, backbone_dim, 1, 1)
                    batch_size = features.size(0)
                    features = features.view(batch_size, -1, 1, 1)
            else:
                # For linear/mlp heads, flatten features (B, C, H, W) -> (B, C*H*W)
                if features.dim() > 2:
                    features = features.flatten(1)
                
            # Project to contrastive space
            embeddings = self.projection_head(features)
            
        # Normalize for contrastive learning
        if normalize:
            normalized_embeddings = self.normalize(embeddings, p=2, dim=1)
        else:
            normalized_embeddings = embeddings
        
        if self.backbone_name.startswith("dino") and self.return_cls:
            return normalized_embeddings, cls_token
        else:
            return normalized_embeddings
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from backbone without projection head."""
        features = self.backbone(x)
        if self.projection_head_type != "cnn" and features.dim() > 2:
            features = features.flatten(1)
        return features


class DualModalityContrastiveModel(nn.Module):
    """
    Model that processes two modalities separately and produces contrastive embeddings.
    
    Args:
        backbone_name: Name of the backbone architecture
        projection_dim: Dimension of the projection head output
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        projection_head_type: Type of projection head
        backbone_kwargs: Additional kwargs for backbone initialization
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet18",
        projection_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        projection_head_type: str = "mlp",
        projection_spatial: Optional[tuple] = None,
        load_dino_weights: bool = True,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Create two separate encoders (can be shared or separate)
        self.encoder = ContrastiveModel(
            backbone_name=backbone_name,
            projection_dim=projection_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            projection_head_type=projection_head_type,
            projection_spatial=projection_spatial,
            load_dino_weights=load_dino_weights,
            backbone_kwargs=backbone_kwargs
        )
    
    def forward(self, modality_0: torch.Tensor, modality_1: torch.Tensor = None, normalize: bool = True) -> tuple:
        """
        Forward pass for dual modality contrastive learning.
        
        Args:
            modality_0: First modality tensor (B, C, H, W)
            modality_1: Second modality tensor (B, C, H, W)
            
        Returns:
            Tuple of (embeddings_0, embeddings_1) where each is (B, projection_dim)
        """
        embeddings_0 = self.encoder(modality_0, normalize=normalize)

        if modality_1 is None:
            return embeddings_0

        if modality_1 is not None:
            embeddings_1 = self.encoder(modality_1, normalize=normalize)
            return embeddings_0, embeddings_1


def create_model(
    model_type: str = "dual_modality",
    backbone_name: str = "resnet18",
    projection_dim: int = 128,
    projection_spatial: Optional[tuple] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    projection_head_type: str = "mlp",
    backbone_kwargs: Optional[Dict[str, Any]] = None,
    load_dino_weights: bool = True,
) -> nn.Module:
    """
    Factory function to create contrastive learning models.
    
    Args:
        model_type: Type of model ('dual_modality', 'single')
        backbone_name: Name of the backbone architecture
        projection_dim: Dimension of the projection head output
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone parameters
        projection_head_type: Type of projection head
        backbone_kwargs: Additional kwargs for backbone initialization
        
    Returns:
        Configured model
    """
    if model_type == "dual_modality":
        return DualModalityContrastiveModel(
            backbone_name=backbone_name,
            projection_dim=projection_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            projection_head_type=projection_head_type,
            projection_spatial=projection_spatial,
            backbone_kwargs=backbone_kwargs,
            load_dino_weights=load_dino_weights,
        )
    elif model_type == "single":
        return ContrastiveModel(
            backbone_name=backbone_name,
            projection_dim=projection_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            projection_head_type=projection_head_type,
            backbone_kwargs=backbone_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
