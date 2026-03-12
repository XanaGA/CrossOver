"""
Simple FPV image encoders.

These models take RGB images and produce a lower- (or equal-) resolution
feature grid with D channels, suitable for aligning with floorplan features.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models


class FPVImageEncoder(nn.Module):
    """
    CNN encoder for FPV images that outputs a 2D feature grid.

    By default this uses a ResNet backbone truncated before the global
    pooling / classification layers, followed by a 1x1 conv to set the
    output channel dimension.

    Args:
        backbone_name: Name of torchvision backbone (e.g. 'resnet18').
        out_channels: Number of channels D in the output feature grid.
        pretrained: If True, load ImageNet-pretrained weights.
        freeze_backbone: If True, freeze backbone parameters.
        backbone_kwargs: Optional kwargs passed to the backbone constructor.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        out_channels: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name.lower()
        backbone_kwargs = backbone_kwargs or {}

        if self.backbone_name.startswith("resnet"):
            resnet = self._create_resnet(self.backbone_name, pretrained, backbone_kwargs)
            # Keep everything up to (and including) layer4, drop avgpool / fc
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            # Infer the number of channels from the last block in layer4
            last_block = resnet.layer4[-1]
            if hasattr(last_block, "conv3"):
                in_channels = last_block.conv3.out_channels
            else:
                in_channels = last_block.conv2.out_channels
        else:
            raise ValueError(f"Unsupported backbone for FPVImageEncoder: {backbone_name}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Simple 1x1 conv head to set output channel dimension
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def _create_resnet(self, name: str, pretrained: bool, kwargs: Dict[str, Any]) -> nn.Module:
        name = name.lower()
        if name == "resnet18":
            return models.resnet18(pretrained=pretrained, **kwargs)
        if name == "resnet34":
            return models.resnet34(pretrained=pretrained, **kwargs)
        if name == "resnet50":
            return models.resnet50(pretrained=pretrained, **kwargs)
        if name == "resnet101":
            return models.resnet101(pretrained=pretrained, **kwargs)
        if name == "resnet152":
            return models.resnet152(pretrained=pretrained, **kwargs)
        raise ValueError(f"Unsupported ResNet variant for FPVImageEncoder: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Feature grid of shape (B, D, H', W'), where H' and W' are
            typically lower than H, W depending on the backbone stride.
        """
        feats = self.backbone(x)
        feats = self.head(feats)
        return feats


def create_fpv_image_encoder(
    backbone_name: str = "resnet18",
    out_channels: int = 128,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    backbone_kwargs: Optional[Dict[str, Any]] = None,
) -> FPVImageEncoder:
    """
    Convenience factory for FPVImageEncoder.
    """
    return FPVImageEncoder(
        backbone_name=backbone_name,
        out_channels=out_channels,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        backbone_kwargs=backbone_kwargs,
    )


__all__ = ["FPVImageEncoder", "create_fpv_image_encoder"]

