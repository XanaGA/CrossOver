import os
from typing import Any, Dict, List, Union
from omegaconf import DictConfig, open_dict
from hydra.utils import to_absolute_path
import torch
import torchvision
import torch.nn as nn
from third_parties.MoGe.modules import MoGe_1_Head
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint
from mmfe_utils.dino_utils import MODEL_TO_NUM_LAYERS, DINOV3_GITHUB_LOCATION

def load_vgg16_backbone(pretrained: bool = True) -> nn.Module:
    """Load a VGG16 backbone from torchvision."""
    encoder = torchvision.models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    model = nn.Sequential(*layers)
    return model

def load_dino_backbone(name: str = "dinov3_vitb16",
                       local_path: str = None,
                       weights_path: str = None,
                       pretrained: bool = True) -> nn.Module:
    """Load a DINO backbone from torchvision."""
    try:
        model = torch.hub.load(
            repo_or_dir=local_path,
            model=name,
            source="local",
            pretrained=False,
        )
        print(f"Loading DINO weights from {local_path}")
    except Exception as e:
        print(f"Warning: Could not load DINO weights from {local_path}, using pretrained weights from {DINOV3_GITHUB_LOCATION}")
        model = torch.hub.load(
            repo_or_dir=DINOV3_GITHUB_LOCATION,
            model=name,
            source="github",
            pretrained=False,
        )
        print(f"Loading DINO weights from {DINOV3_GITHUB_LOCATION}")

    if pretrained and weights_path is not None:
        print(f"Loading DINO weights from {weights_path}")
        state = torch.load(to_absolute_path(weights_path).strip(), map_location="cpu")

        # Common checkpoint formats
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    return model

def load_dino_moge_backbone(name: str = "moge_dinov3_vitb16",
                            local_path: str = None,
                            weights_path: str = None,
                            dim_out: int = 768, 
                            projection_spatial: List[int] = [32, 32],
                            pretrained: bool = True) -> nn.Module:
    """Load a MMFE backbone from torchvision."""
    dino = load_dino_backbone(name[len("moge_"):], local_path,
                                weights_path,
                                pretrained)
    
    moge_head = MoGe_1_Head(
                            num_features=MODEL_TO_NUM_LAYERS[name[len("moge_"):]], 
                            dim_in=dino.blocks[0].attn.qkv.in_features, 
                            dim_out=[dim_out], 
                            dim_proj=512,
                            dim_upsample=[256, 128, 64],
                            dim_times_res_block_hidden=2,
                            num_res_blocks=2,
                            res_block_norm="group_norm",
                            last_res_blocks=0,
                            last_conv_channels=2*dim_out,
                            last_conv_size=1,
                            projection_spatial=projection_spatial
                        )
    return dino, moge_head

def load_mmfe_backbone(loading_src: Union[str, Dict[str, Any]]) -> nn.Module:
    """Load a pretrained MMFE backbone."""
    if isinstance(loading_src, str):
        checkpoint_path = to_absolute_path(loading_src)
        # model = ContrastiveLearningModule.load_from_checkpoint(checkpoint_path=to_absolute_path(checkpoint_path), 
        #                                                         map_location="cpu", load_dino_weights=False)
        model, creation_config = load_contrastive_model_from_checkpoint(checkpoint_path=checkpoint_path)
        return model, creation_config
    elif isinstance(loading_src, DictConfig):
        return ContrastiveLearningModule(**loading_src), None

class GlobalDescriptorBackbone(nn.Module):
    def __init__(self, backbone_configs: Dict[str, Any], backbone: nn.Module = None):
        super().__init__()
        self.backbone_configs = backbone_configs
        self.backbone = backbone
        name = backbone_configs["name"]
        if self.backbone is None:
            if name == "vgg16":
                self.backbone = load_vgg16_backbone(pretrained=backbone_configs["pretrained"])
            elif name.startswith("moge_dino"):
                if backbone_configs.get("kwargs", {}) is not None:
                    dino_local_path = backbone_configs.get("kwargs", {}).get("local_path", None)
                    dino_weights_path = backbone_configs.get("kwargs", {}).get("weights_path", None)
                else:
                    dino_local_path = None
                    dino_weights_path = None

                DINOV3_LOCAL_PATH = os.getenv("DINOV3_LOCAL_PATH", dino_local_path)
                DINOV3_WEIGHTS_PATH = os.getenv("DINOV3_WEIGHTS_PATH", dino_weights_path)

                self.dino, self.moge_head = load_dino_moge_backbone(name=backbone_configs["name"],
                                                    local_path=DINOV3_LOCAL_PATH,
                                                    weights_path=DINOV3_WEIGHTS_PATH,
                                                    dim_out=backbone_configs["kwargs"].get("dim_out", 768) if backbone_configs["kwargs"] is not None else 768,
                                                    projection_spatial=backbone_configs["kwargs"].get("projection_spatial", [32, 32]) if backbone_configs["kwargs"] is not None else [32, 32],
                                                    pretrained=backbone_configs["pretrained"])
            elif name.startswith("dinov3"):
                if backbone_configs.get("kwargs", {}) is not None:
                    dino_local_path = backbone_configs.get("kwargs", {}).get("local_path", None)
                    dino_weights_path = backbone_configs.get("kwargs", {}).get("weights_path", None)
                else:
                    dino_local_path = None
                    dino_weights_path = None

                DINOV3_LOCAL_PATH = os.getenv("DINOV3_LOCAL_PATH", dino_local_path)
                DINOV3_WEIGHTS_PATH = os.getenv("DINOV3_WEIGHTS_PATH", dino_weights_path)

                self.backbone = load_dino_backbone(name=backbone_configs["name"],
                                                local_path=DINOV3_LOCAL_PATH,
                                                weights_path=DINOV3_WEIGHTS_PATH,
                                                pretrained=backbone_configs.get("pretrained", True))
            elif name == "mmfe":
                if backbone_configs.get("kwargs", {}).get("mmfe_creation_config", None) is None:
                    self.backbone, creation_config = load_mmfe_backbone(loading_src=backbone_configs.get("kwargs", {}).get("checkpoint_path", None))
                    with open_dict(self.backbone_configs["kwargs"]):
                        self.backbone_configs["kwargs"]["mmfe_creation_config"] = creation_config
                else:
                    self.backbone, _ = load_mmfe_backbone(loading_src=backbone_configs.get("kwargs", {}).get("mmfe_creation_config", None))
            elif name == "dinov2_vitb14":
                self.backbone = None
                return
            else:
                raise ValueError(f"Unsupported backbone name: {name}")


        ##########################################################################################################################
        # Freeze the backbone
        ##########################################################################################################################
        if backbone_configs["freeze"].lower() == "all":
            if name.startswith("moge_dino"):
                for param in self.dino.parameters():
                    param.requires_grad = False
                for param in self.moge_head.parameters():
                    param.requires_grad = False
            elif name == "mmfe":
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False

        elif backbone_configs["freeze"].lower() == "dino":

            if name.startswith("moge_dino"):
                for param in self.dino.parameters():
                    param.requires_grad = False
            elif name == "mmfe":
                for param in self.backbone.model.encoder.backbone.parameters():
                    param.requires_grad = False
            elif name.startswith("dinov3"):
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"Backbone {name} does not support freeze mode {backbone_configs['freeze']}")
        elif backbone_configs["freeze"].lower() == "none":
            print(f"Trainig the whole backbone")
        else:
            raise ValueError(f"Unsupported freeze mode: {backbone_configs['freeze']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        name = self.backbone_configs["name"]
        if name == "vgg16":
            return self.backbone(x)
        elif name.startswith("moge_dino"):
            x = torch.nn.functional.interpolate(x, (x.shape[-2] // 16 * 16, x.shape[-1] // 16 * 16), mode="bilinear", align_corners=False, antialias=True)
            features = self.dino.get_intermediate_layers(x, n=range(MODEL_TO_NUM_LAYERS[name[len("moge_"):]]), return_class_token=True)
            embeddings = self.moge_head(features, x)[0]
            return embeddings
        elif name.startswith("dinov3"):
            feats = self.backbone.get_intermediate_layers(x, n=range(MODEL_TO_NUM_LAYERS[self.backbone_configs["name"]]), reshape=True, norm=True, return_class_token=False)
            feats = feats[-1]
            return feats
        elif name.startswith("dinov2"):
            feats = self.backbone.get_intermediate_layers(x, n=range(MODEL_TO_NUM_LAYERS[self.backbone_configs["name"]]), reshape=True, norm=True, return_class_token=True)
            feats = feats[-1]
            return feats
        elif name == "mmfe":
            return self.backbone(x)
        else:
            raise ValueError(f"Unsupported backbone name: {name}")