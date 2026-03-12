import torch
import torch.nn.functional as F
from typing import Optional

# DINOv3 model identifiers (mirrors those used in training.models)
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"
MODEL_DINOV2_VITB14 = "dinov2_vitb14"

MODEL_TO_NUM_LAYERS = {
                MODEL_DINOV3_VITS: 12,
                MODEL_DINOV3_VITSP: 12,
                MODEL_DINOV3_VITB: 12,
                MODEL_DINOV3_VITL: 24,
                MODEL_DINOV3_VITHP: 32,
                MODEL_DINOV3_VIT7B: 40,
            }

DINOV2_GITHUB_LOCATION = "facebookresearch/dinov2"
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

available_dinos = [
        MODEL_DINOV3_VITS,
        MODEL_DINOV3_VITSP,
        MODEL_DINOV3_VITB,
        MODEL_DINOV3_VITL,
        MODEL_DINOV3_VITHP,
        MODEL_DINOV3_VIT7B,
        MODEL_DINOV2_VITB14,
    ]


def load_dino(
    name: str,
    load_dino_weights: bool = True,
    local_path: Optional[str] = None,
    dino_weights_path: Optional[str] = None,
):
    """
    Load a DINOv3 backbone by name and optionally apply checkpoint weights.

    Args:
        name: DINOv3 model name (e.g., 'dinov3_vitb16').
        load_dino_weights: If True and a checkpoint path is provided, load weights.
        local_path: Optional local repo path for torch.hub (used first if provided).
        dino_weights_path: Optional path to a checkpoint to load into the model.

    Returns:
        Instantiated DINOv3 model.

    Raises:
        ValueError: If the provided model name is not a supported DINOv3 variant.
    """

    name = name.lower()
    print(f"DINOv3 model: {name}")
    print(f"Available DINOv3 models: {available_dinos}")

    if name not in available_dinos:
        raise ValueError(f"Unsupported DINO variant: {name}")

    # Try local repo first if provided; otherwise fall back to GitHub
    model = None
    if local_path is not None:
        try:
            model = torch.hub.load(
                repo_or_dir=local_path,
                model=name,
                source="local",
                pretrained=False,
            )
            print(f"Loading DINO weights from {local_path}")
        except Exception:
            model = None

    if model is None:
        dino_github = DINOV3_GITHUB_LOCATION if name.startswith("dinov3") else DINOV2_GITHUB_LOCATION
        model = torch.hub.load(
            repo_or_dir=dino_github,
            model=name,
            source="github",
            pretrained=False,
        )
        print(f"Loading DINO weights from {dino_github}")

    if load_dino_weights and dino_weights_path is not None:
        print(f"Loading DINO weights from {dino_weights_path}")
        state = torch.load(dino_weights_path, map_location="cpu")

        # Handle common checkpoint formats
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]

        load_res = model.load_state_dict(state, strict=False)
        if load_res.missing_keys:
            print(f"Missing keys: {load_res.missing_keys}")
        if load_res.unexpected_keys:
            print(f"\nUnexpected keys: {load_res.unexpected_keys}")
        print(f"\nWeights loaded successfully. {len(load_res.missing_keys)} missing keys, {len(load_res.unexpected_keys)} unexpected keys")

    return model

def get_last_feature_dino(model, image, model_name: str):
    """
    Get the last feature of a DINO model.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if model_name.startswith("dinov3") and (image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0):
        image = make_dinov3_size_compatible(image)
    elif model_name.startswith("dinov2") and (image.shape[-2] % 14 != 0 or image.shape[-1] % 14 != 0):
        image = make_dinov2_size_compatible(image)

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(image.cuda(), n=range(model.num_heads), reshape=True, norm=True)
            x = feats[-1]
            # dim = x.shape[0]
            # x = x.view(dim, -1).permute(1, 0)
    return x

def make_dinov3_size_compatible(image):
    'Make the image H and W divisible by 16'
    return _make_size_compatible(image, divisor=16)

def make_dinov2_size_compatible(image):
    'Make the image H and W divisible by 14'
    return _make_size_compatible(image, divisor=14)


def _make_size_compatible(image: torch.Tensor, divisor: int) -> torch.Tensor:
    """
    Resize tensor spatial dims to closest multiple of `divisor`.
    """
    squeeze_batch = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True

    if image.dim() != 4:
        raise ValueError("Expected image tensor with shape (C,H,W) or (B,C,H,W)")

    _, _, height, width = image.shape
    target_h = max(1, int(round(height / divisor))) * divisor
    target_w = max(1, int(round(width / divisor))) * divisor

    if target_h == height and target_w == width:
        return image.squeeze(0) if squeeze_batch else image

    resized = F.interpolate(
        image,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0) if squeeze_batch else resized