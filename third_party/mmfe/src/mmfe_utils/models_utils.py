"""
Utility functions for model construction and weight loading.

Currently provides a helper to load DINOv3 backbones and optionally apply
checkpoint weights, encapsulating the logic used in training models.
"""

import os
from typing import Optional, Union

from hydra.utils import to_absolute_path
import torch
from third_parties.salad.vpr_model import VPRModel



SALAD_GITHUB_LOCATION = "serizba/salad"


def load_salad(
    backbone_name: Union[str, torch.nn.Module] = "dinov2_vitb14",
    local_path_salad: Optional[str] = None,
    salad_weights_path: Optional[str] = None,
    backbone_config: Optional[dict] = None,
    agg_config: Optional[dict] = None,
):
    """
    Load a SALAD model (e.g., dinov2_salad) from a local repo or from GitHub,
    and optionally apply pretrained checkpoint weights.

    Args:
        name: Model name as defined in the SALAD hubconf.py (default: 'dinov2_salad').
        load_salad_weights: If True and a checkpoint path is provided, load weights.
        local_path_salad: Optional local repo path for torch.hub (used first if provided).
        local_path_dino: Optional local repo path for torch.hub (used first if provided).
        salad_weights_path: Optional path to a local checkpoint (.ckpt or .pth).

    Returns:
        Instantiated SALAD model ready for inference.
    """

    # Try loading locally first
    model = None
    if local_path_salad is not None:
        try:
            model = VPRModel(
                backbone_arch=backbone_name,
                backbone_config=backbone_config,
                agg_arch='SALAD',
                agg_config=agg_config,
            )
            print(f"Loaded SALAD model from local path: {local_path_salad}")

            # Optionally load pretrained weights
            if salad_weights_path is not None:
                print(f"Loading pretrained SALAD weights from {salad_weights_path}")
                state = torch.load(salad_weights_path, map_location="cpu")

                # Handle possible Lightning / state_dict formats
                if isinstance(state, dict):
                    if "state_dict" in state:
                        state = state["state_dict"]
                    elif "model" in state:
                        state = state["model"]

                if not isinstance(backbone_name, str) or backbone_name.startswith("dinov3"):
                    DINOV2_LOCAL_PATH = os.getenv("DINOV2_LOCAL_PATH")
                    dinov2_config = {"loading_config": {"local_path": to_absolute_path(DINOV2_LOCAL_PATH)}, 
                                    'return_token': True, 'norm_layer': True,"num_trainable_blocks": 4}
                    og_salad = VPRModel(
                            backbone_arch="dinov2_vitb14",
                            backbone_config=dinov2_config,
                            agg_arch='SALAD',
                            agg_config=agg_config,
                        )
                    load_res = og_salad.load_state_dict(state, strict=False)
                    model.aggregator = og_salad.aggregator
                else:
                    load_res = model.load_state_dict(state, strict=False)


                print(f"Missing keys: {load_res.missing_keys}")
                print(f"\nUnexpected keys: {load_res.unexpected_keys}")
                print(f"\nWeights loaded successfully. {len(load_res.missing_keys)} missing keys, {len(load_res.unexpected_keys)} unexpected keys")
        except Exception as e:
            print(f"Local load failed ({e}), falling back to GitHub.")
            model = None

    # Fallback: load from GitHub
    if model is None:
        github_name = "dinov2_salad" if backbone_name == "dinov2_vitb14" else "dinov3_salad"
        model = torch.hub.load(
            repo_or_dir=SALAD_GITHUB_LOCATION,
            model=github_name,
            source="github",
            pretrained=True,
        )
        print(f"Loaded SALAD model from GitHub: {SALAD_GITHUB_LOCATION}")

    return model
