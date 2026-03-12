#!/usr/bin/env python3
"""
Create a Lightning checkpoint wrapping a DINOv3 backbone into a ContrastiveLearningModule
so it can be loaded by scripts/visualize_embeddings_2D.py without training.

Usage:
  python scripts/dino_to_module.py \
    --dino-weights /path/to/dinov3_vitb16_pretrain.pth \
    --variant dinov3_vitb16 \
    --output /local/home/xanadon/mmfe/outputs/contrastive/checkpoints/dino_vitb16.ckpt

Notes:
- The projection head is set to 'none' (identity), per request.
- The checkpoint contains hyper_parameters expected by load_contrastive_model_from_checkpoint and
  the initialized state_dict of the module.
"""

import argparse
import os
import sys
import torch

# Ensure project src is importable
sys.path.append('/local/home/xanadon/mmfe/src')

from training.lightning_module import create_lightning_module


SUPPORTED_VARIANTS = {
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
}


def infer_variant_from_path(weights_path: str) -> str:
    filename = os.path.basename(weights_path).lower()
    for v in SUPPORTED_VARIANTS:
        if v in filename:
            return v
    raise ValueError(
        f"Could not infer DINO variant from filename '{filename}'. "
        f"Provide --variant (one of: {sorted(SUPPORTED_VARIANTS)})."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Wrap DINO weights into a Lightning checkpoint")
    parser.add_argument("--dino-weights", type=str, required=True, help="Path to DINOv3 .pth weights")
    parser.add_argument("--variant", type=str, default=None, help=f"Backbone variant in {sorted(SUPPORTED_VARIANTS)}. Inferred from filename if omitted.")
    parser.add_argument("--output", type=str, default=None, help="Output .ckpt path. If omitted, saved under outputs/contrastive/checkpoints/")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DINO backbone parameters in the module")
    return parser.parse_args()


def build_hparams(variant: str, dino_weights: str, freeze_backbone: bool) -> dict:
    model_config = {
        "model_type": "dual_modality",
        "backbone_name": variant,
        "projection_dim": 0,  # unused when projection_head_type == 'none'
        "projection_spatial": None,
        "pretrained": False,
        "freeze_backbone": freeze_backbone,
        "projection_head_type": "none",  # head is None/Identity
        "backbone_kwargs": {"dino_weights_path": dino_weights},
    }

    # Loss/optimizer configs are required by loader but unused for inference
    loss_config = {
        "loss_type": "infonce2d",
        "temperature": 0.07,
        "reduction": "mean",
        "block_size": 1,
        "margin": 1.0,
    }

    optimizer_config = {
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
    }

    return {
        "model_config": model_config,
        "loss_config": loss_config,
        "optimizer_config": optimizer_config,
    }


def main():
    args = parse_args()

    if not os.path.isfile(args.dino_weights):
        raise FileNotFoundError(f"Weights not found: {args.dino_weights}")

    variant = args.variant or infer_variant_from_path(args.dino_weights)
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant '{variant}'. Supported: {sorted(SUPPORTED_VARIANTS)}")

    if args.output is None:
        out_dir = "outputs/contrastive/checkpoints"
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"{variant}.ckpt")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    hparams = build_hparams(variant, os.path.abspath(args.dino_weights), args.freeze_backbone)

    # Instantiate module to populate state_dict; this will construct the DINO backbone internally
    module = create_lightning_module(
        model_config=hparams["model_config"],
        loss_config=hparams["loss_config"],
        optimizer_config=hparams["optimizer_config"],
    )
    module.eval()

    # Build lightning-like checkpoint
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": hparams,
    }

    torch.save(checkpoint, args.output)
    print(f"Saved module checkpoint to: {args.output}")


if __name__ == "__main__":
    main()


