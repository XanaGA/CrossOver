#!/usr/bin/env python3
"""
Visualize 2D embedding similarity maps for random samples using a trained checkpoint.

This script:
- Loads a Lightning checkpoint
- Builds a `UnifiedDataset`
- Runs the model to obtain 2D embeddings for a random batch
- Reuses the same visualization logic as `log_validation_examples2D` (both modes)
- Saves the resulting grids to disk

Example:
  python scripts/visualize_embeddings_2D.py \
    --cubicasa-path /local/home/xanadon/mmfe/data/cubicasa5k \
    --structured3d-path /local/home/xanadon/mmfe/data/structure3D/Structured3D_annotation_3d/Structured3D \
    --cubicasa-file /local/home/xanadon/mmfe/data/cubicasa5k/val.txt \
    --structured3d-file /local/home/xanadon/mmfe/data/structure3D/val.json \
    --checkpoint ./outputs/contrastive/checkpoints/latest.ckpt \
    --batch-size 8 --num-rows 4 --image-size 256 256 \
    --output-dir ./outputs/visualizations
"""

import argparse
import os
import glob
import random
from typing import Dict, Any, Tuple, List

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports
import sys

from mmfe_utils.dino_utils import get_last_feature_dino, load_dino
sys.path.append('/home/xavi/mmfe/src')

from dataloading.unified_dataset import UnifiedDataset
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint
from mmfe_utils.data_utils import find_latest_checkpoint, create_val_dataset
from mmfe_utils.viz_utils import rotate_tensor, save_grid, create_row_images, viz_2d_PCA, viz_2d_PCA_rot


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 2D similarity maps from a checkpoint")

    # Dataset paths
    parser.add_argument("--cubicasa-path", type=str, default=None, help="Path to CubiCasa5k root")
    parser.add_argument("--structured3d-path", type=str, default=None, help="Path to Structured3D root (scene_XXXXX folders)")
    parser.add_argument("--aria-synthenv-path", type=str, default=None, help="Path to Aria Synthetic Environments root")
    parser.add_argument("--swiss-dwellings-path", type=str, default=None, help="Path to SwissDwellings split directory")
    parser.add_argument("--scannet-path", type=str, default=None, help="Path to ScanNet root directory")
    parser.add_argument("--zillow-path", type=str, default=None, help="Path to Zillow/ZInD rendered dataset root directory")

    # Split files
    parser.add_argument("--cubicasa-file", type=str, default=None, help="Path to CubiCasa split .txt")
    parser.add_argument("--structured3d-file", type=str, default=None, help="Path to Structured3D split .json")
    parser.add_argument("--aria-synthenv-file", type=str, default=None, help="Optional text file with ASE scene IDs to include")
    parser.add_argument("--swiss-dwellings-file", type=str, default=None, help="Optional text file with SwissDwellings IDs to include")
    parser.add_argument("--scannet-file", type=str, default=None, help="Optional text/JSON file with ScanNet scene IDs to include")
    parser.add_argument("--zillow-file", type=str, default=None, help="Optional text file with Zillow sample IDs to include")

    # Generation / dataset-specific options
    parser.add_argument("--aria-generate", action="store_true", help="Generate ASE modalities from raw files instead of loading pre-rendered images")
    parser.add_argument("--swiss-generate", action="store_true", help="Generate SwissDwellings modalities from raw files instead of loading pre-rendered images")
    parser.add_argument("--scannet-generate", action="store_true", help="Generate ScanNet modalities from raw files instead of loading pre-rendered images")
    parser.add_argument("--zillow-generate", action="store_true", help="Generate Zillow modalities from raw files instead of loading pre-rendered images")
    parser.add_argument("--aria-axis", type=str, default="z", choices=["x", "y", "z"], help="Orthographic projection axis for ASE (default: z)")
    parser.add_argument("--scannet-resolution", type=int, default=1024, help="Resolution for ScanNet generated images (default: 1024)")

    # Model / inference
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .ckpt. If None, latest in outputs/contrastive/checkpoints")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for a single visualization batch")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=42)

    # Visualization
    parser.add_argument("--num-rows", type=int, default=4, help="Number of random rows to show")
    parser.add_argument("--output-dir", type=str, default="./outputs/visualizations", help="Directory to save images")

    return parser.parse_args()

def build_unified_dataset(args) -> UnifiedDataset:
    """
    Build a UnifiedDataset by leveraging the existing `create_val_dataset`
    logic from `mmfe_utils.data_utils`, so dataset definitions stay
    centralized and consistent with training/validation.
    """
    # Default normalization stats used across the project
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    cfg_dict: Dict[str, Any] = {
        "data": {
            "image_size": list(args.image_size),
            "cubicasa": None,
            "structured3d": None,
            "aria_synthenv": None,
            "swiss_dwellings": None,
            "zillow": None,
            "scannet": None,
        },
        "transforms": {
            "mean": mean,
            "std": std,
            # Use same difficulty as default validation pipeline
            "tf_difficulty": "medium",
        },
    }

    if args.cubicasa_path:
        cfg_dict["data"]["cubicasa"] = {
            "path": args.cubicasa_path,
            "val": args.cubicasa_file,
        }

    if args.structured3d_path:
        cfg_dict["data"]["structured3d"] = {
            "path": args.structured3d_path,
            "val": args.structured3d_file,
        }

    if args.aria_synthenv_path:
        cfg_dict["data"]["aria_synthenv"] = {
            "path": args.aria_synthenv_path,
            "val": args.aria_synthenv_file,
        }

    if args.swiss_dwellings_path:
        cfg_dict["data"]["swiss_dwellings"] = {
            "path": args.swiss_dwellings_path,
            "val": args.swiss_dwellings_file,
        }

    if args.zillow_path:
        cfg_dict["data"]["zillow"] = {
            "path": args.zillow_path,
            "val": args.zillow_file,
        }

    if args.scannet_path:
        cfg_dict["data"]["scannet"] = {
            "path": args.scannet_path,
            "val": args.scannet_file,
        }

    cfg = OmegaConf.create(cfg_dict)

    dataset = create_val_dataset(cfg)
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty – check provided dataset paths/files.")
    return dataset


@torch.no_grad()
def run_batch_embeddings(model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device, model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    m0 = batch["modality_0"].to(device)
    m1 = batch["modality_1"].to(device) if "modality_1_noise" not in batch else batch["modality_1_noise"].to(device)
    if model_name.startswith("dino"):
        e0 = get_last_feature_dino(model, m0, model_name)
        e1 = get_last_feature_dino(model, m1, model_name)
    else:
        e0, e1 = model.get_embeddings(m0, m1)
    return e0, e1


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint()

    if args.checkpoint == "dinov3_vitb16":
        dino_weights_path = os.getenv("DINOV3_WEIGHTS_PATH", None)
        dino_weights_path = to_absolute_path(dino_weights_path) if dino_weights_path is not None else None
        model = load_dino(args.checkpoint, load_dino_weights=True, dino_weights_path=dino_weights_path)
        model.to(device)
        model.eval()

    elif args.checkpoint == "dinov2_vitb14":
        dino_weights_path = os.getenv("DINOV2_WEIGHTS_PATH", None)
        dino_weights_path = to_absolute_path(dino_weights_path) if dino_weights_path is not None else None
        model = load_dino(args.checkpoint, load_dino_weights=True, dino_weights_path=dino_weights_path)
        model.to(device)
        model.eval()

    else:
        model = ContrastiveLearningModule.load_from_checkpoint(
                checkpoint_path=to_absolute_path(args.checkpoint), 
                map_location=device, 
                load_dino_weights=False,
                weights_only=False
            )
        model.to(device)

    dataset = build_unified_dataset(args)
    print(f"UnifiedDataset size: {len(dataset)}")

    # Simple dataloader to get one mixed batch
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))

    # Take a single batch
    batch = next(iter(loader))

    # Forward to get 2D embeddings [B,C,H',W']
    e0, e1 = run_batch_embeddings(model, batch, device, args.checkpoint)
    assert e0.dim() == 4 and e1.dim() == 4, "This script expects 2D embeddings (B,C,H,W)."
    # e0 = torch.nn.functional.sigmoid(2*e0)
    # e1 = torch.nn.functional.sigmoid(2*e1)

    # Build rows for both modes
    m0 = batch["modality_0"].to(device)
    m1 = batch["modality_1"].to(device) if "modality_1_noise" not in batch else batch["modality_1_noise"].to(device)
    rows_all_to_all = create_row_images(e0, e1, m0, m1, mode="all_to_all")
    rows_one_to_all = create_row_images(e0, e1, m0, m1, mode="one_to_all")

    # Save similarity maps
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.join(args.output_dir, "similarity_maps_2D")
    save_grid(rows_all_to_all, base + "_all_to_all.png", args.num_rows)
    save_grid(rows_one_to_all, base + "_one_to_all.png", args.num_rows)

    # Save PCA-colorized embedding grids
    viz_2d_PCA(e0, e1, m0, m1, base + "_pca.png", args.num_rows)
    
    # Save rotation equivariance visualization
    m0 = batch["original_modality_0"].to(device) if "original_modality_0" in batch else batch["modality_0"].to(device)
    viz_2d_PCA_rot(m0, base + "_pca_rot.png", 50, model, device, args.checkpoint)

    print("Done.")


if __name__ == "__main__":
    main()



