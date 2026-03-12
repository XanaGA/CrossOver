#!/usr/bin/env python3
"""
Evaluate FPV pose estimation performance.

This script:
- Loads an FPVLightningModule checkpoint
- Creates an aria validation dataset
- For each sample:
  - Computes projected frustums for FPV images
  - Performs template matching with multiple candidate poses
  - Refines the best pose using non-linear optimization
  - Compares estimated pose with ground truth
- Computes and saves evaluation metrics

Usage:
    python scripts/evaluation_scripts/evaluate_fpv_inference.py \
        model.checkpoint=/path/to/fpv_checkpoint.ckpt \
        data.aria_synthenv.path=/path/to/aria/rendered_data \
        data.aria_synthenv.val=/path/to/val.txt \
        eval.n_candidates=1000 \
        eval.refine=true
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dataloading.aria_se_data import AriaSynthEenvDataset
from dataloading.dual_transforms import PairToPIL, PairResize, PairToTensor, PairNormalize
from fpv.fpv_modules import FPVLightningModule, load_fpv_model_from_checkpoint
from fpv.fpv_inference import estimate_pose_template_matching, compute_cosine_similarity_error
from aria_mmfe.code_snippets.plotters import change_params_resolution
import torchvision.transforms as T


def build_aria_dataset(cfg: DictConfig) -> AriaSynthEenvDataset:
    """
    Build an AriaSynthEenvDataset for validation with FPV images.
    """
    root_dir = to_absolute_path(cfg.data.aria_synthenv.path)
    ids_file = cfg.data.aria_synthenv.val
    ids_file = to_absolute_path(ids_file) if ids_file is not None else None

    # Floorplan transforms
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    dual_transform = [
        PairToPIL(),
        PairResize(tuple(cfg.data.image_size)),
        PairToTensor(),
        PairNormalize(mean=mean, std=std),
    ]

    # FPV image transforms
    imagenet_mean = mean.view(1, 3, 1, 1)
    imagenet_std = std.view(1, 3, 1, 1)

    def fpv_to_tensor(fpv_images: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for img in fpv_images:
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        if len(tensors) == 0:
            return torch.empty(0, 3, 0, 0)
        stacked = torch.stack(tensors, dim=0)
        return (stacked - imagenet_mean) / imagenet_std

    n_fpv = int(getattr(cfg.data.aria_synthenv, "n_fpv_images", 1))

    dataset = AriaSynthEenvDataset(
        root_dir=root_dir,
        scene_ids_file=ids_file,
        image_size=tuple(cfg.data.image_size),
        dual_transform=dual_transform,
        n_fpv_images=n_fpv,
        fpv_transforms=[fpv_to_tensor, T.Normalize(mean=mean, std=std)],
        load_depth = cfg.data.aria_synthenv.use_depth
    )
    return dataset


def collate_fpv_batch(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for FPV batches.
    """
    collated = {
        "modality_0": torch.stack([b["modality_0"] for b in batch]),
        "modality_1": torch.stack([b["modality_1"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }
    
    # Collate fpv_dict
    if "fpv_dict" in batch[0]:
        fpv_dicts = [b["fpv_dict"] for b in batch]
        
        # Stack images
        collated["fpv_dict"] = {
            "images": torch.stack([fd["images"] for fd in fpv_dicts]),
        }
        if "depths" in fpv_dicts[0]:
            collated["fpv_dict"]["depths"] = torch.stack([fd["depths"] for fd in fpv_dicts])
        
        # Collate poses
        collated["fpv_dict"]["pose_2D_world"] = {
            "xy": torch.stack([torch.tensor(fd["pose_2D_world"]["xy"]) for fd in fpv_dicts]),
            "theta": torch.stack([torch.tensor(fd["pose_2D_world"]["theta"]) for fd in fpv_dicts]),
        }
        
        # Collate params (need to handle dict of lists)
        params_keys = fpv_dicts[0]["params"].keys()
        collated["fpv_dict"]["params"] = {
            k: torch.tensor([fd["params"][k] for fd in fpv_dicts])
            for k in params_keys
        }
    
    return collated


def compute_pose_error(
    estimated_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute pose estimation errors.
    
    Args:
        estimated_pose: (3,) tensor with (x, y, theta)
        gt_pose: (3,) tensor with (x, y, theta)
        
    Returns:
        Dict with translation_error, rotation_error, total_error
    """
    # Translation error (Euclidean distance)
    trans_error = torch.sqrt(
        (estimated_pose[0] - gt_pose[0])**2 + 
        (estimated_pose[1] - gt_pose[1])**2
    ).item()
    
    # Rotation error (angular difference, wrapped to [-pi, pi])
    rot_diff = estimated_pose[2] - gt_pose[2]
    rot_error = torch.abs(torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))).item()
    
    return {
        "translation_error": trans_error,
        "rotation_error": rot_error,
        "rotation_error_deg": np.degrees(rot_error),
    }


def evaluate_sample(
    model: FPVLightningModule,
    batch: Dict[str, Any],
    batch_idx: int,
    cfg: DictConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate pose estimation for a single batch element.
    """
    # Get data for this batch element
    floorplan = batch["modality_0"][batch_idx:batch_idx+1].to(device)
    fpv_images = batch["fpv_dict"]["images"][batch_idx:batch_idx+1].to(device)
    params = batch["fpv_dict"]["params"]
    
    # Ground truth pose
    gt_xy = batch["fpv_dict"]["pose_2D_world"]["xy"][batch_idx]  # (N, 2)
    gt_theta = batch["fpv_dict"]["pose_2D_world"]["theta"][batch_idx]  # (N,)
    gt_pose = torch.cat([gt_xy[0], gt_theta[0:1]], dim=0).to(device)  # (3,) for first image
    
    # Forward pass
    with torch.no_grad():
        floorplan_feats = model.forward_floorplan(floorplan)
        frustum_data = model.forward_fpv_images(fpv_images,
            gt_depth=batch["fpv_dict"]["depths"][batch_idx:batch_idx+1].to(device) if "depths" in batch["fpv_dict"] else None,
        )
        # frustum_data[0].features = torch.randn_like(frustum_data[0].features) * (frustum_data[0].features.max()-frustum_data[0].features.min()) + frustum_data[0].features.min()
        features = frustum_data[0].features  # [1, 32, 167, 167]
        norm = features.norm(dim=1, keepdim=True)  # [1, 1, 167, 167]
        eps = 1e-8  # avoid division by zero
        features = features / (norm + eps)
        features = features * (norm > 0)
        frustum_data[0].features = features
    
    # Adjust params resolution to match floorplan features
    adjusted_params = change_params_resolution(
        params, 
        (floorplan_feats.shape[2], floorplan_feats.shape[3])
    )
    
    # Get frustum for this batch element
    frustum = frustum_data[0]  # First (and only) batch element
    
    # Estimate pose
    estimated_pose, final_error, info = estimate_pose_template_matching(
        floorplan_feats[0],  # (C, H, W)
        frustum,
        adjusted_params,
        batch_idx=batch_idx,
        n_candidates=cfg.eval.n_candidates,
        refine=cfg.eval.refine,
        device=device,
        gt_pose = gt_pose
    )
    
    # Compute errors
    pose_errors = compute_pose_error(estimated_pose, gt_pose)
    
    # Also compute error at GT pose for reference
    gt_error, _ = compute_cosine_similarity_error(
        floorplan_feats[0], frustum, gt_pose, adjusted_params, batch_idx
    )
    
    return {
        "sample_id": batch["sample_id"][batch_idx],
        "estimated_pose": estimated_pose.cpu().numpy().tolist(),
        "gt_pose": gt_pose.cpu().numpy().tolist(),
        "translation_error": pose_errors["translation_error"],
        "rotation_error": pose_errors["rotation_error"],
        "rotation_error_deg": pose_errors["rotation_error_deg"],
        "final_error": final_error,
        "gt_error": gt_error,
        "template_error": info.get("template_error", None),
        "refined_error": info.get("refined_error", None),
    }


def save_results(results: List[Dict], metrics: Dict, output_dir: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "fpv_inference_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to: {results_file}")
    
    # Save summary metrics
    metrics_file = os.path.join(output_dir, "fpv_inference_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_file}")


@hydra.main(config_path="../../configs", config_name="evaluate_fpv_inference", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    load_dotenv()
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    torch.manual_seed(cfg.eval.seed)
    np.random.seed(cfg.eval.seed)
    
    # Setup device
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = to_absolute_path(cfg.model.checkpoint)
    print(f"\nLoading FPV model from: {checkpoint_path}")
    
    model = load_fpv_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        load_weights=True,
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Create dataset
    print("\nCreating validation dataset...")
    dataset = build_aria_dataset(cfg)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        collate_fn=collate_fpv_batch,
        pin_memory=True,
    )
    
    # Evaluate
    print(f"\nEvaluating with {cfg.eval.n_candidates} candidate poses, refine={cfg.eval.refine}...")
    results = []
    
    num_samples = min(cfg.eval.num_samples, len(dataset)) if cfg.eval.num_samples > 0 else len(dataset)
    samples_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_size = batch["modality_0"].shape[0]
            
            for b in range(batch_size):
                if samples_processed >= num_samples:
                    break
                    
                try:
                    result = evaluate_sample(model, batch, b, cfg, device)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing sample {batch['sample_id'][b]}: {e}")
                    continue
                    
                samples_processed += 1
            
            if samples_processed >= num_samples:
                break
    
    # Compute aggregate metrics
    if results:
        trans_errors = [r["translation_error"] for r in results]
        rot_errors = [r["rotation_error_deg"] for r in results]
        
        metrics = {
            "num_samples": len(results),
            "translation_error": {
                "mean": float(np.mean(trans_errors)),
                "median": float(np.median(trans_errors)),
                "std": float(np.std(trans_errors)),
                "min": float(np.min(trans_errors)),
                "max": float(np.max(trans_errors)),
            },
            "rotation_error_deg": {
                "mean": float(np.mean(rot_errors)),
                "median": float(np.median(rot_errors)),
                "std": float(np.std(rot_errors)),
                "min": float(np.min(rot_errors)),
                "max": float(np.max(rot_errors)),
            },
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"\nTranslation Error (world units):")
        print(f"  Mean:   {metrics['translation_error']['mean']:.4f}")
        print(f"  Median: {metrics['translation_error']['median']:.4f}")
        print(f"  Std:    {metrics['translation_error']['std']:.4f}")
        print(f"\nRotation Error (degrees):")
        print(f"  Mean:   {metrics['rotation_error_deg']['mean']:.2f}")
        print(f"  Median: {metrics['rotation_error_deg']['median']:.2f}")
        print(f"  Std:    {metrics['rotation_error_deg']['std']:.2f}")
        print("="*50)
        
        # Save results
        output_dir = to_absolute_path(cfg.logging.output_dir)
        save_results(results, metrics, output_dir)
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()

