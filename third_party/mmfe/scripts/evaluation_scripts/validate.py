#!/usr/bin/env python3
"""
Validation script (Hydra-based)

This script:
1) Builds a UnifiedDataset for validation from Hydra config
2) Iterates over the dataset and computes validation metrics (loss + retrieval accuracy)
3) Aggregates metrics across batches (average)
4) Stores results to a pandas DataFrame, prints it, and saves it to disk

Run:
  python scripts/validate.py
  # overrides example:
  python scripts/validate.py \
    model.checkpoint=/abs/path/to/ckpt.ckpt \
    data.cubicasa.path=/abs/cubicasa5k data.cubicasa.val=/abs/cubicasa5k/val.txt \
    data.structured3d.path=/abs/Structured3D data.structured3d.val=/abs/Structured3D/val.json \
    data.image_size='[256,256]' val.batch_size=8 val.num_workers=8 \
    logging.output_csv=/abs/outputs/validation_metrics.csv
"""

import os
import sys
from typing import Dict, Any, List

import cv2
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
from tqdm import tqdm

from dataloading.inversible_tf import warp_feature_map_batch
from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import PairRandomAffine, PairToPIL, PairResize, PairGrayscale, PairToTensor, PairNormalize
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint

from torch.utils._pytree import tree_map

from mmfe_utils.tensor_utils import torch_erode
from inference.tta import run_tta, vote_for_best_augmentation

from mmfe_utils.data_utils import create_val_dataset


def move_to_device(batch, device):
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x,
        batch
    )


def visualize_debug_images(modality_0, modality_1_noise, warp_mask_1=None, save_path="debug_viz.png"):
    """
    Visualize modality_0, modality_1_noise, and warp_mask_1 for the first example in the batch.
    
    Args:
        modality_0: Tensor of shape [B, C, H, W] - first modality
        modality_1_noise: Tensor of shape [B, C, H, W] - second modality with noise
        warp_mask_1: Optional tensor of shape [B, H, W] - warp mask
        save_path: Path to save the visualization
    """
    # Take first example from batch
    img0 = modality_0[0].cpu()
    img1_noise = modality_1_noise[0].cpu()
    
    # Denormalize images (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img0_denorm = img0 * std + mean
    img1_noise_denorm = img1_noise * std + mean
    
    # Clamp to valid range [0, 1]
    img0_denorm = torch.clamp(img0_denorm, 0, 1)
    img1_noise_denorm = torch.clamp(img1_noise_denorm, 0, 1)
    
    # Convert to numpy and transpose for matplotlib (H, W, C)
    img0_np = img0_denorm.permute(1, 2, 0).numpy()
    img1_noise_np = img1_noise_denorm.permute(1, 2, 0).numpy()
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot modality_0
    axes[0].imshow(img0_np)
    axes[0].set_title('Modality 0 (Original)')
    axes[0].axis('off')
    
    # Plot modality_1_noise
    axes[1].imshow(img1_noise_np)
    axes[1].set_title('Modality 1 (Noisy)')
    axes[1].axis('off')
    
    # Plot warp_mask_1 if available
    if warp_mask_1 is not None:
        mask_np = warp_mask_1[0].permute(1, 2, 0).cpu().numpy()
        im = axes[2].imshow(mask_np, cmap='gray')
        axes[2].set_title('Warp Mask 1')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, 'No warp mask\n(angle.sum() <= 1)', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Warp Mask 1')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved to: {save_path}")



def compute_batch_metrics(model, batch: Dict[str, Any], cfg: DictConfig = None, batch_idx: int = 0) -> Dict[str, float]:
    modality_0 = batch["modality_0"].to(next(model.parameters()).device)
    modality_1 = batch["modality_1"].to(next(model.parameters()).device)
    modality_1_noise = batch["modality_1_noise"].to(next(model.parameters()).device)

    if cfg is not None and getattr(cfg, 'viz_debug', False):
        # Denormalize images using the same normalization values as used in training
        mean = torch.tensor(cfg.transforms.mean).view(3, 1, 1)
        std = torch.tensor(cfg.transforms.std).view(3, 1, 1)
        
        modality_0_denorm = modality_0[0].cpu() * std + mean
        modality_1_denorm = modality_1[0].cpu() * std + mean
        modality_1_noise_denorm = modality_1_noise[0].cpu() * std + mean
        
        # Clamp to valid range [0, 1] and convert to numpy
        modality_0_denorm = torch.clamp(modality_0_denorm, 0, 1)
        modality_1_denorm = torch.clamp(modality_1_denorm, 0, 1)
        modality_1_noise_denorm = torch.clamp(modality_1_noise_denorm, 0, 1)
        
        modality_0_show = modality_0_denorm.permute(1, 2, 0).numpy()
        modality_1_show = modality_1_denorm.permute(1, 2, 0).numpy()
        modality_1_noise_show = modality_1_noise_denorm.permute(1, 2, 0).numpy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(modality_0_show)
        axs[0].set_title("modality_0")
        axs[0].axis('off')
        axs[1].imshow(modality_1_show)
        axs[1].set_title("modality_1")
        axs[1].axis('off')
        axs[2].imshow(modality_1_noise_show)
        axs[2].set_title("modality_1_noise")
        axs[2].axis('off')
        plt.show()

    with torch.no_grad():
        embeddings_0 = model.get_embeddings(modality_0)
        if cfg.tta != 1:
            aug_step = 360 / cfg.tta
            best_embedding_1_aug, selected_params, best_aug_idx, votes_per_aug = run_tta(embeddings_0, modality_1_noise, 
                                                                                        batch["noise_params"]["valid_mask"], model, 
                                                                                        n_augs=cfg.tta)

            gt_rot = batch["noise_params"]["angle"]
            gt_aug = ((cfg.tta+(360-gt_rot+aug_step/2)//aug_step)%cfg.tta).to(torch.int32)

            accuracy_tta = ((best_aug_idx - gt_aug)==0).sum()/len(best_aug_idx)
            print(f"TTA Accuracy: {accuracy_tta}")
            
            wrapped_embeddings_1_aug, warp_mask_1_aug = warp_feature_map_batch(
                            best_embedding_1_aug, selected_params, image_size=selected_params["image_size"], 
                            align_corners=False, return_mask=True, og_valid_mask=selected_params["valid_mask"]
                        )

            wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(
                            wrapped_embeddings_1_aug, batch["noise_params"], image_size=batch["noise_params"]["image_size"], 
                            align_corners=False, return_mask=True, og_valid_mask=warp_mask_1_aug
                        )

            # wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(
            #     best_embedding_1_aug, [selected_params, batch["noise_params"]], image_size=batch["noise_params"]["image_size"], 
            #     align_corners=False, return_mask=True, og_valid_mask=selected_params["valid_mask"]
            # )


            warp_mask_1 = torch_erode(warp_mask_1, kernel_size=3, iterations=1)
            warp_mask_1 = TF.resize(warp_mask_1, size=wrapped_embeddings_1.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
            
        else:
            embeddings_1_noisy = model.get_embeddings(modality_1_noise)
            params_noise = batch["noise_params"]
            warp_mask_1 = None
        
            if params_noise["angle"].abs().sum() > 1:
                # We are in medium or hard difficulty
                wrapped_embeddings_1, warp_mask_1 = warp_feature_map_batch(
                            embeddings_1_noisy, params_noise, image_size=params_noise["image_size"], 
                            align_corners=False, og_valid_mask=params_noise["valid_mask"], return_mask=True
                        )
                warp_mask_1 = torch_erode(warp_mask_1, kernel_size=3, iterations=1)
                warp_mask_1 = TF.resize(warp_mask_1, size=wrapped_embeddings_1.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
            else:
                wrapped_embeddings_1 = embeddings_1_noisy
                warp_mask_1 = torch.ones(embeddings_1_noisy.shape[0], 1, embeddings_1_noisy.shape[2], embeddings_1_noisy.shape[3])
                warp_mask_1 = warp_mask_1.to(embeddings_1_noisy.device)
        
        # Debug visualization if enabled
        if cfg is not None and getattr(cfg, 'viz_debug', False):
            # Create output directory for debug visualizations
            debug_dir = os.path.join(os.path.dirname(to_absolute_path(cfg.logging.output_csv)), "debug_viz")
            os.makedirs(debug_dir, exist_ok=True)
            
            save_path = os.path.join(debug_dir, f"batch_{batch_idx:04d}_debug.png")
            visualize_debug_images(modality_1, modality_1_noise, warp_mask_1, save_path)

        resized_mask_selected = TF.resize(batch["noise_params"]["valid_mask"], size=embeddings_0.shape[-2:], interpolation=TF.InterpolationMode.NEAREST)
        accs = model.compute_retrieval_accuracy2D(embeddings_0, wrapped_embeddings_1, 
                                                    sample_percentage=0.5, topk=3, distance_th=3.0, 
                                                    valid_mask=(warp_mask_1 * resized_mask_selected).bool())
        loss = model.loss_fn(embeddings_0, wrapped_embeddings_1, warp_mask_1)

        metrics = {
            "val_loss": float(loss.item()),
            "val_acc_all": float(accs["acc"].item()),
            "val_acc_self": float(accs["acc_self"].item()),
            "val_acc_others": float(accs["acc_others"].item()),
            "val_acc_no_self": float(accs["acc_no_self"].item()),
        }

        if cfg.tta != 1:
            metrics["tta_accuracy"] = float(accuracy_tta.item())
        else:
            metrics["tta_accuracy"] = 1.0

        print("Metrics: ", metrics)
        
        # Add Euclidean distance and distance threshold metrics if they exist
        if "AEPE" in accs:
            metrics["val_AEPE"] = float(accs["AEPE"].item())
        
        # Add distance threshold metrics if they exist
        for key in accs.keys():
            if key.startswith("PCK@"):
                metrics[f"val_{key}"] = float(accs[key].item())
        
        # Add top-k metrics if they exist
        for key in accs.keys():
            if key.startswith("acc_top"):
                metrics[f"val_{key}"] = float(accs[key].item())

    return metrics


@hydra.main(config_path="../../configs", config_name="validate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    output_csv_abs = to_absolute_path(cfg.logging.output_csv)
    os.makedirs(os.path.dirname(output_csv_abs), exist_ok=True)

    # Resolve device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(cfg, "runtime") and getattr(cfg.runtime, "device", None) in ("cuda", "cpu"):
        device = cfg.runtime.device

    # Load model
    if not cfg.model.checkpoint:
        raise ValueError("model.checkpoint must be provided (path to .ckpt)")
    # model = load_contrastive_model_from_checkpoint(to_absolute_path(cfg.model.checkpoint))
    model = ContrastiveLearningModule.load_from_checkpoint(checkpoint_path=to_absolute_path(cfg.model.checkpoint), 
                                                            map_location=device, load_dino_weights=False, weights_only=False)
    model.to(device)
    model.eval()

    # Dataset & loader
    val_dataset = create_val_dataset(cfg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.val.batch_size),
        shuffle=False,
        num_workers=int(cfg.val.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    # Iterate and collect metrics per batch
    per_batch: List[Dict[str, float]] = []
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        # Move everything in the batch to the device
        batch = move_to_device(batch, device)
        metrics = compute_batch_metrics(model, batch, cfg, batch_idx)
        per_batch.append(metrics)

    # Aggregate metrics (average and std)
    df = pd.DataFrame(per_batch)
    means = df.mean(numeric_only=True)
    stds = df.std(numeric_only=True)
    summary_row = {k: float(v) for k, v in means.to_dict().items()}
    summary_row["batches"] = len(per_batch)
    std_row = {f"std_{k}": float(v) for k, v in stds.to_dict().items()}
    # Combined summary dict with both means and stds
    combined_summary = {**summary_row, **std_row}

    # Display and save
    print("Validation metrics per batch:")
    print(df)
    print("\nAverages:")
    print(summary_row)
    print("\nStandard deviations:")
    print(std_row)

    df.to_csv(output_csv_abs, index=False)

    # Also save aggregates (mean and std) as a small CSV next to the main CSV
    agg_csv_path = os.path.splitext(output_csv_abs)[0] + "_agg.csv"
    agg_df = pd.DataFrame({"mean": means, "std": stds})
    agg_df.to_csv(agg_csv_path)

    # Also save a small JSON summary next to CSV
    json_path = os.path.splitext(output_csv_abs)[0] + "_summary.json"
    try:
        import json
        with open(json_path, "w") as f:
            json.dump(combined_summary, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save summary JSON: {e}")


if __name__ == "__main__":
    main()
