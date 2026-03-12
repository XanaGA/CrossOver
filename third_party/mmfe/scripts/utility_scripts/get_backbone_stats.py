#!/usr/bin/env python3
"""
Extract backbone feature statistics from RoMa models.

This script:
- Initializes a RoMaFineTuner or RoMaV1 model (doesn't need to be pretrained)
- Loads a validation dataset (e.g., Structured3D)
- Extracts features from the backbone for all samples
- Computes aggregated statistics: max, min, mean per channel, and correlation statistics
- Saves results to a JSON file

Run example:
  python scripts/utility_scripts/get_backbone_stats.py \
    model.model_type=romafinetuner \
    model.backbone.backbone_name=mmfe \
    model.backbone.mmfe_checkpoint_path=/path/to/mmfe.ckpt \
    data.structured3d.path=/path/to/Structured3D \
    data.structured3d.val=/path/to/val.json \
    runtime.batch_size=8 \
    output.results_json=outputs/metrics/backbone_stats.json
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra
from dotenv import load_dotenv
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for script usage
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from roma.roma_pl_module import RoMaFineTuner, analyze_channel_correlations
from mmfe_utils.data_utils import create_val_dataset

try:
    from romatch import roma_indoor
except ImportError:
    print("RoMaV1 not found")

try:
    from romav2 import RoMaV2
except ImportError:
    print("RoMaV2 not found")


def initialize_model(cfg: DictConfig, device: str) -> torch.nn.Module:
    """
    Initialize RoMa model based on configuration.
    
    Args:
        cfg: Configuration object
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Initialized model
    """
    model_type = cfg.model.model_type.lower()
    
    if model_type == "roma_v1":
        # Initialize RoMaV1 directly
        model = roma_indoor(device=device)
        model.to(device)
        model.eval()
        return model
    
    elif model_type == "romafinetuner":
        # Initialize RoMaFineTuner

        # model = RoMaFineTuner.load_from_checkpoint(
        #     checkpoint_path=to_absolute_path(cfg.model.backbone.checkpoint), 
        #     map_location=device, 
        #     weights_only=False,
        #     mmfe_roma_checkpoint_path=None
        # )
        model = RoMaFineTuner(
            matcher_name=cfg.model.matcher_name,
            lr=1e-4,
            weight_decay=1e-4,
            backbone_kwargs = cfg.model.backbone
        )
        model.to(device)
        model.eval()
        
        # # If checkpoint is provided, load from checkpoint
        # if cfg.model.get('checkpoint') is not None and cfg.model.checkpoint:
        #     checkpoint_path = to_absolute_path(cfg.model.checkpoint)
        #     model = RoMaFineTuner.load_from_checkpoint(
        #         checkpoint_path=checkpoint_path,
        #         map_location=device,
        #         weights_only=False,
        #         mmfe_roma_checkpoint_path=backbone_kwargs.get("mmfe_checkpoint_path", None) if backbone_kwargs else None
        #     )
        # else:
        #     # Initialize without pretrained weights
        #     # Ensure backbone_kwargs is a dict (not None) for RoMaFineTuner
        #     if not backbone_kwargs:
        #         backbone_kwargs = {}
            
        #     model = RoMaFineTuner(
        #         matcher_name=cfg.model.matcher_name,
        #         lr=1e-4,  # Not used for inference
        #         weight_decay=1e-4,  # Not used for inference
        #         backbone_kwargs=backbone_kwargs,
        #         mmfe_roma_checkpoint_path=backbone_kwargs.get("mmfe_checkpoint_path", None) if backbone_kwargs else None
        #     )
        
        # model.to(device)
        # model.eval()
        return model
    
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'roma_v1' or 'romafinetuner'")


def extract_backbone_features(model: torch.nn.Module, images: torch.Tensor, device: str, model_type: str) -> torch.Tensor:
    """
    Extract features from the backbone encoder.
    
    Args:
        model: RoMa model (RoMaFineTuner or roma_indoor)
        images: Input images tensor [B, C, H, W]
        device: Device to use
        model_type: "roma_v1" or "romafinetuner"
    
    Returns:
        Features tensor [B, num_patches, channels]
    """
    images = images.to(device)
    
    with torch.no_grad():
        if model_type == "roma_v1":
            # Access encoder.dinov2_vitl14
            encoder = model.encoder
        elif model_type == "romafinetuner":
            encoder = model.model.encoder
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
            
        if isinstance(encoder.dinov2_vitl14, list):
            backbone = encoder.dinov2_vitl14[0].to(device)
        else:
            backbone = encoder.dinov2_vitl14.to(device)

        model_dtype = next(backbone.parameters()).dtype
        images = images.to(dtype=model_dtype)
        
        # Call forward_features if available, otherwise forward
        if hasattr(backbone, 'forward_features'):
            output = backbone.forward_features(images)
        else:
            output = backbone(images)
        
        # Extract patch tokens
        if isinstance(output, dict):
            features = output['x_norm_patchtokens']  # [B, num_patches, channels]
        else:
            # If output is not a dict, assume it's the patch tokens directly
            # Reshape if needed: [B, C, H, W] -> [B, H*W, C]
            if output.dim() == 4:
                B, C, H, W = output.shape
                features = output.permute(0, 2, 3, 1).reshape(B, H*W, C)
            else:
                features = output
        
    
    return features


def compute_batch_statistics(features: torch.Tensor) -> Dict[str, Any]:
    """
    Compute statistics for a batch of features.
    
    Args:
        features: Features tensor [B, num_patches, channels]
    
    Returns:
        Dictionary with batch statistics
    """
    features_flat_batch = features.view(features.shape[0], -1)
    features_flat_channel = features.permute(2, 0, 1).reshape(features.shape[2], -1)
    # Per-channel statistics (across batch and patches)
    batch_max = features_flat_batch.max(dim=-1).values.cpu().numpy()  # [batch_size]
    batch_min = features_flat_batch.min(dim=-1).values.cpu().numpy()  # [batch_size]
    batch_mean = features_flat_batch.mean(dim=-1).cpu().numpy()  # [batch_size]
    channel_max = features_flat_channel.max(dim=-1).values.cpu().numpy()  # [channels]
    channel_min = features_flat_channel.min(dim=-1).values.cpu().numpy()  # [channels]
    channel_mean = features_flat_channel.mean(dim=-1).cpu().numpy()  # [channels]
    
    # Correlation statistics
    corr_matrix, corr_stats = analyze_channel_correlations(features)
    
    return {
        'batch_max': batch_max,
        'batch_min': batch_min,
        'batch_mean': batch_mean,
        'channel_max': channel_max,
        'channel_min': channel_min,
        'channel_mean': channel_mean,
        'correlation_stats': corr_stats,
        'num_samples': features.shape[0] * features.shape[1],  # batch_size * num_patches
        'batch_size': features.shape[0],
        'num_patches': features.shape[1],
        'num_channels': features.shape[2]
    }


def aggregate_statistics(all_batch_stats: list) -> Dict[str, Any]:
    """
    Aggregate statistics across all batches.
    
    Args:
        all_batch_stats: List of batch statistics dictionaries
    
    Returns:
        Aggregated statistics dictionary
    """
    if len(all_batch_stats) == 0:
        raise ValueError("No batch statistics provided")
    
    # Get dimensions from first batch
    num_channels = all_batch_stats[0]['num_channels']
    
    # Aggregate per-channel statistics
    # For max: take global max across all batches
    all_max = np.concatenate([stats['batch_max'] for stats in all_batch_stats], axis=0) 
    global_max = all_max.max()  
    
    # For min: take global min across all batches
    all_min = np.concatenate([stats['batch_min'] for stats in all_batch_stats], axis=0)
    global_min = all_min.min()  
    
    # For mean: weighted average by number of samples
    total_samples = sum(stats['num_samples'] for stats in all_batch_stats)
    weighted_mean = 0
    for stats in all_batch_stats:
        weight = stats['num_samples'] / total_samples
        weighted_mean += weight * stats['batch_mean'].mean()
    
    # Aggregate per-channel stats across batches
    channel_max_stack = np.stack([stats['channel_max'] for stats in all_batch_stats], axis=0)  # [num_batches, num_channels]
    channel_min_stack = np.stack([stats['channel_min'] for stats in all_batch_stats], axis=0)
    channel_mean_stack = np.stack([stats['channel_mean'] for stats in all_batch_stats], axis=0)

    channel_stats = {
        'max': {
            'mean': channel_max_stack.mean(axis=0).tolist(),
            'std': channel_max_stack.std(axis=0).tolist(),
        },
        'min': {
            'mean': channel_min_stack.mean(axis=0).tolist(),
            'std': channel_min_stack.std(axis=0).tolist(),
        },
        'mean': {
            'mean': channel_mean_stack.mean(axis=0).tolist(),
            'std': channel_mean_stack.std(axis=0).tolist(),
        },
    }

    # Aggregate correlation statistics
    # Average correlation statistics across batches
    corr_keys = ['mean_abs_correlation', 'std_correlation', 'max_correlation', 
                 'median_abs_correlation', 'high_corr_fraction', 'very_high_corr_fraction']
    aggregated_corr_stats = {}
    for key in corr_keys:
        values = [stats['correlation_stats'][key] for stats in all_batch_stats]
        aggregated_corr_stats[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    return {
        'global_stats': {
            'max': float(global_max),
            'min': float(global_min),
            'mean': float(weighted_mean)
        },
        'correlation_stats': aggregated_corr_stats,
        'total_samples': int(total_samples),
        'num_batches': len(all_batch_stats),
        'num_channels': int(num_channels),
        'channel_stats': channel_stats
    }


def save_channel_histograms(channel_stats: Dict[str, Dict[str, Any]], output_dir: str, show_std: bool = False, 
                            max_bins: int = 64) -> Dict[str, str]:
    """
    Create and save histograms with error bars for per-channel statistics.
    If channels > max_bins, aggregates them to ensure readability.

    Args:
        channel_stats: Dict containing per-channel mean/std for max, min, mean.
        output_dir: Directory to save histogram images.
        max_bins: Maximum number of bars to plot. If channels exceed this, they are binned.

    Returns:
        Dictionary mapping metric name to saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    for metric, values in channel_stats.items():
        means = np.array(values.get('mean', []))
        stds = np.array(values.get('std', []))
        
        if means.size == 0:
            continue

        num_channels = len(means)
        
        # Aggregate if too many channels
        if num_channels > max_bins:
            # Split channels into roughly equal chunks
            mean_chunks = np.array_split(means, max_bins)
            std_chunks = np.array_split(stds, max_bins)
            
            # Compute average of means and stds for each chunk
            # Note: We average the STDs to visualize the "average variation" of channels in this block
            plot_means = np.array([np.mean(chunk) for chunk in mean_chunks])
            plot_stds = np.array([np.mean(chunk) for chunk in std_chunks]) if show_std else None
            
            # Calculate bin size for labeling
            bin_size = num_channels / max_bins
            xlabel_text = f"Channel Bins (Aggregated ~{bin_size:.1f} channels/bin)"
            title_suffix = f"(Aggregated to {max_bins} bins)"
        else:
            plot_means = means
            plot_stds = stds if show_std else None
            xlabel_text = "Channel"
            title_suffix = ""

        x_pos = np.arange(len(plot_means))

        plt.figure(figsize=(12, 6)) # Increased height slightly for readability
        plt.bar(x_pos, plot_means, yerr=plot_stds, capsize=3, alpha=0.7, color="#4C72B0")
        
        plt.xlabel(xlabel_text)
        plt.ylabel(f"{metric} value")
        plt.title(f"Per-channel {metric} {title_suffix}")
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"{metric}_histogram.png")
        plt.savefig(filename, dpi=200)
        plt.close()
        saved_paths[metric] = filename
        
        print(f"Saved {metric} histogram to {filename}")

    return saved_paths


def save_results(stats: Dict[str, Any], output_path: str, metadata: Dict[str, Any]) -> None:
    """
    Save aggregated statistics to JSON file.
    
    Args:
        stats: Aggregated statistics dictionary
        output_path: Path to output JSON file
        metadata: Additional metadata to include
    """
    results = {
        'metadata': metadata,
        'statistics': stats
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


@hydra.main(config_path="../../configs", config_name="get_backbone_stats", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Resolve device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(cfg, "runtime") and getattr(cfg.runtime, "device", None) in ("cuda", "cpu"):
        device = cfg.runtime.device
    
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = initialize_model(cfg, device)
    model_type = cfg.model.model_type.lower()
    print(f"Model type: {model_type}")
    
    # Create validation dataset
    print("Creating validation dataset...")
    if (cfg.model.matcher_name == "romav1" and cfg.model.backbone.backbone_name == "dinov2_vitb14" and cfg.model.backbone.use_downsampled_dino):
        cfg.data.image_size = (266, 266)
    else:
        cfg.data.image_size = (256, 256)
    val_dataset = create_val_dataset(cfg)
    print(f"Dataset size: {len(val_dataset)} samples")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        num_workers=cfg.runtime.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Process all batches
    print("Processing batches and computing statistics...")
    all_batch_stats = []
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        if cfg.runtime.max_batches and batch_idx >= cfg.runtime.max_batches:
            break
        
        # Extract features from modality_0 (floor plan images)
        images = batch['modality_0']
        
        try:
            features = extract_backbone_features(model, images, device, model_type)
            
            # Compute batch statistics
            batch_stats = compute_batch_statistics(features)
            all_batch_stats.append(batch_stats)
            
            batch_count += 1
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    if len(all_batch_stats) == 0:
        raise ValueError("No batches were successfully processed")
    
    # Aggregate statistics
    print("Aggregating statistics...")
    aggregated_stats = aggregate_statistics(all_batch_stats)
    histogram_dir = os.path.join(os.path.dirname(to_absolute_path(cfg.output.results_json)), "backbone_histograms")
    histogram_paths = save_channel_histograms(aggregated_stats.get('channel_stats', {}), histogram_dir)
    aggregated_stats['channel_histograms'] = histogram_paths
    
    # Prepare metadata
    backbone_name = 'default'
    if model_type == 'romafinetuner' and hasattr(cfg.model, 'backbone'):
        backbone_kwargs = cfg.model.backbone
        if hasattr(backbone_kwargs, '_content'):
            backbone_kwargs = dict(backbone_kwargs)
        if backbone_kwargs:
            backbone_name = backbone_kwargs.get('backbone_name', 'default')
    
    metadata = {
        'model_type': model_type,
        'matcher_name': cfg.model.get('matcher_name', 'N/A') if model_type == 'romafinetuner' else 'roma_v1',
        'backbone_name': backbone_name,
        'dataset_size': len(val_dataset),
        'num_batches_processed': batch_count,
        'batch_size': cfg.runtime.batch_size,
        'image_size': cfg.data.image_size[0]
    }
    
    # Save results
    output_path = to_absolute_path(cfg.output.results_json)
    save_results(aggregated_stats, output_path, metadata)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Processed {batch_count} batches")
    print(f"Total samples: {aggregated_stats['total_samples']}")
    print(f"Number of channels: {aggregated_stats['num_channels']}")
    print(f"\nBasic Statistics: {aggregated_stats['global_stats']}")
    print(f"\nCorrelation Statistics:")
    for key, value in aggregated_stats['correlation_stats'].items():
        print(f"  {key}: mean={value['mean']:.4f}, std={value['std']:.4f}")


if __name__ == "__main__":
    main()

