#!/usr/bin/env python3
"""
Script to visualize vector representations of dataset examples using PCA.

This script loads a trained contrastive learning model and generates embeddings
for examples from the dataset, then visualizes them in 2D using PCA.

Usage:
    python scripts/visualize_embeddings_pca.py \
        --cubicasa-path /path/to/CubiCasa5k \
        --structured3d-path /path/to/Structured3D \
        --cubicasa-file /path/to/cubicasa/train.txt \
        --structured3d-file /path/to/structured3d/train.json \
        --checkpoint /path/to/checkpoint.ckpt \
        --num-examples 100 \
        --output-dir ./outputs/visualizations
"""

import argparse
import os
import glob
from typing import Optional, List, Tuple
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Import from the project
import sys
sys.path.append('/local/home/xanadon/mmfe/src')

from mmfe_utils.data_utils import find_latest_checkpoint
from dataloading.unified_dataset import UnifiedDataset
from training.lightning_module import ContrastiveLearningModule, load_contrastive_model_from_checkpoint
from dataloading.dual_transforms import  PairToTensor, PairResize, PairGrayscale, PairNormalize, PairToPIL

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize embeddings using PCA")
    
    # Dataset paths
    parser.add_argument("--cubicasa-path", type=str, default=None,
                       help="Path to CubiCasa5k dataset root")
    parser.add_argument("--structured3d-path", type=str, default=None,
                       help="Path to Structured3D dataset root")
    
    # Split files
    parser.add_argument("--cubicasa-file", type=str, default=None,
                       help="Path to CubiCasa5k training split file")
    parser.add_argument("--structured3d-file", type=str, default=None,
                       help="Path to Structured3D training split file")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint. If None, uses latest in outputs/contrastive/checkpoints")
    parser.add_argument("--num-examples", type=int, default=100,
                       help="Number of examples to visualize")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256],
                       metavar=("HEIGHT", "WIDTH"), help="Image size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    
    # Visualization parameters
    parser.add_argument("--output-dir", type=str, default="./outputs/visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"],
                       help="Dimensionality reduction method")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()


def create_dataset(args) -> UnifiedDataset:
    """Create dataset for visualization."""
    dataset_configs = []
    
    # Add CubiCasa5k if path provided
    if args.cubicasa_path:
        dataset_configs.append({
            "type": "cubicasa5k",
            "args": {
                "root_dir": args.cubicasa_path,
                "sample_ids_file": args.cubicasa_file,
                "image_size": args.image_size,
                "generate": False,  # Use pre-generated images
            },
        })
    
    # Add Structured3D if path provided
    if args.structured3d_path:
        dataset_configs.append({
            "type": "structured3d",
            "args": {
                "root_dir": args.structured3d_path,
                "scene_ids_file": args.structured3d_file,
                "image_size": args.image_size,
                "generate": False,  # Use pre-generated images
            },
        })
    
    if not dataset_configs:
        raise ValueError("At least one dataset (CubiCasa5k or Structured3D) must be provided")
    

    # Data transforms
    dual_transform = [
        PairToPIL(),
        PairResize(args.image_size),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
        # PairRandomRotation(degrees=180),
        PairNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Add transform to all dataset configs
    # for config in dataset_configs:
    #     config["args"]["dual_transform"] = dual_transform
    
    # Create dataset
    dataset = UnifiedDataset(dataset_configs=dataset_configs, common_transform=dual_transform)
    print(f"Created dataset with {len(dataset)} examples")
    
    return dataset


def extract_embeddings(model: ContrastiveLearningModule, dataset: UnifiedDataset, 
                      num_examples: int, device: str) -> Tuple[np.ndarray, List[str], List[str], List[str], List[np.ndarray]]:
    """Extract embeddings from the model for the given number of examples using all modalities."""
    model.to(device)
    
    # Sample random indices
    total_examples = len(dataset)
    if num_examples > total_examples:
        print(f"Warning: Requested {num_examples} examples but dataset only has {total_examples}")
        num_examples = total_examples
    
    indices = random.sample(range(total_examples), num_examples)
    
    all_embeddings = []
    all_dataset_sources = []
    all_modality_types = []
    all_furniture_pcts = []
    all_sample_ids = []
    all_thumbnails: List[np.ndarray] = []
    
    print(f"Extracting embeddings for {num_examples} examples using all modalities...")
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            # Get all modalities for this sample
            modalities_dict = dataset.get_all_sample_modalities(sample_idx)
            
            dataset_source = modalities_dict.get("source_dataset", "unknown")
            global_index = modalities_dict.get("global_index", sample_idx)
            local_index = modalities_dict.get("local_index", sample_idx)
            
            # Create a unique sample identifier
            sample_id = f"{dataset_source}_{local_index}"
            
            # Process each modality type
            for modality_type, furniture_dict in modalities_dict.items():
                if modality_type in ["source_dataset", "global_index", "local_index"]:
                    continue
                    
                # Process each furniture percentage for this modality
                for furniture_pct, modality_tensor in furniture_dict.items():
                    if isinstance(modality_tensor, torch.Tensor):
                        # Add batch dimension if needed
                        if modality_tensor.dim() == 3:
                            modality_tensor = modality_tensor.unsqueeze(0)
                        
                        # Move to device
                        modality_tensor = modality_tensor.to(device)
                        
                        # Get embedding using the model's encoder (we'll use encoder_0 for all modalities)
                        embedding = model.model.encoder(modality_tensor)
                        
                        # Store embedding
                        all_embeddings.append(embedding.cpu().numpy())
                        
                        # Store metadata
                        all_dataset_sources.append(dataset_source)
                        all_modality_types.append(modality_type)
                        all_furniture_pcts.append(furniture_pct)
                        all_sample_ids.append(sample_id)

                        # Create a small thumbnail for visualization from the (normalized) tensor
                        # Unnormalize using ImageNet stats used in create_dataset
                        with torch.no_grad():
                            img_tensor = modality_tensor[0].detach().cpu().clone()
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            img_tensor = img_tensor * std + mean
                            img_tensor = img_tensor.clamp(0.0, 1.0)
                            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            all_thumbnails.append(img_np)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} samples...")
    
    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    else:
        raise ValueError("No embeddings were extracted. Check if modalities are loading correctly.")
    
    print(f"Extracted embeddings shape: {all_embeddings.shape}")
    print(f"Dataset sources: {set(all_dataset_sources)}")
    print(f"Modality types: {set(all_modality_types)}")
    print(f"Furniture percentages: {set(all_furniture_pcts)}")
    print(f"Unique samples: {len(set(all_sample_ids))}")
    
    return all_embeddings, all_dataset_sources, all_modality_types, all_sample_ids, all_thumbnails


def reduce_dimensions(embeddings: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
    """Reduce dimensionality of embeddings."""
    print(f"Reducing dimensions using {method.upper()}...")
    
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    if method == "pca":
        explained_variance = reducer.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance):.3f}")
    
    return reduced_embeddings


def create_visualizations(reduced_embeddings: np.ndarray, dataset_labels: List[str], 
                         modality_type_labels: List[str], sample_ids: List[str],
                         thumbnails: List[np.ndarray], output_dir: str, method: str):
    """Create and save visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'dataset': dataset_labels,
        'modality_type': modality_type_labels,
        'sample_id': sample_ids
    })
    
    # Set up the plotting style
    plt.style.use('default')
    
    # 1. Plot by dataset source
    plt.figure(figsize=(12, 8))
    for dataset in df['dataset'].unique():
        mask = df['dataset'] == dataset
        plt.scatter(df[mask]['x'], df[mask]['y'], label=dataset, alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Embedding Visualization by Dataset Source ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    dataset_plot_path = os.path.join(output_dir, f'embeddings_by_dataset_{method}.png')
    plt.savefig(dataset_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dataset plot: {dataset_plot_path}")
    
    # 2. Plot by modality type
    plt.figure(figsize=(12, 8))
    for modality_type in df['modality_type'].unique():
        mask = df['modality_type'] == modality_type
        plt.scatter(df[mask]['x'], df[mask]['y'], label=modality_type, alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Embedding Visualization by Modality Type ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    modality_plot_path = os.path.join(output_dir, f'embeddings_by_modality_{method}.png')
    plt.savefig(modality_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved modality plot: {modality_plot_path}")
    
    # 3. Plot by sample (same color for all modalities of the same sample)
    plt.figure(figsize=(12, 8))
    
    # Get unique samples and create color/shape combinations
    unique_samples = df['sample_id'].unique()
    n_samples = len(unique_samples)
    
    # Create color palette
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_samples)))
    if n_samples > 20:
        # Repeat colors if we have more than 20 samples
        colors = np.tile(colors, (n_samples // 20 + 1, 1))[:n_samples]
    
    # Create shape markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', '|', '_']
    
    for i, sample_id in enumerate(unique_samples):
        mask = df['sample_id'] == sample_id
        sample_data = df[mask]
        
        # Use different markers for different modalities within the same sample
        for j, (_, row) in enumerate(sample_data.iterrows()):
            marker = markers[j % len(markers)]
            plt.scatter(row['x'], row['y'], 
                       c=[colors[i]], marker=marker,
                       alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Embedding Visualization by Sample ({method.upper()})\nSame color = same sample, different shapes = different modalities')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    sample_plot_path = os.path.join(output_dir, f'embeddings_by_sample_{method}.png')
    plt.savefig(sample_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample plot: {sample_plot_path}")
    
    # 4. Plot thumbnails at each embedding location
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_title(f'Embedding Visualization with Thumbnails ({method.upper()})')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.grid(True, alpha=0.3)

    # Normalize coordinates for better spacing if needed
    x_vals = df['x'].values
    y_vals = df['y'].values

    # Determine a reasonable thumbnail zoom based on data spread
    x_span = max(x_vals) - min(x_vals) if len(x_vals) > 0 else 1.0
    y_span = max(y_vals) - min(y_vals) if len(y_vals) > 0 else 1.0
    span = max(x_span, y_span)
    base_zoom = 0.15 if span == 0 else max(0.05, min(0.3, 0.15 * (2.0 / span)))

    # Freeze axis limits so annotations do not change data limits
    if len(x_vals) > 0 and len(y_vals) > 0:
        x_span = max(x_vals) - min(x_vals)
        y_span = max(y_vals) - min(y_vals)
        x_pad = 0.05 * (x_span if x_span > 0 else 1.0)
        y_pad = 0.05 * (y_span if y_span > 0 else 1.0)
        ax.set_xlim(min(x_vals) - x_pad, max(x_vals) + x_pad)
        ax.set_ylim(min(y_vals) - y_pad, max(y_vals) + y_pad)
    ax.autoscale(False)

    for (x, y), img in zip(zip(x_vals, y_vals), thumbnails):
        try:
            image_box = OffsetImage(img, zoom=base_zoom, resample=True)
            ab = AnnotationBbox(
                image_box,
                (x, y),
                frameon=True,
                xycoords='data',
                pad=0.0,
                annotation_clip=True,
            )
            ab.set_clip_on(True)
            ab.set_zorder(3)
            ax.add_artist(ab)
        except Exception:
            # Fallback to a small scatter point if image fails
            ax.scatter([x], [y], s=10, c='gray', alpha=0.6)

    plt.tight_layout()
    thumb_plot_path = os.path.join(output_dir, f'embeddings_thumbnails_{method}.png')
    plt.savefig(thumb_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved thumbnails plot: {thumb_plot_path}")

    # 5. Save data as CSV for further analysis
    csv_path = os.path.join(output_dir, f'embeddings_data_{method}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved data CSV: {csv_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Find checkpoint if not provided
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint()
    
    # Load model
    model = load_contrastive_model_from_checkpoint(args.checkpoint)
    
    # Create dataset
    dataset = create_dataset(args)
    
    # Extract embeddings
    embeddings, dataset_labels, modality_type_labels, sample_ids, thumbnails = extract_embeddings(
        model, dataset, args.num_examples, device
    )
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(embeddings, args.method)
    
    # Create visualizations
    create_visualizations(reduced_embeddings, dataset_labels, modality_type_labels, sample_ids,
                         thumbnails, args.output_dir, args.method)
    
    print(f"\nVisualization complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
