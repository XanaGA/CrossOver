#!/usr/bin/env python3
"""
Utility script to initialize NetVLAD clusters from training data.

This script extracts features from a random subset of training data and
uses k-means clustering to initialize the NetVLAD cluster centers.

Supports multiple datasets simultaneously (CubiCasa5k, Structured3D, Aria).

Usage:
    # Single dataset
    python scripts/utility_scripts/init_netvlad_clusters.py \
        --backbone vgg16 \
        --num_clusters 64 \
        --num_descriptors 50000 \
        --cubicasa_root data/cubicasa5k \
        --cubicasa_samples data/cubicasa5k/train.txt \
        --output outputs/clusters
    
    # Multiple datasets
    python scripts/utility_scripts/init_netvlad_clusters.py \
        --backbone vgg16 \
        --num_clusters 64 \
        --cubicasa_root data/cubicasa5k \
        --s3d_root data/structure3D \
        --aria_root data/aria/SyntheticEnv \
        --output outputs/clusters
"""

import os
import sys
import argparse
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from global_descriptors.backbones import GlobalDescriptorBackbone, load_dino_backbone, load_vgg16_backbone
from training.models import ContrastiveModel
from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import PairToPIL, PairResize, PairGrayscale, PairToTensor, PairNormalize

def create_dataset(args) -> UnifiedDataset:
    # Simple transforms for feature extraction
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    transforms = [
        PairToPIL(),
        PairResize(tuple(args.image_size)),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
        PairNormalize(mean=mean, std=std),
    ]
    
    # Build dataset configs dynamically based on what's provided
    dataset_configs = []
    
    # CubiCasa5k dataset
    if args.cubicasa_root is not None:
        print(f"  - Adding CubiCasa5k from: {args.cubicasa_root}")
        config = {
            "type": "cubicasa5k",
            "args": {
                "root_dir": args.cubicasa_root,
                "image_size": tuple(args.image_size),
            }
        }
        if args.cubicasa_samples:
            config["args"]["sample_ids_file"] = args.cubicasa_samples
        dataset_configs.append(config)
    
    # Structured3D dataset
    if args.s3d_root is not None:
        print(f"  - Adding Structured3D from: {args.s3d_root}")
        config = {
            "type": "structured3d",
            "args": {
                "root_dir": args.s3d_root,
                "image_size": tuple(args.image_size),
            }
        }
        if args.s3d_samples:
            config["args"]["scene_ids_file"] = args.s3d_samples
        dataset_configs.append(config)
    
    # Aria SyntheticEnv dataset
    if args.aria_root is not None:
        print(f"  - Adding Aria SyntheticEnv from: {args.aria_root}")
        config = {
            "type": "aria_synthenv",
            "args": {
                "root_dir": args.aria_root,
                "image_size": tuple(args.image_size),
            }
        }
        if args.aria_samples:
            config["args"]["scene_ids_file"] = args.aria_samples
        dataset_configs.append(config)
    
    # Check that at least one dataset is provided
    if len(dataset_configs) == 0:
        raise ValueError(
            "No datasets specified! Provide at least one of: "
            "--cubicasa_root, --s3d_root, --aria_root"
        )
    
    dataset = UnifiedDataset(
        dataset_configs=dataset_configs,
        common_transform=transforms,
        invertible_transform=None,
    )

    return dataset

def extract_descriptors(
    backbone: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_descriptors: int,
    device: torch.device,
) -> np.ndarray:
    """
    Extract local descriptors from images using the backbone.
    
    Args:
        backbone: Backbone model that outputs 2D feature maps
        dataloader: DataLoader for images
        num_descriptors: Total number of descriptors to extract
        device: Device to run on
        
    Returns:
        Array of descriptors (num_descriptors, feature_dim)
    """
    backbone.eval()
    
    # Get feature dimension from first batch
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_features = backbone(sample_batch["modality_0"][:1].to(device))
        _, feature_dim, h, w = sample_features.shape
    
    descriptors_per_image = min(100, h * w)
    num_images_needed = (num_descriptors + descriptors_per_image - 1) // descriptors_per_image
    
    print(f"Feature dimension: {feature_dim}")
    print(f"Feature map size: {h} x {w}")
    print(f"Extracting {descriptors_per_image} descriptors per image")
    print(f"Need {num_images_needed} images to get {num_descriptors} descriptors")
    
    all_descriptors = []
    total_extracted = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting descriptors"):
            if total_extracted >= num_descriptors:
                break
            
            images = batch["modality_0"].to(device)
            
            # Extract features (B, C, H, W)
            features = backbone(images)
            
            # Reshape to (B, C, H*W) and permute to (B, H*W, C)
            batch_size, c, h, w = features.shape
            features = features.view(batch_size, c, -1).permute(0, 2, 1)  # (B, H*W, C)
            
            # Sample random spatial locations for each image
            for img_features in features:
                if total_extracted >= num_descriptors:
                    break
                
                # img_features: (H*W, C)
                num_to_sample = min(descriptors_per_image, num_descriptors - total_extracted)
                
                # Random sampling
                indices = np.random.choice(img_features.shape[0], num_to_sample, replace=False)
                sampled = img_features[indices].cpu().numpy()
                
                all_descriptors.append(sampled)
                total_extracted += num_to_sample
    
    # Concatenate all descriptors
    descriptors = np.concatenate(all_descriptors, axis=0)[:num_descriptors]
    
    print(f"Extracted {descriptors.shape[0]} descriptors with dimension {descriptors.shape[1]}")
    
    return descriptors


def cluster_descriptors(
    descriptors: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """
    Cluster descriptors using MiniBatch k-means.
    
    Args:
        descriptors: Array of descriptors (N, D)
        num_clusters: Number of clusters
        
    Returns:
        Cluster centers (num_clusters, D)
    """
    print(f"\nClustering {descriptors.shape[0]} descriptors into {num_clusters} clusters...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        max_iter=100,
        batch_size=1000,
        verbose=1,
        random_state=42,
    )
    
    kmeans.fit(descriptors)
    
    print(f"✓ Clustering complete!")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    
    return kmeans.cluster_centers_


def main():
    parser = argparse.ArgumentParser(description="Initialize NetVLAD clusters")
    
    # Model arguments
    parser.add_argument("--backbone", type=str, default="vgg16",
                       help="Backbone architecture")
    parser.add_argument("--local_path", type=str, default=None,
                       help="Local path to DINO weights")
    parser.add_argument("--weights_path", type=str, default=None,
                       help="Path to DINO weights")
    parser.add_argument("--pretrained", action="store_true",
                       help="Use pretrained backbone")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to MMFE checkpoint")
    
    # Clustering arguments
    parser.add_argument("--num_clusters", type=int, default=64,
                       help="Number of NetVLAD clusters")
    parser.add_argument("--num_descriptors", type=int, default=50000,
                       help="Number of descriptors to use for clustering")
    
    # Data arguments - support multiple datasets
    # CubiCasa5k
    parser.add_argument("--cubicasa_root", type=str, default=None,
                       help="Root directory of CubiCasa5k dataset")
    parser.add_argument("--cubicasa_samples", type=str, default=None,
                       help="File with CubiCasa5k sample IDs")
    
    # Structured3D
    parser.add_argument("--s3d_root", type=str, default=None,
                       help="Root directory of Structured3D dataset")
    parser.add_argument("--s3d_samples", type=str, default=None,
                       help="File with Structured3D scene IDs")
    
    # Aria SyntheticEnv
    parser.add_argument("--aria_root", type=str, default=None,
                       help="Root directory of Aria SyntheticEnv dataset")
    parser.add_argument("--aria_samples", type=str, default=None,
                       help="File with Aria scene IDs")
    
    # Common data arguments
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                       help="Image size (H W)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Output arguments
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for cluster centers (filename will be auto-generated)")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create backbone
    backbone_configs = {
        "name": args.backbone,
        "pretrained": args.pretrained,
        "kwargs": {
            "local_path": args.local_path,
            "weights_path": args.weights_path,
            "checkpoint_path": args.checkpoint_path,
        },
        "freeze": "all",
    }
    backbone_configs = OmegaConf.create(backbone_configs)
    backbone = GlobalDescriptorBackbone(backbone_configs=backbone_configs)
    
    backbone = backbone.to(device)
    backbone.eval()
    
    # Create dataset(s)
    print(f"\nCreating unified dataset...")
    
    dataset = create_dataset(args)
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Extract descriptors
    print(f"\nExtracting {args.num_descriptors} descriptors...")
    descriptors = extract_descriptors(
        backbone=backbone,
        dataloader=dataloader,
        num_descriptors=args.num_descriptors,
        device=device,
    )
    
    # Cluster descriptors
    cluster_centers = cluster_descriptors(
        descriptors=descriptors,
        num_clusters=args.num_clusters,
    )
    
    # Save cluster centers
    output_path = os.path.join(args.output, f"netvlad_clusters_{args.backbone}_{args.num_clusters}.pth")
    print(f"\nSaving cluster centers to: {output_path}")
    
    save_dict = {
        "centroids": cluster_centers,
        "descriptors": descriptors[:10000],  
        "config": {
            "backbone": args.backbone,
            "num_clusters": args.num_clusters,
            "num_descriptors": args.num_descriptors,
            "image_size": args.image_size,
        }
    }
    
    os.makedirs(args.output, exist_ok=True)
    torch.save(save_dict, output_path)
    
    print(f"✓ Done! Cluster centers saved to {output_path}")
    print(f"\nTo use these clusters, load them and call netvlad.init_params(centroids, descriptors)")


if __name__ == "__main__":
    main()

