#!/usr/bin/env python3
"""
Test script for UnifiedDataset.

This script demonstrates how to:
1. Create a UnifiedDataset instance combining multiple datasets
2. Iterate over all items across datasets
3. Display images with matplotlib
4. Show basic statistics about the unified dataset

Usage examples:
    python tests/test_unified_dataloader.py \
        --cubicasa-path /path/to/CubiCasa5k \
        --structured3d-path /path/to/Structured3D \
        --aria-synthenv-path /path/to/aria/SyntheticEnv/original_data \
        --swiss-dwellings-path /path/to/SwissDwellings/modified-swiss-dwellings-v2/train \
        --image-size 512 512

    # You can also run with only one dataset
    python tests/test_unified_dataloader.py --cubicasa-path /path/to/CubiCasa5k
    python tests/test_unified_dataloader.py --structured3d-path /path/to/Structured3D
    python tests/test_unified_dataloader.py --aria-synthenv-path /path/to/aria/SyntheticEnv/original_data
    python tests/test_unified_dataloader.py --swiss-dwellings-path /path/to/SwissDwellings/modified-swiss-dwellings-v2/train
"""

import argparse
import os
import sys
import time
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataloading.dual_transforms import  PairToPIL, PairToTensor, PairResize, PairGrayscale

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.unified_dataset import UnifiedDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test UnifiedDataset by iterating over items from multiple datasets"
    )
    parser.add_argument(
        "--cubicasa-path",
        type=str,
        help="Path to CubiCasa5k dataset root directory (optional)"
    )
    parser.add_argument(
        "--cubicasa-ids",
        type=str,
        help="Optional text file with CubiCasa5k IDs to include"
    )
    parser.add_argument(
        "--structured3d-path",
        type=str,
        help="Path to Structured3D dataset root directory (optional)"
    )
    parser.add_argument(
        "--structured3d-ids",
        type=str,
        help="Optional text file with Structured3D scene IDs to include"
    )
    parser.add_argument(
        "--aria-synthenv-path",
        type=str,
        help="Path to Aria Synthetic Environments root directory (optional)"
    )
    parser.add_argument(
        "--aria-synthenv-ids",
        type=str,
        help="Optional text file with ASE scene IDs to include"
    )
    parser.add_argument(
        "--scannet-path",
        type=str,
        help="Path to ScanNet dataset root directory (optional)"
    )
    parser.add_argument(
        "--scannet-ids",
        type=str,
        help="Optional text/JSON file with ScanNet scene IDs to include"
    )
    parser.add_argument(
        "--scannet-generate",
        action="store_true",
        help="Generate ScanNet modalities from raw files instead of loading pre-rendered images"
    )
    parser.add_argument(
        "--scannet-resolution",
        type=int,
        default=1024,
        help="Resolution for ScanNet generated images (default: 1024)"
    )
    parser.add_argument(
        "--scannet-label-filter",
        type=str,
        help="Optional label filter for ScanNet floorplan rendering (e.g., 'wall')"
    )
    parser.add_argument(
        "--aria-generate",
        action="store_true",
        help="Generate ASE modalities from raw files instead of loading pre-rendered images"
    )
    parser.add_argument(
        "--swiss-dwellings-path",
        type=str,
        help="Path to SwissDwellings split directory (e.g., .../modified-swiss-dwellings-v2/train)"
    )
    parser.add_argument(
        "--swiss-dwellings-ids",
        type=str,
        help="Optional text file with SwissDwellings IDs to include"
    )
    parser.add_argument(
        "--swiss-generate",
        action="store_true",
        help="Generate SwissDwellings modalities from raw files instead of loading pre-rendered images"
    )
    parser.add_argument(
        "--zillow-path",
        type=str,
        help="Path to Zillow/ZInD rendered dataset root directory (optional)"
    )
    parser.add_argument(
        "--zillow-ids",
        type=str,
        help="Optional text file with Zillow sample IDs to include"
    )
    parser.add_argument(
        "--zillow-generate",
        action="store_true",
        help="Generate Zillow modalities from raw files instead of loading pre-rendered images"
    )
    parser.add_argument(
        "--aria-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Orthographic projection axis for ASE (default: z)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to HEIGHT x WIDTH (e.g., --image-size 512 512)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for Structured3D rendering (default: 100)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Generate grayscale/no-color for Structured3D"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum number of unified items to display (default: all items)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save images (if not provided, images are only displayed)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display images interactively (useful for saving only)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for dataloader (if not provided, processes items individually)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for the unified dataset"
    )
    return parser.parse_args()


def create_unified_dataset(args) -> UnifiedDataset:
    """Create and configure the unified dataset from provided paths."""

    # Basic transforms for visualization
    transform = [
        PairToPIL(),
        PairResize((512, 512)) if args.image_size is None else PairResize(args.image_size),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
    ]

    dataset_configs: List[Dict[str, Any]] = []

    if args.cubicasa_path:
        if not os.path.exists(args.cubicasa_path):
            raise FileNotFoundError(f"CubiCasa5k path does not exist: {args.cubicasa_path}")
        dataset_configs.append({
            "type": "cubicasa5k",
            "args": {
                "root_dir": args.cubicasa_path,
                "sample_ids_file": args.cubicasa_ids,
                "image_size": args.image_size,
                "dual_transform": transform,
                # set a deterministic pair for visualization
                # "modality_pairs": [("drawing", "drawing")],
            },
        })

    if args.structured3d_path:
        if not os.path.exists(args.structured3d_path):
            raise FileNotFoundError(f"Structured3D path does not exist: {args.structured3d_path}")
        dataset_configs.append({
            "type": "structured3d",
            "args": {
                "root_dir": args.structured3d_path,
                "scene_ids_file": args.structured3d_ids,
                "no_color": args.no_color,
                "image_size": args.image_size,
                "dpi": args.dpi,
                "dual_transform": transform,
                # set a deterministic pair for visualization
                # "modality_pairs": [("floorplan", "floorplan")],
            },
        })

    if args.aria_synthenv_path:
        if not os.path.exists(args.aria_synthenv_path):
            raise FileNotFoundError(f"Aria SynthEnv path does not exist: {args.aria_synthenv_path}")
        dataset_configs.append({
            "type": "aria_synthenv",
            "args": {
                "root_dir": args.aria_synthenv_path,
                "scene_ids_file": args.aria_synthenv_ids,
                "image_size": args.image_size,
                "dual_transform": transform,
                "generate": args.aria_generate,
                "ortho_axis": args.aria_axis,
                # "scene_ids_file": "data/aria/SyntheticEnv/original_data/train.txt",
            },
        })

    if args.swiss_dwellings_path:
        if not os.path.exists(args.swiss_dwellings_path):
            raise FileNotFoundError(f"SwissDwellings path does not exist: {args.swiss_dwellings_path}")
        dataset_configs.append({
            "type": "swiss_dwellings",
            "args": {
                "root_dir": args.swiss_dwellings_path,
                "sample_ids_file": args.swiss_dwellings_ids,
                "image_size": args.image_size,
                "dual_transform": transform,
                "generate": args.swiss_generate,
            },
        })

    if args.scannet_path:
        if not os.path.exists(args.scannet_path):
            raise FileNotFoundError(f"ScanNet path does not exist: {args.scannet_path}")
        dataset_configs.append({
            "type": "scannet",
            "args": {
                "root_dir": args.scannet_path,
                "scene_ids_file": args.scannet_ids,
                "image_size": args.image_size,
                "dual_transform": transform,
                "generate": args.scannet_generate,
                "resolution": args.scannet_resolution,
                # "modality_pairs": [("density_map_mesh", "density_map_mesh_noisy")],
            },
        })

    if args.zillow_path:
        if not os.path.exists(args.zillow_path):
            raise FileNotFoundError(f"Zillow path does not exist: {args.zillow_path}")
        dataset_configs.append({
            "type": "zillow",
            "args": {
                "root_dir": args.zillow_path,
                "sample_ids_file": args.zillow_ids,
                "image_size": args.image_size,
                "dual_transform": transform,
                "generate": args.zillow_generate,
            },
        })

    if len(dataset_configs) == 0:
        raise ValueError("Provide at least one dataset path via --cubicasa-path, --structured3d-path, --aria-synthenv-path, --swiss-dwellings-path, --scannet-path, or --zillow-path")

    print("Creating UnifiedDataset with:")
    if args.cubicasa_path:
        print(f"  - CubiCasa5k: {args.cubicasa_path}")
    if args.structured3d_path:
        print(f"  - Structured3D: {args.structured3d_path}")
    if args.aria_synthenv_path:
        print(f"  - AriaSynthEnv: {args.aria_synthenv_path}")
    if args.swiss_dwellings_path:
        print(f"  - SwissDwellings: {args.swiss_dwellings_path}")
    if args.scannet_path:
        print(f"  - ScanNet: {args.scannet_path}")
    if args.zillow_path:
        print(f"  - Zillow: {args.zillow_path}")
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")
    print(f"  DPI (S3D): {args.dpi}")
    print(f"  No color (S3D): {args.no_color}")

    unified = UnifiedDataset(dataset_configs=dataset_configs)
    return unified


def display_item(unified: UnifiedDataset, index: int, args, save_dir: Optional[str] = None):
    """Display a single unified item."""

    start_time = time.time()
    sample = unified[index]
    load_time = time.time() - start_time

    source = sample.get("source_dataset", "unknown")
    sample_id = sample.get("sample_id", "?")
    m0_image = sample["modality_0"]
    m1_image = sample["modality_1"]
    m0_type = sample["m0_type"]
    m1_type = sample["m1_type"]
    if "m0_description" in sample:
        m0_description = sample["m0_description"]
    else:
        m0_description = None
    if "m1_description" in sample:
        m1_description = sample["m1_description"]
    else:
        m1_description = None

    # Convert tensors back to numpy for display
    def tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            np_array = tensor.numpy()
            # Ensure values are in [0, 1] for display
            if np_array.max() <= 1.0:
                np_array = (np_array * 255).astype(np.uint8)
            else:
                np_array = np_array.astype(np.uint8)
        else:
            np_array = np.array(tensor)
        return np_array

    m0_np = tensor_to_numpy(m0_image)
    m1_np = tensor_to_numpy(m1_image)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Modality 0
    ax1.imshow(m0_np.transpose(1, 2, 0))
    if m0_description is not None:
        ax1.set_title(f"[{source}] {m0_type}\n{m0_description}\nSize: {m0_np.shape[1]}x{m0_np.shape[2]}")
    else:
        ax1.set_title(f"[{source}] {m0_type}\nSize: {m0_np.shape[1]}x{m0_np.shape[2]}")
    ax1.axis('off')

    # Modality 1
    ax2.imshow(m1_np.transpose(1, 2, 0))
    if m1_description is not None:
        ax2.set_title(f"{m1_type}\n{m1_description}\nSize: {m1_np.shape[1]}x{m1_np.shape[2]}")
    else:
        ax2.set_title(f"{m1_type}\nSize: {m1_np.shape[1]}x{m1_np.shape[2]}")
    ax2.axis('off')

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{source}_{sample_id}_{m0_type}_{m1_type}.png"
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")

    # Display if not in no-display mode
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    return load_time


def print_item_info(sample, index: int):
    source = sample.get("source_dataset", "unknown")
    print(f"\nItem {index}: {sample.get('sample_id', '?')} [{source}]")
    print(f"  Directory: {sample.get('sample_dir', '?')}")
    print(f"  Modality pair: {sample['m0_type']} + {sample['m1_type']}")

    # Modality 0 info
    m0_image = sample['modality_0']
    if isinstance(m0_image, torch.Tensor):
        print(f"  {sample['m0_type']} tensor shape: {m0_image.shape}")
        print(f"  {sample['m0_type']} tensor dtype: {m0_image.dtype}")
    else:
        print(f"  {sample['m0_type']} numpy shape: {m0_image.shape}")
        print(f"  {sample['m0_type']} numpy dtype: {m0_image.dtype}")

    # Modality 1 info
    m1_image = sample['modality_1']
    if isinstance(m1_image, torch.Tensor):
        print(f"  {sample['m1_type']} tensor shape: {m1_image.shape}")
        print(f"  {sample['m1_type']} tensor dtype: {m1_image.dtype}")
    else:
        print(f"  {sample['m1_type']} numpy shape: {m1_image.shape}")
        print(f"  {sample['m1_type']} numpy dtype: {m1_image.dtype}")


def main():
    args = parse_args()

    # Check if GPU is available
    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")

    if args.save_dir and not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create save directory {args.save_dir}: {e}")
            sys.exit(1)

    # Create unified dataset
    try:
        dataset = create_unified_dataset(args)
    except Exception as e:
        print(f"Error creating unified dataset: {e}")
        sys.exit(1)

    print(f"\nUnified dataset created successfully!")
    print(f"Total items: {len(dataset)}")

    # Determine how many items to process
    num_items = len(dataset)
    if args.max_items:
        num_items = min(num_items, args.max_items)

    print(f"Processing {num_items} items...")

    # Use dataloader if batch_size is provided
    if args.batch_size is not None:
        # Create a subset of the dataset if max_items is specified
        if args.max_items:
            dataset = torch.utils.data.Subset(dataset, range(num_items))
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging, can be increased for speed
            drop_last=False
        )
        
        print(f"Using dataloader with batch size: {args.batch_size}")
        print(f"Number of batches: {len(dataloader)}")
        
        # Process batches
        total_start_time = time.time()
        total_batches = 0
        total_items = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_size = len(batch['modality_0']) if isinstance(batch['modality_0'], list) else batch['modality_0'].size(0)
            total_batches += 1
            total_items += batch_size
            
            print(f"Batch {batch_idx + 1}/{len(dataloader)}: {batch_size} items")
            
            # Print image sizes for each item in the batch
            for i in range(batch_size):
                m0_shape = batch['modality_0'][i].shape if hasattr(batch['modality_0'][i], 'shape') else f"list of {len(batch['modality_0'][i])}"
                m1_shape = batch['modality_1'][i].shape if hasattr(batch['modality_1'][i], 'shape') else f"list of {len(batch['modality_1'][i])}"
                source = batch.get('source_dataset', ['unknown'])[i] if isinstance(batch.get('source_dataset', 'unknown'), list) else batch.get('source_dataset', 'unknown')
                sample_id = batch.get('sample_id', ['?'])[i] if isinstance(batch.get('sample_id', '?'), list) else batch.get('sample_id', '?')
                
                print(f"  Item {i+1}: {source}_{sample_id} - M0: {m0_shape}, M1: {m1_shape}")
            
            # Stop if we've processed enough items
            if args.max_items and total_items >= args.max_items:
                break
        
        total_time = time.time() - total_start_time
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total batches processed: {total_batches}")
        print(f"Total items processed: {total_items}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per batch: {total_time/total_batches:.2f}s")
        print(f"Average time per item: {total_time/total_items:.2f}s")
        
    else:
        # Original single-item processing
        load_times = []
        total_start_time = time.time()

        for i in range(num_items):
            print(f"\nProcessing item {i+1}/{num_items}...")

            try:
                load_time = display_item(dataset, i, args, args.save_dir)
                load_times.append(load_time)

                # Print detailed info for first few items
                if i < 3:
                    sample = dataset[i]
                    print_item_info(sample, i)

            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue

        total_time = time.time() - total_start_time

        # Print statistics
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total items processed: {len(load_times)}")
        if load_times:
            print(f"Total time: {total_time:.2f}s")
            print(f"Average load time per item: {np.mean(load_times):.2f}s")
            print(f"Min load time: {np.min(load_times):.2f}s")
            print(f"Max load time: {np.max(load_times):.2f}s")


if __name__ == "__main__":
    main()


