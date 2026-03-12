#!/usr/bin/env python3
"""
Test script for Structured3DDataset.

This script demonstrates how to:
1. Create a Structured3DDataset instance
2. Iterate over all scenes
3. Display images with matplotlib
4. Show basic statistics about the dataset

Usage:
    python tests/test_dataloader_structured3D.py --path /path/to/Structured3D
    python tests/test_dataloader_structured3D.py --path /path/to/Structured3D --show-modalities
"""

import argparse
import os
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloading.dual_transforms import PairToTensor, PairResize

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.s3d_data import Structured3DDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Structured3DDataset by iterating over all scenes"
    )
    parser.add_argument(
        "--path", 
        required=True,
        help="Path to Structured3D dataset root directory"
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        help="Optional path to a text file listing scene IDs to include"
    )
    parser.add_argument(
        "--show-modalities", 
        action="store_true",
        help="Display both modalities side by side"
    )
    parser.add_argument(
        "--no-color", 
        action="store_true",
        help="Generate grayscale/no-color images"
    )
    parser.add_argument(
        "--bbox-pct", 
        type=float, 
        default=1.0,
        help="Percentage of bounding boxes to display (0.0 to 1.0, default: 1.0)"
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
        help="DPI for image generation (default: 100)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        help="Maximum number of scenes to display (default: all scenes)"
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
        help="Batch size for dataloader (if not provided, processes scenes individually)"
    )
    
    return parser.parse_args()


def create_dataset(args) -> Structured3DDataset:
    """Create and configure the dataset."""
    
    # Basic transforms for visualization
    transform = [
        PairResize((512, 512)) if args.image_size is None else PairResize(args.image_size),
        PairToTensor(),
    ]
    
    print(f"Creating dataset with:")
    print(f"  Root directory: {args.path}")
    print(f"  No color: {args.no_color}")
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")
    print(f"  DPI: {args.dpi}")
    
    dataset = Structured3DDataset(  
        root_dir=args.path,
        scene_ids_file=args.ids_file,
        no_color=args.no_color,
        image_size=args.image_size,
        dpi=args.dpi,
        dual_transform=transform,
        # modality_pairs=[("floorplan", "lidar")]
    )
    
    return dataset


def display_scene(dataset: Structured3DDataset, index: int, args, save_dir: Optional[str] = None):
    """Display a single scene."""
    
    start_time = time.time()
    
    # Get the sample
    sample = dataset[index]
    scene_id = sample["sample_id"]
    m0_image = sample["modality_0"]
    m1_image = sample["modality_1"]
    m0_type = sample["m0_type"]
    m1_type = sample["m1_type"]
    
    load_time = time.time() - start_time
    
    # Convert tensors back to numpy for display
    def tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            # CHW -> HWC
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
    ax1.set_title(f"Scene {int(scene_id):05d} - {m0_type}\n"
                 f"Size: {m0_np.shape[0]}x{m0_np.shape[1]}")
    ax1.axis('off')
    
    # Modality 1
    ax2.imshow(m1_np.transpose(1, 2, 0))
    ax2.set_title(f"{m1_type}\n"
                 f"Size: {m1_np.shape[0]}x{m1_np.shape[1]}")
    ax2.axis('off')
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"scene_{int(scene_id):05d}_{m0_type}_{m1_type}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    
    # Display if not in no-display mode
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)
    
    return load_time


def main():
    args = parse_args()
    
    # Validate arguments
    if not os.path.exists(args.path):
        print(f"Error: Dataset path does not exist: {args.path}")
        sys.exit(1)
    
    if args.save_dir and not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create save directory {args.save_dir}: {e}")
            sys.exit(1)
    
    # Create dataset
    try:
        dataset = create_dataset(args)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)
    
    print(f"\nDataset created successfully!")
    print(f"Total scenes: {len(dataset)}")
    
    # Determine how many scenes to process
    num_scenes = len(dataset)
    if args.max_samples:
        num_scenes = min(num_scenes, args.max_samples)
    
    print(f"Processing {num_scenes} scenes...")

    # Use dataloader if batch_size is provided
    if args.batch_size is not None:
        # Create a subset of the dataset if max_samples is specified
        if args.max_samples:
            dataset = torch.utils.data.Subset(dataset, range(num_scenes))
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
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
                scene_id = batch.get('sample_id', ['?'])[i] if isinstance(batch.get('sample_id', '?'), list) else batch.get('sample_id', '?')[i]
                
                print(f"  Item {i+1}: {scene_id} - M0: {m0_shape}, M1: {m1_shape}")
            
            # Stop if we've processed enough scenes
            if args.max_samples and total_items >= args.max_samples:
                break
        
        total_time = time.time() - total_start_time
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total batches processed: {total_batches}")
        print(f"Total scenes processed: {total_items}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per batch: {total_time/total_batches:.2f}s")
        print(f"Average time per scene: {total_time/total_items:.2f}s")
        
    else:
        # Original single-scene processing
        load_times = []
        total_start_time = time.time()
        
        for i in range(num_scenes):
            print(f"\nProcessing scene {i+1}/{num_scenes}...")
            
            try:
                load_time = display_scene(dataset, i, args, args.save_dir)
                load_times.append(load_time)
                
            except Exception as e:
                print(f"Error processing scene {i}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total scenes processed: {len(load_times)}")
        if load_times:
            print(f"Total time: {total_time:.2f}s")
            print(f"Average load time per scene: {np.mean(load_times):.2f}s")
            print(f"Min load time: {np.min(load_times):.2f}s")
            print(f"Max load time: {np.max(load_times):.2f}s")
    
    if args.save_dir:
        print(f"Images saved to: {args.save_dir}")
    
    print(f"\nDataset configuration:")
    print(f"  No color: {args.no_color}")
    print(f"  BBox percentage: {args.bbox_pct}")
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")
    print(f"  DPI: {args.dpi}")


if __name__ == "__main__":
    main() 