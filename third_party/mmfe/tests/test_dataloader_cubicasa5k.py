#!/usr/bin/env python3
"""
Test script for Cubicasa5kDataset.

This script demonstrates how to:
1. Create a Cubicasa5kDataset instance
2. Iterate over all samples
3. Display images with matplotlib
4. Show basic statistics about the dataset

Usage:
    python tests/test_dataloader_cubicasa5k.py --path /path/to/CubiCasa5k --use-original-size
    python tests/test_dataloader_cubicasa5k.py --path /path/to/CubiCasa5k --image-size 512 512
"""

import argparse
import os
import sys
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from mmfe_utils.tensor_utils import tensor_to_numpy

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.cubicasa_data import Cubicasa5kDataset
from dataloading.dual_transforms import  PairToTensor, PairResize



def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Cubicasa5kDataset by iterating over all samples"
    )
    parser.add_argument(
        "--path", 
        required=True,
        help="Path to CubiCasa5k dataset root directory"
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        help="Optional path to a text file listing scene IDs to include"
    )
    parser.add_argument(
        "--use-original-size", 
        action="store_true",
        help="Use F1_original.png instead of F1_scaled.png"
    )
    parser.add_argument(
        "--image-size", 
        type=int, 
        nargs=2, 
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to HEIGHT x WIDTH (e.g., --image-size 512 512)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        help="Maximum number of samples to display (default: all samples)"
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
        help="Batch size for dataloader (if not provided, processes samples individually)"
    )
    
    return parser.parse_args()


def create_dataset(args) -> Cubicasa5kDataset:
    """Create and configure the dataset."""
    
    # Basic transforms for visualization
    transform = [
        PairResize((512, 512)) if args.image_size is None else PairResize(args.image_size),
        PairToTensor(),
    ]
    
    print(f"Creating dataset with:")
    print(f"  Root directory: {args.path}")
    print(f"  Use original size: {args.use_original_size}")
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")
    
    dataset = Cubicasa5kDataset(
        root_dir=args.path,
        sample_ids_file=args.ids_file,
        image_size=args.image_size,
        use_original_size=args.use_original_size,
        dual_transform=transform,
        modality_pairs=[("gt_svg_annotations", "lidar_points")]
    )
    
    return dataset


def display_sample(dataset: Cubicasa5kDataset, index: int, args, save_dir: Optional[str] = None):
    """Display a single sample."""
    
    start_time = time.time()
    
    # Get the sample
    sample = dataset[index]
    sample_id = sample["sample_id"]
    m0_image = sample["modality_0"]
    m1_image = sample["modality_1"]
    m0_type = sample["m0_type"]
    m1_type = sample["m1_type"]
    
    load_time = time.time() - start_time
    
    m0_np = tensor_to_numpy(m0_image)
    m1_np = tensor_to_numpy(m1_image)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Modality 0
    ax1.imshow(m0_np)
    ax1.set_title(f"Sample {sample_id} - {m0_type}\n"
                f"Size: {m0_np.shape[0]}x{m0_np.shape[1]}")
    ax1.axis('off')
    
    # Modality 1
    ax2.imshow(m1_np)
    ax2.set_title(f"{m1_type}\n"
                f"Size: {m0_np.shape[0]}x{m0_np.shape[1]}")
    ax2.axis('off')
        
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sample_{sample_id}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    
    # Display if not in no-display mode
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)
    
    return load_time


def print_sample_info(sample, index: int):
    """Print detailed information about a sample."""
    print(f"\nSample {index}: {sample['sample_id']}")
    print(f"  Directory: {sample['sample_dir']}")
    print(f"  Modality pair: {sample['m0_type']} + {sample['m1_type']}")
    
    # Print modality 0 info
    m0_image = sample['modality_0']
    if isinstance(m0_image, torch.Tensor):
        print(f"  {sample['m0_type']} tensor shape: {m0_image.shape}")
        print(f"  {sample['m0_type']} tensor dtype: {m0_image.dtype}")
    else:
        print(f"  {sample['m0_type']} numpy shape: {m0_image.shape}")
        print(f"  {sample['m0_type']} numpy dtype: {m0_image.dtype}")
    
    # Print modality 1 info
    m1_image = sample['modality_1']
    if isinstance(m1_image, torch.Tensor):
        print(f"  {sample['m1_type']} tensor shape: {m1_image.shape}")
        print(f"  {sample['m1_type']} tensor dtype: {m1_image.dtype}")
    else:
        print(f"  {sample['m1_type']} numpy shape: {m1_image.shape}")
        print(f"  {sample['m1_type']} numpy dtype: {m1_image.dtype}")


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
    print(f"Total samples: {len(dataset)}")
    
    # Determine how many samples to process
    num_samples = len(dataset)
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)
    
    print(f"Processing {num_samples} samples...")

    # Use dataloader if batch_size is provided
    if args.batch_size is not None:
        # Create a subset of the dataset if max_samples is specified
        if args.max_samples:
            dataset = torch.utils.data.Subset(dataset, range(num_samples))
        
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
                sample_id = batch.get('sample_id', ['?'])[i] if isinstance(batch.get('sample_id', '?'), list) else batch.get('sample_id', '?')
                
                print(f"  Item {i+1}: {sample_id} - M0: {m0_shape}, M1: {m1_shape}")
            
            # Stop if we've processed enough samples
            if args.max_samples and total_items >= args.max_samples:
                break
        
        total_time = time.time() - total_start_time
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total batches processed: {total_batches}")
        print(f"Total samples processed: {total_items}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per batch: {total_time/total_batches:.2f}s")
        print(f"Average time per sample: {total_time/total_items:.2f}s")
        
    else:
        # Original single-sample processing
        load_times = []
        total_start_time = time.time()
        
        for i in range(num_samples):
            print(f"\nProcessing sample {i+1}/{num_samples}...")
            
            try:
                load_time = display_sample(dataset, i, args, args.save_dir)
                load_times.append(load_time)
                
                # Print detailed info for first few samples
                if i < 3:
                    sample = dataset[i]
                    print_sample_info(sample, i)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Print statistics
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total samples processed: {len(load_times)}")
        if load_times:
            print(f"Total time: {total_time:.2f}s")
            print(f"Average load time per sample: {np.mean(load_times):.2f}s")
            print(f"Min load time: {np.min(load_times):.2f}s")
            print(f"Max load time: {np.max(load_times):.2f}s")
    
    if args.save_dir:
        print(f"Images saved to: {args.save_dir}")
    
    print(f"\nDataset configuration:")
    print(f"  Use original size: {args.use_original_size}")
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")


if __name__ == "__main__":
    main()
