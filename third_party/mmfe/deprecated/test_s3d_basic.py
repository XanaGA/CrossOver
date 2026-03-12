#!/usr/bin/env python3
"""
Basic test script for Structured3DDataset without matplotlib dependency.
"""

import os
import sys

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.s3d_data import Structured3DDataset


def main():
    # Test with a small subset
    dataset_path = "data/structure3D"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return
    
    print(f"Testing Structured3DDataset with path: {dataset_path}")
    
    try:
        # Create dataset
        dataset = Structured3DDataset(
            root_dir=dataset_path,
            to_tensor=True
        )
        
        print(f"Dataset created successfully!")
        print(f"Total scenes: {len(dataset)}")
        print(f"Available modalities: {dataset.modalities}")
        print(f"Modality pairs: {dataset.modality_pairs}")
        
        if len(dataset) == 0:
            print("No scenes found!")
            return
        
        # Test first sample
        print("\nTesting first sample...")
        sample = dataset[0]
        
        print(f"Scene ID: {sample['sample_id']}")
        print(f"Modality 0 type: {sample['m0_type']}")
        print(f"Modality 1 type: {sample['m1_type']}")
        print(f"Modality 0 shape: {sample['modality_0'].shape}")
        print(f"Modality 1 shape: {sample['modality_1'].shape}")
        print(f"Original size: {sample['original_size']}")
        print(f"Current size: {sample['current_size']}")
        
        # Test multiple samples to see different modality pairs
        print("\nTesting multiple samples for modality pair variety...")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: {sample['m0_type']} + {sample['m1_type']}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
