#!/usr/bin/env python3
"""
Train/Validation Split Generator for ARIA Synthetic Environment Dataset

This script generates random train/validation splits for the ARIA dataset
and saves them to text files that can be used as scene_ids_file parameters
for the AriaSynthEenvDataset class.

The ARIA dataset contains 5000 scenes numbered from 0 to 4999.
"""

import os
import random
import argparse
from typing import List, Tuple


def generate_train_val_split(
    total_scenes: int = 5000,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    output_dir: str = ".",
    scene_id_prefix: str = ""
) -> Tuple[List[str], List[str]]:
    """
    Generate random train/validation splits for ARIA dataset.
    
    Args:
        total_scenes: Total number of scenes in the dataset (default: 5000)
        train_ratio: Ratio of scenes to use for training (default: 0.8)
        random_seed: Random seed for reproducible splits (default: 42)
        output_dir: Directory to save the split files (default: current directory)
        scene_id_prefix: Optional prefix for scene IDs (default: "")
    
    Returns:
        Tuple of (train_scene_ids, val_scene_ids) as lists of strings
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Generate all scene IDs
    all_scene_ids = [f"{scene_id_prefix}{i}" for i in range(total_scenes)]
    
    # Randomly shuffle the scene IDs
    shuffled_scene_ids = all_scene_ids.copy()
    random.shuffle(shuffled_scene_ids)
    
    # Calculate split indices
    train_size = int(total_scenes * train_ratio)
    train_scene_ids = shuffled_scene_ids[:train_size]
    val_scene_ids = shuffled_scene_ids[train_size:]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training split
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, "w") as f:
        for scene_id in train_scene_ids:
            f.write(f"{scene_id}\n")
    
    # Save validation split
    val_file = os.path.join(output_dir, "val.txt")
    with open(val_file, "w") as f:
        for scene_id in val_scene_ids:
            f.write(f"{scene_id}\n")
    
    print(f"Generated train/val split:")
    print(f"  Training scenes: {len(train_scene_ids)} ({train_ratio:.1%})")
    print(f"  Validation scenes: {len(val_scene_ids)} ({1-train_ratio:.1%})")
    print(f"  Train file: {train_file}")
    print(f"  Val file: {val_file}")
    print(f"  Random seed: {random_seed}")
    
    return train_scene_ids, val_scene_ids


def main():
    """Main function to run the train/val split generation."""
    parser = argparse.ArgumentParser(
        description="Generate train/validation splits for ARIA dataset"
    )
    parser.add_argument(
        "--total-scenes",
        type=int,
        default=5000,
        help="Total number of scenes in the dataset (default: 5000)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of scenes to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/aria/SyntheticEnv/",
        help="Directory to save the split files (default: current directory)"
    )
    parser.add_argument(
        "--scene-id-prefix",
        type=str,
        default="",
        help="Optional prefix for scene IDs (default: empty)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if args.total_scenes <= 0:
        raise ValueError("total_scenes must be positive")
    
    # Generate splits
    train_ids, val_ids = generate_train_val_split(
        total_scenes=args.total_scenes,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        scene_id_prefix=args.scene_id_prefix
    )
    
    print("\nSplit generation completed successfully!")
    print(f"Use these files as scene_ids_file parameter in AriaSynthEenvDataset:")
    print(f"  Training: {os.path.join(args.output_dir, 'train.txt')}")
    print(f"  Validation: {os.path.join(args.output_dir, 'val.txt')}")


if __name__ == "__main__":
    main()
