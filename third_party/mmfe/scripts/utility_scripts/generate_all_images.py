#!/usr/bin/env python3
"""
Script to generate and save required images for provided datasets (CubiCasa5k, Structured3D, Aria SynthEnv, SwissDwellings).

This script iterates through samples in the selected datasets and generates images for
furniture percentages: 0%, 25%, 50%, 75%, and 100% (configurable).
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.unified_dataset import UnifiedDataset


def create_unified_dataset(
    cubicasa_root: str = None,
    s3d_root: str = None,
    aria_root: str = None,
    swiss_root: str = None,
    zillow_root: str = None,
    scannet_root: str = None,
    cubicasa_ids_file: str = None,
    s3d_ids_file: str = None,
    aria_ids_file: str = None,
    swiss_ids_file: str = None,
    zillow_ids_file: str = None,
    scannet_ids_file: str = None,
) -> UnifiedDataset:
    """
    Create a unified dataset with any subset of Cubicasa5k, Structured3D, Aria SynthEnv, and SwissDwellings datasets.

    Only datasets with provided root paths will be included.
    """
    dataset_configs: List[Dict[str, Any]] = []

    # Add Cubicasa5k dataset if provided
    if cubicasa_root:
        cubicasa_config = {
            "type": "cubicasa5k",
            "args": {
                "root_dir": cubicasa_root,
                "generate": True,  # Must be True for generation
            }
        }
        if cubicasa_ids_file:
            cubicasa_config["args"]["sample_ids_file"] = cubicasa_ids_file
        dataset_configs.append(cubicasa_config)

    # Add Structured3D dataset if provided
    if s3d_root:
        s3d_config = {
            "type": "structured3d",
            "args": {
                "root_dir": s3d_root,
                "generate": True,  # Must be True for generation
            }
        }
        if s3d_ids_file:
            s3d_config["args"]["scene_ids_file"] = s3d_ids_file
        dataset_configs.append(s3d_config)

    # Add Aria SynthEnv dataset if provided
    if aria_root:
        aria_config = {
            "type": "aria_synthenv",
            "args": {
                "root_dir": aria_root,
                "generate": True,  # Must be True for generation
                "ortho_axis": "z",
            }
        }
        if aria_ids_file:
            aria_config["args"]["scene_ids_file"] = aria_ids_file
        dataset_configs.append(aria_config)

    # Add SwissDwellings dataset if provided
    if swiss_root:
        swiss_config = {
            "type": "swiss_dwellings",
            "args": {
                "root_dir": swiss_root,
                "generate": True,  # Must be True for generation
            }
        }
        if swiss_ids_file:
            swiss_config["args"]["sample_ids_file"] = swiss_ids_file
        dataset_configs.append(swiss_config)

    # Add Zillow dataset if provided
    if zillow_root:
        zillow_config = {
            "type": "zillow",
            "args": {
                "root_dir": zillow_root,
                "generate": True,
            }
        }
        if zillow_ids_file:
            zillow_config["args"]["sample_ids_file"] = zillow_ids_file
        dataset_configs.append(zillow_config)

    # Add ScanNet dataset if provided
    if scannet_root:
        scannet_config = {
            "type": "scannet",
            "args": {
                "root_dir": scannet_root,
                "generate": True,  # Must be True for generation
            }
        }
        if scannet_ids_file:
            scannet_config["args"]["scene_ids_file"] = scannet_ids_file
        dataset_configs.append(scannet_config)

    if len(dataset_configs) == 0:
        raise ValueError(
            "Provide at least one dataset root via --cubicasa-root, --s3d-root, --aria-root, --swiss-root, --zillow-root, or --scannet-root"
        )

    return UnifiedDataset(dataset_configs=dataset_configs)




def main():
    parser = argparse.ArgumentParser(description="Generate and save required images for provided datasets")
    parser.add_argument(
        "--cubicasa-root",
        required=False,
        help="Path to Cubicasa5k dataset root directory"
    )
    parser.add_argument(
        "--s3d-root", 
        required=False,
        help="Path to Structured3D dataset root directory"
    )
    parser.add_argument(
        "--aria-root", 
        required=False,
        help="Path to Aria SynthEnv dataset root directory"
    )
    parser.add_argument(
        "--swiss-root", 
        required=False,
        help="Path to SwissDwellings split directory (e.g., .../modified-swiss-dwellings-v2/train)"
    )
    parser.add_argument(
        "--zillow-root",
        required=False,
        help="Path to Zillow/ZInD rendered dataset root directory"
    )
    parser.add_argument(
        "--scannet-root",
        required=False,
        help="Path to ScanNet dataset root directory"
    )
    parser.add_argument(
        "--cubicasa-ids-file",
        help="Optional path to Cubicasa5k sample IDs file"
    )
    parser.add_argument(
        "--s3d-ids-file",
        help="Optional path to Structured3D scene IDs file"
    )
    parser.add_argument(
        "--aria-ids-file",
        help="Optional path to Aria SynthEnv scene IDs file"
    )
    parser.add_argument(
        "--swiss-ids-file",
        help="Optional path to SwissDwellings sample IDs file"
    )
    parser.add_argument(
        "--zillow-ids-file",
        help="Optional path to Zillow sample IDs file"
    )
    parser.add_argument(
        "--scannet-ids-file",
        help="Optional path to ScanNet scene IDs file"
    )
    parser.add_argument(
        "--furniture-percentages",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Furniture percentages to generate (default: 0.0 0.25 0.5 0.75 1.0)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without actually generating files"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("IMAGE GENERATION SCRIPT")
    print("="*60)
    print(f"Cubicasa5k root: {args.cubicasa_root}")
    print(f"Structured3D root: {args.s3d_root}")
    print(f"Aria SynthEnv root: {args.aria_root}")
    print(f"SwissDwellings root: {args.swiss_root}")
    print(f"Zillow root: {args.zillow_root}")
    print(f"ScanNet root: {args.scannet_root}")
    print(f"Furniture percentages: {args.furniture_percentages}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)
    
    # Create unified dataset
    print("\nCreating unified dataset...")
    try:
        unified_dataset = create_unified_dataset(
            cubicasa_root=args.cubicasa_root,
            s3d_root=args.s3d_root,
            aria_root=args.aria_root,
            swiss_root=args.swiss_root,
            zillow_root=args.zillow_root,
            scannet_root=args.scannet_root,
            cubicasa_ids_file=args.cubicasa_ids_file,
            s3d_ids_file=args.s3d_ids_file,
            aria_ids_file=args.aria_ids_file,
            swiss_ids_file=args.swiss_ids_file,
            zillow_ids_file=args.zillow_ids_file,
            scannet_ids_file=args.scannet_ids_file,
        )
        print(f"✓ Unified dataset created with {len(unified_dataset)} total samples")
        
        # Print dataset breakdown
        for i, (name, dataset) in enumerate(unified_dataset._datasets):
            print(f"  - {name}: {len(dataset)} samples")
            
    except Exception as e:
        print(f"✗ Failed to create unified dataset: {e}")
        return 1
    
    # Determine how many samples to process
    total_samples = len(unified_dataset)
    samples_to_process = min(args.max_samples, total_samples) if args.max_samples else total_samples
    
    print(f"\nProcessing {samples_to_process} samples...")
    
    # Statistics tracking
    stats = {
        "total_samples": samples_to_process,
        "processed_samples": 0,
        "processed_images": 0,
        "errors": 0,
        "start_time": time.time()
    }
    
    # Process each sample
    for sample_idx in range(samples_to_process):
        sample_failed = False
        try:
            # Get sample info
            item = unified_dataset[sample_idx]
            source_dataset = item.get("source_dataset", "unknown")
            sample_id = item.get("sample_id", f"sample_{sample_idx}")
            
            print(f"\n[{sample_idx + 1}/{samples_to_process}] Processing {source_dataset} sample: {sample_id}")
            if source_dataset in ("aria_synthenv", "swiss_dwellings", "zillow", "scannet"):
                if args.dry_run:
                    print("  Would generate cached modalities")
                    if source_dataset == "aria_synthenv":
                        stats["processed_images"] += 2
                    elif source_dataset == "swiss_dwellings":
                        stats["processed_images"] += 3
                    elif source_dataset == "scannet":
                        stats["processed_images"] += 5  # floorplan, density_map, density_map_point4, density_map_mesh, density_map_mesh_noisy
                    else:  # zillow
                        stats["processed_images"] += 1
                else:
                    try:
                        result = unified_dataset.generate_and_save(sample_idx, furniture_pct=None)
                        print("  ✓ Generated images")
                        if source_dataset == "aria_synthenv":
                            ann_like = result.get('ann_path')
                            pts_like = result.get('points_path')
                            if ann_like:
                                print(f"    - Annotations: {ann_like}")
                            if pts_like:
                                print(f"    - Points: {pts_like}")
                            stats["processed_images"] += 2
                        elif source_dataset == "swiss_dwellings":
                            ann_like = result.get('binary_path')
                            col_like = result.get('colored_path')
                            pts_like = result.get('points_path')
                            if ann_like:
                                print(f"    - Binary: {ann_like}")
                            if col_like:
                                print(f"    - Colored: {col_like}")
                            if pts_like:
                                print(f"    - Points: {pts_like}")
                            stats["processed_images"] += 3
                        elif source_dataset == "scannet":
                            floorplan_path = result.get('floorplan_path')
                            density_path = result.get('density_map_path')
                            density_large_path = result.get('density_map_point4_path')
                            density_mesh_path = result.get('density_map_mesh_path')
                            density_mesh_noisy_path = result.get('density_map_mesh_noisy_path')
                            if floorplan_path:
                                print(f"    - Floorplan: {floorplan_path}")
                            if density_path:
                                print(f"    - Density: {density_path}")
                            if density_large_path:
                                print(f"    - Density Large: {density_large_path}")
                            if density_mesh_path:
                                print(f"    - Density Mesh: {density_mesh_path}")
                            if density_mesh_noisy_path:
                                print(f"    - Density Mesh Noisy: {density_mesh_noisy_path}")
                            stats["processed_images"] += 5
                        else:  # zillow
                            pts_like = result.get('points_path')
                            if pts_like:
                                print(f"    - Points: {pts_like}")
                            stats["processed_images"] += 1
                    except Exception as e:
                        print(f"  ✗ Failed to generate images: {e}")
                        stats["errors"] += 1
                        sample_failed = True

            else:
                # Process each furniture percentage
                for furniture_pct in args.furniture_percentages:
                    if args.dry_run:
                        print(f"  Would generate images for furniture_pct={furniture_pct}")
                        stats["processed_images"] += 2  # 2 images per percentage
                    else:
                        try:
                            result = unified_dataset.generate_and_save(sample_idx, furniture_pct)
                            print(f"  ✓ Generated images for furniture_pct={furniture_pct}")
                            print(f"    - Annotations: {result.get('ann_path', 'N/A')}")
                            print(f"    - Points: {result.get('points_path', 'N/A')}")
                            stats["processed_images"] += 2  # 2 images per percentage
                        except Exception as e:
                            print(f"  ✗ Failed to generate images for furniture_pct={furniture_pct}: {e}")
                            stats["errors"] += 1
                            sample_failed = True
                
        except Exception as e:
            print(f"✗ Failed to process sample {sample_idx}: {e}")
            stats["errors"] += 1
            continue

        if not sample_failed:
            stats["processed_samples"] += 1
    
    # Print final statistics
    elapsed_time = time.time() - stats["start_time"]
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total samples processed: {stats['processed_samples']}/{stats['total_samples']}")
    print(f"Total images generated: {stats['processed_images']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time/max(1, stats['processed_samples']):.2f} seconds")
    print(f"Average time per image: {elapsed_time/max(1, stats['processed_images']):.2f} seconds")
    
    if stats["errors"] > 0:
        print(f"\n⚠️  {stats['errors']} errors occurred during generation. Check the output above for details.")
        return 1
    else:
        print("\n✓ All images generated successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
