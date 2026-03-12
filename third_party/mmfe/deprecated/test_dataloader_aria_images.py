#!/usr/bin/env python3
"""
Test script for AriaSynthEenvDataset.

This script demonstrates how to:
1. Create an AriaSynthEenvDataset instance
2. Iterate over all samples
3. Display images with matplotlib
4. Show basic statistics about the dataset

Usage:
    python tests/test_dataloader_aria_images.py --path /path/to/aria_synthenv
    python tests/test_dataloader_aria_images.py --path /path/to/aria_synthenv --image-size 512 512
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

from dataloading.aria_se_data import AriaSynthEenvDataset
from dataloading.dual_transforms import PairToPIL, PairToTensor, PairResize
from aria_mmfe.aria_images.aria_cv_tools import (
    filter_points_by_fustrum,
    get_pinhole_matrix_from_ase,
    points_to_image_coords_from_params,
)
from scipy.spatial.transform import Rotation as R



def parse_args():
    parser = argparse.ArgumentParser(
        description="Test AriaSynthEenvDataset by iterating over all samples"
    )
    parser.add_argument(
        "--path", 
        required=True,
        help="Path to Aria SyntheticEnv dataset root directory",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        help="Optional path to a text file listing scene IDs to include",
    )
    parser.add_argument(
        "--image-size", 
        type=int, 
        nargs=2, 
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to HEIGHT x WIDTH (e.g., --image-size 512 512)",
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        help="Maximum number of samples to display (default: all samples)",
    )
    parser.add_argument(
        "--save-dir", 
        type=str,
        help="Directory to save images (if not provided, images are only displayed)",
    )
    parser.add_argument(
        "--no-display", 
        action="store_true",
        help="Don't display images interactively (useful for saving only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for dataloader (if not provided, processes samples individually)",
    )
    
    return parser.parse_args()


def create_dataset(args) -> AriaSynthEenvDataset:
    """Create and configure the dataset."""
    
    # Basic transforms for visualization
    transform = [
        PairToPIL(),
        PairResize((512, 512)) if args.image_size is None else PairResize(args.image_size),
        PairToTensor(),
    ]
    
    print("Creating AriaSynthEenvDataset with:")
    print(f"  Root directory: {args.path}")
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")
    print("  n_fpv_images: 1")
    
    dataset = AriaSynthEenvDataset(
        root_dir=args.path,
        scene_ids_file=args.ids_file,
        image_size=args.image_size,
        dual_transform=transform,
        generate=False,
        n_fpv_images=1,
    )
    
    return dataset


def display_sample(dataset: AriaSynthEenvDataset, index: int, args, save_dir: Optional[str] = None):
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
    
    # ---------------- Base modalities figure ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Modality 0
    ax1.imshow(m0_np)
    ax1.set_title(
        f"Sample {sample_id} - {m0_type}\n"
        f"Size: {m0_np.shape[0]}x{m0_np.shape[1]}"
    )
    ax1.axis("off")
    
    # Modality 1
    ax2.imshow(m1_np)
    ax2.set_title(
        f"{m1_type}\n"
        f"Size: {m0_np.shape[0]}x{m0_np.shape[1]}"
    )
    ax2.axis("off")
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sample_{sample_id}.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    
    # Display if not in no-display mode
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    # ---------------- FPV visualization (if available) ----------------
    fpv_dict = sample.get("fpv_dict", None)
    if fpv_dict is not None:
        fpv_images = fpv_dict.get("images", [])
        poses_floorplan = fpv_dict.get("poses_floorplan", None)
        scene_points_2d = fpv_dict.get("scene_points_2d", None)
        scene_points_3d = fpv_dict.get("scene_points_3d", None)
        pose_3d = fpv_dict.get("pose_3D", None)
        pose_2d = fpv_dict.get("pose_2D", None)
        params = fpv_dict.get("params", None)

        # First FPV image (if any)
        fpv_np = None
        if len(fpv_images) > 0:
            fpv_img = fpv_images[0]
            try:
                # Reuse tensor_to_numpy for tensors; fallback to np.array otherwise
                fpv_np = tensor_to_numpy(fpv_img)
            except Exception:
                fpv_np = np.array(fpv_img)

        if fpv_np is not None and params is not None:
            h_params = float(params["h"])
            w_params = float(params["w"])

            # Use modality_0 as floorplan base for overlay
            floor_map_all = m0_np.copy()
            floor_map_filt = m0_np.copy()
            h_img, w_img = floor_map_all.shape[:2]

            # Scale floorplan-space coordinates to displayed image resolution
            scale_x = w_img / w_params
            scale_y = h_img / h_params

            def _scale_coords(arr):
                arr = np.asarray(arr, dtype=np.float32)
                return np.stack(
                    [arr[:, 0] * scale_x, arr[:, 1] * scale_y],
                    axis=1,
                )

            # ---------------- Frustum-filtered points in scene (world) space ----------------
            scene_points_2d_filt = None
            if scene_points_3d is not None and pose_3d is not None:
                cam_xyz_all = np.asarray(pose_3d.get("xyz", None))
                cam_quat_all = np.asarray(pose_3d.get("quat", None))

                if cam_xyz_all.ndim == 2 and cam_xyz_all.shape[0] > 0:
                    cam_xyz0 = cam_xyz_all[0]
                    cam_quat0 = cam_quat_all[0]

                    # Build camera->world pose matrix for first FPV pose
                    R_wc = R.from_quat(cam_quat0).as_matrix()
                    pose_cam = np.eye(4, dtype=np.float32)
                    pose_cam[:3, :3] = R_wc.astype(np.float32)
                    pose_cam[:3, 3] = cam_xyz0.astype(np.float32)

                    # Camera intrinsics and frustum filtering in one step
                    K = get_pinhole_matrix_from_ase()
                    h_fpv, w_fpv = fpv_np.shape[:2]
                    pts_w_f = filter_points_by_fustrum(
                        np.asarray(scene_points_3d, dtype=np.float32),
                        pose_cam,
                        K,
                        (h_fpv, w_fpv),
                    )

                    if pts_w_f is not None and len(pts_w_f) > 0:
                        # Project onto floorplan using params
                        scene_points_2d_filt = points_to_image_coords_from_params(
                            pts_w_f, params
                        )

            # ---------------- 3-panel visualization: RGB, full map, filtered map ----------------
            fig_fpv, (ax_rgb, ax_map_all, ax_map_filt) = plt.subplots(
                1, 3, figsize=(18, 6)
            )

            # Panel 1: FPV RGB image
            ax_rgb.imshow(fpv_np)
            ax_rgb.set_title(f"FPV image (sample {sample_id})")
            ax_rgb.axis("off")

            # Common pose data
            pose_xy = None
            pose_theta = None
            if pose_2d is not None:
                pose_xy = pose_2d.get("xy", None)
                pose_theta = pose_2d.get("theta", None)
            if pose_xy is None and poses_floorplan is not None:
                pose_xy = poses_floorplan

            # Panel 2: Floorplan with ALL sampled points + pose
            ax_map_all.imshow(floor_map_all)
            ax_map_all.set_title("Floorplan + all sampled scene points")
            ax_map_all.axis("off")

            if scene_points_2d is not None and len(scene_points_2d) > 0:
                pts_scaled_all = _scale_coords(scene_points_2d)
                ax_map_all.scatter(
                    pts_scaled_all[:, 0],
                    pts_scaled_all[:, 1],
                    s=1,
                    c="black",
                    alpha=0.3,
                    label="scene points",
                )

            if pose_xy is not None:
                cams_scaled_all = _scale_coords(pose_xy)
                ax_map_all.scatter(
                    cams_scaled_all[:, 0],
                    cams_scaled_all[:, 1],
                    s=40,
                    c="red",
                    marker="x",
                    label="camera poses",
                )
                if pose_theta is not None and len(pose_theta) > 0:
                    theta0 = float(pose_theta[0])
                    cx, cy = cams_scaled_all[0]
                    arrow_len = 0.07 * min(h_img, w_img)
                    dx = arrow_len * np.cos(theta0)
                    dy = arrow_len * np.sin(theta0)
                    ax_map_all.arrow(
                        cx,
                        cy,
                        dx,
                        dy,
                        head_width=arrow_len * 0.25,
                        head_length=arrow_len * 0.25,
                        fc="red",
                        ec="red",
                        linewidth=2,
                    )

            if (
                (scene_points_2d is not None and len(scene_points_2d) > 0)
                or (pose_xy is not None)
            ):
                ax_map_all.legend(loc="upper right")

            # Panel 3: Floorplan with ONLY frustum-filtered points + pose
            ax_map_filt.imshow(floor_map_filt)
            ax_map_filt.set_title("Floorplan + frustum-filtered scene points")
            ax_map_filt.axis("off")

            if scene_points_2d_filt is not None and len(scene_points_2d_filt) > 0:
                pts_scaled_f = _scale_coords(scene_points_2d_filt)
                ax_map_filt.scatter(
                    pts_scaled_f[:, 0],
                    pts_scaled_f[:, 1],
                    s=1,
                    c="cyan",
                    alpha=0.7,
                    label="filtered points",
                )

            if pose_xy is not None:
                cams_scaled_f = _scale_coords(pose_xy)
                ax_map_filt.scatter(
                    cams_scaled_f[:, 0],
                    cams_scaled_f[:, 1],
                    s=40,
                    c="red",
                    marker="x",
                    label="camera poses",
                )
                if pose_theta is not None and len(pose_theta) > 0:
                    theta0 = float(pose_theta[0])
                    cx, cy = cams_scaled_f[0]
                    arrow_len = 0.07 * min(h_img, w_img)
                    dx = arrow_len * np.cos(theta0)
                    dy = arrow_len * np.sin(theta0)
                    ax_map_filt.arrow(
                        cx,
                        cy,
                        dx,
                        dy,
                        head_width=arrow_len * 0.25,
                        head_length=arrow_len * 0.25,
                        fc="red",
                        ec="red",
                        linewidth=2,
                    )

            if (
                scene_points_2d_filt is not None
                and len(scene_points_2d_filt) > 0
                or (pose_xy is not None)
            ):
                ax_map_filt.legend(loc="upper right")

            # Save FPV + map visualization if requested
            if save_dir:
                fpv_save_path = os.path.join(save_dir, f"sample_{sample_id}_fpv_triplet.png")
                fig_fpv.savefig(fpv_save_path, bbox_inches="tight", dpi=150)
                print(f"Saved FPV triplet visualization: {fpv_save_path}")

            # Display or close
            if not args.no_display:
                plt.show()
            else:
                plt.close(fig_fpv)
    
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
    print(f"  Image size: {args.image_size if args.image_size else 'auto'}")
    print("  n_fpv_images: 1")


if __name__ == "__main__":
    main()
