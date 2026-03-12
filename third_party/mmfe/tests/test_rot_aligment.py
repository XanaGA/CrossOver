#!/usr/bin/env python3
"""
Test alignment between two modality embeddings using an estimated affine transformation.

This script:
- Loads a model checkpoint and a tiny validation dataset
- Computes embeddings for the first batch and selects only the first image
- Applies an affine transformation (rotation, translation, scale) to modality_1 embedding
- Estimates the affine transformation between modality_0 and transformed modality_1 using a PCA-based 2D projection
- Evaluates alignment performance by comparing transformed image corners with ground truth
- Saves comprehensive PCA RGB visualizations including corner alignment analysis

Run example:
  python tests/test_rot_aligment.py \
    --checkpoint /abs/path/model.ckpt \
    --cubicasa-path /abs/cubicasa5k --cubicasa-file /abs/cubicasa5k/val.txt \
    --structured3d-path /abs/Structured3D --structured3d-file /abs/Structured3D/val.json \
    --image-size 256 256 \
    --angle 30 --tx 10 --ty 5 --scale 1.1 \
    --threshold 5.0 \
    --output-dir outputs/visualizations/aligment
"""

import os
import sys
import argparse

import cv2
from hydra.utils import to_absolute_path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from mmfe_utils.tensor_utils import torch_erode

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import PairToPIL, PairResize, PairGrayscale, PairToTensor, PairNormalize
from training.lightning_module import ContrastiveLearningModule
from mmfe_utils.viz_utils import pca_rgb
from mmfe_utils.aligment import (find_nn, inverse_affine_matrix, 
                            rot_to_affine, apply_affine_2d_map, 
                            apply_affine_2d_points, estimate_affine_matrix,
                            evaluate_corner_alignment)

def create_val_dataset(args) -> UnifiedDataset:
    # Transforms similar to train_contrastive.py (validation path)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    dual_transform_val = [
        PairToPIL(),
        PairResize(tuple(args.image_size)),
        PairGrayscale(num_output_channels=3),
        PairToTensor(),
        PairNormalize(mean=mean, std=std),
    ]

    val_configs = [
        {
            "type": "cubicasa5k",
            "args": {
                "root_dir": to_absolute_path(args.cubicasa_path),
                "sample_ids_file": to_absolute_path(args.cubicasa_file),
                "image_size": tuple(args.image_size),
                "dual_transform": dual_transform_val,
            },
        },
        {
            "type": "structured3d",
            "args": {
                "root_dir": to_absolute_path(args.structured3d_path),
                "scene_ids_file": to_absolute_path(args.structured3d_file),
                "image_size": tuple(args.image_size),
                "dual_transform": dual_transform_val,
            },
        },
    ]

    dataset = UnifiedDataset(dataset_configs=val_configs)
    return dataset

def angle_from_rotation(R: np.ndarray) -> float:
    """Return rotation angle in degrees from a 2x2 rotation matrix."""
    theta = np.arctan2(R[1, 0], R[0, 0])
    return float(np.degrees(theta))

def main():
    parser = argparse.ArgumentParser(description="Test alignment between modality embeddings with rotation estimation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--rotate-before', action='store_true', help='Rotate modality_1 embedding before alignment')
    parser.add_argument('--cubicasa-path', type=str, default="data/cubicasa5k/", help='Path to CubiCasa5k dataset root')
    parser.add_argument('--cubicasa-file', type=str, default="data/cubicasa5k/val.txt", help='Path to CubiCasa5k split file')
    parser.add_argument('--structured3d-path', type=str, default="data/structure3D/Structured3D_bbox/Structured3D/", help='Path to Structured3D dataset root')
    parser.add_argument('--structured3d-file', type=str, default="data/structure3D/val.json", help='Path to Structured3D split file')
    parser.add_argument('--angle', type=float, default=45.0, help='Rotation angle (degrees) to apply to modality_1 embedding')
    parser.add_argument('--tx', type=float, default=0.0, help='Translation in x direction (pixels)')
    parser.add_argument('--ty', type=float, default=0.0, help='Translation in y direction (pixels)')
    parser.add_argument('--scale', type=float, default=1.0, help='Uniform scale factor')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations/aligment', help='Output directory')
    parser.add_argument('--use-gt-correspondances', action='store_true', help='Use ground truth correspondances')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256], help='Image size')
    parser.add_argument('--method', type=str, default='ransac', choices=['ransac', 'lmeds', 'best'], help='Method to use for affine matrix estimation')
    parser.add_argument('--threshold', type=float, default=5.0, help='Distance threshold for corner alignment accuracy (pixels)')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    model = ContrastiveLearningModule.load_from_checkpoint(checkpoint_path=to_absolute_path(args.checkpoint), 
                                                            map_location=device, load_dino_weights=False)
    model.to(device)
    model.eval()

    # Use only the first image
    val_dataset = create_val_dataset(args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    theta = np.deg2rad(args.angle)

    # # Invert the scale to be more intuitive
    # args.scale = 1.0 / args.scale
    
    # Create rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    # Create scale matrix (uniform scaling)
    S = np.array([[args.scale, 0],
                  [0, args.scale]])
    
    # Create translation vector
    t = np.array([args.tx, args.ty])
    
    # Combine rotation and scale: A = R * S
    A = R @ S
    
    # Create 2x3 affine transformation matrix
    true_affine = np.array([[A[0, 0], A[0, 1], t[0]],
                           [A[1, 0], A[1, 1], t[1]]])
    
    for batch in val_loader:
        image0, image1 = batch["modality_0"], batch["modality_1"]
        if args.rotate_before:
            norm_filler = (1-torch.tensor([0.485, 0.456, 0.406]))/torch.tensor([0.229, 0.224, 0.225])
            image1 = apply_affine_2d_map(image1, true_affine).unsqueeze(0)
            mask = apply_affine_2d_map(torch.ones(1, image1.shape[-1], image1.shape[-2]), true_affine)
            mask = torch_erode(mask, kernel_size=3)
            black_threshold = 0.5
            image1 = torch.where(~(mask.repeat(1, 3, 1, 1) < black_threshold), image1, norm_filler.view(1, 3, 1, 1))

        e0, e1 = model.get_embeddings(image0, image1)  # (C,H,W)
        e0 = e0[0]
        e1 = e1[0]
        break

    if args.rotate_before:
        e1_rot = e1
        mask = None
    else:
        e1_rot = apply_affine_2d_map(e1, true_affine)
        mask = apply_affine_2d_map(torch.ones(1, e1.shape[1], e1.shape[2]), true_affine)
        mask = torch_erode(mask, kernel_size=3)

    idx0, _, idx1, _ = find_nn(e0, e1_rot, mask=mask, top_k=100)
    
    center = np.array([e1.shape[1]//2, e1.shape[2]//2])
    
    if args.use_gt_correspondances:
        # idx1 = torch.tensor(rotate_points(idx0.cpu().numpy(), center, true_affine))
        idx1 = apply_affine_2d_points(idx0, true_affine, center=center)

    # R_est = estimate_rotation(idx0.cpu(), idx1.cpu(), center).T
    aff_est = estimate_affine_matrix(idx0.cpu(), idx1.cpu(), center=center, method=args.method)
    theta_est = angle_from_rotation(aff_est)
    
    # Evaluate corner alignment performance
    eval_results = evaluate_corner_alignment(true_affine, aff_est, 
                                           img_shape=(args.image_size[1], args.image_size[0]), 
                                           threshold=args.threshold)

    # Estimate rotation via PCA-2 vector fields
    v0, v1r = pca_rgb([e0, e1_rot])

    print(f"Applied affine transform - Rotation: {args.angle:.2f}°, Translation: ({args.tx:.1f}, {args.ty:.1f}), Scale: {args.scale:.2f}")
    print(f"Estimated rotation: {theta_est:.2f}°")
    print(f"\n=== Corner Alignment Evaluation ===")
    print(f"Accuracy (threshold={args.threshold}px): {eval_results['accuracy']:.1f}%")
    print(f"Mean corner distance: {eval_results['mean_distance']:.2f} pixels")
    print(f"Max corner distance: {eval_results['max_distance']:.2f} pixels")
    print(f"RMS error: {eval_results['rms_error']:.2f} pixels")
    print(f"Individual corner distances: {eval_results['distances']}")


    #############################################################################
    # VISUALIZATIONS
    #############################################################################
    # Save PCA RGB visualizations similar to aligment.py
    out_dir = os.path.abspath(args.output_dir)

    # Create comprehensive visualizations    
    # Get original images (denormalize them for display)
    def denormalize_tensor(tensor, mean, std):
        """Denormalize a tensor for visualization"""
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    img0_display = denormalize_tensor(image0[0], mean, std).permute(1, 2, 0).cpu().numpy()
    img1_display = denormalize_tensor(image1[0], mean, std)

    if args.rotate_before:
        img1_rotated_display = img1_display.permute(1, 2, 0).cpu().numpy()
    else:
        # Apply affine transformation to img1_display for better visualization
        img1_rotated_display = apply_affine_2d_map(img1_display, true_affine).permute(1, 2, 0).cpu().numpy()
    
    # Get embedding dimensions
    emb_h, emb_w = e0.shape[1], e0.shape[2]
    img_h, img_w = img0_display.shape[0], img0_display.shape[1]
    
    # Calculate scaling factors
    scale_x = img_w / emb_w
    scale_y = img_h / emb_h
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    
    # Top row: Original images with selected points
    axs[0, 0].imshow(img0_display)
    axs[0, 0].set_title('Modality 0 - Original Image')
    axs[0, 0].axis('off')

    # Max number of points to display
    max_points = min(idx0.shape[0], idx1.shape[0], 10)
    
    # Scale embedding coordinates to image coordinates
    for i in range(max_points):
        emb_x, emb_y = idx0[i].cpu().numpy()
        img_x = emb_x * scale_x
        img_y = emb_y * scale_y
        axs[0, 0].scatter(img_x, img_y, color='red', s=50, marker='o', edgecolors='white', linewidth=2)
        # Add point numbers
        axs[0, 0].text(img_x + 5, img_y - 5, str(i), color='white', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    axs[0, 1].imshow(img1_rotated_display)
    axs[0, 1].set_title(f'Modality 1 - Affine Transform (R:{args.angle:.1f}°, T:({args.tx:.1f},{args.ty:.1f}), S:{args.scale:.2f})')
    axs[0, 1].axis('off')
    
    # Scale embedding coordinates to image coordinates for modality 1
    for i in range(max_points):
        emb_x, emb_y = idx1[i].cpu().numpy()
        img_x = emb_x * scale_x
        img_y = emb_y * scale_y
        axs[0, 1].scatter(img_x, img_y, color='blue', s=50, marker='o', edgecolors='white', linewidth=2)
        # Add point numbers
        axs[0, 1].text(img_x + 5, img_y - 5, str(i), color='white', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
    
    # Show correspondence lines with corrected alignment
    corrected_img1 = apply_affine_2d_map(img1_rotated_display, inverse_affine_matrix(aff_est)).permute(1, 2, 0).cpu().numpy()
    
    axs[0, 2].imshow(corrected_img1)
    axs[0, 2].set_title(f'Modality 1 - Corrected by Estimated Affine')
    axs[0, 2].axis('off')
    
    # Draw correspondence lines with corrected points Maximum 10
    for i in range(max_points):
        # Point in modality 0
        emb_x0, emb_y0 = idx0[i].cpu().numpy()
        img_x0 = emb_x0 * scale_x
        img_y0 = emb_y0 * scale_y
        
        # Point in modality 1 - apply inverse of estimated rotation
        emb_x1, emb_y1 = idx1[i].cpu().numpy()
        img_x1 = emb_x1 * scale_x
        img_y1 = emb_y1 * scale_y
        
        # Apply inverse rotation to the point coordinates
        point_center = np.array([img_w // 2, img_h // 2])
        corrected_point_1 = apply_affine_2d_points(np.array([img_x1, img_y1]), inverse_affine_matrix(aff_est), center=point_center).squeeze(0)
        img_x1_corrected, img_y1_corrected = corrected_point_1
        
        # Draw line connecting corresponding points
        axs[0, 2].plot([img_x0, img_x1_corrected], [img_y0, img_y1_corrected], 'yellow', linewidth=2, alpha=0.7)
        axs[0, 2].scatter(img_x0, img_y0, color='red', s=50, marker='o', edgecolors='white', linewidth=2)
        axs[0, 2].scatter(img_x1_corrected, img_y1_corrected, color='green', s=50, marker='s', edgecolors='white', linewidth=2)
        
        # Add point numbers
        axs[0, 2].text(img_x0 + 5, img_y0 - 5, str(i), color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
        axs[0, 2].text(img_x1_corrected + 5, img_y1_corrected - 5, str(i), color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))
    
    # Second row: PCA visualizations
    mag0 = np.linalg.norm(v0, axis=-1)
    mag1r = np.linalg.norm(v1r, axis=-1)
    
    axs[1, 0].imshow(mag0, cmap='viridis')
    axs[1, 0].set_title('Modality 0 - PCA2 Magnitude')
    axs[1, 0].axis('off')
    
    # Draw the idx0 as red dots on the PCA visualization
    for i in range(max_points):
        x, y = idx0[i].cpu()
        axs[1, 0].scatter(x, y, color='red', s=50, marker='o', edgecolors='white', linewidth=2)
        axs[1, 0].text(x + 2, y - 2, str(i), color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))
    
    axs[1, 1].imshow(mag1r, cmap='viridis')
    axs[1, 1].set_title('Modality 1 Affine Transformed - PCA2 Magnitude')
    axs[1, 1].axis('off')
    
    # Draw the idx1 as blue dots on the PCA visualization
    for i in range(max_points):
        x, y = idx1[i].cpu()
        axs[1, 1].scatter(x, y, color='blue', s=50, marker='o', edgecolors='white', linewidth=2)
        axs[1, 1].text(x + 2, y - 2, str(i), color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))
    
    # Show affine transformation estimation info
    axs[1, 2].text(0.1, 0.9, f'Applied Transform:', fontsize=12, fontweight='bold')
    axs[1, 2].text(0.1, 0.8, f'  Rotation: {args.angle:.2f}°', fontsize=11)
    axs[1, 2].text(0.1, 0.7, f'  Translation: ({args.tx:.1f}, {args.ty:.1f})', fontsize=11)
    axs[1, 2].text(0.1, 0.6, f'  Scale: {args.scale:.2f}', fontsize=11)
    axs[1, 2].text(0.1, 0.5, f'Estimated Rotation: {theta_est:.2f}°', fontsize=12, fontweight='bold')
    axs[1, 2].text(0.1, 0.4, f'Rotation Error: {abs(args.angle - theta_est):.2f}°', fontsize=11)
    axs[1, 2].text(0.1, 0.3, f'Corner Accuracy: {eval_results["accuracy"]:.1f}%', fontsize=11, color='green' if eval_results["accuracy"] > 75 else 'red')
    axs[1, 2].text(0.1, 0.2, f'Mean Corner Dist: {eval_results["mean_distance"]:.2f}px', fontsize=10)
    axs[1, 2].text(0.1, 0.1, f'Image Size: {img_w}x{img_h}', fontsize=10)
    axs[1, 2].set_xlim(0, 1)
    axs[1, 2].set_ylim(0, 1)
    axs[1, 2].axis('off')
    axs[1, 2].set_title('Affine Transformation Estimation Results')
    
    # Third row: Corner transformation visualization
    # Show original image with corners
    axs[2, 0].imshow(img0_display)
    axs[2, 0].set_title('Original Image Corners')
    axs[2, 0].axis('off')
    
    # Draw original corners
    corners = eval_results['corners_original']
    corner_labels = ['TL', 'TR', 'BL', 'BR']
    for i, (corner, label) in enumerate(zip(corners, corner_labels)):
        axs[2, 0].scatter(corner[0], corner[1], color='red', s=100, marker='o', edgecolors='white', linewidth=2)
        axs[2, 0].text(corner[0] + 10, corner[1] - 10, label, color='white', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    # Show transformed image with ground truth and predicted corners
    axs[2, 1].imshow(img1_rotated_display)
    axs[2, 1].set_title('Transformed Image - GT vs Pred Corners')
    axs[2, 1].axis('off')
    
    # Draw ground truth corners
    corners_gt = eval_results['corners_gt']
    for i, (corner, label) in enumerate(zip(corners_gt, corner_labels)):
        axs[2, 1].scatter(corner[0], corner[1], color='green', s=100, marker='o', edgecolors='white', linewidth=2)
        axs[2, 1].text(corner[0] + 10, corner[1] - 10, f'{label}_GT', color='white', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.7))
    
    # Draw predicted corners
    corners_pred = eval_results['corners_pred']
    for i, (corner, label) in enumerate(zip(corners_pred, corner_labels)):
        axs[2, 1].scatter(corner[0], corner[1], color='blue', s=100, marker='s', edgecolors='white', linewidth=2)
        axs[2, 1].text(corner[0] + 10, corner[1] + 15, f'{label}_Pred', color='white', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.7))
    
    # Draw lines connecting GT and predicted corners
    for i, (gt_corner, pred_corner) in enumerate(zip(corners_gt, corners_pred)):
        axs[2, 1].plot([gt_corner[0], pred_corner[0]], [gt_corner[1], pred_corner[1]], 
                      'yellow', linewidth=2, alpha=0.7)
    
    # Corner error visualization
    axs[2, 2].bar(range(4), eval_results['distances'], color=['red', 'green', 'blue', 'orange'])
    axs[2, 2].set_title('Corner Distance Errors')
    axs[2, 2].set_xlabel('Corner')
    axs[2, 2].set_ylabel('Distance (pixels)')
    axs[2, 2].set_xticks(range(4))
    axs[2, 2].set_xticklabels(corner_labels)
    axs[2, 2].axhline(y=args.threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({args.threshold}px)')
    axs[2, 2].legend()
    
    # Add distance values on top of bars
    for i, distance in enumerate(eval_results['distances']):
        axs[2, 2].text(i, distance + 0.1, f'{distance:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'alignment_visualization.png')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved comprehensive visualization: {fig_path}")
        


if __name__ == "__main__":
    main()


