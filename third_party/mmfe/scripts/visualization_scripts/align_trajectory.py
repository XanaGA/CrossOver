#!/usr/bin/env python3
"""
Visualize trajectory alignment across FPV images, floorplan, and window modality.

This script:
- Loads an Aria dataset example
- For each FPV image in sequence, displays:
  1. FPV image (undistorted, devignetted)
  2. Floorplan (points modality) with trajectory overlaid up to current frame
  3. Window modality with trajectory overlaid (aligned using estimated transformation)
- Interactive: press any key to advance to next frame

Usage:
    python scripts/visualization_scripts/align_trajectory.py \
        model.checkpoint=/path/to/model.ckpt \
        data.aria_synthenv.path=/path/to/data \
        data.aria_synthenv.train=/path/to/train.txt \
        data.image_size='[256,256]' \
        data.affine_transform.common_degrees=0 \
        data.affine_transform.noise_degrees=180
"""

import os
import sys
import cv2
import numpy as np
import torch
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R

from inference.tta import run_tta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import hydra

from dataloading.unified_dataset import UnifiedDataset
from dataloading.dual_transforms import (
    PairToPIL, PairResize, PairToTensor, PairNormalize, PairRandomAffine
)
from training.lightning_module import ContrastiveLearningModule
from mmfe_utils.aligment import (
    compose_affine_matrices, find_nn, estimate_affine_matrix, apply_affine_2d_points, inverse_affine_matrix, params_to_affine_matrix
)
from aria_mmfe.ase_data.ase_utils import read_trajectory, read_points, read_3d_boxes
from aria_mmfe.code_snippets.plotters import (
    change_params_resolution,
    render_pointcloud_and_boxes_orthographic_cv,
    overlay_trajectory,
    overlay_single_pose,
    world_to_pixel
)
from aria_mmfe.aria_images.aria_cv_tools import (
    get_device_camera_transform,
    pose_from_xyzq,
    xyzq_from_pose,
    points_to_image_coords_from_params
)


def estimate_alignment(
    model: ContrastiveLearningModule,
    img0: torch.Tensor,
    img1: torch.Tensor,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
    n_tta: int = 0
) -> np.ndarray:
    """
    Estimate affine transformation matrix between two images using their embeddings.
    
    Args:
        model: The contrastive learning model
        img0: First image tensor (C, H, W)
        img1: Second image tensor (C, H, W)
        device: Device to use
        mask: Optional mask for img1
        n_tta: Number of TTA augmentations
    Returns:
        Affine transformation matrix (2, 3) as numpy array
    """
    # Get embeddings
    with torch.no_grad():
        if n_tta > 0:
            e0 = model.get_embeddings(
                img0.unsqueeze(0).to(device),
            )
            if mask is not None:
                mask = mask.unsqueeze(0).to(device)
            else:
                mask = torch.ones(1, 1, img1.shape[1], img1.shape[2]).to(device) # BxCxHxW

            e1, selected_params, best_aug_idx, votes_per_aug = run_tta(e0, img1.unsqueeze(0).to(device), 
                                                                        mask, model, 
                                                                        n_augs=n_tta,
                                                                        # filler=norm_filler[0],
                                                                        model_name="mmfe",
                                                                        )

            

        else:
            e0, e1 = model.get_embeddings(
            img0.unsqueeze(0).to(device),
            img1.unsqueeze(0).to(device)
        )
    
    # Remove sample dimension if present
    if e0.dim() == 4:
        e0 = e0.squeeze(0)  # (C, H, W)
    if e1.dim() == 4:
        e1 = e1.squeeze(0)  # (C, H, W)
    
    # Find nearest neighbors
    idx0, _, idx1, _ = find_nn(e0, e1, mask=None, top_k=100)
    
    # Estimate affine transformation
    H, W = e0.shape[1], e0.shape[2]
    center = np.array([H // 2, W // 2])
    
    aff_est = estimate_affine_matrix(
        idx0.cpu().numpy(),
        idx1.cpu().numpy(),
        center=center,
        method="ransac",
        # centered = False
    )

    if n_tta > 0:
        params_i = {
            "angle": selected_params["angle"][0].cpu().numpy(),
            "scale": selected_params["scale"][0].cpu().numpy(),
            "translate_x": selected_params["translate"][0][1].cpu().numpy(),
            "translate_y": selected_params["translate"][0][0].cpu().numpy(),
        }
        aff_correcting_aug = params_to_affine_matrix(params_i)
        aff_est = compose_affine_matrices(inverse_affine_matrix(aff_correcting_aug), aff_est).cpu().numpy()

    aff_est = inverse_affine_matrix(aff_est)

    return aff_est


def build_aria_dataset(cfg: DictConfig):
    """
    Build a UnifiedDataset containing only aria_synthenv with transforms.
    
    Similar to _build_aria_dataset in train_fpv_img.py but with support for
    common and noise affine transformations like in create_datasets.
    
    Returns:
        dataset: UnifiedDataset
        noise_transform: Optional list of noise transforms
    """
    root_dir = to_absolute_path(cfg.data.aria_synthenv.path)
    ids_file = getattr(cfg.data.aria_synthenv, "train", None)
    ids_file = to_absolute_path(ids_file) if ids_file is not None else None
    
    # Normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Common transforms (applied to both modalities)
    common_transform = [
        PairToPIL(),
        PairResize(tuple(cfg.data.image_size)),
        PairToTensor(),
    ]
    
    # Add common affine transform if specified
    if cfg.data.affine_transform.get("common_degrees") is not None:
        common_transform.append(
            PairRandomAffine(
                degrees=cfg.data.affine_transform.common_degrees,
                translate=cfg.data.affine_transform.get("common_translate", [0.0, 0.0]),
                scale=cfg.data.affine_transform.get("common_scale", [1.0, 1.0]),
            )
        )
    
    common_transform.append(PairNormalize(mean=mean, std=std))
    
    # Noise transforms (applied only to modality_1)
    noise_transform = None
    if cfg.data.affine_transform.get("noise_degrees") is not None:
        filler = (1 - mean) / std
        noise_transform = [
            PairRandomAffine(
                degrees=cfg.data.affine_transform.noise_degrees,
                translate=cfg.data.affine_transform.get("noise_translate", [0.0, 0.0]),
                scale=cfg.data.affine_transform.get("noise_scale", [1.0, 1.0]),
                filler=filler,
            )
        ]
    
    # FPV image transforms
    imagenet_mean = mean.view(1, 3, 1, 1)
    imagenet_std = std.view(1, 3, 1, 1)
    
    def fpv_to_tensor(fpv_images):
        tensors = []
        for img in fpv_images:
            # img: H x W x C, uint8
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        if len(tensors) == 0:
            return torch.empty(0, 3, 0, 0)
        stacked = torch.stack(tensors, dim=0)  # (N, 3, H, W)
        return (stacked - imagenet_mean) / imagenet_std
    
    n_fpv = int(getattr(cfg.data.aria_synthenv, "n_fpv_images", 1))
    
    # Create dataset config for UnifiedDataset
    dataset_configs = [{
        "type": "aria_synthenv",
        "args": {
            "root_dir": root_dir,
            "scene_ids_file": ids_file,
            "modality_pairs": [("points", "window")],
            "image_size": tuple(cfg.data.image_size),
            "dual_transform": None,  # UnifiedDataset will apply common_transform
            "n_fpv_images": n_fpv,
            "fpv_transforms": [fpv_to_tensor],
        }
    }]
    
    dataset = UnifiedDataset(
        dataset_configs=dataset_configs,
        common_transform=common_transform,
        invertible_transform=noise_transform,
    )
    
    return dataset, noise_transform


def render_trajectory_up_to_frame(
    base_map: np.ndarray,
    traj_df,
    params: Dict,
    frame_idx: int,
    aff_transform: np.ndarray = None,
    apply_device2camera: bool = True,
) -> np.ndarray:
    """
    Render trajectory up to a specific frame index.
    
    Args:
        base_map: Base floorplan image
        traj_df: Trajectory dataframe
        params: Projection parameters
        frame_idx: Frame index to render up to (inclusive)
        apply_device2camera: Whether to apply device-to-camera transform
        
    Returns:
        Image with trajectory overlaid
    """
    img = base_map.copy()
    
    # Get trajectory up to frame_idx
    if frame_idx >= len(traj_df):
        frame_idx = len(traj_df) - 1
    
    # Render trajectory for all frames up to current
    traj_pts = []
    quats = []
    
    for i in range(frame_idx + 1):
        row = traj_df.iloc[i]
        
        if apply_device2camera:
            from projectaria_tools.projects import ase
            calibration = ase.get_ase_rgb_calibration()
            T_device_cam = calibration.get_transform_device_camera().to_matrix()
            pts_xyz = row[['tx_world_device', 'ty_world_device', 'tz_world_device']].values
            q = row[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].values
            pose_device = pose_from_xyzq(pts_xyz, q)
            pose_cam = pose_device @ T_device_cam
            pts_cam, q = xyzq_from_pose(pose_cam)
            traj_pts.append(pts_cam)
            quats.append(q)
        else:
            traj_pts.append(row[['tx_world_device', 'ty_world_device']].values)
            quats.append(row[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].values)
    
    if len(traj_pts) == 0:
        return img
    
    traj_pts = np.array(traj_pts)
    quats = np.array(quats)
    
    # Convert to pixel coordinates
    traj_pixels = world_to_pixel(traj_pts, params)

    if aff_transform is not None:
        traj_pixels = apply_affine_2d_points(traj_pixels, aff_transform, 
                                            center=np.array([img.shape[1]//2, img.shape[0]//2]),
                                            ).cpu().numpy().astype(np.int32)
    
    # Draw trajectory line
    if len(traj_pixels) > 1:
        cv2.polylines(img, [traj_pixels], isClosed=False, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    
    # Draw orientation arrows
    r = R.from_quat(quats)
    camera_forward_vector = np.array([0, 0, 1])
    view_dirs_world = r.apply(camera_forward_vector)
    
    arrow_len_px = 15
    start_pt = traj_pixels[-1]
    dir_x = view_dirs_world[-1, 0]
    dir_y = view_dirs_world[-1, 1]
    norm = np.hypot(dir_x, dir_y)
    
    dir_x /= norm
    dir_y /= norm
    
    end_x = int(start_pt[0] + dir_x * arrow_len_px)
    end_y = int(start_pt[1] - dir_y * arrow_len_px)

    cv2.arrowedLine(img, tuple(start_pt), (end_x, end_y), (255, 0, 0), 1, tipLength=0.3, line_type=cv2.LINE_AA)


    # Highlight current frame
    if len(traj_pixels) > 0:
        current_pt = traj_pixels[-1]
        cv2.circle(img, tuple(current_pt), 5, (0, 255, 0), -1)
    
    return img


def transform_trajectory_poses(
    traj_df,
    params: Dict,
    affine_matrix: np.ndarray,
    frame_idx: int,
    apply_device2camera: bool = True
) -> np.ndarray:
    """
    Transform trajectory poses using an affine transformation.
    
    Args:
        traj_df: Trajectory dataframe
        params: Projection parameters
        affine_matrix: Affine transformation matrix (2, 3)
        frame_idx: Frame index to transform up to
        apply_device2camera: Whether to apply device-to-camera transform
        
    Returns:
        Transformed trajectory pixel coordinates (N, 2)
    """
    if frame_idx >= len(traj_df):
        frame_idx = len(traj_df) - 1
    
    traj_pts = []
    
    for i in range(frame_idx + 1):
        row = traj_df.iloc[i]
        
        if apply_device2camera:
            from projectaria_tools.projects import ase
            calibration = ase.get_ase_rgb_calibration()
            T_device_cam = calibration.get_transform_device_camera().to_matrix()
            pts_xyz = row[['tx_world_device', 'ty_world_device', 'tz_world_device']].values
            q = row[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].values
            pose_device = pose_from_xyzq(pts_xyz, q)
            pose_cam = pose_device @ T_device_cam
            pts_cam, _ = xyzq_from_pose(pose_cam)
            traj_pts.append(pts_cam)
        else:
            traj_pts.append(row[['tx_world_device', 'ty_world_device']].values)
    
    if len(traj_pts) == 0:
        return np.array([])
    
    traj_pts = np.array(traj_pts)
    
    # Convert to pixel coordinates in original image
    traj_pixels = world_to_pixel(traj_pts, params)
    
    # Transform using affine matrix
    H, W = params['h'], params['w']
    center = np.array([H // 2, W // 2])
    
    traj_pixels_tensor = torch.from_numpy(traj_pixels).float() if isinstance(traj_pixels, np.ndarray) else traj_pixels
    affine_tensor = torch.from_numpy(affine_matrix).float() if isinstance(affine_matrix, np.ndarray) else affine_matrix
    
    transformed_pixels = apply_affine_2d_points(
        traj_pixels_tensor,
        affine_tensor,
        center=center
    )
    
    return transformed_pixels.numpy()


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize a tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor.clone()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and change to HWC format
    img = np.ascontiguousarray(tensor.permute(1, 2, 0).cpu().numpy())
    img = (img * 255).astype(np.uint8)
    
    return img


@hydra.main(config_path="../../configs", config_name="align_trajectory", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main visualization entry point."""
    print(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if not hasattr(cfg, "model") or not cfg.model.get("checkpoint"):
        raise ValueError("model.checkpoint must be provided")
    
    checkpoint_path = to_absolute_path(cfg.model.checkpoint)
    print(f"Loading model from: {checkpoint_path}")
    
    model = ContrastiveLearningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
        load_dino_weights=False,
        weights_only=False
    )
    model.to(device)
    model.eval()
    
    # Build dataset
    print("Building dataset...")
    dataset, noise_transform = build_aria_dataset(cfg)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Get first example
    sample_idx = 3
    sample = dataset[sample_idx]
    
    if "fpv_dict" not in sample:
        raise ValueError("FPV images not found in sample. Make sure n_fpv_images > 0")
    
    scene_id = sample["sample_id"]
    print(f"Visualizing scene: {scene_id}")
    
    # Load trajectory and floorplan data
    # Get root_dir from config (UnifiedDataset doesn't expose root_dir directly)
    root_dir = to_absolute_path(cfg.data.aria_synthenv.path)
    parent_dir = os.path.dirname(root_dir)
    original_root = os.path.join(parent_dir, "original_data") if os.path.isdir(os.path.join(parent_dir, "original_data")) else root_dir
    
    traj_df = read_trajectory(original_root, scene_id)
    
    # Get modalities
    modality_0 = sample["modality_0_noise"]  # Points modality (no transformation)
    modality_1 = sample["original_modality_1"]  # Window modality (with noise transformation)
    fpv_dict = sample["fpv_dict"]
    fpv_params = fpv_dict["params"]
    
    # Get params for trajectory rendering (from fpv_dict)
    params = change_params_resolution(fpv_params, cfg.data.image_size)
    
    # Create base_map from original_modality_0 (points image)
    # Convert from tensor to numpy if needed
    if isinstance(modality_0, torch.Tensor):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        modality_0_denorm = modality_0 * std + mean
        modality_0_denorm = torch.clamp(modality_0_denorm, 0, 1)
        # Convert to numpy HWC
        base_map = (modality_0_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    else:
        base_map = modality_0.copy() if hasattr(modality_0, 'copy') else modality_0
    
    # Ensure base_map is RGB
    if len(base_map.shape) == 2:
        base_map = cv2.cvtColor(base_map, cv2.COLOR_GRAY2RGB)
    elif base_map.shape[2] == 3:
        # Already RGB
        pass
    
    # Estimate alignment between points and window modalities
    print("Estimating alignment between points and window modalities...")
    aff_est = estimate_alignment(model, modality_0, modality_1, torch.device(device), n_tta=cfg.model.n_tta)
    print(f"Estimated affine matrix:\n{aff_est}")
    inverse_aff_est = inverse_affine_matrix(aff_est)
    print(f"Inverse estimated affine matrix:\n{inverse_aff_est}")
    
    # Load all FPV images in order directly from the scene directory
    # (The dataset randomly samples, but we want all images in order)
    scene_rendered_dir = os.path.join(root_dir, scene_id)
    rgb_dir = os.path.join(scene_rendered_dir, "images", "train", "rgb")
    
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    # Collect all RGB frames with valid indices
    all_rgb_files = sorted(
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    
    frame_candidates = []
    for fname in all_rgb_files:
        stem = os.path.splitext(fname)[0]
        i_str = stem.replace("processed", "")
        try:
            frame_idx = int(i_str)
        except ValueError:
            continue
        if 0 <= frame_idx < len(traj_df):
            frame_candidates.append((fname, frame_idx))
    
    # Sort by frame index
    frame_candidates.sort(key=lambda x: x[1])
    
    if not frame_candidates:
        raise RuntimeError(f"No valid RGB frames found for scene {scene_id}")
    
    print(f"Found {len(frame_candidates)} FPV images")
    print("Press any key to advance to next frame, 'q' to quit")
    
    # Convert params to numpy for plotting functions
    params_np = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                params_np[k] = v.item()
            else:
                params_np[k] = v.cpu().numpy()
        elif isinstance(v, (list, tuple)):
            # Handle lists/tuples that might contain tensors
            params_np[k] = [x.item() if torch.is_tensor(x) and x.numel() == 1 else x.cpu().numpy() if torch.is_tensor(x) else x for x in v]
        else:
            params_np[k] = v


    # Get ground truth affine matrix
    # Get the transform params (assuming single sample in sample)
    noise_params = sample["noise_params"]
    if isinstance(noise_params.get("angle"), (list, torch.Tensor)):
        actual_angle = float(noise_params["angle"][0].cpu().numpy()) if torch.is_tensor(noise_params["angle"][0]) else noise_params["angle"][0]
        actual_tx = float(noise_params["translate"][1][0].cpu().numpy()) if torch.is_tensor(noise_params["translate"][1][0]) else noise_params["translate"][1][0]
        actual_ty = float(noise_params["translate"][0][0].cpu().numpy()) if torch.is_tensor(noise_params["translate"][0][0]) else noise_params["translate"][0][0]
        actual_scale = float(noise_params["scale"][0].cpu().numpy()) if torch.is_tensor(noise_params["scale"][0]) else noise_params["scale"][0]
    else:
        actual_angle = float(noise_params["angle"]) if torch.is_tensor(noise_params["angle"]) else noise_params["angle"]
        actual_tx = float(noise_params["translate"][1]) if torch.is_tensor(noise_params["translate"][1]) else noise_params["translate"][1]
        actual_ty = float(noise_params["translate"][0]) if torch.is_tensor(noise_params["translate"][0]) else noise_params["translate"][0]
        actual_scale = float(noise_params["scale"]) if torch.is_tensor(noise_params["scale"]) else noise_params["scale"]
    
    # Create ground truth affine matrix
    theta_actual = np.deg2rad(actual_angle)
    R_actual = np.array([[np.cos(theta_actual), -np.sin(theta_actual)],
                        [np.sin(theta_actual),  np.cos(theta_actual)]])
    S_actual = np.array([[actual_scale, 0], [0, actual_scale]])
    t_actual = np.array([actual_tx, actual_ty])
    A_actual = R_actual @ S_actual
    H_img, W_img = cfg.data.image_size
    aff_gt = np.array([[A_actual[0, 0], A_actual[0, 1], t_actual[1]/(H_img//2)],
                        [A_actual[1, 0], A_actual[1, 1], t_actual[0]/(W_img//2)]])

    print(f"GT affine matrix:\n{aff_gt}")
    inverse_aff_gt = inverse_affine_matrix(aff_gt)
    print(f"Inverse GT affine matrix:\n{inverse_aff_gt}")

    
    # Process each FPV image in order
    for file_idx, (fname, frame_idx) in enumerate(frame_candidates):
        # Load and process FPV image
        img_path = os.path.join(rgb_dir, fname)
        img_rgb = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
        if img_rgb is None:
            continue
        
        
        fpv_img = img_rgb
        
        # Render floorplan with trajectory up to current frame
        # Show all poses up to and including current frame
        floorplan_img = render_trajectory_up_to_frame(
            base_map.copy(),
            traj_df,
            params_np,
            frame_idx,
            aff_gt
        )
        
        # Convert floorplan to RGB if needed
        if len(floorplan_img.shape) == 2:
            floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_GRAY2RGB)
        elif floorplan_img.shape[2] == 3 and floorplan_img.dtype == np.uint8:
            # BGR to RGB
            floorplan_img = cv2.cvtColor(floorplan_img, cv2.COLOR_BGR2RGB)
        
        # Resize floorplan to match image size
        floorplan_img = cv2.resize(floorplan_img, tuple(cfg.data.image_size), interpolation=cv2.INTER_AREA)
        
        # Render window modality with transformed trajectory
        window_img = denormalize_tensor(modality_1)
        
        # Transform trajectory poses for window modality (estimated alignment)
        traj_pixels_transformed = transform_trajectory_poses(
            traj_df,
            params_np,
            compose_affine_matrices(inverse_aff_est, aff_gt),
            frame_idx
        )
        
        # Draw transformed trajectory on window image (estimated)
        if len(traj_pixels_transformed) > 1:
            traj_pixels_int = traj_pixels_transformed.astype(np.int32)
            # Clip to image bounds
            h, w = window_img.shape[:2]
            traj_pixels_int = np.clip(traj_pixels_int, [0, 0], [w-1, h-1])
            cv2.polylines(window_img, [traj_pixels_int], isClosed=False, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        
        # Highlight current frame (estimated)
        if len(traj_pixels_transformed) > 0:
            current_pt = traj_pixels_transformed[-1].astype(np.int32)
            h, w = window_img.shape[:2]
            current_pt = np.clip(current_pt, [0, 0], [w-1, h-1])
            cv2.circle(window_img, tuple(current_pt), 5, (0, 255, 0), -1)
        
        # Transform trajectory with ground truth
        traj_pixels_gt = transform_trajectory_poses(
            traj_df,
            params_np,
            compose_affine_matrices(inverse_aff_gt, aff_gt),
            frame_idx
        )
        
        # Draw ground truth trajectory in different color
        if len(traj_pixels_gt) > 1:
            traj_pixels_gt_int = traj_pixels_gt.astype(np.int32)
            h, w = window_img.shape[:2]
            traj_pixels_gt_int = np.clip(traj_pixels_gt_int, [0, 0], [w-1, h-1])
            cv2.polylines(window_img, [traj_pixels_gt_int], isClosed=False, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        # Highlight current frame (ground truth)
        if len(traj_pixels_gt) > 0:
            current_pt_gt = traj_pixels_gt[-1].astype(np.int32)
            h, w = window_img.shape[:2]
            current_pt_gt = np.clip(current_pt_gt, [0, 0], [w-1, h-1])
            cv2.circle(window_img, tuple(current_pt_gt), 3, (255, 255, 0), -1)
        
        # Resize FPV image to match
        fpv_img_resized = cv2.resize(fpv_img, tuple(cfg.data.image_size), interpolation=cv2.INTER_AREA)
        
        # Create side-by-side visualization
        combined = np.hstack([fpv_img_resized, floorplan_img, window_img])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(combined, "FPV Image", (10, 30), font, 1, (255, 255, 255), 2)
        # cv2.putText(combined, "Floorplan (Points)", (cfg.data.image_size[1] + 10, 30), font, 1, (255, 255, 255), 2)
        # cv2.putText(combined, "Window (Aligned)", (2 * cfg.data.image_size[1] + 10, 30), font, 1, (255, 255, 255), 2)
        # cv2.putText(combined, f"Frame: {frame_idx}/{len(traj_df)-1} ({file_idx+1}/{len(frame_candidates)})", 
        #            (10, combined.shape[0] - 20), font, 0.7, (255, 255, 255), 2)
    
        legend_y = 60
        cv2.putText(combined, "GT: Blue", (2 * cfg.data.image_size[1] + 10, legend_y), font, 0.6, (255, 0, 0), 2)
        cv2.putText(combined, "Est: Red", (2 * cfg.data.image_size[1] + 10, legend_y + 25), font, 0.6, (0, 0, 255), 2)
        
        # Display
        cv2.imshow("Trajectory Alignment", combined)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Visualization complete!")


if __name__ == "__main__":
    main()

