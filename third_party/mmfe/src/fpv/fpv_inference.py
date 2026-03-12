"""
Inference utilities for FPV pose estimation.

This module provides functions for:
- Computing squared error between frustum features and floorplan features
- Non-linear pose optimization using scipy.optimize.least_squares
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
import cv2
import matplotlib.pyplot as plt

from fpv.fpv_3D_utils import (
    FrustumData,
    transform_xy_local_to_world,
    world_to_pixel_fpv,
)
from aria_mmfe.aria_images.aria_cv_tools import world_to_pixel
from aria_mmfe.code_snippets.plotters import change_params_resolution


def normalize_coords(coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Normalize pixel coordinates from [0, W-1] / [0, H-1] to [-1, 1] for grid_sample.
    
    Args:
        coords: (..., 2) Pixel coordinates (x, y)
        H: Height of the floorplan
        W: Width of the floorplan
        
    Returns:
        Normalized coordinates in [-1, 1] range
    """
    norm_coords = coords.clone().float()
    norm_coords[..., 0] = 2.0 * norm_coords[..., 0] / (W - 1) - 1.0
    norm_coords[..., 1] = 2.0 * norm_coords[..., 1] / (H - 1) - 1.0
    return norm_coords


def compute_cosine_similarity_error(
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    pose: torch.Tensor,
    params: Dict,
    batch_idx: int = 0,
) -> Tuple[float, torch.Tensor]:
    """
    Compute cosine similarity error between transformed frustum features and floorplan features.
    
    This follows the logic from FrustumRegressionLoss.forward().
    
    Args:
        floorplan_features: (C, H_fp, W_fp) Floorplan feature map for one sample
        frustum: FrustumData containing features, coords_proj_xy, valid_mask_xy
        pose: (3,) tensor with (x, y, theta) in world coordinates
        params: Dict with projection parameters (scale, x_min_w, y_min_w, etc.)
        batch_idx: Index into batched params
        
    Returns:
        Tuple of (scalar error, masked_cosine_similarity_error tensor)
    """
    C, H_fp, W_fp = floorplan_features.shape
    device = floorplan_features.device
    
    # Get frustum data
    pred_features = frustum.features  # (N, C, H_g, W_g)
    coords_local = frustum.coords_proj_xy  # (N, H_g, W_g, 2)
    camera_mask = frustum.valid_mask_xy  # (N, 1, H_g, W_g)
    
    # Ensure pose has correct shape for transform functions
    pose = pose.view(1, 3).to(device)  # (1, 3)
    
    # Transform local coords to world coords
    xy_world = transform_xy_local_to_world(coords_local, pose)  # (N, H_g, W_g, 2)
    
    # Create single-sample params dict for world_to_piel_fpv
    single_params = {
        k: v[batch_idx:batch_idx+1] if torch.is_tensor(v) else [v[batch_idx]] if isinstance(v, list) else v
        for k, v in params.items()
    }
    
    # Convert world to pixel coordinates
    xy_pixel = world_to_pixel_fpv(xy_world, single_params, 0)  # (N, H_g, W_g, 2)
    
    # Normalize coordinates to [-1, 1] for grid_sample
    grid = normalize_coords(xy_pixel, H_fp, W_fp)  # (N, H_g, W_g, 2)
    
    # Expand floorplan features to match number of images
    N = pred_features.shape[0]
    map_expanded = floorplan_features.unsqueeze(0).expand(N, -1, -1, -1)  # (N, C, H_fp, W_fp)
    
    # Sample floorplan features at projected locations
    target_features = F.grid_sample(
        map_expanded,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # (N, C, H_g, W_g)
    
    # Compute validity mask
    # A pixel is valid if:
    # A) It is visible in the camera (camera_mask)
    # B) It projects INSIDE the floorplan boundaries (grid values in [-1, 1])
    in_bounds_x = (grid[..., 0] >= -1) & (grid[..., 0] <= 1)
    in_bounds_y = (grid[..., 1] >= -1) & (grid[..., 1] <= 1)
    floorplan_mask = (in_bounds_x & in_bounds_y).unsqueeze(1)  # (N, 1, H_g, W_g)

    zero_mask = (pred_features.abs().sum(dim=1, keepdim=True) != 0)
    
    # Combined mask
    final_mask = camera_mask * floorplan_mask.float() * zero_mask.float()
    
    # If no valid pixels, return large error
    if final_mask.sum() == 0:
        return float('inf'), torch.zeros_like(pred_features)
    
    # Compute cosine similarity error
    pred = F.normalize(pred_features, dim=1)
    tgt  = F.normalize(target_features, dim=1)
    cosine_similarity = (pred * tgt).sum(dim=1, keepdim=True)  # (N,1,H,W)
    
    # Apply mask
    masked_cosine_similarity_error = (1 - cosine_similarity) * final_mask
    
    # Average over valid elements
    num_valid_elements = final_mask.sum()
    error = masked_cosine_similarity_error.sum() / (num_valid_elements + 1e-6)
    
    return error.item(), masked_cosine_similarity_error


def compute_cosine_similarity_error_for_poses(
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    poses: torch.Tensor,
    params: Dict,
    batch_idx: int = 0,
) -> torch.Tensor:
    """
    Compute cosine similarity error for multiple poses efficiently.
    
    Args:
        floorplan_features: (C, H_fp, W_fp) Floorplan feature map
        frustum: FrustumData with features (N, C, H_g, W_g)
        poses: (K, 3) tensor of candidate poses
        params: Projection parameters
        batch_idx: Index into batched params
        
    Returns:
        (K,) tensor of cosine similarity errors for each pose
    """
    K = poses.shape[0]
    errors = torch.zeros(K, device=poses.device)
    
    for k in range(K):
        error, _ = compute_cosine_similarity_error(
            floorplan_features, frustum, poses[k], params, batch_idx
        )
        errors[k] = error
        
    return errors


def _residual_function(
    pose_vec: np.ndarray,
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    params: Dict,
    batch_idx: int,
    device: torch.device,
) -> np.ndarray:
    """
    Residual function for scipy.optimize.least_squares.
    
    Args:
        pose_vec: (3,) numpy array with (x, y, theta)
        floorplan_features: (C, H_fp, W_fp) feature map
        frustum: FrustumData
        params: Projection parameters
        batch_idx: Index into batched params
        device: Torch device
        
    Returns:
        Flattened residuals as numpy array
    """
    pose = torch.tensor(pose_vec, dtype=torch.float32, device=device)
    
    C, H_fp, W_fp = floorplan_features.shape
    
    # Get frustum data
    pred_features = frustum.features  # (N, C, H_g, W_g)
    coords_local = frustum.coords_proj_xy  # (N, H_g, W_g, 2)
    camera_mask = frustum.valid_mask_xy  # (N, 1, H_g, W_g)
    
    # Transform coordinates
    pose_expanded = pose.view(1, 3)
    xy_world = transform_xy_local_to_world(coords_local, pose_expanded)
    
    # Create single-sample params
    single_params = {
        k: v[batch_idx:batch_idx+1] if torch.is_tensor(v) else [v[batch_idx]] if isinstance(v, list) else v
        for k, v in params.items()
    }
    
    xy_pixel = world_to_pixel_fpv(xy_world, single_params, 0)
    grid = normalize_coords(xy_pixel, H_fp, W_fp)
    
    N = pred_features.shape[0]
    map_expanded = floorplan_features.unsqueeze(0).expand(N, -1, -1, -1)
    
    target_features = F.grid_sample(
        map_expanded,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # Compute validity mask
    in_bounds_x = (grid[..., 0] >= -1) & (grid[..., 0] <= 1)
    in_bounds_y = (grid[..., 1] >= -1) & (grid[..., 1] <= 1)
    floorplan_mask = (in_bounds_x & in_bounds_y).unsqueeze(1)
    final_mask = camera_mask * floorplan_mask.float()
    
    # Compute residuals (difference, not squared)
    diff = pred_features - target_features
    masked_diff = diff * final_mask
    
    # Return flattened residuals
    return masked_diff.detach().cpu().numpy().flatten()


def optimize_pose(
    initial_pose: torch.Tensor,
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    params: Dict,
    batch_idx: int = 0,
    method: str = 'trf',
    max_nfev: int = 100,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    verbose: int = 0,
) -> Tuple[torch.Tensor, float]:
    """
    Optimize 2D pose using non-linear least squares.
    
    Args:
        initial_pose: (3,) tensor with initial (x, y, theta) guess
        floorplan_features: (C, H_fp, W_fp) floorplan feature map
        frustum: FrustumData containing frustum features and coordinates
        params: Projection parameters dict
        batch_idx: Index into batched params
        method: Optimization method ('trf', 'dogbox', or 'lm')
        max_nfev: Maximum number of function evaluations
        ftol: Tolerance for termination by change in cost function
        xtol: Tolerance for termination by change in variables
        verbose: Verbosity level (0, 1, or 2)
        
    Returns:
        Tuple of (optimized_pose tensor, final_error scalar)
    """
    device = floorplan_features.device
    
    # Get bounds from params
    scale = params['scale'][batch_idx] if torch.is_tensor(params['scale']) else params['scale'][batch_idx]
    x_min = params['x_min_w'][batch_idx] if torch.is_tensor(params['x_min_w']) else params['x_min_w'][batch_idx]
    y_min = params['y_min_w'][batch_idx] if torch.is_tensor(params['y_min_w']) else params['y_min_w'][batch_idx]
    W_fp = params['w'][batch_idx] if torch.is_tensor(params['w']) else params['w'][batch_idx]
    H_fp = params['h'][batch_idx] if torch.is_tensor(params['h']) else params['h'][batch_idx]
    
    # Convert to float if tensor
    if torch.is_tensor(scale):
        scale = scale.item()
    if torch.is_tensor(x_min):
        x_min = x_min.item()
    if torch.is_tensor(y_min):
        y_min = y_min.item()
    if torch.is_tensor(W_fp):
        W_fp = W_fp.item()
    if torch.is_tensor(H_fp):
        H_fp = H_fp.item()
    
    x_max = x_min + W_fp / scale
    y_max = y_min + H_fp / scale
    
    # Set bounds: x in [x_min, x_max], y in [y_min, y_max], theta in [-pi, pi]
    bounds = (
        [x_min, y_min, -np.pi],
        [x_max, y_max, np.pi]
    )
    
    # Initial guess
    x0 = initial_pose.detach().cpu().numpy()
    
    # Run optimization
    result = least_squares(
        _residual_function,
        x0,
        args=(floorplan_features, frustum, params, batch_idx, device),
        bounds=bounds,
        method=method,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        verbose=verbose,
    )
    
    # Convert result back to tensor
    optimized_pose = torch.tensor(result.x, dtype=torch.float32, device=device)
    
    # Compute final error
    final_error, _ = compute_cosine_similarity_error(
        floorplan_features, frustum, optimized_pose, params, batch_idx
    )
    
    return optimized_pose, final_error


def visualize_template_matching_poses(
    candidate_poses: torch.Tensor,
    params: Dict,
    batch_idx: int = 0,
    errors: Optional[torch.Tensor] = None,
    best_idx: Optional[int] = None,
    floorplan_image: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize candidate poses overlaid on the floorplan.
    
    Args:
        candidate_poses: (K, 3) tensor of candidate poses (x, y, theta)
        params: Projection parameters dict
        batch_idx: Index into batched params
        errors: Optional (K,) tensor of errors for each pose
        best_idx: Optional index of the best pose
        floorplan_image: Optional floorplan image to overlay on. If None, creates blank image.
    """
    device = candidate_poses.device
    
    # Extract params for this batch
    if torch.is_tensor(params['h']):
        H = int(params['h'][batch_idx].item())
        W = int(params['w'][batch_idx].item())
    else:
        H = int(params['h'][batch_idx])
        W = int(params['w'][batch_idx])
    
    # Create single-sample params dict for world_to_pixel
    # world_to_pixel expects scalar values, not batched tensors
    np_params = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                np_params[k] = v.item()
            else:
                # Extract single value from batch
                np_params[k] = v[batch_idx].item() if v.dim() > 0 else v.item()
        elif isinstance(v, list):
            np_params[k] = v[batch_idx] if len(v) > batch_idx else v[0]
        else:
            np_params[k] = v
    
    # Create or use provided floorplan image
    if floorplan_image is None:
        vis_img = np.ones((H, W, 3), dtype=np.uint8) * 255  # White background
    else:
        vis_img = floorplan_image.copy()
        if vis_img.shape[:2] != (H, W):
            vis_img = cv2.resize(vis_img, (W, H), interpolation=cv2.INTER_AREA)
    
    # Resize the map to 256x256 and update params (same as test_aria_3D.py)
    vis_img = cv2.resize(vis_img, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Convert params to torch format for change_params_resolution (same as test_aria_3D.py)
    torch_params = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            # Extract single value from batch
            if v.dim() > 0 and v.shape[0] > batch_idx:
                torch_params[k] = torch.tensor([v[batch_idx].item()], device=device, dtype=torch.float32)
            else:
                torch_params[k] = torch.tensor([v.item()], device=device, dtype=torch.float32)
        elif isinstance(v, list):
            torch_params[k] = torch.tensor([v[batch_idx]], device=device, dtype=torch.float32)
        else:
            torch_params[k] = torch.tensor([v], device=device, dtype=torch.float32)
    
    # Update params for new resolution
    torch_params = change_params_resolution(torch_params, (256, 256))
    
    # Convert back to numpy format for world_to_pixel
    np_params = {}
    for k, v in torch_params.items():
        if torch.is_tensor(v):
            np_params[k] = v[0].item() if v.numel() > 0 else v.item()
        else:
            np_params[k] = v
    
    # Update H and W to match resized image
    H, W = 256, 256
    
    # Convert poses to numpy
    poses_np = candidate_poses.detach().cpu().numpy()
    K = len(poses_np)
    
    # Separate negative poses and ground truth
    negative_poses = poses_np[:-1]  # All except last
    gt_pose = poses_np[-1:]  # Last one (ground truth)
    
    # Convert world coordinates to pixel coordinates
    if len(negative_poses) > 0:
        neg_xy = negative_poses[:, :2]  # (K-1, 2)
        neg_pixels = world_to_pixel(neg_xy, np_params)  # (K-1, 2)
        neg_thetas = negative_poses[:, 2]  # (K-1,)
    
    gt_xy = gt_pose[:, :2]  # (1, 2)
    gt_pixels = world_to_pixel(gt_xy, np_params)  # (1, 2)
    gt_theta = gt_pose[0, 2]
    
    # Draw negative poses in red
    arrow_len_px = 0.07 * min(H, W)
    for i in range(len(negative_poses)):
        px, py = neg_pixels[i]
        theta = neg_thetas[i]
        
        # Convert theta to direction vector
        # theta is rotation angle in world coordinates
        dx = np.cos(theta)
        dy = np.sin(theta)
        norm = np.hypot(dx, dy)
        if norm < 1e-6:
            continue
        dx /= norm
        dy /= norm
        
        end_x = int(px + dx * arrow_len_px)
        end_y = int(py - dy * arrow_len_px)  # Flip y for image coordinates
        
        # Draw circle and arrow in red
        cv2.circle(vis_img, (int(px), int(py)), 1, (0, 0, 255), -1)
        cv2.arrowedLine(vis_img, (int(px), int(py)), (end_x, end_y), 
                        (0, 0, 255), 1, tipLength=0.3)
    
    # Draw ground truth pose in green
    px, py = gt_pixels[0]
    dx = -np.sin(gt_theta)
    dy = np.cos(gt_theta)
    norm = np.hypot(dx, dy)
    if norm >= 1e-6:
        dx /= norm
        dy /= norm
        end_x = int(px + dx * arrow_len_px)
        end_y = int(py - dy * arrow_len_px)
        
        # Draw circle and arrow in green
        cv2.circle(vis_img, (int(px), int(py)), 1, (0, 255, 0), -1)
        cv2.arrowedLine(vis_img, (int(px), int(py)), (end_x, end_y), 
                        (0, 255, 0), 1, tipLength=0.3)
    
    # Highlight best pose if provided
    if best_idx is not None and errors is not None:
        if best_idx < len(negative_poses):
            # Best pose is one of the negative poses
            px, py = neg_pixels[best_idx]
            cv2.circle(vis_img, (int(px), int(py)), 2, (255, 0, 255), 2)  # Magenta circle
        elif best_idx == len(negative_poses):
            # Best pose is the ground truth
            px, py = gt_pixels[0]
            cv2.circle(vis_img, (int(px), int(py)), 2, (255, 0, 255), 2)  # Magenta circle
    
    # Convert BGR to RGB for matplotlib
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    # Display using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_img_rgb)
    plt.title(f"Template Matching Candidates (Red: negatives, Green: GT, Magenta: best)")
    if errors is not None:
        plt.xlabel(f"Best error: {errors[best_idx].item():.4f}" if best_idx is not None else "")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def template_matching(
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    candidate_poses: torch.Tensor,
    params: Dict,
    batch_idx: int = 0,
    visualize: bool = True,
    floorplan_image: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, float, int]:
    """
    Find the best pose from a set of candidates using template matching.
    
    Args:
        floorplan_features: (C, H_fp, W_fp) floorplan feature map
        frustum: FrustumData
        candidate_poses: (K, 3) tensor of candidate poses
        params: Projection parameters
        batch_idx: Index into batched params
        visualize: If True, visualize candidate poses on floorplan
        floorplan_image: Optional floorplan image for visualization
        
    Returns:
        Tuple of (best_pose, best_error, best_idx)
    """
    errors = compute_cosine_similarity_error_for_poses(
        floorplan_features, frustum, candidate_poses, params, batch_idx
    )
    
    best_idx = errors.argmin().item()
    best_pose = candidate_poses[best_idx]
    best_error = errors[best_idx].item()
    
    if visualize:
        visualize_template_matching_poses(
            candidate_poses, params, batch_idx, errors, best_idx, floorplan_image
        )
    
    return best_pose, best_error, best_idx


def estimate_pose_template_matching(
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    params: Dict,
    batch_idx: int = 0,
    n_candidates: int = 1000,
    refine: bool = False,
    device: Optional[torch.device] = None,
    gt_pose = None
) -> Tuple[torch.Tensor, float, Dict]:
    """
    Full pose estimation pipeline: template matching + optional refinement.
    
    Args:
        floorplan_features: (C, H_fp, W_fp) floorplan feature map
        frustum: FrustumData
        params: Projection parameters (batched)
        batch_idx: Index into batched params
        n_candidates: Number of random candidate poses for template matching
        refine: Whether to refine with non-linear optimization
        device: Torch device
        
    Returns:
        Tuple of (estimated_pose, final_error, info_dict)
    """
    if device is None:
        device = floorplan_features.device
    
    # Generate random candidate poses
    from fpv.fpv_3D_utils import sample_random_poses
    
    # Create single-sample params for sampling
    single_params = {
        k: v[batch_idx:batch_idx+1] if torch.is_tensor(v) else [v[batch_idx]] if isinstance(v, list) else v
        for k, v in params.items()
    }
    
    candidate_poses = sample_random_poses(n_candidates, single_params, device)
    candidate_poses = candidate_poses.squeeze(0)  # (K, 3)
    candidate_poses = torch.cat([candidate_poses, gt_pose.unsqueeze(0)])
    # candidate_poses = gt_pose.unsqueeze(0)
    
    # Template matching
    best_pose, template_error, best_idx = template_matching(
        floorplan_features, frustum, candidate_poses, params, batch_idx
    )

    print(f"Best idx: {best_idx}")
    
    info = {
        'template_error': template_error,
        'template_pose': best_pose.clone(),
        'best_candidate_idx': best_idx,
    }
    
    # Optional refinement
    if refine:
        refined_pose, refined_error = optimize_pose(
            best_pose, floorplan_features, frustum, params, batch_idx
        )
        info['refined_error'] = refined_error
        info['refined_pose'] = refined_pose.clone()
        return refined_pose, refined_error, info
    
    return best_pose, template_error, info


def estimate_pose_ransac(
    floorplan_features: torch.Tensor,
    frustum: FrustumData,
    params: Dict,
    batch_idx: int = 0,
    n_candidates: int = 1000,
    refine: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float, Dict]:
    """
    Full pose estimation pipeline: template matching + optional refinement.
    
    Args:
        floorplan_features: (C, H_fp, W_fp) floorplan feature map
        frustum: FrustumData
        params: Projection parameters (batched)
        batch_idx: Index into batched params
        n_candidates: Number of random candidate poses for template matching
        refine: Whether to refine with non-linear optimization
        device: Torch device
        
    Returns:
        Tuple of (estimated_pose, final_error, info_dict)
    """
    if device is None:
        device = floorplan_features.device
    
    # TODO: Instead of sampling poses and doing template matching apply RANSAC
    # For now, use template matching as placeholder
    from fpv.fpv_3D_utils import sample_random_poses
    
    single_params = {
        k: v[batch_idx:batch_idx+1] if torch.is_tensor(v) else [v[batch_idx]] if isinstance(v, list) else v
        for k, v in params.items()
    }
    
    candidate_poses = sample_random_poses(n_candidates, single_params, device)
    candidate_poses = candidate_poses.squeeze(0)  # (K, 3)
    
    best_pose, template_error, best_idx = template_matching(
        floorplan_features, frustum, candidate_poses, params, batch_idx
    )
    
    info = {
        'template_error': template_error,
        'template_pose': best_pose.clone(),
        'best_candidate_idx': best_idx,
    }
    
    # Optional refinement
    if refine:
        refined_pose, refined_error = optimize_pose(
            best_pose, floorplan_features, frustum, params, batch_idx
        )
        info['refined_error'] = refined_error
        info['refined_pose'] = refined_pose.clone()
        return refined_pose, refined_error, info
    
    return best_pose, template_error, info

__all__ = [
    'compute_squared_error',
    'compute_squared_error_for_poses',
    'optimize_pose',
    'template_matching',
    'estimate_pose_template_matching',
    'estimate_pose_ransac',
]









