from typing import Tuple, Union, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

def estimate_affine_matrix(p, q, center: np.ndarray = None, method: str = "ransac", invert: bool = False, centered: bool = True):
    """
    p, q: arrays shape (N,2) of corresponding points (N>=2)
    returns affine matrix (2,3)
    """
    if method == "ransac":
        method = cv2.RANSAC
    elif method == "lmeds":
        method = cv2.LMEDS
    elif method == "best":
        method = cv2.LMEDS
        p = p[:2]
        q = q[:2]
    else:
        raise ValueError(f"Unsupported method: {method}")
    if center is not None:
        p = p - center
        q = q - center

    # Go from RowColumn to XY (OpenCV convention)
    if isinstance(p, torch.Tensor):
        # p = p[:, [1, 0]] 
        # q = q[:, [1, 0]] 
        matrix = cv2.estimateAffinePartial2D(p.cpu().numpy(), q.cpu().numpy(), method=method)[0]
    elif isinstance(p, np.ndarray):
        # p = p[:, ::-1]  
        # q = q[:, ::-1] 
        matrix = cv2.estimateAffinePartial2D(p, q, method=method)[0]
    else:
        raise ValueError(f"Unsupported type: {type(p)} for points")

    # Compute the inverse of the matrix
    if invert:
        matrix = inverse_affine_matrix(matrix)

    if centered:
    # Normalize the translation by the center (points are centered, if not it would be 2*center)
        matrix[:2, 2] /= center
    else:
        matrix[:2, 2] /= 2*center
    return matrix

def estimate_affine_matrix_multiple(idx0: torch.Tensor, idx1: torch.Tensor, center: np.ndarray = None, method: str = "ransac", ransac_threshold: float = 3.0):
    """
    Estimate affine transformation for multiple augmentations and return the one with most inliers.
    
    Args:
        idx0: Corresponding points in image0 (N_AUGS, N_pts, 2)
        idx1: Corresponding points in image1 (N_AUGS, N_pts, 2)
        center: Center point for normalization (optional)
        method: Method for estimation ("ransac" or "lmeds")
        ransac_threshold: Threshold for RANSAC inlier detection (only used for counting inliers)
    
    Returns:
        best_affine: Affine matrix (2,3) from the augmentation with most inliers
        best_aug_idx: Index of the best augmentation
        n_inliers: Number of inliers for the best augmentation
    """
    if isinstance(idx0, torch.Tensor):
        idx0 = idx0.cpu().numpy()
    if isinstance(idx1, torch.Tensor):
        idx1 = idx1.cpu().numpy()
    
    n_augs = idx0.shape[0]
    best_affine = None
    best_aug_idx = 0
    max_inliers = 0
    
    # Convert method to OpenCV constant
    if method == "ransac":
        cv_method = cv2.RANSAC
    elif method == "lmeds":
        cv_method = cv2.LMEDS
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Process each augmentation
    for aug_idx in range(n_augs):
        p = idx0[aug_idx]  # (N_pts, 2)
        q = idx1[aug_idx]  # (N_pts, 2)
        
        # Filter out NaN points (from padding)
        valid_mask = ~(np.isnan(p).any(axis=1) | np.isnan(q).any(axis=1))
        p_valid = p[valid_mask]
        q_valid = q[valid_mask]
        
        if len(p_valid) < 2:
            # Not enough points for estimation
            continue
        
        # Center points if center is provided
        if center is not None:
            p_centered = p_valid - center
            q_centered = q_valid - center
        else:
            p_centered = p_valid
            q_centered = q_valid
        
        # Estimate affine transformation
        result = cv2.estimateAffinePartial2D(p_centered, q_centered, method=cv_method, ransacReprojThreshold=ransac_threshold)
        if result[0] is None:
            # Estimation failed
            continue
        
        matrix = result[0]
        inliers = result[1]  # Boolean mask of inliers
        
        # Count inliers
        n_inliers = np.sum(inliers) if inliers is not None else len(p_valid)
        
        # Update best if this has more inliers
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_aug_idx = aug_idx
            # Normalize translation by center if center is provided
            if center is not None:
                matrix[:2, 2] /= center
            best_affine = matrix
    
    if best_affine is None:
        raise ValueError("Failed to estimate affine transformation for any augmentation")
    
    return best_affine, best_aug_idx, max_inliers

def find_nn(
    feat0: torch.Tensor,
    feat1: torch.Tensor,
    top_k: int = 2,
    mask: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds the top_k best, mutually exclusive nearest neighbor pairs between
    feat0 and feat1. Mutually exclusive means that once a point from feat0 or
    feat1 is used in a pair, it cannot be used in any other pair.

    This implementation uses a greedy approach on the cosine similarity matrix.

    Args:
        feat0 (torch.Tensor): Features of the points to find neighbors for.
                              Shape: (D, H, W)
        feat1 (torch.Tensor): Features of the candidate points to search within.
                              Shape: (D, H, W)
        top_k (int): The number of unique nearest neighbor pairs to find.

    Returns:
        A tuple containing:
        - The (y, x) coordinates of the selected points in feat0. Shape: (k, 2)
        - The features of the selected points from feat0. Shape: (k, D)
        - The (y, x) coordinates of the nearest neighbors in feat1. Shape: (k, 2)
        - The features of the nearest neighbor points from feat1. Shape: (k, D)
        (Note: The returned number of pairs k might be less than top_k if not
         enough valid points are available).
    """
    # Get dimensions
    D, H, W = feat0.shape
    num_points = H * W
    if mask is None:
        mask = torch.ones(H, W, device=feat0.device, dtype=torch.bool)
    
    # Ensure top_k is not larger than the number of possible pairs
    top_k = min(top_k, num_points)

    # 1. Reshape and Normalize Features (same as before)
    feat0_flat = feat0.view(D, -1).T
    feat1_flat = feat1.view(D, -1).T
    feat0_norm = feat0_flat / torch.norm(feat0_flat, dim=1, keepdim=True)
    feat1_norm = feat1_flat / torch.norm(feat1_flat, dim=1, keepdim=True)
    mask_flat = mask.view(-1).bool()

    # 2. Compute Cosine Similarity (same as before)
    sim = feat0_norm @ feat1_norm.T
    
    # 3. Greedy Selection of Top-K Unique Matches
    # We will iteratively find the best match and then remove the involved
    # points from future consideration.
    
    # Create a mutable copy of the similarity matrix
    sim_copy = sim.clone()

    sim_copy[~mask_flat] = 0
    sim_copy[:, ~mask_flat] = 0
    
    # Lists to store the flat indices of the selected pairs
    indices0_list = []
    indices1_list = []
    
    for _ in range(top_k):
        # Find the flat index of the current maximum similarity score
        # Note: If sim_copy becomes empty or all values are -1, this could fail.
        # A check on the max value could be added if that's a concern.
        max_val = sim_copy.max()
        if max_val <= -1: # Stop if no valid pairs are left
            break
            
        flat_idx = torch.argmax(sim_copy)
        
        # Convert the flat index to 2D indices (for feat0 and feat1)
        idx0 = (flat_idx // num_points).item()
        idx1 = (flat_idx % num_points).item()
        
        # Store the found indices
        indices0_list.append(idx0)
        indices1_list.append(idx1)
        
        # Invalidate the selected row and column to ensure uniqueness
        # This prevents these points from being selected again.
        sim_copy[idx0, :] = -1
        sim_copy[:, idx1] = -1

    # Convert lists to tensors
    indices0 = torch.tensor(indices0_list, device=feat0.device, dtype=torch.long)
    indices1 = torch.tensor(indices1_list, device=feat0.device, dtype=torch.long)

    # 4. Gather Results (same logic as before, but with the new indices)
    features0 = feat0_flat[indices0]
    features1 = feat1_flat[indices1]

    coords0_y = indices0 // W
    coords0_x = indices0 % W
    coords0 = torch.stack([coords0_x, coords0_y], dim=1)

    coords1_y = indices1 // W
    coords1_x = indices1 % W
    coords1 = torch.stack([coords1_x, coords1_y], dim=1)

    return coords0, features0, coords1, features1

def affine_to_params(affine_matrix: torch.Tensor) -> dict:
    """
    Convert an affine matrix to a dictionary of parameters.
    Input:
    - affine_matrix: (2,3) torch tensor
    Output:
    - params: dict
    """
    # Extract rotation + scale components
    a = affine_matrix[0, 0]
    b = affine_matrix[0, 1]
    c = affine_matrix[1, 0]
    d = affine_matrix[1, 1]

    # Scale (uniform)
    scale = torch.sqrt(a*a + c*c)

    # Rotation angle (in radians)
    theta = torch.arctan2(c, a)

    # Or convert to degrees
    angle_deg = torch.rad2deg(theta)
    return {'angle_rad': theta.cpu().item(), 'angle': angle_deg.cpu().item(), 
            'scale': scale.cpu().item(), 
            'translate_x': affine_matrix[0, 2].cpu().item(), 'translate_y': affine_matrix[1, 2].cpu().item()}

def params_to_affine_matrix(params: dict, return_np: bool = False) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a dictionary of parameters to an affine matrix.
    Input:
    - params: dict
    Output:
    - affine_matrix: (2,3) torch tensor
    """
    np_mat = np.array([[params['scale']*np.cos(np.deg2rad(params['angle'])), -params['scale']*np.sin(np.deg2rad(params['angle'])), params['translate_x']],
                        [params['scale']*np.sin(np.deg2rad(params['angle'])), params['scale']*np.cos(np.deg2rad(params['angle'])), params['translate_y']]])
    if return_np:
        return np_mat
    else:
        return torch.from_numpy(np_mat).to(dtype=torch.float32, device=params['translate_x'].device)

def compose_affine_matrices(affine_matrix1: torch.Tensor, affine_matrix2: torch.Tensor) -> torch.Tensor:
    """
    Compose two affine matrices.
    Input:
    - affine_matrix1: (2,3) torch tensor
    - affine_matrix2: (2,3) torch tensor
    Output:
    - affine_matrix: (2,3) torch tensor
    """
    if isinstance(affine_matrix1, np.ndarray):
        affine_matrix1 = torch.from_numpy(affine_matrix1).to(dtype=torch.float32)
    if isinstance(affine_matrix2, np.ndarray):
        affine_matrix2 = torch.from_numpy(affine_matrix2).to(dtype=torch.float32)
    mat1 = torch.vstack([affine_matrix1, torch.tensor([0,0,1])])
    mat2 = torch.vstack([affine_matrix2, torch.tensor([0,0,1])])
    return (mat1 @ mat2)[:2, :]

def inverse_affine_matrix(affine_matrix, return_homo: bool = False):
    """
    Inverse an affine matrix.
    Input:
    - affine_matrix: (2,3) torch tensor
    Output:
    - affine_matrix_inv: (2,3) torch tensor
    """
    res = None
    if isinstance(affine_matrix, np.ndarray):
        rot = affine_matrix[:2, :2]
        trans = affine_matrix[:2, 2]
        inv_rot = np.linalg.inv(rot)
        inv_trans = -inv_rot @ trans
        res = np.vstack([np.hstack([inv_rot, inv_trans.reshape(-1, 1)]), np.array([0,0,1])])
    else:
        if affine_matrix.shape == (2,3):
            affine_matrix = torch.vstack([affine_matrix, torch.tensor([0,0,1])])
        rot = affine_matrix[:2, :2]
        trans = affine_matrix[:2, 2]
        inv_rot = torch.linalg.inv(rot)
        inv_trans = -inv_rot @ trans
        res = torch.vstack([torch.hstack([inv_rot, inv_trans.reshape(-1, 1)]), torch.tensor([0,0,1])])

    if return_homo:
        return res
    else:
        return res[:2, :]

def apply_affine_2d_points(pts: torch.Tensor, affine_matrix_og: torch.Tensor, center: np.ndarray = None, device = None) -> torch.Tensor:
    """
    Apply an affine transformation to a set of 2D points (Nx2).
    Input:
    - pts: (N,2) torch tensor
    - affine_matrix: (2,3) torch tensor
    Output:
    - pts_transformed: (N,2) torch tensor
    """
    if isinstance(affine_matrix_og, np.ndarray):
        affine_matrix = torch.from_numpy(affine_matrix_og)
    else:
        affine_matrix = affine_matrix_og.clone()

    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    if device == None:
        device = pts.device
    if pts.dim() == 1:
        pts = pts.unsqueeze(0)

    if affine_matrix.shape == (2,2):
        affine_matrix = rot_to_affine(affine_matrix)

    # affine_matrix = inverse_affine_matrix(affine_matrix, return_homo=True)
    affine_matrix = torch.vstack([affine_matrix, torch.tensor([0,0,1])])

    if pts.shape[1] == 2:
        pts = torch.hstack([pts, torch.ones(pts.shape[0], 1).to(device)])

    pts = pts.to(torch.float32)
    affine_matrix = affine_matrix.to(device).to(torch.float32)

    # Scale the affine translation according to the image dimensions
    center = torch.from_numpy(center.astype(np.float32)).to(device)
    # It is not 2*center because when applying the matrix we first center the points
    affine_matrix[:2, 2] *= center

    center = torch.hstack([center, torch.zeros(1).to(device)])
    homogeneous_pts = (center + (affine_matrix @ (pts - center).T).T)
    homogeneous_pts = homogeneous_pts / homogeneous_pts[:, 2:3]
    return homogeneous_pts[:, :2]

def apply_affine_2d_map(tensor_input: torch.Tensor, affine_matrix: np.ndarray) -> torch.Tensor:
    """
    Apply affine transformation to embedding maps or images with bilinear sampling.
    Supports both single images (C,H,W) and batched images (B,C,H,W).
    Padding mode 'border' to avoid introducing NaNs.
    
    Args:
        tensor_input: Input tensor of shape (C,H,W) or (B,C,H,W)
        affine_matrix: 2x3 or 2x2 affine transformation matrix
        padding: Padding mode for grid_sample ('zeros', 'border', 'reflection')
    
    Returns:
        Transformed tensor with same shape as input
    """
    if isinstance(tensor_input, np.ndarray):
        if tensor_input.ndim == 3:  # (H,W,C) -> (C,H,W)
            tensor_input = torch.from_numpy(tensor_input).permute(2, 0, 1)
        else:  # (B,H,W,C) -> (B,C,H,W)
            tensor_input = torch.from_numpy(tensor_input).permute(0, 3, 1, 2)
    
    device = tensor_input.device
    dtype = tensor_input.dtype

    if affine_matrix.shape == (2, 2):
        affine_matrix = rot_to_affine(affine_matrix)

    if isinstance(affine_matrix, np.ndarray):
        affine_matrix = torch.from_numpy(affine_matrix).to(device)

    # Handle different input dimensions
    if tensor_input.dim() == 3:  # Single image (C,H,W)
        nchw = (1, tensor_input.shape[0], tensor_input.shape[1], tensor_input.shape[2])
        tensor_bchw = tensor_input.unsqueeze(0)
    elif tensor_input.dim() == 4:  # Batched images (B,C,H,W)
        nchw = tensor_input.shape
        tensor_bchw = tensor_input
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor_input.dim()}. Expected 3 (C,H,W) or 4 (B,C,H,W)")

    # Create grid for the entire batch
    params = affine_to_params(affine_matrix)

    transformed = TF.affine(tensor_bchw, params['angle'], 
                            [(tensor_input.shape[-1]//2)*params['translate_x'], (tensor_input.shape[-2]//2)*params['translate_y']], 
                            params['scale'], 0.0, interpolation=TF.InterpolationMode.BILINEAR).to(device)

    #########################################################################################
    # Deprecated                        
    # batch_size = tensor_bchw.shape[0]
    # grid = F.affine_grid(affine_matrix.unsqueeze(0).expand(batch_size, -1, -1), size=nchw, align_corners=False).to(torch.float32)
    
    # # Apply transformation
    # transformed = F.grid_sample(tensor_bchw, grid, mode='bilinear', padding_mode=padding, align_corners=False)
    #########################################################################################
    
    # Return with same shape as input
    if transformed.shape[0] == 1:
        return transformed.squeeze(0)
    else:
        return transformed

def rot_to_affine(affine_matrix: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert an affine matrix to a homogeneous matrix.
    """
    if isinstance(affine_matrix, np.ndarray):
        return np.hstack([affine_matrix, np.array([[0, 0]]).T])
    else:
        return torch.hstack([affine_matrix, torch.tensor([[0, 0]]).T.to(affine_matrix.device)])


def evaluate_corner_alignment(true_affine: np.ndarray, pred_affine: np.ndarray, 
                            img_shape: tuple, threshold: Union[float, List[float]] = 5.0) -> dict:
    """
    Evaluate alignment performance by comparing transformed image corners.
    
    Args:
        true_affine: Ground truth 2x3 affine transformation matrix
        pred_affine: Predicted 2x3 affine transformation matrix
        img_shape: (height, width) of the image
        threshold: Distance threshold(s) for accuracy calculation (in pixels). 
                  Can be a single float or a list of floats.
    
    Returns:
        dict: Contains distances, accuracy metrics, and other metrics
    """
    h, w = img_shape
    
    # Define the four corners of the image (top-left, top-right, bottom-left, bottom-right)
    corners = np.array([
        [0, 0],      # top-left
        [w-1, 0],    # top-right
        [0, h-1],    # bottom-left
        [w-1, h-1]   # bottom-right
    ], dtype=np.float32)
    
    # Transform corners with ground truth transformation
    corners_gt = apply_affine_2d_points(corners, true_affine, center=np.array([w/2, h/2]))
    
    # Transform corners with predicted transformation
    corners_pred = apply_affine_2d_points(corners, pred_affine, center=np.array([w/2, h/2]))
    
    # Compute distances between corresponding corners
    distances = np.linalg.norm(corners_gt - corners_pred, axis=1)
    
    # Convert threshold to list if it's a single value
    if isinstance(threshold, (int, float)):
        thresholds = [threshold]
    else:
        thresholds = threshold
    
    # Calculate accuracy for each threshold
    result = {
        'corners_original': corners,
        'corners_gt': corners_gt,
        'corners_pred': corners_pred,
        'distances': distances,
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'rms_error': np.sqrt(np.mean(distances**2)),
        'median_error': np.median(distances)
    }
    
    # Add accuracy metrics for each threshold
    for thresh in thresholds:
        accuracy = np.mean(distances < thresh) * 100
        result[f'accuracy@{thresh}'] = accuracy
    
    return result

def evaluate_corner_alignment_batch(true_affines: np.ndarray, pred_affines: np.ndarray, 
                                  img_shape: tuple, threshold: float = 5.0) -> dict:
    """
    Evaluate alignment performance by comparing transformed image corners for a batch.
    
    Args:
        true_affines: Ground truth 2x3 affine transformation matrices, shape (B, 2, 3)
        pred_affines: Predicted 2x3 affine transformation matrices, shape (B, 2, 3)
        img_shape: (height, width) of the image
        threshold: Distance threshold for accuracy calculation (in pixels)
    
    Returns:
        dict: Contains batch-averaged metrics
    """
    batch_size = true_affines.shape[0]
    h, w = img_shape
    
    all_accuracies = []
    all_mean_distances = []
    all_max_distances = []
    all_rms_errors = []
    
    for i in range(batch_size):
        result = evaluate_corner_alignment(true_affines[i], pred_affines[i], img_shape, threshold)
        all_accuracies.append(result['accuracy'])
        all_mean_distances.append(result['mean_distance'])
        all_max_distances.append(result['max_distance'])
        all_rms_errors.append(result['rms_error'])
    
    return {
        'accuracy': np.mean(all_accuracies),
        'mean_distance': np.mean(all_mean_distances),
        'max_distance': np.mean(all_max_distances),
        'rms_error': np.mean(all_rms_errors),
        'accuracy_std': np.std(all_accuracies),
        'mean_distance_std': np.std(all_mean_distances),
        'max_distance_std': np.std(all_max_distances),
        'rms_error_std': np.std(all_rms_errors),
        'threshold': threshold,
        'batch_size': batch_size
    }

if __name__ == "__main__":
    pass
