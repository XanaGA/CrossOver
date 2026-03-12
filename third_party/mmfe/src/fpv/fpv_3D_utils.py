from typing import List
import torch
import torch.nn.functional as F
import numpy as np
import math

# Constants for SNAP grid definition (You may want to make these arguments)
FRUSTUM_WIDTH_M = 5  # Width of the local map in meters (X axis)
FRUSTUM_DEPTH_M = 8   # Depth of the local map in meters (Z axis)
CELL_SIZE_M = 0.05        # Resolution of the neural map
HEIGHT_MIN_M = -2.0      # Min height relative to camera
HEIGHT_MAX_M = 2.0       # Max height relative to camera
NUM_HEIGHT_PLANES = 35   # K points in the vertical column
DEPTH_MIN = 0.1          # Min depth for log-distribution (meters)
DEPTH_MAX = FRUSTUM_DEPTH_M         # Max depth for log-distribution (meters)

import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass

from aria_mmfe.aria_images.aria_cv_tools import load_aria_vignette, mask_from_vignette, undistort_image_fisheye

# Fixed Aria Camera Constants
ARIA_IMAGE_WIDTH = 704
ARIA_IMAGE_HEIGHT = 704
ARIA_FOCAL_LENGTH = 297.6375381033778

@dataclass
class FrustumConfig:
    """Configuration for the local frustum grid."""
    width_m: float = FRUSTUM_WIDTH_M         # Width (Left-Right)
    depth_m: float = FRUSTUM_DEPTH_M          # Depth (Forward)
    cell_size_m: float = CELL_SIZE_M       # Resolution
    h_min_m: float = HEIGHT_MIN_M          # Min height (relative to camera)
    h_max_m: float = HEIGHT_MAX_M          # Max height
    num_height_planes: int = NUM_HEIGHT_PLANES    # Vertical points per cell
    depth_min_log: float = DEPTH_MIN     # Min depth for scoring
    depth_max_log: float = DEPTH_MAX    # Max depth for scoring

@dataclass
class FrustumCoordinates:
    """
    Stores pre-computed geometric data for the frustum grid.
    All tensors are fixed and reused across batches.
    """
    # The original 3D grid in the Local Ground Frame (X=Right, Y=Forward, Z=Up)
    # Shape: (1, GridDepth, GridWidth, 3)
    coords_3d_local: torch.Tensor
    
    # The 3D depths of the points in the Camera Frame (Z-Forward)
    # Shape: (1, 1, GridDepth, GridWidth, NumHeights) - reshaped for broadcasting
    cam_depths: torch.Tensor

    # Projected 2D pixel coordinates normalized to [-1, 1]
    # Shape: (1, GridDepth, GridWidth, NumHeights, 2)
    coords_uv_norm: torch.Tensor

    # Mask indicating if a point projects inside the image plane
    # Shape: (1, 1, GridDepth, GridWidth, NumHeights)
    valid_mask_3D: torch.Tensor

    # Projected 2D pixel coordinates normalized to [-1, 1]
    # Shape: (1, GridDepth, GridWidth, NumHeights, 2)
    coords_xy_local: torch.Tensor

    # Mask indicating if a point projects inside the image plane
    # Shape: (1, 1, GridDepth, GridWidth, NumHeights)
    valid_mask_xy: torch.Tensor

    def to(self, device: torch.device) -> "FrustumCoordinates":
        return FrustumCoordinates(
            coords_3d_local=self.coords_3d_local.to(device),
            cam_depths=self.cam_depths.to(device),
            coords_uv_norm=self.coords_uv_norm.to(device),
            valid_mask_3D=self.valid_mask_3D.to(device),
            coords_xy_local=self.coords_xy_local.to(device),
            valid_mask_xy=self.valid_mask_xy.to(device),
        )
    
@dataclass
class FrustumData:
    """Holds the computed frustum data."""
    features: torch.Tensor        # (B, C, GridDepth, GridWidth) - Max-pooled features
    coords_proj_xy: torch.Tensor  # (B, GridDepth, GridWidth, 2) - Projected in XY
    valid_mask_xy: torch.Tensor  # (B, GridDepth, GridWidth) - Valid mask in XY

# -----------------------------------------------------------------------------
# 1. Projected Frustum Computation (SNAP Ground-Level Encoder)
# -----------------------------------------------------------------------------

def precompute_frustum_grid(config: FrustumConfig, device: torch.device) -> FrustumCoordinates:
    """
    Generates the 3D grid, projects it to 2D, and stores coordinates.
    This should be called once (e.g., in __init__).
    """

    vignette_img = load_aria_vignette(rotate=False, as_pil=False, binary=False)
    mask = mask_from_vignette(vignette_img, binary=False)
    # Undistort the mask
    undistorted_mask, _ = undistort_image_fisheye(mask)

    # --- A. Define Local Grid (Local Ground Frame: Z is UP) ---
    # X: Right, Y: Forward, Z: Up (relative to camera center)
    
    xs = torch.arange(-config.width_m / 2, config.width_m / 2, config.cell_size_m, device=device)
    ys = torch.arange(config.depth_min_log, config.depth_m + config.depth_min_log, config.cell_size_m, device=device) # Start 1m forward
    zs = torch.linspace(config.h_min_m, config.h_max_m, config.num_height_planes, device=device)
    
    grid_depth, grid_width = len(ys), len(xs)
    
    # Meshgrid (Y=Depth, X=Width)
    local_y, local_x = torch.meshgrid(ys, xs, indexing='ij') 
    
    # Expand to 3D Volume
    # Dimensions: (GridDepth, GridWidth, NumHeights)
    pts_x = local_x.unsqueeze(-1).expand(-1, -1, config.num_height_planes) # Local X (Right)
    pts_y = local_y.unsqueeze(-1).expand(-1, -1, config.num_height_planes) # Local Y (Forward)
    pts_z = zs.view(1, 1, -1).expand(grid_depth, grid_width, -1)           # Local Z (Up)
    
    # Store Local Coords (X, Y, Z=0) for final output alignment
    # We store the 'base' of the column for the BEV map coordinates
    coords_3d_local = torch.stack([pts_x, pts_y, pts_z], dim=-1) # (GridDepth, GridWidth, 3)

    # --- B. Convert to Camera Optical Frame (CV Convention) ---
    # CV Convention: X_cam=Right, Y_cam=Down, Z_cam=Forward
    # Mapping from Local (X, Y_fwd, Z_up):
    # X_cam = Local X
    # Y_cam = -Local Z (Up becomes Down)
    # Z_cam = Local Y (Forward)
    
    cam_x = pts_x
    cam_y = -pts_z
    cam_z = pts_y # This is the depth we need for weighting
    
    # --- C. Pinhole Projection ---
    fx = ARIA_FOCAL_LENGTH
    fy = ARIA_FOCAL_LENGTH
    cx = ARIA_IMAGE_WIDTH / 2.0
    cy = ARIA_IMAGE_HEIGHT / 2.0
    
    # Avoid division by zero
    depths = cam_z.clamp(min=0.1)
    
    u = (cam_x * fx) / depths + cx
    v = (cam_y * fy) / depths + cy
    
    # --- D. Normalize and Create Mask ---
    undistorted_mask = torch.tensor(undistorted_mask, device=device)

    u_long = u.long()
    v_long = v.long()

    valid_mask_3D = (
        (u_long >= 0) & (u_long < ARIA_IMAGE_WIDTH) &
        (v_long >= 0) & (v_long < ARIA_IMAGE_HEIGHT)
    )

    # Clamp indices to avoid OOB access
    u_safe = u_long.clamp(0, ARIA_IMAGE_WIDTH - 1)
    v_safe = v_long.clamp(0, ARIA_IMAGE_HEIGHT - 1)

    # Now this is SAFE
    valid_mask_3D = valid_mask_3D & (undistorted_mask[v_safe, u_safe] > 0)

    valid_mask_3D = valid_mask_3D.float()

    # Normalize to [-1, 1] for grid_sample
    norm_u = (u / (ARIA_IMAGE_WIDTH - 1)) * 2 - 1
    norm_v = (v / (ARIA_IMAGE_HEIGHT - 1)) * 2 - 1
    
    # Stack UVs: (GridDepth, GridWidth, NumHeights, 2)
    coords_uv_norm = torch.stack([norm_u, norm_v], dim=-1)

    # Only 2D cell with some valid point on top are considered valid
    valid_mask_xy = valid_mask_3D.any(dim=2)
    
    # Store depths for log-weighting: (GridDepth, GridWidth, NumHeights)
    cam_depths = cam_z

    return FrustumCoordinates(
        coords_3d_local=coords_3d_local,
        cam_depths=cam_depths,
        coords_uv_norm=coords_uv_norm,
        valid_mask_3D=valid_mask_3D,
        coords_xy_local=coords_3d_local[:, :, 0, :2],
        valid_mask_xy=valid_mask_xy
    )

# def compute_projected_frustum_mlp(
#         fpv_depth_logits: torch.Tensor,  # (B, N=1, D, H, W)
#         fpv_features: torch.Tensor,      # (B, N=1, C, H, W)
#         coords: FrustumCoordinates,
#         config: FrustumConfig,
#         mlp: torch.nn.Module,            # REQUIRED
#     ) -> List[FrustumData]:

#     assert mlp is not None, "SNAP requires an MLP"

#     B, N, C, H, W = fpv_features.shape
#     assert N == 1, "This function is monocular SNAP"

#     D = fpv_depth_logits.shape[2]
#     T = B * N

#     device = fpv_features.device
#     coords = coords.to(device)

#     # ---------------------------------------------------------
#     # 1. Flatten batch
#     # ---------------------------------------------------------
#     feats = fpv_features.view(T, C, H, W)
#     depth_logits = fpv_depth_logits.view(T, D, H, W)

#     # ---------------------------------------------------------
#     # 2. Project frustum points into image
#     # ---------------------------------------------------------
#     grid_uv = coords.coords_uv_norm.expand(T, -1, -1, -1, -1)
#     grid_uv_flat = grid_uv.reshape(T, -1, 1, 2)

#     # ---------------------------------------------------------
#     # 3. Sample image features
#     # ---------------------------------------------------------
#     sampled_feats = F.grid_sample(
#         feats, grid_uv_flat,
#         align_corners=True,
#         padding_mode="zeros",
#     )
#     sampled_feats = sampled_feats.view(
#         T, C,
#         grid_uv.shape[1],
#         grid_uv.shape[2],
#         grid_uv.shape[3],
#     )

#     # ---------------------------------------------------------
#     # 4. Sample depth logits (NO softmax)
#     # ---------------------------------------------------------
#     sampled_depth_logits = F.grid_sample(
#         depth_logits, grid_uv_flat,
#         align_corners=True,
#         padding_mode="zeros",
#     )
#     sampled_depth_logits = sampled_depth_logits.view(
#         T, D,
#         grid_uv.shape[1],
#         grid_uv.shape[2],
#         grid_uv.shape[3],
#     )

#     # ---------------------------------------------------------
#     # 5. Log-depth interpolation (correct as-is)
#     # ---------------------------------------------------------
#     point_depths = coords.cam_depths[None].expand_as(
#         sampled_depth_logits[:, :1]
#     )

#     log_min = math.log(config.depth_min_log)
#     log_max = math.log(config.depth_max_log)

#     t = (
#         torch.log(point_depths.clamp(
#             config.depth_min_log,
#             config.depth_max_log
#         )) - log_min
#     ) / (log_max - log_min)
#     t = t.clamp(0, 1)

#     flat_logits = sampled_depth_logits.permute(0, 2, 3, 4, 1).reshape(T, -1, D)
#     flat_t = t.view(T, -1)

#     idx_f = (flat_t * (D - 1)).floor().long()
#     idx_c = (flat_t * (D - 1)).ceil().long()
#     a = flat_t * (D - 1) - idx_f.float()

#     s_f = torch.gather(flat_logits, 2, idx_f.unsqueeze(-1)).squeeze(-1)
#     s_c = torch.gather(flat_logits, 2, idx_c.unsqueeze(-1)).squeeze(-1)

#     point_logits = (1 - a) * s_f + a * s_c
#     point_logits = point_logits.view(
#         T, 1,
#         grid_uv.shape[1],
#         grid_uv.shape[2],
#         grid_uv.shape[3],
#     )

#     # ---------------------------------------------------------
#     # 6. SNAP point MLP
#     # ---------------------------------------------------------
#     valid = coords.valid_mask_3D.expand_as(point_logits)

#     xk = torch.cat([sampled_feats, point_logits], dim=1)
#     xk = xk * valid

#     xk = xk.permute(0, 2, 3, 4, 1)   # (T, Dg, Wg, K, C+1)
#     xk = mlp(xk)                    # (T, Dg, Wg, K, C_out)
#     xk = xk.permute(0, 4, 1, 2, 3)

#     xk = xk.masked_fill(~valid, float("-inf"))

#     # ---------------------------------------------------------
#     # 7. Vertical max pooling (SNAP Eq. 2)
#     # ---------------------------------------------------------
#     bev = torch.max(xk, dim=-1).values   # (T, C_out, Dg, Wg)

#     bev = bev.view(B, N, bev.shape[1], bev.shape[2], bev.shape[3])

#     # ---------------------------------------------------------
#     # 8. Package output
#     # ---------------------------------------------------------
#     outputs = []
#     for b in range(B):
#         outputs.append(
#             FrustumData(
#                 features=bev[b],
#                 coords_proj_xy=coords.coords_xy_local[None],
#                 valid_mask_xy=coords.valid_mask_xy[None],
#             )
#         )

#     return outputs


# def compute_projected_frustum_softmax(
#     fpv_depth_logits: torch.Tensor, # (B, N, D, H, W)
#     fpv_features: torch.Tensor,     # (B, N, C, H, W)
#     coords: FrustumCoordinates,
#     config: FrustumConfig,
# ) -> List[FrustumData]:
#     """
#     Computes Query Neural Maps (M^Q) for a batch of sequences.
#     """
#     B, N, C, H, W = fpv_features.shape
#     D = fpv_depth_logits.shape[2]
#     TotalImages = B * N

#     coords = coords.to(fpv_features.device)
    
#     # 1. Flatten Batch and Sequence dimensions
#     fpv_feats_flat = fpv_features.view(TotalImages, C, H, W)
#     depth_logits_flat = fpv_depth_logits.view(TotalImages, D, H, W)
    
#     # 2. Expand pre-computed coords to TotalImages
#     # Coords are (1, ...), we expand to (TotalImages, ...)
#     grid_uv = coords.coords_uv_norm.expand(TotalImages, -1, -1, -1, -1)
    
#     # Flatten spatial dims for grid_sample: (TotalImages, TotalPoints, 1, 2)
#     grid_uv_flat = grid_uv.reshape(TotalImages, -1, 1, 2).to(fpv_features.device)
    
#     # --- 3. Feature Sampling ---
#     sampled_feats = F.grid_sample(fpv_feats_flat, grid_uv_flat, align_corners=True, padding_mode='zeros')
#     # Reshape back: (TotalImages, C, GridDepth, GridWidth, NumHeights)
#     sampled_feats = sampled_feats.view(TotalImages, C, grid_uv.shape[1], grid_uv.shape[2], grid_uv.shape[3])
    
#     # --- 4. Depth Probability Sampling ---
#     depth_probs = F.softmax(depth_logits_flat, dim=1)
#     sampled_depth_dist = F.grid_sample(depth_probs, grid_uv_flat, align_corners=True, padding_mode='zeros')
#     # Reshape: (TotalImages, D, GridDepth, GridWidth, NumHeights)
#     sampled_depth_dist = sampled_depth_dist.view(TotalImages, D, grid_uv.shape[1], grid_uv.shape[2], grid_uv.shape[3])

#     # --- 5. Log-Space Depth Interpolation ---
#     point_depths = coords.cam_depths[None].expand(TotalImages, -1, -1, -1, -1)
    
#     log_min = math.log(config.depth_min_log)
#     log_max = math.log(config.depth_max_log)
    
#     t_vals = (torch.log(point_depths.clamp(min=config.depth_min_log, max=config.depth_max_log)) - log_min) / (log_max - log_min)
#     t_vals = t_vals.clamp(0, 1).to(fpv_features.device) # (TotalImages, 1, GridDepth, GridWidth, NumHeights)
    
#     # Manual 1D interpolation
#     flat_dist = sampled_depth_dist.permute(0, 2, 3, 4, 1).reshape(TotalImages, -1, D)
#     flat_t = t_vals.view(TotalImages, -1)
    
#     float_idx = flat_t * (D - 1)
#     idx_floor = float_idx.floor().long()
#     idx_ceil = float_idx.ceil().long()
#     alpha = float_idx - idx_floor.float()
    
#     val_floor = torch.gather(flat_dist, 2, idx_floor.unsqueeze(-1)).squeeze(-1)
#     val_ceil = torch.gather(flat_dist, 2, idx_ceil.unsqueeze(-1)).squeeze(-1)
    
#     point_probs = (1 - alpha) * val_floor + alpha * val_ceil
#     point_probs = point_probs.view(TotalImages, 1, grid_uv.shape[1], grid_uv.shape[2], grid_uv.shape[3])
    
#     # --- 6. Masking and Vertical Pooling ---
#     valid_mask_3D = coords.valid_mask_3D[None].expand(TotalImages, -1, -1, -1, -1)
#     weighted_feats = sampled_feats * point_probs * valid_mask_3D
    
#     # [cite_start]Max pool along height column 
#     pooled_features, _ = torch.max(weighted_feats, dim=-1) # (TotalImages, C, GridDepth, GridWidth)
    
#     # --- 7. Structure Output ---
#     # We have flat tensors. We need to split them back into Scenes (B) and Images (N)
    
#     # Reshape back to (B, N, ...)
#     batch_features = pooled_features.view(B, N, C, pooled_features.shape[2], pooled_features.shape[3])
     
#     # Expand 3D local coords to (B, N, ...)
#     # coords.coords_3d_local is (1, GridDepth, GridWidth, 3)
    
#     scene_list = []
    
#     for b in range(B):
#         # Slice per scene (batch item)
#         # Features: (N, C, Depth, Width)
#         # Coords 3D: (N, Depth, Width, 3)
#         # Coords 2D: (N, Depth, Width, Heights, 2)
        
#         data = FrustumData(
#             features=batch_features[b],
#             coords_proj_xy=coords.coords_xy_local[None].expand(N, -1, -1, -1),
#             valid_mask_xy=coords.valid_mask_xy[None].expand(N, -1, -1, -1)
#         )
#         scene_list.append(data)
        
#     return scene_list

def compute_projected_frustum(
    fpv_depth_logits: torch.Tensor, # (B, N, D, H, W)
    fpv_features: torch.Tensor,     # (B, N, C, H, W)
    coords: FrustumCoordinates,
    config: FrustumConfig,
    gt_depth: torch.Tensor = None,
) -> List[FrustumData]:
    """
    Computes Query Neural Maps (M^Q) for a batch of sequences.
    """
    temperature = 0.5
    B, N, C, H, W = fpv_features.shape
    D = fpv_depth_logits.shape[2]
    TotalImages = B * N

    coords = coords.to(fpv_features.device)
    
    # 1. Flatten Batch and Sequence dimensions
    fpv_feats_flat = fpv_features.view(TotalImages, C, H, W)
    depth_logits_flat = fpv_depth_logits.view(TotalImages, D, H, W)
    
    # 2. Expand pre-computed coords to TotalImages
    # Coords are (1, ...), we expand to (TotalImages, ...)
    grid_uv = coords.coords_uv_norm.expand(TotalImages, -1, -1, -1, -1)
    
    # Flatten spatial dims for grid_sample: (TotalImages, TotalPoints, 1, 2)
    grid_uv_flat = grid_uv.reshape(TotalImages, -1, 1, 2).to(fpv_features.device)
    
    # --- 3. Feature Sampling ---
    sampled_feats = F.grid_sample(fpv_feats_flat, grid_uv_flat, align_corners=True, padding_mode='zeros')
    # Reshape back: (TotalImages, C, GridDepth, GridWidth, NumHeights)
    sampled_feats = sampled_feats.view(TotalImages, C, grid_uv.shape[1], grid_uv.shape[2], grid_uv.shape[3])
    
    # --- 4. Depth Probability Sampling ---
    # depth_probs = F.softmax(depth_logits_flat / temperature, dim=1)
    # gt_depth: (B, N, H, W) → flatten to (TotalImages, H, W)
    if gt_depth is not None:
        # Flatten like everything else
        gt_depth_flat = gt_depth.view(TotalImages, H, W)

        # Clamp to valid range
        gt_depth_clamped = gt_depth_flat.clamp(
            min=config.depth_min_log,
            max=config.depth_max_log
        )

        # Log-space normalization (EXACTLY like your code)
        log_min = math.log(config.depth_min_log)
        log_max = math.log(config.depth_max_log)

        t = (torch.log(gt_depth_clamped) - log_min) / (log_max - log_min)
        t = t.clamp(0, 1)

        # Convert to depth bin index
        gt_depth_idx = torch.round(t * (D - 1)).long()

        depth_probs = torch.zeros(
            TotalImages, D, H, W,
            device=fpv_features.device
        )

        depth_probs.scatter_(1, gt_depth_idx.unsqueeze(1), 1.0)
    else:
        depth_probs = F.softmax(depth_logits_flat / temperature, dim=1)
    sampled_occupancy = F.grid_sample(depth_probs, grid_uv_flat, align_corners=True, padding_mode='zeros')
    # Reshape: (TotalImages, D, GridDepth, GridWidth, NumHeights)
    sampled_occupancy = sampled_occupancy.view(TotalImages, D, grid_uv.shape[1], grid_uv.shape[2], grid_uv.shape[3])

    # --- 5. Log-Space Depth Interpolation ---
    point_depths = coords.cam_depths[None].expand(TotalImages, -1, -1, -1, -1)
    
    log_min = math.log(config.depth_min_log)
    log_max = math.log(config.depth_max_log)
    
    t_vals = (torch.log(point_depths.clamp(min=config.depth_min_log, max=config.depth_max_log)) - log_min) / (log_max - log_min)
    t_vals = t_vals.clamp(0, 1).to(fpv_features.device) # (TotalImages, 1, GridDepth, GridWidth, NumHeights)
    
    # Manual 1D interpolation
    flat_occ = sampled_occupancy.permute(0, 2, 3, 4, 1).reshape(TotalImages, -1, D)
    flat_t = t_vals.view(TotalImages, -1)
    
    float_idx = flat_t * (D - 1)
    idx_floor = float_idx.floor().long()
    idx_ceil = float_idx.ceil().long()
    alpha = float_idx - idx_floor.float()
    
    val_floor = torch.gather(flat_occ, 2, idx_floor.unsqueeze(-1)).squeeze(-1)
    val_ceil = torch.gather(flat_occ, 2, idx_ceil.unsqueeze(-1)).squeeze(-1)
    
    point_probs = (1 - alpha) * val_floor + alpha * val_ceil
    point_probs = point_probs.view(TotalImages, 1, grid_uv.shape[1], grid_uv.shape[2], grid_uv.shape[3])
    
    # --- 6. Masking and Vertical Pooling ---
    valid_mask_3D = coords.valid_mask_3D[None].expand(TotalImages, -1, -1, -1, -1)
    weighted_feats = sampled_feats * point_probs * valid_mask_3D
    
    # [cite_start]Max pool along height column 
    pooled_features, _ = torch.max(weighted_feats, dim=-1) # (TotalImages, C, GridDepth, GridWidth)
    
    # --- 7. Structure Output ---
    # We have flat tensors. We need to split them back into Scenes (B) and Images (N)
    
    # Reshape back to (B, N, ...)
    batch_features = pooled_features.view(B, N, C, pooled_features.shape[2], pooled_features.shape[3])
     
    # Expand 3D local coords to (B, N, ...)
    # coords.coords_3d_local is (1, GridDepth, GridWidth, 3)
    
    scene_list = []
    
    for b in range(B):
        # Slice per scene (batch item)
        # Features: (N, C, Depth, Width)
        # Coords 3D: (N, Depth, Width, 3)
        # Coords 2D: (N, Depth, Width, Heights, 2)
        
        data = FrustumData(
            features=batch_features[b],
            coords_proj_xy=coords.coords_xy_local[None].expand(N, -1, -1, -1),
            valid_mask_xy=coords.valid_mask_xy[None].expand(N, -1, -1, -1)
        )
        scene_list.append(data)
        
    return scene_list


# -----------------------------------------------------------------------------
# 2. Transform Frustums (World/Floorplan Coords)
# -----------------------------------------------------------------------------

def transform_xy_local_to_world(xy_local, pose):
    """
    xy_local: (..., 2)
    pose: (..., 3) -> (tx, ty, theta)
    """
    tx, ty, theta = pose[..., 0], pose[..., 1], pose[..., 2]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    x, y = xy_local[..., 0], xy_local[..., 1]

    x_w = cos_t[..., None, None] * x - sin_t[..., None, None] * y + tx[..., None, None]
    y_w = sin_t[..., None, None] * x + cos_t[..., None, None] * y + ty[..., None, None]

    return torch.stack([x_w, y_w], dim=-1)


def world_to_pixel_fpv(pts_world, params, index_params):
    """
    Convert world coordinates (..., 2) → pixel coordinates (..., 2)
    Supports batched params of shape (B,)
    """
    is_torch = isinstance(pts_world, torch.Tensor)

    x = pts_world[..., 0]
    y = pts_world[..., 1]

    px = params["x_offset"][index_params] + (x - params["x_min_w"][index_params]) * params["scale"][index_params]

    py = params["y_offset"][index_params] + (y - params["y_min_w"][index_params]) * params["scale"][index_params]

    # flip y-axis (h is batched!)
    py = (params["h"][index_params] - 1) - py

    if is_torch:
        return torch.stack((px, py), dim=-1).to(torch.int32)
    else:
        return np.stack((px, py), axis=-1).astype(np.int32)



def transform_fustrums_to_floorplan(
    frustums,               # List[FrustumData], length B
    poses: torch.Tensor,    # (B, N_poses, 3)
    params: dict
):
    """
    Transforms frustum coords_xy_local into floorplan pixel coordinates.
    Returns: List[List[FrustumData]]
    """

    B = len(frustums)
    assert B == poses.shape[0], "Mismatch between frustums and poses batch size"

    results = []

    for b in range(B):
        frustum_b = frustums[b]
        poses_b = poses[b]  # (N_poses, 3)

        # ---- Local → World ----
        xy_world = transform_xy_local_to_world(
            frustum_b.coords_proj_xy, poses_b
        )  # (D, W, 2)

        # ---- World → Pixel ----
        xy_pixel = world_to_pixel_fpv(xy_world, params, b)
        # xy_pixel = xy_world

        # ---- Create transformed frustum ----
        transformed_frustum = FrustumData(
            features=frustum_b.features,
            coords_proj_xy=xy_pixel,
            valid_mask_xy=frustum_b.valid_mask_xy,
        )

        results.append(transformed_frustum)

    return results


# -----------------------------------------------------------------------------
# 3. Random Pose Sampling
# -----------------------------------------------------------------------------

def sample_random_poses(
    n_poses: int, 
    floorplan_params: dict, 
    device: torch.device,
) -> torch.Tensor:
    
    # Use a variable to ensure all tensors have the same first dimension
    batch_size = len(floorplan_params["scale"])

    # Helper function to ensure tensors are (B, 1) for easy broadcasting
    def prepare(x):
        if torch.is_tensor(x):
            return x.view(batch_size, 1).to(device)
        return x

    # Prepare all parameters
    W_fp = prepare(floorplan_params['w'])
    H_fp = prepare(floorplan_params['h'])
    scale = prepare(floorplan_params['scale'])
    x_min = prepare(floorplan_params['x_min_w'])
    y_min = prepare(floorplan_params['y_min_w'])
    
    # Calculate bounds - these will now be (B, 1)
    x_max = x_min + (W_fp / scale)
    y_max = y_min + (H_fp / scale)

    # Sample (B, N)
    # Both tx and ty will now correctly broadcast (B, N) * (B, 1) + (B, 1)
    tx = torch.rand((batch_size, n_poses), device=device) * (x_max - x_min) + x_min
    ty = torch.rand((batch_size, n_poses), device=device) * (y_max - y_min) + y_min
    th = torch.rand((batch_size, n_poses), device=device) * 2 * np.pi - np.pi
    
    return torch.stack([tx, ty, th], dim=-1)


def expand_neg_fustrums(frustums: List[FrustumData], n_poses: int) -> List[FrustumData]:
    results = []
    for frustum in frustums:
        results.append(FrustumData(
            features=frustum.features,
            coords_proj_xy=frustum.coords_proj_xy.unsqueeze(1).repeat(1, n_poses, 1, 1, 1),
            valid_mask_xy=frustum.valid_mask_xy.unsqueeze(1).repeat(1, n_poses, 1, 1, 1),
        ))
    return results

# -----------------------------------------------------------------------------
# 4. Updated Validation Step
# -----------------------------------------------------------------------------
"""
def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
    floorplan = batch["modality_0"]
    fpv_images = batch["fpv_dict"]["images"].to(self.device)
    fpv_params = batch["fpv_dict"]["params"]

    # 1. Create Frustum Config from Params
    config = FrustumConfig.from_params(fpv_params)

    with torch.no_grad():
        floorplan_feats = self.floorplan_encoder(floorplan)

    fpv_feats = self.encode_image(floorplan) 
    fpv_depth = self.encode_image(floorplan, moge_head="depth")   

    # 2. Compute Frustum with Z-UP & Config
    frustum_data = compute_projected_frustrum(fpv_depth, fpv_feats, config)

    # 3. Transform (Poses are pixels)
    gt_poses = batch["fpv_dict"]["poses_floorplan"] # (B, 1, 3)
    pos_grid = transform_fustrums(frustum_data, gt_poses, fpv_params)

    # 4. Negative Poses (Random Pixels)
    neg_poses = sample_random_poses(10, fpv_params, self.device)
    neg_grid = transform_fustrums(frustum_data, neg_poses, fpv_params)

    loss = self.criterion(floorplan_feats, frustum_data.features, pos_grid, neg_grid)
    return loss
"""