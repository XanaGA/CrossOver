import torch
import torch.nn.functional as F
import math
import torch.nn as nn

try:
    from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
    from romatch.models.encoders import CNNandDinov2
    from romatch.models.matcher import CosKernel, GP, Decoder, ConvRefiner, RegressionMatcher
except ImportError:
    print("romatch not found")

def transform_params_to_identity(params):
    """
    Convert transform params to identity, leaving the mask unchanged.
    Input:
    - params: dict
    Output:
    - params: dict
    """
    params["angle"] = torch.zeros_like(params["angle"])
    params["translate"] = [torch.zeros_like(params["translate"][0]), torch.zeros_like(params["translate"][1])]
    params["scale"] = torch.ones_like(params["scale"])
    return params

def create_grid(B, H, W, device):
    """Creates a normalized meshgrid [-1, 1]."""
    y_range = torch.linspace(-1, 1, H, device=device)
    x_range = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    # Stack to [B, 3, H*W] for matrix multiplication: (x, y, 1)
    grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    grid = grid.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1)
    return grid


def _build_affine_matrix(angle_deg, scale, translate_x, translate_y, H, W, device):
    """
    Builds a 3x3 Forward Affine Matrix in *pixel coordinates*:
    
        p' = T(translate) · T(center) · R(angle) · S(scale) · T(-center) · p

    Args:
        angle_deg: Tensor of shape (B,) in degrees.
        scale: Tensor of shape (B,)
        translate_x: Tensor of shape (B,)
        translate_y: Tensor of shape (B,)
        H, W: int
        device: torch.device
    """
    B = angle_deg.shape[0]

    # Convert to radians
    angle = angle_deg * math.pi / 180.0
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # Center of image
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    
    # We can construct the combined matrix directly to avoid 4x matmul overhead
    # Matrix M = [ [a, b, tx], [c, d, ty], [0, 0, 1] ]
    
    # 1. Rotation/Scale part
    # In Y-down system, this matrix rotates CLOCKWISE for positive angle
    a = cos_a * scale
    b = -sin_a * scale
    c = sin_a * scale
    d = cos_a * scale

    # 2. Translation part
    # The full chain: T_trans @ T_pos @ RS @ T_neg
    # The translation component (col 2) becomes: 
    #     RS * (-center) + center + translate
    #     = [a * -cx + b * -cy + cx + tx]
    #       [c * -cx + d * -cy + cy + ty]
    
    tx = (1 - a) * cx - b * cy + translate_x
    ty = (1 - d) * cy - c * cx + translate_y

    # Assemble into (B, 3, 3) tensor
    M = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
    
    M[:, 0, 0] = a
    M[:, 0, 1] = b
    M[:, 0, 2] = tx
    
    M[:, 1, 0] = c
    M[:, 1, 1] = d
    M[:, 1, 2] = ty
    
    M[:, 2, 2] = 1.0

    return M


def compute_gt_warp(params0, params1, H, W, device):
    """
    Computes dense warp from modality_0_noise -> modality_1_noise.
    
    Returns:
        warp_gt: [B, H, W, 2] Normalized coordinates in range [-1, 1].
                 (-1, -1) is top-left, (1, 1) is bottom-right.
        valid_mask: [B, H, W, 1] 1.0 if the warp points to a valid pixel, 0.0 otherwise.
    """

    B = params0["angle"].shape[0]

    # --- Extract parameters for batch ---
    a0 = params0["angle"].float().to(device)
    s0 = params0["scale"].float().to(device)
    tx0 = params0["translate"][0].float().to(device)
    ty0 = params0["translate"][1].float().to(device)

    a1 = params1["angle"].float().to(device)
    s1 = params1["scale"].float().to(device)
    tx1 = params1["translate"][0].float().to(device)
    ty1 = params1["translate"][1].float().to(device)

    # --- Build affine matrices in pixel coordinates ---
    # Note: These are still built in pixel space, which is correct for the math
    M0 = _build_affine_matrix(a0, s0, tx0, ty0, H, W, device)   # original -> noise0
    M1 = _build_affine_matrix(a1, s1, tx1, ty1, H, W, device)   # original -> noise1

    # Relative transform noise0 -> noise1
    M0_inv = torch.linalg.inv(M0.to(dtype=torch.float32))
    M_rel = M1 @ M0_inv                                    # (B,3,3)

    # --- Build pixel grid for noise0 image ---
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    ones = torch.ones_like(xs)

    grid = torch.stack([xs, ys, ones], dim=-1).to(dtype=torch.float32)     # [H,W,3]
    grid = grid.reshape(1, -1, 3).repeat(B, 1, 1)   # [B, H*W, 3]

    # Apply transform
    warped = grid @ M_rel.permute(0,2,1)            # [B, H*W, 3]
    warped = warped[...,:2]                         # pixel coordinates

    warp_gt = warped.reshape(B, H, W, 2)

    # Validity mask: Check bounds in PIXEL space (before normalization)
    x = warp_gt[...,0]
    y = warp_gt[...,1]

    # Check validity (0 to W-1)
    valid = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    valid = valid.float().unsqueeze(-1)

    # --- NORMALIZE TO [-1, 1] ---
    # Formula: x_norm = 2 * (x / (W - 1)) - 1
    # This matches align_corners=True behavior in grid_sample
    warp_gt[..., 0] = 2 * warp_gt[..., 0] / (W - 1) - 1
    warp_gt[..., 1] = 2 * warp_gt[..., 1] / (H - 1) - 1

    return warp_gt, valid
