import torch
import torch.nn.functional as F
import torchvision.transforms as T
import math
from typing import Union, List, Tuple, Optional

def make_valid_mask(params, device="cpu", dtype=torch.float32):
    """
    Generate a binary mask (1=valid, 0=invalid) after applying the given affine
    to the input image. This matches how the image itself was transformed.

    params: dict with affine parameters ('angle','translate','scale','shear','image_size')
    """
    W, H = params["image_size"]

    # mask has 1 everywhere
    mask = torch.ones(1, H, W, device=device, dtype=dtype)

    # Apply same affine as to the image
    mask_t = T.functional.affine(
        mask,
        angle=params["angle"],
        translate=params["translate"],
        scale=params["scale"],
        shear=params["shear"],
        interpolation=T.functional.InterpolationMode.NEAREST,  # keep it binary
        fill=0,
    )
    return mask_t


def _pixel_to_norm(coords_pixel: torch.Tensor, size: int, align_corners: bool):
    """
    coords_pixel: arbitrary real-valued pixel coordinates in feature-space indexing (0 .. size-1)
    returns normalized coords in [-1,1]
    """
    if align_corners:
        # normalized coordinate mapping when align_corners=True
        return 2.0 * coords_pixel / (size - 1) - 1.0
    else:
        # normalized coordinate mapping when align_corners=False
        # PyTorch's grid_sample expects normalized coords where -1 and 1 map to centers of corner pixels
        # mapping for fractional pixel index x is:
        return 2.0 * (coords_pixel + 0.5) / size - 1.0


def compute_inverse_affine_grid(
    params: dict,
    feature_size: Tuple[int, int],
    image_size: Tuple[int, int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    align_corners: bool = False
) -> torch.Tensor:
    """
    Given sampled image-space affine params (as returned by PairRandomAffine._sample),
    compute a grid suitable for grid_sample that samples from the *transformed* feature map
    to produce an aligned feature map in original image coordinates (i.e. apply inverse warp
    to the transformed features).
    Returns grid shaped (1, h_feat, w_feat, 2), ready for F.grid_sample.

    - params: dict with keys 'angle' (deg), 'translate' (tx,ty in pixels), 'scale' (s),
              'image_size'=(W,H)
    - feature_size: (h_feat, w_feat)
    - image_size: (W,H)  <- must match params['image_size'] normally.
    - align_corners: passed to grid_sample (default False to match torch default)
    """

    # sizes
    h_feat, w_feat = feature_size
    W_img, H_img = image_size  # note order
    W_img = W_img if isinstance(W_img, int) else W_img[0]
    H_img = H_img if isinstance(H_img, int) else H_img[0]

    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32

    # sampling stride from feature to image
    Sx = float(W_img) / float(w_feat)
    Sy = float(H_img) / float(h_feat)

    # feature-grid indices (i: rows -> y, j: cols -> x)
    # grid_x: shape (h_feat, w_feat) values 0..w_feat-1
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h_feat, device=device, dtype=dtype),
        torch.arange(w_feat, device=device, dtype=dtype),
        indexing='ij'
    )

    # map feature coords -> image pixel coords (center-align)
    # x_image = (j + 0.5) * Sx - 0.5
    x_img = (grid_x + 0.5) * Sx - 0.5
    y_img = (grid_y + 0.5) * Sy - 0.5

    # center used by torchvision.transforms.functional.affine
    # We use (W-1)/2, (H-1)/2 which matches the usual pixel-centered rotation center.
    cx = (W_img - 1.) / 2.0
    cy = (H_img - 1.) / 2.0

    angle_rad = math.radians(params.get('angle', 0.0))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    s = float(params.get('scale', 1.0))
    tx, ty = params.get('translate', (0.0, 0.0))

    # apply forward image-space transform to the original-image coordinates to locate
    # where they appear in the transformed image (this gives the sampling locations inside the transformed image)
    x_rel = x_img - cx
    y_rel = y_img - cy

    x_rot = cos_a * x_rel - sin_a * y_rel
    y_rot = sin_a * x_rel + cos_a * y_rel

    x_src_img = s * x_rot + cx + tx
    y_src_img = s * y_rot + cy + ty

    # map source image pixel coords -> source feature coords (inverse of earlier)
    x_src_feat = (x_src_img + 0.5) / Sx - 0.5
    y_src_feat = (y_src_img + 0.5) / Sy - 0.5

    # normalized coords for grid_sample
    x_norm = _pixel_to_norm(x_src_feat, w_feat, align_corners=align_corners)
    y_norm = _pixel_to_norm(y_src_feat, h_feat, align_corners=align_corners)

    # stack to grid in (x,y) order
    grid = torch.stack((x_norm, y_norm), dim=-1)  # (h_feat, w_feat, 2)
    grid = grid.unsqueeze(0)  # batch dim = 1

    return grid  # (1, h_feat, w_feat, 2)


def warp_feature_map(
    feat: torch.Tensor,
    params: Union[dict, List[dict]],
    image_size: Tuple[int, int],
    align_corners: bool = False,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    og_valid_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Warp `feat` (C,h_feat,w_feat) or (N,C,h_feat,w_feat) that was produced by the encoder on the
    *transformed* image back to original-image coordinates using the inverse of the sampled affine.

    - feat: torch tensor (C,h,w) or (N,C,h,w)
    - params: either a single params dict (applies to all batch elements) or a list of dicts length N
    - image_size: (W,H) of the *original image* before transform (must match params['image_size'])
    - returns warped feature map(s) with same shape as feat
    """
    single = False
    if feat.dim() == 3:
        # make batch dim
        feat = feat.unsqueeze(0)
        single = True
    N, C, h_feat, w_feat = feat.shape
    device = feat.device
    dtype = feat.dtype

    # Build grid(s)
    if isinstance(params, dict):
        grid = compute_inverse_affine_grid(params, (h_feat, w_feat), image_size,
                                           device=device, dtype=dtype, align_corners=align_corners)
        # broadcast to N
        grid = grid.expand(N, -1, -1, -1).to(device)
    elif isinstance(params, list):
        assert len(params) == N, "If passing list of params it must match batch size"
        grids = []
        for p in params:
            g = compute_inverse_affine_grid(p, (h_feat, w_feat), image_size,
                                            device=device, dtype=dtype, align_corners=align_corners)
            grids.append(g)
        grid = torch.cat(grids, dim=0)  # (N, h, w, 2)
    else:
        raise TypeError("params must be dict or list[dict]")

    # sample
    warped = F.grid_sample(feat, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    og_valid_mask = og_valid_mask.unsqueeze(0) if og_valid_mask.dim() == 3 else og_valid_mask
    warped_mask = F.grid_sample(
                        og_valid_mask, grid, mode="nearest", padding_mode="zeros", align_corners=align_corners
                    )
    warped_mask = (warped_mask > 0.5).int()

    if single:
        return warped.squeeze(0), warped_mask.squeeze(0)
    return warped, warped_mask



def compute_inverse_affine_grid_batch(params, feature_size, align_corners=False, device=None, dtype=None):
    """
    Batchified version: compute inverse affine grid for all B examples
    feat: (B,C,H_feat,W_feat)
    params: dict with tensors Bx? or list of dicts to apply transforms sequentially
    Returns: grid of shape (B, H_feat, W_feat, 2)
    """
    # Handle both single params dict and list of params dicts
    if not isinstance(params, list):
        params = [params]
    
    B = params[0]["angle"].shape[0]
    H_feat, W_feat = feature_size
    W_img, H_img = params[0]["image_size"]
    W_img = W_img if isinstance(W_img, int) else W_img[0]
    H_img = H_img if isinstance(H_img, int) else H_img[0]

    if device is None:
        device = params[0]["angle"].device
    if dtype is None:
        dtype = params[0]["angle"].dtype

    # Feature -> image scaling
    Sx = float(W_img) / float(W_feat)
    Sy = float(H_img) / float(H_feat)

    # feature grid coordinates (H_feat,W_feat)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H_feat, device=device, dtype=dtype),
        torch.arange(W_feat, device=device, dtype=dtype),
        indexing="ij"
    )
    # expand to batch
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # B,H_feat,W_feat
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # map to image coordinates (pixel centers)
    x_img = (grid_x + 0.5) * Sx - 0.5
    y_img = (grid_y + 0.5) * Sy - 0.5

    # center
    cx = (W_img - 1.) / 2.0
    cy = (H_img - 1.) / 2.0

    # relative coords
    x_rel = x_img - cx
    y_rel = y_img - cy

    # Apply transforms sequentially
    for param_dict in params:
        
        # expand params
        translate = param_dict["translate"] if isinstance(param_dict["translate"], list) else param_dict["translate"].permute(1,0)
        angle = param_dict["angle"].view(B,1,1)
        scale = param_dict["scale"].view(B,1,1)
        shear = param_dict.get("shear", torch.zeros_like(angle)).view(B,1,1)
        tx = translate[0].view(B,1,1)
        ty = translate[1].view(B,1,1)

        angle_rad = angle * math.pi / 180.0
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # forward affine (image space)
        x_img = scale * (cos_a * x_rel - sin_a * y_rel) + cx + tx
        y_img = scale * (sin_a * x_rel + cos_a * y_rel) + cy + ty

    # map back to feature coordinates
    x_src_feat = (x_img + 0.5) / Sx - 0.5
    y_src_feat = (y_img + 0.5) / Sy - 0.5

    # normalize to [-1,1]
    if align_corners:
        x_norm = 2.0 * x_src_feat / (W_feat - 1) - 1.0
        y_norm = 2.0 * y_src_feat / (H_feat - 1) - 1.0
    else:
        x_norm = 2.0 * (x_src_feat + 0.5) / W_feat - 1.0
        y_norm = 2.0 * (y_src_feat + 0.5) / H_feat - 1.0

    grid = torch.stack((x_norm, y_norm), dim=-1)  # B,H_feat,W_feat,2
    return grid

def warp_feature_map_batch(
    feat: torch.Tensor,
    params: dict,
    image_size: Tuple[int,int],
    align_corners: bool = False,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    return_mask: bool = True,
    og_valid_mask: torch.Tensor = None,
    return_grid: bool = False
):
    """
    Warp a batch of features (B,C,H_feat,W_feat) using batchified params dict.
    Returns warped_feat (B,C,H_feat,W_feat) and optionally warped_mask (B,1,H_feat,W_feat)
    """
    B, C, H_feat, W_feat = feat.shape
    device = feat.device
    dtype = feat.dtype
    res = None

    if isinstance(params, list):
        params = [{**p, "image_size": image_size} for p in params]
    else:
        params = [{**params, "image_size": image_size}]

    # compute grid batch
    grid = compute_inverse_affine_grid_batch(
        params,
        feature_size=(H_feat, W_feat),
        align_corners=align_corners,
        device=device, dtype=dtype
    )

    # warp features
    warped_feat = F.grid_sample(feat, grid.to(dtype=torch.float32), mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Normalize the warped features
    # warped_feat = warped_feat/warped_feat.norm(dim=-1)[..., None]

    if return_mask:
        
        B, C, H_feat, W_feat = og_valid_mask.shape
        for p in params:
            grid = compute_inverse_affine_grid_batch(
                {**p, "image_size": image_size},
                feature_size=(H_feat, W_feat),
                align_corners=align_corners,
                device=device, dtype=dtype
            )
            og_valid_mask = og_valid_mask.unsqueeze(1) if og_valid_mask.dim() == 3 else og_valid_mask
            og_valid_mask = F.grid_sample(og_valid_mask, grid.to(dtype=torch.float32), mode="nearest", padding_mode="zeros", align_corners=align_corners)
            og_valid_mask = (og_valid_mask > 0.4).float()

        warped_mask = og_valid_mask

        res = (grid, warped_mask) if return_grid else (warped_feat, warped_mask)
        return res

    res = grid if return_grid else warped_feat
    return res