import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Optional
import cv2
from mmfe_utils.tensor_utils import tensor_to_numpy_image, cosine_map
import os
from sklearn.decomposition import PCA
import matplotlib.cm as mpl_cm

def show_equivariance_debug(
    modality_0: torch.Tensor,
    modality_1: torch.Tensor,
    modality_0_noise: torch.Tensor,
    modality_1_noise: torch.Tensor,
    warp_mask_0: torch.Tensor,
    warp_mask_1: torch.Tensor,
    noise_params: dict,
    align_corners: bool = False,
    wait_key: int = 0
):
    """
    Visualize equivariance/debug with OpenCV windows for a single batch.

    Displays:
    - Original and noise images for both modalities
    - Wrapped noise masks
    - Wrapped noise images overlaid on originals

    Close windows with key press.
    """
    from dataloading.inversible_tf import warp_feature_map_batch

    index = 0

    # Show the original and noise images
    modality_0_np = tensor_to_numpy_image(modality_0[index])
    modality_1_np = tensor_to_numpy_image(modality_1[index])
    modality_0_noise_np = tensor_to_numpy_image(modality_0_noise[index])
    modality_1_noise_np = tensor_to_numpy_image(modality_1_noise[index])
    cv2.imshow("modality_0", modality_0_np)
    cv2.imshow("modality_1", modality_1_np)
    cv2.imshow("modality_0_noise", modality_0_noise_np)
    cv2.imshow("modality_1_noise", modality_1_noise_np)

    # Show the wrapped noise masks
    warp_mask_0_np = tensor_to_numpy_image(warp_mask_0[index])
    warp_mask_1_np = tensor_to_numpy_image(warp_mask_1[index])
    cv2.imshow("warp_mask_0", warp_mask_0_np)
    cv2.imshow("warp_mask_1", warp_mask_1_np)

    # Apply wrapping to the original images
    wrapped_image_0_to_og, _ = warp_feature_map_batch(
        modality_0_noise,
        noise_params,
        image_size=noise_params['image_size'],
        align_corners=align_corners,
        og_valid_mask=noise_params['valid_mask'],
        return_mask=True,
    )
    wrapped_image_1_to_og, _ = warp_feature_map_batch(
        modality_1_noise,
        noise_params,
        image_size=noise_params['image_size'],
        align_corners=align_corners,
        og_valid_mask=noise_params['valid_mask'],
        return_mask=True,
    )

    # Show the wrapped original images
    wrapped_image_0_to_og_np = tensor_to_numpy_image(wrapped_image_0_to_og[index])
    wrapped_image_1_to_og_np = tensor_to_numpy_image(wrapped_image_1_to_og[index])
    cv2.imshow("wrapped_image_0_to_og", wrapped_image_0_to_og_np)
    cv2.imshow("wrapped_image_1_to_og", wrapped_image_1_to_og_np)

    # Overlap the wrapped original images with the original images
    wrapped_image_0_to_og_np = cv2.addWeighted(wrapped_image_0_to_og_np, 0.5, modality_0_np, 0.5, 0)
    wrapped_image_1_to_og_np = cv2.addWeighted(wrapped_image_1_to_og_np, 0.5, modality_1_np, 0.5, 0)
    cv2.imshow("wrapped_image_0_to_og_overlay", wrapped_image_0_to_og_np)
    cv2.imshow("wrapped_image_1_to_og_overlay", wrapped_image_1_to_og_np)

    cv2.waitKey(wait_key)
    cv2.destroyAllWindows()

def save_grid(rows: List[np.ndarray], out_path: str, max_rows: int) -> None:
    if not rows:
        return
    rows = rows[:max_rows]

    # Insert a thin horizontal separator between rows
    stitched: List[np.ndarray] = []
    for i, row in enumerate(rows):
        stitched.append(row)
        if i < len(rows) - 1:
            sep = np.ones((10, row.shape[1], 3), dtype=np.float32)
            stitched.append(sep)

    grid = np.concatenate(stitched, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Convert from [0,1] float to uint8 for saving
    grid_to_save = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
    grid_to_save = cv2.cvtColor(grid_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, grid_to_save)
    print(f"Saved: {out_path}")

def create_row_images(e0_bchw: torch.Tensor, e1_bchw: torch.Tensor, m0_imgs: torch.Tensor, m1_imgs: torch.Tensor, mode: str) -> List[np.ndarray]:
    # Use first sample shapes to scale maps to image size
    sample_np = tensor_to_numpy_image(m0_imgs[0])
    img_h, img_w = sample_np.shape[:2]
    b = e0_bchw.shape[0]

    rows: List[np.ndarray] = []
    for idx in torch.randperm(b).tolist():
        img0 = tensor_to_numpy_image(m0_imgs[idx])
        img1 = tensor_to_numpy_image(m1_imgs[idx])

        # Negative example within batch
        neg_idx = (idx + random.randint(1, b)) % b
        img_neg = tensor_to_numpy_image(m0_imgs[neg_idx])

        if mode == "one_to_all":
            i = random.randint(0, e0_bchw[idx].shape[1]-1)
            j = random.randint(0, e0_bchw[idx].shape[2]-1)
            e0 = e0_bchw[idx][:, i, j].cpu()

            # Mark query location (small square)
            ratio = img0.shape[1] / e0_bchw[idx].shape[1]
            ii = int(i * ratio)
            jj = int(j * ratio)
            img0[ii-4:ii+4, jj-4:jj+4] = [1, 0, 0]
            # img1[ii-3:ii+3, jj-3:jj+3] = [1, 0, 0]
            # img_neg[ii-3:ii+3, jj-3:jj+3] = [1, 0, 0]
        elif mode == "all_to_all":
            e0 = e0_bchw[idx].cpu()
        else:
            raise ValueError(f"Invalid mode: {mode}")

        cmap_pos = get_color_map(e0, e1_bchw[idx].cpu())
        cmap_neg = get_color_map(e0, e0_bchw[neg_idx].cpu())

        cmap_pos = cv2.resize(cmap_pos, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        cmap_neg = cv2.resize(cmap_neg, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        sep = np.ones((img_h, 10, 3), dtype=np.float32)
        row = np.concatenate([img0, img1, sep, cmap_pos, sep, img_neg, sep, cmap_neg], axis=1)
        rows.append(row)

    return rows

# Proper Viridis heatmap using matplotlib 'viridis'
def heatmap_colormap(gray01: np.ndarray) -> np.ndarray:
    # Ensure [0,1]
    if isinstance(gray01, torch.Tensor):
        gray01 = gray01.cpu().numpy()
    gray01 = np.clip(gray01, 0.0, 1.0)
    cmap = mpl_cm.get_cmap('viridis')  # perceptually uniform colormap
    colored = cmap(gray01)  # returns RGBA in [0,1]
    return colored[..., :3].astype(np.float32)

def get_color_map(e0: torch.Tensor, e1: torch.Tensor) -> np.ndarray:
    if isinstance(e0, np.ndarray):
        e0 = torch.from_numpy(e0)
    if isinstance(e1, np.ndarray):
        e1 = torch.from_numpy(e1)
    cos_hw = cosine_map(e0, e1)  # [-1,1]
    cos01 = (cos_hw + 1.0) * 0.5  # [0,1]
    color_map = heatmap_colormap(cos01)  # [H',W',3] in [0,1]
    color_map = (color_map * 255.0).astype(np.uint8)
    color_map = color_map.astype(np.float32) / 255.0
    return color_map

def pca_rgb(emb_list: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Fit PCA over concatenated spatial pixels from provided [C,H,W] tensors, map to RGB in [0,1].
    Returns list of HxWx3 float32 arrays aligned with input order.
    """
    if len(emb_list) == 0:
        return []
    # Prepare data matrix: concat all (H*W, C)
    flat_list = []
    shapes_hw = []
    for emb in emb_list:
        c, h, w = emb.shape
        shapes_hw.append((h, w))
        flat_list.append(emb.permute(1, 2, 0).reshape(-1, c))
    X = torch.cat(flat_list, dim=0).cpu().numpy()
    # PCA to 3 comps
    pca = PCA(n_components=3, random_state=42)
    X3 = pca.fit_transform(X)
    # Normalize globally to [0,1]
    X3 = (X3 - X3.min()) / (X3.max() - X3.min())
    # Split back and reshape
    rgb_images: List[np.ndarray] = []
    offset = 0
    for (h, w) in shapes_hw:
        n = h * w
        rgb = X3[offset:offset+n].reshape(h, w, 3).astype(np.float32)
        # Normalize by image
        # rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb_images.append(rgb)
        offset += n
    return rgb_images


def viz_2d_PCA(e0_bchw: torch.Tensor, e1_bchw: torch.Tensor, m0_imgs: torch.Tensor, m1_imgs: torch.Tensor, out_path: str = None, max_rows: int = 4) -> None:
    """
    Visualize PCA-colorized embeddings (D->RGB) per row:
    - modality 0 PCA RGB
    - modality 1 PCA RGB
    - modality 0 of a different sample PCA RGB
    Saves a grid PNG to out_path.
    """
    # Target image size
    sample_np = tensor_to_numpy_image(m0_imgs[0])
    img_h, img_w = sample_np.shape[:2]
    b = e0_bchw.shape[0]

    rows: List[np.ndarray] = []
    for idx in torch.randperm(b).tolist():
        neg_idx = (idx + random.randint(1, b)) % b

        e0 = e0_bchw[idx].cpu()
        e1 = e1_bchw[idx].cpu()
        e0_neg = e0_bchw[neg_idx].cpu()

        # PCA over concatenated pixels from the three maps
        rgb0, rgb1, rgb_neg = pca_rgb([e0, e1, e0_neg])

        # Resize to match input image size for neat grids
        rgb0 = cv2.resize(rgb0, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        rgb1 = cv2.resize(rgb1, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        rgb_neg = cv2.resize(rgb_neg, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # Original images for each
        img0 = tensor_to_numpy_image(m0_imgs[idx])
        img1 = tensor_to_numpy_image(m1_imgs[idx])
        img_neg = tensor_to_numpy_image(m0_imgs[neg_idx])

        sep = np.ones((img_h, 10, 3), dtype=np.float32)
        row = np.concatenate([img0, rgb0, sep, img1, rgb1, sep, img_neg, rgb_neg], axis=1)
        rows.append(row)

    if out_path is not None:
        save_grid(rows, out_path, max_rows)
    return rows


def rotate_tensor(tensor: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a (C, H, W) or (B, C, H, W) tensor by degrees."""
    if angle == 0:
        return tensor
    
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # 2D Affine rotation matrix
    rot_mat = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0]
    ], dtype=tensor.dtype, device=tensor.device)
    
    is_3d = tensor.dim() == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)
    
    grid = F.affine_grid(rot_mat.unsqueeze(0), tensor.size(), align_corners=False)
    rotated = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
    
    return rotated.squeeze(0) if is_3d else rotated


def viz_2d_PCA_rot(m0_imgs: torch.Tensor, out_path: str, n_examples: int, 
                   model: torch.nn.Module, device: torch.device, model_name: str) -> None:
    """
    Visualize rotation equivariance by showing a 3x4 grid for each example:
    - Row 1: Original image rotated by 0°, 90°, 180°, 270°
    - Row 2: PCA visualization of embeddings computed from rotated images
    - Row 3: PCA visualization of spatially rotated original embeddings
    
    Each example is saved as a separate grid file.
    
    Args:
        e0_bchw: Original embeddings [B, C, H, W]
        m0_imgs: Original images [B, C, H, W]
        out_path: Base path for output files (will append "_example_N.png")
        n_examples: Number of examples to visualize
        model: Model to compute embeddings for rotated images
        device: Device to run computations on
        model_name: Model name (used to determine if DINO model)
    """
    from mmfe_utils.dino_utils import get_last_feature_dino
    
    # Target image size
    sample_np = tensor_to_numpy_image(m0_imgs[0])
    img_h, img_w = sample_np.shape[:2]
    b = m0_imgs.shape[0]
    
    # Rotation angles
    angles = [0, 90, 180, 270]
    
    # Get random examples
    indices = torch.randperm(b).tolist()[:n_examples]
    
    # Create output directory if needed
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    for example_idx, idx in enumerate(indices):
        # Get original image and embedding
        m0_orig = m0_imgs[idx:idx+1].to(device)  # [1, C, H, W]
        m0_orig = rotate_tensor(m0_orig[0], 90).unsqueeze(0)  # [1, C, H, W]
        
        # Prepare lists for each row
        row1_images = []  # Rotated original images
        row2_embeddings = []  # Embeddings computed from rotated images
        row3_embeddings = []  # Spatially rotated original embeddings
        
        # Process each rotation angle
        for angle in angles:
            # Row 1: Rotate the original image
            m0_rot = rotate_tensor(m0_orig[0], angle).unsqueeze(0)  # [1, C, H, W]
            img_rot = tensor_to_numpy_image(m0_rot[0])
            row1_images.append(img_rot)
            
            # Row 2: Compute embedding from rotated image
            with torch.no_grad():
                if model_name.startswith("dino"):
                    e0_rot_computed = get_last_feature_dino(model, m0_rot, model_name)[0]  # [C, H', W']
                else:
                    e0_rot_result = model.get_embeddings(m0_rot, None)
                    # Handle both tuple and single tensor returns (model may return either)
                    if isinstance(e0_rot_result, tuple):
                        e0_rot_computed = e0_rot_result[0][0]  # [C, H', W'] from (e0, e1) where each is [1, C, H', W']
                    else:
                        e0_rot_computed = e0_rot_result[0]  # [C, H', W'] from [1, C, H', W']
            if angle == 0:
                e0_orig = e0_rot_computed.cpu()
            row2_embeddings.append(e0_rot_computed.cpu())
            
            # Row 3: Rotate the original embedding spatially
            e0_rot_spatial = rotate_tensor(e0_orig, angle)
            row3_embeddings.append(e0_rot_spatial.cpu())
        
        # Perform PCA over all embeddings (row2 and row3) for consistent coloring
        all_embeddings = row2_embeddings + row3_embeddings
        pca_rgbs = pca_rgb(all_embeddings)
        row2_rgbs = pca_rgbs[:4]
        row3_rgbs = pca_rgbs[4:]
        
        # Resize PCA visualizations to match image size
        for i in range(4):
            row2_rgbs[i] = cv2.resize(row2_rgbs[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            row3_rgbs[i] = cv2.resize(row3_rgbs[i], (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Build the 3x4 grid
        sep_v = np.ones((img_h, 10, 3), dtype=np.float32)  # Vertical separator
        
        # Build each row (4 images + 3 vertical separators)
        row1 = np.concatenate([row1_images[0], sep_v, row1_images[1], sep_v, row1_images[2], sep_v, row1_images[3]], axis=1)
        row2 = np.concatenate([row2_rgbs[0], sep_v, row2_rgbs[1], sep_v, row2_rgbs[2], sep_v, row2_rgbs[3]], axis=1)
        row3 = np.concatenate([row3_rgbs[0], sep_v, row3_rgbs[1], sep_v, row3_rgbs[2], sep_v, row3_rgbs[3]], axis=1)
        
        # Horizontal separator should match row width (4 * img_w + 3 * 10)
        row_width = row1.shape[1]
        sep_h = np.ones((10, row_width, 3), dtype=np.float32)  # Horizontal separator
        
        # Combine rows
        grid = np.concatenate([row1, sep_h, row2, sep_h, row3], axis=0)
        
        # Save grid
        base_path, ext = os.path.splitext(out_path)
        out_file = f"{base_path}_example_{example_idx}{ext}"
        grid_to_save = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
        grid_to_save = cv2.cvtColor(grid_to_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_file, grid_to_save)
        print(f"Saved: {out_file}")


def resize_with_padding(img, target_width, target_height, pad_color=255):
    """
    Resize image while maintaining aspect ratio and adding padding.
    
    Args:
        img: Input image
        target_width: Target width
        target_height: Target height
        pad_color: Padding color (default: 255 for white)
    
    Returns:
        Resized image with padding
    """
    h, w = img.shape[:2]
    aspect = w / h
    
    if aspect > target_width / target_height:
        # Width is the limiting factor
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        # Height is the limiting factor
        new_h = target_height
        new_w = int(target_height * aspect)
    
    # Resize the image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create canvas with padding color
    if len(img.shape) == 3:
        canvas = np.ones((target_height, target_width, img.shape[2]), dtype=img.dtype) * pad_color
    else:
        canvas = np.ones((target_height, target_width), dtype=img.dtype) * pad_color
    
    # Center the resized image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas