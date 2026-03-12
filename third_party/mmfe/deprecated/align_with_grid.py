def apply_affine_2d_map(tensor_input: torch.Tensor, affine_matrix: np.ndarray, padding: str = 'zeros') -> torch.Tensor:
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
    batch_size = tensor_bchw.shape[0]
    grid = F.affine_grid(affine_matrix.unsqueeze(0).expand(batch_size, -1, -1), size=nchw, align_corners=False).to(torch.float32)
    
    # Apply transformation
    transformed = F.grid_sample(tensor_bchw, grid, mode='bilinear', padding_mode=padding, align_corners=False)
    
    # Return with same shape as input
    if transformed.shape[0] == 1:
        return transformed.squeeze(0)
    else:
        return transformed


def apply_affine_image(img: np.ndarray, affine_matrix: np.ndarray, return_np: bool = True) -> np.ndarray:
    """
    Transforms and input image using the affine matrix returned by estimateAffinePartial2D(), 
    which is of the form:
    [cos(θ)⋅s    -sin(θ)⋅s    tx
     sin(θ)⋅s     cos(θ)⋅s    ty]

    Input:
    - img: (H,W,3) numpy array
    - affine_matrix: (2,3) numpy array

    Output:
    - img_transformed: (H,W,3) numpy array
    """
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
    if isinstance(affine_matrix, torch.Tensor):
        affine_matrix = affine_matrix.cpu().numpy()
    if affine_matrix.shape == (2,2):
        affine_matrix = rot_to_affine(affine_matrix)
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Apply the affine transformation using OpenCV's warpAffine
    img_transformed = cv2.warpAffine(img, affine_matrix, (w, h))
    
    if return_np:
        return img_transformed
    else:
        return torch.from_numpy(img_transformed).permute(2, 0, 1)


def angle_to_rot(angle: float) -> np.ndarray:
    """
    Convert an angle in degrees to an affine matrix.
    """
    theta = np.deg2rad(angle)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
