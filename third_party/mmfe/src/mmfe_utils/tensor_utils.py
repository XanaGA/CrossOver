from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

def norm_tensor_to_pil(tensor, mean=None, std=None):
    """
    Denormalizes a (C, H, W) tensor (using ImageNet stats) 
    and converts it to a PIL Image.
    """
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device)
    # 1. Denormalize: (tensor * std) + mean
    tensor = (tensor.permute(1, 2, 0) * std) + mean
    
    # 2. Clamp values to [0.0, 1.0] range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # 3. Move to CPU, convert to NumPy, transpose to (H, W, C)
    np_image = tensor.cpu().numpy() # It has aleady permuted
    
    # 4. Scale from [0.0, 1.0] to [0, 255] and cast to uint8
    np_image_uint8 = (np_image * 255).astype(np.uint8)
    
    # 5. Create PIL Image
    return Image.fromarray(np_image_uint8)

def torch_dilate(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1):
    """Dilates a binary mask using max pooling.
    
    mask: Tensor of shape (B, 1, H, W) with values 0 or 1
    """
    padding = kernel_size // 2
    for _ in range(iterations):
        mask = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=padding)
    return mask

def torch_erode(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1):
    """Erodes a binary mask using min pooling (implemented via negation + max_pool)."""
    padding = kernel_size // 2
    for _ in range(iterations):
        mask = -F.max_pool2d(-mask.float(), kernel_size, stride=1, padding=padding)
    return mask

def numpy_to_torch(image_np: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy image (H, W, C) or (H, W) into a float32 torch.Tensor in CHW, scaled to [0, 1].
    """
    if image_np.ndim == 2:
        image_np = image_np[:, :, None]
    # Ensure uint8-like scaling if not already float
    tensor = torch.from_numpy(image_np)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    # Heuristically scale to [0, 1] if values look like 0..255
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    # HWC -> CHW
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor

# Convert tensors back to numpy
def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        # CHW -> HWC
        np_array = tensor.cpu().permute(1, 2, 0).numpy()
        # Ensure values are in [0, 1] for display
        if np_array.max() <= 1.0:
            np_array = (np_array * 255).astype(np.uint8)
        else:
            np_array = np_array.astype(np.uint8)
    else:
        np_array = np.array(tensor)
    return np_array

# Convert tensor to numpy image
def tensor_to_numpy_image(tensor, normalize: bool = True):
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]  # Take first image
    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    
    if normalize:
        # Per channel normalization
        img = img / img.max(axis=(0,1))
        img = np.clip(img, 0, 1)
    return img

# Cosine similarity map between two embedding maps [C,H,W] each -> [H,W] in [-1,1]
def cosine_map(e0: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
    # e0,e1: [C,H,W]
    # Compute per-spatial-location cosine similarity across channel dimension
    # dot: [H,W]
    eps = 1e-8
    if len(e0.shape) == 1 and len(e1.shape) == 3:
        e0 = e0[:, None, None]
    elif e0.shape != e1.shape:
        raise ValueError(f"Embedding shapes must match for all_to_all or one_to_all mode. Got {e0.shape} and {e1.shape}")
    
    dot = (e0 * e1).sum(dim=0)
    norm0 = torch.linalg.vector_norm(e0, dim=0).clamp_min(eps)
    norm1 = torch.linalg.vector_norm(e1, dim=0).clamp_min(eps)
    cos = dot / (norm0 * norm1)
    cos = torch.clamp(cos, -1.0, 1.0)
    return cos
