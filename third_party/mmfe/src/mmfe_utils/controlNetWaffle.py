import torch
import numpy as np
from PIL import Image

def to_pil(img):
    # Accept np.ndarray HxWxC in RGB [0..255] or torch.Tensor CxHxW [0..1]
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    elif torch.is_tensor(img):
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.detach().cpu().clamp(0, 1)
            img = (img * 255).to(torch.uint8)
            img = img.permute(1, 2, 0).numpy()
            return Image.fromarray(img)
        elif img.dim() == 3 and img.shape[-1] in (1, 3):
            img = img.detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(img)
        else:
            raise ValueError("Unsupported tensor image shape")
    else:
        raise TypeError("Unsupported image type")

def preprocess_control_image(control_image, width, height, red_threshold=0.01):
    """
    Preprocess the control image for the controlnet.
    """
    # Convert to desired to Waffle ControlNet format
    control_img = control_image
    # Invert the control image
    control_img = 1 - control_img
    # find pixels that are white (use isclose for floats)
    # white_mask = torch.isclose(control_img, torch.tensor(1.0), atol=0.9).all(dim=0)  # shape (H, W), bool
    white_mask = (control_img > red_threshold).all(dim=0)  # shape (H, W), bool

    # assign red to those pixels — prepare a (C,1) tensor so broadcasting works
    if white_mask.any():
        red = torch.tensor([1.0, 0.0, 0.0], dtype=control_img.dtype, device=control_img.device)[:, None]  # (C,1)
        control_img[:, white_mask] = red   # selects (C, N) and broadcasts red to (C, N)

    # Convert to PIL and resize
    control_img_pil = to_pil(control_img*255).convert("RGB").resize((width, height), Image.BILINEAR)

    # Show the control image
    # control_img_pil.show()
    return control_img_pil