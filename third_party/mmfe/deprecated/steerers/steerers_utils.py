import torch
import numpy as np
import matplotlib.pyplot as plt

def dict_to_device(batch, device="cuda"):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = dict_to_device(value, device)
    return batch


def visualize_keypoint_rotation(kpts_A_orig: torch.Tensor, kpts_A_rot: torch.Tensor, angle: float):
    """
    Visualize keypoint rotation with before/after plots and arrows showing transformation.
    
    Args:
        kpts_A_orig: Original keypoints [N, 2] in normalized coordinates [-1, 1]
        kpts_A_rot: Rotated keypoints [N, 2] in normalized coordinates [-1, 1]  
        angle: Rotation angle in degrees
    """
    # Subsample 10 keypoints to show
    kpts_A_orig = kpts_A_orig[:10, :]
    kpts_A_rot = kpts_A_rot[:10, :]
    
    # Choose 10 different colors
    colors = plt.cm.viridis(np.linspace(0, 1, 10))[:, :3]
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    
    # Before rotation
    axs[0].scatter(kpts_A_orig[:, 0], kpts_A_orig[:, 1], c=colors, label='Original', s=10)
    axs[0].set_title(f"Keypoints Before {angle}° Rotation")
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])
    axs[0].set_aspect('equal')
    axs[0].legend()
    
    # After rotation
    axs[1].scatter(kpts_A_rot[:, 0], kpts_A_rot[:, 1], c=colors, label='Rotated', s=10)
    axs[1].set_title(f"Keypoints After {angle}° Rotation")
    axs[1].set_xlim([-1, 1])
    axs[1].set_ylim([-1, 1])
    axs[1].set_aspect('equal')
    axs[1].legend()
    
    # Both together with arrows from original to rotated points
    axs[2].scatter(kpts_A_orig[:, 0], kpts_A_orig[:, 1], c=colors, label='Original', s=10)
    axs[2].scatter(kpts_A_rot[:, 0], kpts_A_rot[:, 1], c=colors, label='Rotated', s=10)
    
    # Draw arrows from original to rotated points
    for i in range(len(kpts_A_orig)):
        axs[2].arrow(
            kpts_A_orig[i, 0], kpts_A_orig[i, 1],
            kpts_A_rot[i, 0] - kpts_A_orig[i, 0],
            kpts_A_rot[i, 1] - kpts_A_orig[i, 1],
            color=colors[i], head_width=0.03, head_length=0.05, length_includes_head=True, alpha=0.8
        )
    
    axs[2].set_title(f"Keypoints Before and After {angle}° Rotation")
    axs[2].set_xlim([-1, 1])
    axs[2].set_ylim([-1, 1])
    axs[2].set_aspect('equal')
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()


def visualize_image_and_keypoints(batch: dict, kpts_A: torch.Tensor, kpts_B: torch.Tensor, nbr_rot_A: int, nbr_rot_B: int, nbr_rot: int):
    """
    Visualize images with their keypoints and show correspondence between keypoints.
    
    Args:
        batch: Dictionary containing 'im_A' and 'im_B' image tensors [B, C, H, W]
        kpts_A: Keypoints for image A [B, N, 2] in normalized coordinates [-1, 1]
        kpts_B: Keypoints for image B [B, N, 2] in normalized coordinates [-1, 1]
    """
    # Take first batch element and subsample 10 keypoints
    im_A = batch["im_A"][0].detach().cpu()  # [C, H, W]
    im_B = batch["im_B"][0].detach().cpu()  # [C, H, W]
    
    kpts_A_sub = kpts_A[0, :10, :].detach().cpu()  # [10, 2]
    kpts_B_sub = kpts_B[0, :10, :].detach().cpu()  # [10, 2]
    
    # Convert images from [C, H, W] to [H, W, C] for display
    if im_A.shape[0] == 3:  # RGB
        im_A = im_A.permute(1, 2, 0)
        im_B = im_B.permute(1, 2, 0)
    else:  # Grayscale
        im_A = im_A.squeeze(0)
        im_B = im_B.squeeze(0)
    
    # Normalize images to [0, 1] for display
    im_A = (im_A - im_A.min()) / (im_A.max() - im_A.min())
    im_B = (im_B - im_B.min()) / (im_B.max() - im_B.min())
    
    # Convert normalized keypoints [-1, 1] to pixel coordinates
    H, W = im_A.shape[:2] if im_A.ndim == 3 else im_A.shape
    kpts_A_pix = torch.stack([
        ((kpts_A_sub[:, 0] + 1.0) * 0.5) * (W - 1),
        ((kpts_A_sub[:, 1] + 1.0) * 0.5) * (H - 1)
    ], dim=1)
    
    H, W = im_B.shape[:2] if im_B.ndim == 3 else im_B.shape
    kpts_B_pix = torch.stack([
        ((kpts_B_sub[:, 0] + 1.0) * 0.5) * (W - 1),
        ((kpts_B_sub[:, 1] + 1.0) * 0.5) * (H - 1)
    ], dim=1)
    
    # Choose colors for keypoints
    colors = plt.cm.viridis(np.linspace(0, 1, 10))[:, :3]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # First image: im_A with keypoints
    if im_A.ndim == 3:
        axs[0].imshow(im_A)
    else:
        axs[0].imshow(im_A, cmap='gray')
    axs[0].scatter(kpts_A_pix[:, 1], kpts_A_pix[:, 0], c=colors, s=50, edgecolors='white', linewidth=1, marker='o')
    axs[0].set_title(f"Image A Keypoints {nbr_rot_A} turns")
    axs[0].axis('off')
    
    # Second image: im_B with keypoints
    if im_B.ndim == 3:
        axs[1].imshow(im_B)
    else:
        axs[1].imshow(im_B, cmap='gray')
    axs[1].scatter(kpts_B_pix[:, 1], kpts_B_pix[:, 0], c=colors, s=50, edgecolors='white', linewidth=1, marker='s')
    axs[1].set_title(f"Image B with Keypoints {nbr_rot_B} turns")
    axs[1].axis('off')
    
    # Third plot: keypoints only with arrows showing correspondence
    # Create a combined coordinate space for visualization
    # Use pixel coordinates for correspondence plot
    axs[2].scatter(kpts_A_pix[:, 1], kpts_A_pix[:, 0], c=colors, s=50, label='Image A keypoints', marker='o')
    axs[2].scatter(kpts_B_pix[:, 1], kpts_B_pix[:, 0], c=colors, s=50, label='Image B keypoints', marker='s')

    for i in range(len(kpts_A_pix)):
        axs[2].arrow(
            kpts_B_pix[i, 1], kpts_B_pix[i, 0],        
            kpts_A_pix[i, 1] - kpts_B_pix[i, 1],       
            kpts_A_pix[i, 0] - kpts_B_pix[i, 0],       
            color=colors[i], head_width=5, head_length=7, 
            length_includes_head=True, alpha=0.7, linewidth=2
        )

    axs[2].set_xlim([0, W])
    axs[2].set_ylim([H, 0])  # Flip y-axis to match image coordinates
    axs[2].set_aspect('equal')
    axs[2].set_title(f"Keypoint Correspondence: Turns from B to A = {nbr_rot}")

    
    plt.tight_layout()
    plt.show()


def rotate_keypoints(keypoints: torch.Tensor, angle: float, continuous_rot: bool = False, debug: bool = False) -> torch.Tensor:
    # rotate keypoints back so that GT annotations can be used
        kpts_A = keypoints.clone()

        if continuous_rot:
            rot_A_rad = angle
        else:
            rot_A_rad = np.deg2rad(angle)

        R_A = torch.tensor([[np.cos(rot_A_rad), -np.sin(rot_A_rad)],
                        [np.sin(rot_A_rad), np.cos(rot_A_rad)]],
                        dtype=kpts_A.dtype,
                        device=kpts_A.device)


        if debug:
            # Save original keypoints for visualization
            kpts_A_orig = kpts_A.clone().detach().cpu()
            kpts_A_rot = (kpts_A @ R_A.T).detach().cpu()
            
            # Visualize the rotation using the separate function
            visualize_keypoint_rotation(kpts_A_orig[0], kpts_A_rot[0], angle)

        # Overwrite kpts_A with rotated versions for downstream code
        kpts_A = kpts_A @ R_A.T

        return kpts_A


def filter_kpts_inside_image(
    kpts_A: torch.Tensor,
    kpts_B: torch.Tensor,
    im_B: torch.Tensor,
    min_keep: int = 1,
    random_select: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter keypoints keeping only those whose B-coordinates lie inside image B.

    Args:
        kpts_A: [B, N, 2] normalized coords in [-1, 1]
        kpts_B: [B, N, 2] normalized coords in [-1, 1]
        im_B:  [B, C, H, W] image tensor to provide bounds
        min_keep: ensure at least this many keypoints per batch element
        random_select: if more valid than kept, randomly sample; otherwise take first

    Returns:
        (kpts_A_filtered, kpts_B_filtered): both [B, N_kept, 2]
    """
    if kpts_A.ndim != 3 or kpts_B.ndim != 3:
        raise ValueError("kpts must be [B, N, 2]")
    if im_B.ndim != 4:
        raise ValueError("im_B must be [B, C, H, W]")

    Bsz, N, _ = kpts_B.shape
    H, W = im_B.shape[-2], im_B.shape[-1]

    # Convert normalized coords [-1,1] -> pixel coords [0, W-1], [0, H-1]
    kpts_B_pix_x = ((kpts_B[..., 0] + 1.0) * 0.5) * (W - 1)
    kpts_B_pix_y = ((kpts_B[..., 1] + 1.0) * 0.5) * (H - 1)
    inside_mask = (
        (kpts_B_pix_x >= 0) & (kpts_B_pix_x <= (W - 1)) &
        (kpts_B_pix_y >= 0) & (kpts_B_pix_y <= (H - 1))
    )  # [B, N]

    valid_counts = inside_mask.sum(dim=1)
    min_valid = int(valid_counts.min().item()) if Bsz > 0 else 0
    min_valid = max(min_valid, int(min_keep))

    kpts_A_filtered = []
    kpts_B_filtered = []
    for b in range(Bsz):
        valid_idx = torch.nonzero(inside_mask[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() < min_valid:
            if valid_idx.numel() == 0:
                # Fallback: select first min_valid indices
                valid_idx = torch.arange(min_valid, device=kpts_B.device)
            else:
                pad_needed = min_valid - valid_idx.numel()
                pad_idx = valid_idx[-1].repeat(pad_needed)
                valid_idx = torch.cat([valid_idx, pad_idx], dim=0)
        else:
            if random_select:
                rand_idx = torch.randperm(valid_idx.numel(), device=kpts_B.device)[:min_valid]
                valid_idx = valid_idx[rand_idx]
            else:
                valid_idx = valid_idx[:min_valid]

        kpts_A_filtered.append(kpts_A[b, valid_idx])
        kpts_B_filtered.append(kpts_B[b, valid_idx])

    kpts_A_out = torch.stack(kpts_A_filtered, dim=0)
    kpts_B_out = torch.stack(kpts_B_filtered, dim=0)
    return kpts_A_out, kpts_B_out