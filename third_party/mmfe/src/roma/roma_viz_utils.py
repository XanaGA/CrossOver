import torch
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def visualize_debug(batch, gt_warp, gt_mask, step_name="train_debug"):
    """
    Visualizes:
        1. Source Image (with probe points)
        2. Target Image (with the SAME points mapped via gt_warp)
        3. Source Image + Mask Overlay (showing which source pixels land inside target)
    """
    # Detach and move to CPU
    im0 = batch["modality_0"][0].detach().cpu()       # [C, H, W]
    im1 = batch["modality_1_noise"][0].detach().cpu() # [C, H, W]
    
    # warp is [H, W, 2] in range [-1, 1]
    warp = gt_warp[0].detach().cpu()                  
    mask = gt_mask[0].detach().cpu().squeeze()        # [H, W]

    C, H, W = im0.shape

    # ---------------------------------------------------------
    # 1) Sample sparse correspondences
    # ---------------------------------------------------------
    num_points = 20
    
    # Pick random points in the Source grid
    ys_src = torch.randint(0, H, (num_points,))
    xs_src = torch.randint(0, W, (num_points,))
    
    # 2) Get their mapped location in Target using gt_warp
    mapped_coords_norm = warp[ys_src, xs_src, :] # [N, 2]

    # 3) Denormalize mapped coords to Pixels
    xs_tgt_px = (mapped_coords_norm[:, 0] + 1) * (W - 1) / 2.0
    ys_tgt_px = (mapped_coords_norm[:, 1] + 1) * (H - 1) / 2.0

    # Filter invalid points
    valid_idx = (
        (xs_tgt_px >= 0) & (xs_tgt_px < W) &
        (ys_tgt_px >= 0) & (ys_tgt_px < H) &
        (mask[ys_src, xs_src] > 0.5)
    )

    ys_src, xs_src = ys_src[valid_idx], xs_src[valid_idx]
    xs_tgt_px, ys_tgt_px = xs_tgt_px[valid_idx], ys_tgt_px[valid_idx]

    # ---------------------------------------------------------
    # 4) PLOT
    # ---------------------------------------------------------
    colors = plt.cm.jet(torch.linspace(0, 1, len(ys_src)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Helper to normalize images for matplotlib
    def prep_im(im):
        im = im - im.min()
        im = im / (im.max() + 1e-6)
        return im.permute(1, 2, 0).numpy()

    img0_disp = prep_im(im0)
    img1_disp = prep_im(im1)

    # A) Source Image + Points
    axes[0].imshow(img0_disp)
    axes[0].scatter(xs_src, ys_src, c=colors, s=50, edgecolors='white', marker='o')
    axes[0].set_title("Source (Modality 0)\nProbe Points")

    # B) Target Image + Mapped Points
    axes[1].imshow(img1_disp)
    axes[1].scatter(xs_tgt_px, ys_tgt_px, c=colors, s=50, edgecolors='white', marker='X')
    axes[1].set_title("Target (Modality 1)\nMapped Locations")
    
    # C) Source Image + Mask Overlay
    # We overlay the mask on top of the Source Image
    axes[2].imshow(img0_disp)
    
    # Create a colored overlay for the mask
    # Valid = Transparent, Invalid = Dark/Red overlay
    mask_np = mask.numpy()
    overlay = np.zeros((H, W, 4))
    overlay[mask_np < 0.5] = [0, 0, 0, 0.7] # Darken invalid regions
    # Alternatively: overlay[mask_np > 0.5] = [0, 1, 0, 0.3] # Tint valid regions green
    
    axes[2].imshow(overlay)
    
    # Re-plot source points to verify they are in the valid (bright) area
    axes[2].scatter(xs_src, ys_src, c=colors, s=50, edgecolors='white', marker='o')
    axes[2].set_title("Source + Valid Mask Overlay\n(Dark regions = Invalid)")

    plt.suptitle(f"Debug Step: {step_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


def render_wandb_image(batch, pred_warp, gt_warp, gt_mask):
    """
    Creates a matplotlib figure showing Source, Target, and Predicted Matches,
    then renders it to a numpy array for W&B logging.
    """
    # 1. Unpack data (cpu, detach)
    im0 = batch['modality_0'][0].detach().cpu().permute(1, 2, 0).numpy() # H,W,C
    im1 = batch['modality_1_noise'][0].detach().cpu().permute(1, 2, 0).numpy()
    
    # pred_warp is [B, H, W, 2] -> take 0th -> [H, W, 2]
    warp = pred_warp[0].detach().cpu() 
    mask = gt_mask[0].detach().cpu().squeeze()
    
    H, W, _ = im0.shape

    # 2. Select sparse probe points
    num_points = 40
    ys_src = torch.randint(0, H, (num_points,))
    xs_src = torch.randint(0, W, (num_points,))

    # 3. Sample predicted locations
    # pred_warp contains (x, y) normalized coords
    mapped_coords = warp[ys_src, xs_src, :] # [N, 2]
    
    # Denormalize [-1, 1] -> [0, W]
    xs_tgt = (mapped_coords[:, 0] + 1) * (W - 1) / 2.0
    ys_tgt = (mapped_coords[:, 1] + 1) * (H - 1) / 2.0

    # 4. Filter invalid (masked) points
    valid = mask[ys_src, xs_src] > 0.5
    xs_src, ys_src = xs_src[valid], ys_src[valid]
    xs_tgt, ys_tgt = xs_tgt[valid], ys_tgt[valid]

    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Helper to normalize image for display
    def norm_im(im): return (im - im.min()) / (im.max() - im.min() + 1e-6)
    
    # Colormap
    colors = plt.cm.hsv(np.linspace(0, 1, len(xs_src)))

    # Source
    axes[0].imshow(norm_im(im0))
    axes[0].scatter(xs_src, ys_src, c=colors, s=40, edgecolors='white', marker='o')
    axes[0].set_title("Source Image (Input)")
    axes[0].axis('off')

    # Target
    axes[1].imshow(norm_im(im1))
    axes[1].scatter(xs_tgt, ys_tgt, c=colors, s=40, edgecolors='white', marker='X')
    axes[1].set_title("Target Image\n(X marks where Source points were mapped)")
    axes[1].axis('off')

    plt.tight_layout()
    
    # 6. Render to Buffer -> Numpy
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    image_pil = Image.open(buf)
    image_np = np.array(image_pil)
    
    return image_np
