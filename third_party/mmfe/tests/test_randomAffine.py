import torch
import torch.nn.functional as F
import cv2
import numpy as np
from dataloading.dual_transforms import PairRandomAffine
from dataloading.inversible_tf import warp_feature_map

# --- Load image ---
img = cv2.imread("tests/test_images/test_image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C,H,W

# --- Duplicate for pair ---
img0 = img.clone()
img1 = img.clone()

# --- Apply random affine ---
pair_aff = PairRandomAffine(degrees=8.0, translate=(0.08, 0.08), scale=(0.98, 1.02))
img0_t, params = pair_aff(img0, return_transform=True)

# --- Forward valid mask ---
H, W = params["image_size"][1], params["image_size"][0]
# mask_forward = torch.ones(1, H, W)
# mask_forward = F.interpolate(mask_forward.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
# params["valid_mask"] = mask_forward
mask_forward = params['valid_mask']

# --- Fake encoder: downsample to feature map size ---
feat0_t = F.interpolate(img0_t.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
feat0 = F.interpolate(img0.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)
feat1 = F.interpolate(img1.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)

# --- Warp features back to original coordinate frame ---
feat0_warped, warp_mask = warp_feature_map(
    feat0_t, params, image_size=params['image_size'], align_corners=False, og_valid_mask=params['valid_mask']
)

# Combine masks if desired
combined_mask = warp_mask  # or multiply with forward mask if available

# --- Visualization helper ---
def show_tensor(img_tensor, window_name="img"):
    """
    img_tensor: C,H,W tensor, values 0-1
    """
    if not isinstance(img_tensor, list):
        img_tensor = [img_tensor]
        window_name = [window_name]

    for img_t, w_name in zip(img_tensor, window_name):
        img_np = img_t.detach().cpu().numpy()
        if img_np.shape[0] == 1:
            img_np = img_np[0]
        else:
            img_np = np.transpose(img_np, (1,2,0))
        img_np = (img_np * 255).astype(np.uint8)
        cv2.imshow(w_name, img_np)

def show_mask(mask_tensor, window_name="mask"):
    mask_np = mask_tensor.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    cv2.imshow(window_name, mask_np)

# --- Upsample feature maps for visualization ---
def upsample_feat(feat, target_size=(256,256)):
    return F.interpolate(feat.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

# --- Show images ---
#show_tensor(img0, "Original img0")
#show_tensor(img0_t, "Transformed img0")
#show_tensor(img1, "Original img1")
show_tensor([img0, img0_t, img1], ["Original img0", "Transformed img0", "Original img1"])

show_tensor(upsample_feat(feat0_t), "Feature map of transformed img0")
show_tensor(upsample_feat(feat0), "Feature map of original img0")
show_tensor(upsample_feat(feat1), "Feature map of original img1")
show_tensor(upsample_feat(feat0_warped), "Warped feature map")

show_mask(mask_forward, "Forward valid mask")
show_mask(upsample_feat(warp_mask.repeat(3,1,1).float()), "Warp valid mask")
cv2.waitKey(0)


# --- Compute mean absolute difference only on valid pixels ---
valid_idx = combined_mask.bool().squeeze(0)
mad = (feat0_warped - feat1)[...,valid_idx].abs().mean().item()
print("Mean absolute difference on valid pixels:", mad)

