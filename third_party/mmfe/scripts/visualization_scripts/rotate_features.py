#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import tempfile
import shutil
from typing import Dict, Any

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

# Project imports
from training.lightning_module import ContrastiveLearningModule
from mmfe_utils.tensor_utils import tensor_to_numpy_image, get_color_map
from mmfe_utils.viz_utils import pca_rgb
from mmfe_utils.data_utils import create_val_dataset

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

def create_rotation_grid(
    img_0: torch.Tensor, img_1: torch.Tensor, 
    p0_rot: np.ndarray, p1_rot: np.ndarray,
    p0_comp: np.ndarray, p1_comp: np.ndarray,
    c0_rot: np.ndarray, c1_rot: np.ndarray
) -> np.ndarray:
    """Composes a 2-row visualization grid."""
    img_0_np = tensor_to_numpy_image(img_0)
    img_1_np = tensor_to_numpy_image(img_1)
    h, w = img_0_np.shape[:2]

    # Resize all feature viz to match input image
    vizes = [p0_rot, p1_rot, p0_comp, p1_comp, c0_rot, c1_rot]
    r = [cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST) for v in vizes]

    sep_h = np.ones((h, 10, 3), dtype=np.float32)
    # Row: [Input | Rotated Reference Features | Actually Computed Features | Error Map]
    row0 = np.concatenate([img_0_np, sep_h, r[0], sep_h, r[2], sep_h, r[4]], axis=1)
    row1 = np.concatenate([img_1_np, sep_h, r[1], sep_h, r[3], sep_h, r[5]], axis=1)
    
    sep_v = np.ones((10, row0.shape[1], 3), dtype=np.float32)
    return np.concatenate([row0, sep_v, row1], axis=0)

def create_rotation_video(model, dataset, sample_idx, cfg, device):
    """Generates an MP4 for a single dataset sample."""
    sample = dataset[sample_idx]
    m0 = sample['modality_0'].unsqueeze(0).to(device)
    m1 = sample['modality_1'].unsqueeze(0).to(device)
    
    # Pre-compute 0-degree reference
    with torch.no_grad():
        f0_ref = model.get_embeddings(m0)
        f1_ref = model.get_embeddings(m1)
    
    temp_dir = tempfile.mkdtemp()
    try:
        n_rot = cfg.video.num_rotations
        angles = np.linspace(0, 360, n_rot, endpoint=False)
        
        computed_f0, computed_f1 = [], []
        rotated_f0, rotated_f1 = [], []
        cmaps_0, cmaps_1 = [], []

        print(f"  -> Extracting features for {n_rot} rotations...")
        for angle in angles:
            # 1. Compute features from rotated input
            m0_in = rotate_tensor(m0[0], angle).unsqueeze(0)
            m1_in = rotate_tensor(m1[0], angle).unsqueeze(0)
            with torch.no_grad():
                f0_c = model.get_embeddings(m0_in)[0]
                f1_c = model.get_embeddings(m1_in)[0]
            
            # 2. Mathematically rotate the reference features (The "Goal")
            f0_r = rotate_tensor(f0_ref[0], angle)
            f1_r = rotate_tensor(f1_ref[0], angle)

            computed_f0.append(f0_c); computed_f1.append(f1_c)
            rotated_f0.append(f0_r); rotated_f1.append(f1_r)
            cmaps_0.append(get_color_map(f0_r, f0_c))
            cmaps_1.append(get_color_map(f1_r, f1_c))

        # Perform PCA across all frames to keep colors consistent
        pca_m0 = pca_rgb(computed_f0 + rotated_f0)
        pca_m1 = pca_rgb(computed_f1 + rotated_f1)

        for i, angle in enumerate(angles):
            grid = create_rotation_grid(
                rotate_tensor(m0[0], angle), rotate_tensor(m1[0], angle),
                pca_m0[n_rot + i], pca_m1[n_rot + i], # Rotated refs
                pca_m0[i], pca_m1[i],                 # Computed
                cmaps_0[i], cmaps_1[i]
            )
            frame_path = os.path.join(temp_dir, f"f_{i:03d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor((np.clip(grid, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Assemble Video
        out_path = os.path.join(to_absolute_path(cfg.video.output_dir), f"sample_{sample_idx}.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        frames = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir)])
        img = cv2.imread(frames[0])
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), cfg.video.fps, (img.shape[1], img.shape[0]))
        for f in frames: vw.write(cv2.imread(f))
        vw.release()
        return out_path
    finally:
        shutil.rmtree(temp_dir)

@hydra.main(config_path="../../configs", config_name="rotate_video", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {cfg.model.checkpoint}")
    model = ContrastiveLearningModule.load_from_checkpoint(
        checkpoint_path=to_absolute_path(cfg.model.checkpoint),
        map_location=device, load_dino_weights=False, weights_only=False
    ).to(device).eval()

    dataset = create_val_dataset(cfg)
    
    n_viz = min(cfg.video.num_samples, len(dataset))
    print(f"Starting video generation for {n_viz} samples...")
    
    for i in range(n_viz):
        try:
            path = create_rotation_video(model, dataset, i, cfg, device)
            print(f"  [+] Saved: {path}")
        except Exception as e:
            print(f"  [!] Failed sample {i}: {e}")

if __name__ == "__main__":
    main()