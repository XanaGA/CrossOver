import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import os
import sys
from pathlib import Path
import subprocess


# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


def generate_netvlad_clusters(cfg: DictConfig) -> str:
    """
    Generate NetVLAD clusters by running init_netvlad_clusters.py.
    
    Args:
        cfg: Training configuration
        
    Returns:
        Path to generated cluster file
    """
    print("\n" + "="*80)
    print("NetVLAD clusters not found. Generating clusters...")
    print("="*80 + "\n")
    
    # Build command for init_netvlad_clusters.py
    project_root = Path.cwd()
    script_path = os.path.join(project_root, 'scripts/utility_scripts', 'init_netvlad_clusters.py')
    
    cmd = [
        sys.executable,  # Use same Python interpreter
        script_path,
        "--backbone", cfg.backbone.name,
        "--num_clusters", str(cfg.descriptor.kwargs.get("num_clusters", 64)),
        "--num_descriptors", "50000",
        "--image_size", str(cfg.data.image_size[0]), str(cfg.data.image_size[1]),
        "--output", os.path.join(cfg.logging.output_dir, "clusters"),
    ]
    
    # Add DINO-specific arguments if using DINO
    if cfg.backbone.name.startswith("dino"):
        local_path = os.getenv("DINOV3_LOCAL_PATH")
        weights_path = os.getenv("DINOV3_WEIGHTS_PATH")
        if local_path is not None:
            cmd.extend(["--local_path", to_absolute_path(local_path)])
        if weights_path is not None:
            cmd.extend(["--weights_path", to_absolute_path(weights_path)])
    
    if cfg.backbone.pretrained:
        cmd.append("--pretrained")

    if cfg.backbone.name == "mmfe":
        cmd.append("--checkpoint_path")
        cmd.append(to_absolute_path(cfg.backbone.kwargs.checkpoint_path))
    
    # Add dataset arguments
    if cfg.data.get("cubicasa") is not None and cfg.data.cubicasa.path is not None:
        cmd.extend(["--cubicasa_root", to_absolute_path(cfg.data.cubicasa.path)])
        if cfg.data.cubicasa.get("train"):
            cmd.extend(["--cubicasa_samples", to_absolute_path(cfg.data.cubicasa.train)])
    
    if cfg.data.get("structured3d") is not None and cfg.data.structured3d.path is not None:
        cmd.extend(["--s3d_root", to_absolute_path(cfg.data.structured3d.path)])
        if cfg.data.structured3d.get("train"):
            cmd.extend(["--s3d_samples", to_absolute_path(cfg.data.structured3d.train)])
    
    if cfg.data.get("aria_synthenv") is not None and cfg.data.aria_synthenv.path is not None:
        cmd.extend(["--aria_root", to_absolute_path(cfg.data.aria_synthenv.path)])
        if cfg.data.aria_synthenv.get("train"):
            cmd.extend(["--aria_samples", to_absolute_path(cfg.data.aria_synthenv.train)])
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the script
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate clusters. Exit code: {result.returncode}")
    
    # Construct the path to generated clusters
    num_clusters = cfg.descriptor.kwargs.get("num_clusters", 64)
    cluster_filename = f"netvlad_clusters_{cfg.backbone.name}_{num_clusters}.pth"
    cluster_path = os.path.join(cfg.logging.output_dir, "clusters", cluster_filename)
    
    print("\n" + "="*80)
    print(f"✓ Clusters generated successfully: {cluster_path}")
    print("="*80 + "\n")
    
    return cluster_path
