import torch
from fpv.fpv_3D_utils import FrustumConfig, precompute_frustum_grid
from fpv.fpv_viz_utils import visualize_frustum_2d, visualize_frustum_3d

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FrustumConfig()

    print("Precomputing Frustum...")
    coords = precompute_frustum_grid(config, device)

    # Validate shapes match your requirements
    print(f"3D Coords Shape: {coords.coords_3d_local.shape}") # [200, 200, 10, 3]
    print(f"UV Norm Shape:   {coords.coords_uv_norm.shape}")   # [200, 200, 10, 2]

    visualize_frustum_2d(coords)
    visualize_frustum_3d(coords)