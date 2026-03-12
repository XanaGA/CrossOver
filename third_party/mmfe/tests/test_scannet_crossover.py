import os
import cv2
import numpy as np
import open3d as o3d
import torch

from third_parties.CrossOver.render_utils import crop_image, load_and_center_mesh, render_pointcloud_density, render_scene, resize_with_padding

# -------------------------------------------------------------------
# Choose a scene
# -------------------------------------------------------------------
base_path = "data/scannet_zip"
chunk = "chunk_01"
scene = "scene0001_00"

scene_path = os.path.join(base_path, chunk, scene)
print("Using scene:", scene_path)

# -------------------------------------------------------------------
# 3. Load data3D.npz
# -------------------------------------------------------------------
npz_path = os.path.join(scene_path, "data3D.npz")
data3D = np.load(npz_path, allow_pickle=True)

# -------------------------------------------------------------------
# 4. Load data3D_pointbind.pt (PyTorch tensor or dict)
# -------------------------------------------------------------------
pt_path = os.path.join(scene_path, "data3D_pointbind.pt")
data3D_pointbind = torch.load(pt_path, weights_only=False)

print("\nLoaded data3D_pointbind.pt:")
print(type(data3D_pointbind))

if isinstance(data3D_pointbind, dict):
    print("Keys:", data3D_pointbind.keys())


# -------------------------------------------------------------------
# 5. Scene-level exploration
# -------------------------------------------------------------------
# Scene-level exploration
scene = data3D["scene"].item()
scene_pb = data3D_pointbind["scene"]
padding = 50
resolution = 1024

# Mesh
mesh_img = render_scene(os.path.join(scene_path, "floor+obj.ply"))
mesh_img = crop_image(np.array(mesh_img), pad=0)
mesh_img = resize_with_padding(mesh_img, resolution, padding)
cv2.imshow("Mesh Render", mesh_img)

# Point cloud
coords = scene["pcl_coords"]
pc_img = render_pointcloud_density(coords, padding=padding, resolution=resolution, point_size=1)
cv2.imshow("PointCloud Render", pc_img)

# Point cloud from mesh (sampled points)
mesh = load_and_center_mesh(os.path.join(scene_path, "floor+obj.ply"))
# Sample points from mesh surface (sample number of points proportional to mesh area)
num_samples = min(50000, len(coords))  # Use similar number of points as original point cloud
mesh_points = mesh.sample(num_samples)
mesh_pc_img = render_pointcloud_density(mesh_points, padding=padding, resolution=resolution, point_size=1, axis="z", rotate=False)
cv2.imshow("PointCloud From Mesh Render", mesh_pc_img)

cv2.waitKey(0)

