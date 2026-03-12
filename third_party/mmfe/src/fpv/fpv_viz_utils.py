import cv2
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from aria_mmfe.aria_images.aria_cv_tools import load_aria_vignette, mask_from_vignette, undistort_image_fisheye
from fpv.fpv_3D_utils import FrustumCoordinates

def visualize_frustum_2d(frustum_coords: FrustumCoordinates):
    # 1. Prepare the Background (Undistorted Vignette)
    vignette_img = load_aria_vignette(rotate=False, as_pil=False, binary=False)
    mask = mask_from_vignette(vignette_img, binary=False)
    # Undistort the mask
    undistorted_mask, _ = undistort_image_fisheye(mask)
    
    # Convert to BGR for colored plotting
    vis_img = cv2.cvtColor(undistorted_mask, cv2.COLOR_GRAY2BGR)
    h, w = vis_img.shape[:2]

    # 2. Prepare Coordinates
    # Reshape from [200, 200, 10, 2] -> [N, 2]
    uv_norm = frustum_coords.coords_uv_norm.cpu().numpy().reshape(-1, 2)
    valid_mask = frustum_coords.valid_mask_3D.cpu().numpy().reshape(-1)

    # Convert normalized [-1, 1] to pixel coordinates [0, W/H]
    u_px = (uv_norm[:, 0] + 1) * 0.5 * (w - 1)
    v_px = (uv_norm[:, 1] + 1) * 0.5 * (h - 1)

    # 3. Plotting (Subsampled for performance and clarity)
    # Total points = 400,000. Let's plot every 50th point.
    step = 1 
    for i in range(0, len(u_px), step):
        pt = (int(u_px[i]), int(v_px[i]))
        
        # Check if point is within image bounds before drawing
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            color = (0, 255, 0) if valid_mask[i] > 0.5 else (0, 0, 255) # Green vs Red
            cv2.circle(vis_img, pt, 1, color, -1)

    # OpenCV uses BGR, matplotlib expects RGB
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_img_rgb)
    plt.title("Frustum Projection (Green=Valid, Red=Invalid)")
    plt.axis("off")
    plt.show()


def visualize_frustum_3d(frustum_coords: FrustumCoordinates):
    geometries = []

    # 1. Add Coordinate Frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(axes)

    # 2. Visualize 3D Frustum Points (coords_3d_local)
    # Shape: [200, 200, 10, 3] -> [N, 3]
    pts_3d = frustum_coords.coords_3d_local.cpu().numpy().reshape(-1, 3)
    mask_3d = frustum_coords.valid_mask_3D.cpu().numpy().reshape(-1)
    
    pcd_3d = o3d.geometry.PointCloud()
    pcd_3d.points = o3d.utility.Vector3dVector(pts_3d)
    
    # Colors: Valid=Green [0,1,0], Invalid=Red [1,0,0]
    colors_3d = np.zeros((pts_3d.shape[0], 3))
    colors_3d[mask_3d > 0.5] = [0, 1, 0]
    colors_3d[mask_3d <= 0.5] = [1, 0, 0]
    pcd_3d.colors = o3d.utility.Vector3dVector(colors_3d)
    geometries.append(pcd_3d)

    # 3. Visualize 2D Ground Grid (coords_xy_local)
    # Shape: [200, 200, 2] -> [M, 2]. We add Z=0 for 3D space.
    pts_xy = frustum_coords.coords_xy_local.cpu().numpy().reshape(-1, 2)
    mask_xy = frustum_coords.valid_mask_xy.cpu().numpy().reshape(-1)
    
    # Append 0 for Z dimension to visualize on the "floor"
    pts_xy_3d = np.hstack([pts_xy, np.zeros((pts_xy.shape[0], 1))])
    
    pcd_xy = o3d.geometry.PointCloud()
    pcd_xy.points = o3d.utility.Vector3dVector(pts_xy_3d)
    
    # Colors: Valid=Blue [0,0,1], Invalid=Black [0,0,0]
    colors_xy = np.zeros((pts_xy_3d.shape[0], 3))
    colors_xy[mask_xy > 0.5] = [0, 0, 1]
    colors_xy[mask_xy <= 0.5] = [0, 0, 0]
    pcd_xy.colors = o3d.utility.Vector3dVector(colors_xy)
    geometries.append(pcd_xy)

    # 4. Render
    o3d.visualization.draw_geometries(geometries, window_name="Frustum 3D (Green/Red) & XY Local (Blue/Black)")