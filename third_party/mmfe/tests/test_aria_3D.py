import os
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from projectaria_tools.projects import ase

from scipy.spatial.transform import Rotation as R
import torch

from aria_mmfe.aria_images.aria_cv_tools import (devignette_image_numpy, filter_uvs_by_mask, get_device_camera_transform, load_aria_vignette, mask_from_vignette, 
                                                matrix_from_pinhole_calibration, undistort_image_fisheye, 
                                                unproject_2d_points_fisheye, unproject_2d_points_pinhole, pose_from_xyzq, xyzq_from_pose)
from aria_mmfe.ase_data.ase_utils import read_3d_boxes
from aria_mmfe.code_snippets.plotters import (change_params_resolution, overlay_grid, overlay_points, 
                                            overlay_trajectory, overlay_single_pose, 
                                            render_pointcloud_and_boxes_orthographic_cv)
from fpv.fpv_3D_utils import FrustumConfig, precompute_frustum_grid, transform_fustrums_to_floorplan, compute_projected_frustum

USE_ASE_FISHEYE = False # True for ASE fisheye, False for pinhole (Not implemented yet)
# ----------------------------- Visualizer (Open3D) -------------------------

def visualize_with_open3d(root_dir, scene_id, use_ase_fisheye=True, example_frame_index=0):
    """ Visual test harness that shows:
    - semidense pointcloud
    - camera poses from trajectory.csv as coordinate frames
    - an example unprojected depth frame (converted to point cloud and shown as a small overlay)

    Parameters:
    - use_ase_fisheye: if True, attempt to use ASE fisheye calibration + unprojection; otherwise use dataset pinhole K fallback
    - example_frame_index: index in the rgb directory to use for demonstration (int)
    """
    try:
        import open3d as o3d
    except Exception:
        raise RuntimeError("open3d required for visualization. Install via pip install open3d")

    # ---------------- read data ----------------
    # original_data_path = os.path.join((os.path.dirname(root_dir)), "original_data")
    original_data_path = root_dir
    points_path = os.path.join(original_data_path, scene_id, "semidense_points.csv.gz")
    boxes = read_3d_boxes(original_data_path, scene_id)
    vignette_img = load_aria_vignette()

    points = np.array([])
    try:
        df_pts = pd.read_csv(points_path, compression='gzip')
        points = df_pts.iloc[:, 2:5].values
    except Exception as e:
        print(f"Failed to read points: {e}")

    traj_path = os.path.join(root_dir, scene_id, "images", "train", "trajectory.csv")
    traj_df = pd.read_csv(traj_path)

    rgb_dir = os.path.join(root_dir, scene_id, "images", "train", "rgb")
    depth_dir = os.path.join(root_dir, scene_id, "images", "train", "depth")

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg'))])

    if len(rgb_files) == 0:
        raise RuntimeError("No rgb files found")

    # choose example frame
    example_fname = rgb_files[example_frame_index % len(rgb_files)]
    example_depth_fname = depth_files[example_frame_index % len(depth_files)] if len(depth_files)>0 else None

    # ---------------- ASE calibration (optional) ----------------
    ase_calib = ase.get_ase_rgb_calibration()
    print("Using ASE calibration for fisheye operations")

    # ---------------- build visual elements ----------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Scene {scene_id}")

    if points.size:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(pcd)

    # add cameras as coordinate frames
    calibration = ase.get_ase_rgb_calibration()
    T_device_cam = calibration.get_transform_device_camera().to_matrix()
    T_cam_device = np.linalg.inv(T_device_cam)
    camera_frames = []
    for idx, row in traj_df.iterrows():
        tx = row['tx_world_device']; ty = row['ty_world_device']; tz = row['tz_world_device']
        q = row[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].values
        pose = pose_from_xyzq(np.array([tx,ty,tz]), q)
        pose = pose @ T_device_cam
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.transform(pose)
        vis.add_geometry(frame)
        camera_frames.append(frame)

    # unproject example depth frame
    if example_depth_fname is not None:
        depth_path = os.path.join(depth_dir, example_depth_fname)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print("Could not read depth image")
        else:

            # rotate 90 deg clockwise to match actual orientation discussed (images are rotated CCW in storage)
            # depth_raw = cv2.rotate(depth_raw, cv2.ROTATE_90_CLOCKWISE)
            rgb_path = os.path.join(rgb_dir, example_fname)
            rgb = cv2.imread(rgb_path)
            rgb = devignette_image_numpy(rgb, vignette_img) 
            if not USE_ASE_FISHEYE:
                rgb, pinhole_calib = undistort_image_fisheye(rgb)
                depth_raw, pinhole_calib = undistort_image_fisheye(depth_raw)
                mask = mask_from_vignette(vignette_img, binary=True)
                mask, pinhole_calib = undistort_image_fisheye(mask)

            h,w = depth_raw.shape[:2]
            # create pixel grid and sample a subset to unproject
            grid_step = 8
            us, vs = np.meshgrid(np.arange(0, w, grid_step), np.arange(0, h, grid_step))
            us = us.flatten(); vs = vs.flatten()
            if not USE_ASE_FISHEYE:
                valid_uvs = filter_uvs_by_mask([us, vs], mask)
                us = valid_uvs[:,0]; vs = valid_uvs[:,1]

            depths = depth_raw[vs, us].astype(np.float32)

            # Construct pts Nx3
            pts_uvd = np.stack([us.astype(np.float32), vs.astype(np.float32), depths.astype(np.float32)], axis=1)

            # get camera pose for this frame from traj_df by index name matching 'vignette{index}' if possible
            # fallback: take the first row
            try:
                # try to infer index number from filename
                stem = os.path.splitext(example_fname)[0]
                i_str = stem.replace('vignette', '')
                frame_idx = int(i_str)
                row_pose = traj_df.iloc[frame_idx]
            except Exception:
                row_pose = traj_df.iloc[0]

            tx = row_pose['tx_world_device']; ty = row_pose['ty_world_device']; tz = row_pose['tz_world_device']
            q = row_pose[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].values
            pose = pose_from_xyzq(np.array([tx,ty,tz]), q)
            pose = pose @ T_device_cam
            if use_ase_fisheye and ase_calib is not None:
                world_pts, valid_uvs = unproject_2d_points_fisheye(pts_uvd, pose, ase_calibration=ase_calib)
            else:
                pinhole_K = matrix_from_pinhole_calibration(pinhole_calib)
                world_pts = unproject_2d_points_pinhole(pts_uvd, pose, pinhole_K)

            # create Open3D point cloud for these points and color from rgb
            pcd_example = o3d.geometry.PointCloud()
            pcd_example.points = o3d.utility.Vector3dVector(world_pts)

            colors = rgb[valid_uvs[:,1], valid_uvs[:,0], ::-1].astype(np.float32) / 255.0  # BGR->RGB
            pcd_example.colors = o3d.utility.Vector3dVector(colors)
            # pcd_example.paint_uniform_color([1,0,0])
            vis.add_geometry(pcd_example)

    vis.run()
    vis.destroy_window()

    visualize_2d(traj_df, pcd, pcd_example, boxes, example_frame_index)


# ----------------------------- 2D visualization -----------------

def visualize_2d(traj_df, floorplan_pcd, local_3d_pcd, boxes, index_frame: int = None):
    """
    Simple top-down 2D matplotlib visualization showing:
      - Camera positions (X,Y)
      - Floorplan / semidense points projected to XY
      - Local depth-unprojection points projected to XY
    """
    points = np.asarray(floorplan_pcd.points)

    base_map, _, params = render_pointcloud_and_boxes_orthographic_cv(
        points, boxes, axis="z", image_size=(1024, 1024), return_params=True
    )

    base_map = overlay_grid(base_map.copy(), params)

    if index_frame is None:
        map_with_traj = overlay_trajectory(base_map.copy(), traj_df, params)
    else:
        map_with_traj = overlay_single_pose(base_map.copy(), traj_df, params, index_frame)

    full_map = overlay_points(map_with_traj.copy(), np.asarray(local_3d_pcd.points)[:, :2], params)

    ### Show global map
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {}
    for k, v in params.items():
        out[k] = torch.tensor([v], device=device, dtype=torch.float32)

    params = out

    # OpenCV BGR → RGB
    full_map_rgb = cv2.cvtColor(full_map, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(full_map_rgb)
    plt.title("GT Pose and Depth")
    plt.axis("off")

    # Plot the transformed fustrum
    config = FrustumConfig()
    coords = precompute_frustum_grid(config, device)

    fpv_feats = torch.rand((1, 1, 32, 704, 704)).to(device)
    fpv_depth = torch.rand((1, 1, 10, 704, 704)).to(device)
    frustum_data = compute_projected_frustum(fpv_depth, fpv_feats, coords, FrustumConfig())

    T_device_cam = get_device_camera_transform()
    row = traj_df.iloc[index_frame]
    pts_xyz = row[["tx_world_device", "ty_world_device", "tz_world_device"]].values
    quat = row[
        ["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
    ].values

    pose_device = pose_from_xyzq(np.asarray(pts_xyz, dtype=np.float64),
                                    np.asarray(quat, dtype=np.float64))
    pose_cam = pose_device @ T_device_cam
    cam_xyz, cam_quat = xyzq_from_pose(pose_cam)
    cam_quat = np.asarray(cam_quat, dtype=np.float64)


    # Compute 2D orientation theta in floorplan/image coordinates.
    # Mirror the logic from overlay_single_pose: use camera forward [0,0,1].
    r = R.from_quat(cam_quat)
    forward = r.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    fx, fy = forward[0], forward[1]
    dx_px = fx
    dy_px = fy
    theta = float(np.arctan2(dy_px, dx_px))
    theta = theta - np.pi / 2

    gt_poses = torch.cat([torch.tensor(cam_xyz[..., :2]), torch.tensor(theta)[None]], dim=-1)[None, None].to(device) # (B, N, 3)
    
    # Resize the map
    map_with_traj = cv2.resize(map_with_traj,(256, 256), interpolation=cv2.INTER_AREA)
    params = change_params_resolution(params, (256, 256))

    transformed_frustrum = transform_fustrums_to_floorplan(frustum_data, gt_poses, params)
    
    fustrum_px = transformed_frustrum[0].coords_proj_xy
    valid_mask = transformed_frustrum[0].valid_mask_xy.squeeze(0)
    fustrum_px_in = fustrum_px[valid_mask]
    fustrum_px_out = fustrum_px[~valid_mask.bool()]

    fustrum_px_in = np.asarray(fustrum_px_in.squeeze(0).view(-1,2).detach().cpu())
    fustrum_px_out = np.asarray(fustrum_px_out.squeeze(0).view(-1,2).detach().cpu())

    full_map = overlay_points(map_with_traj.copy(), fustrum_px_in, colors=np.array([0,255,0])[None].repeat(len(fustrum_px_in), 0))
    full_map = overlay_points(full_map.copy(), fustrum_px_out)

    full_map_rgb = cv2.cvtColor(full_map, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(full_map_rgb)
    plt.title("GT Pose and TF Fustrum")
    plt.axis("off")

    plt.show()

# ----------------------------- Helper read_points for user -----------------

def read_points(root_dir, scene_id):
    pc_root_dir = os.path.join((os.path.dirname(root_dir)), "original_data")
    points_path = os.path.join(pc_root_dir, scene_id, "semidense_points.csv.gz")
    try:
        df = pd.read_csv(points_path, compression='gzip')
        points = df.iloc[:, 2:5].values
        return points
    except Exception as e:
        print(f"Error reading points: {e}")
        return np.array([])

# ----------------------------- If run as script ----------------------------
if __name__ == '__main__':
    # Example usage - update root_directory and scene_name to your data location
    root_directory = "/home/xavi/mmfe/data/aria/SyntheticEnv/original_data"
    scene_name = "7"

    # Visualize using ASE fisheye if available; set False to use pinhole fallback
    visualize_with_open3d(root_directory, scene_name, use_ase_fisheye=USE_ASE_FISHEYE, example_frame_index=0)
