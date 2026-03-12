import os
import gzip
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R

from aria_mmfe.aria_images.aria_cv_tools import devignette_image_numpy, load_aria_vignette, mask_from_vignette, undistort_image_fisheye, world_to_pixel, get_projection_params
from aria_mmfe.ase_data.ase_utils import read_3d_boxes, read_trajectory, read_points
from aria_mmfe.code_snippets.plotters import render_pointcloud, overlay_trajectory, overlay_single_pose, render_pointcloud_and_boxes_orthographic_cv
from mmfe_utils.viz_utils import resize_with_padding

UNDISTORT = True # False for ASE fisheye, True for pinhole
SHOW_DEPTH = True
SHOW_MASK = True

class FloorplanVisualizer:
    def __init__(self, root_dir, image_size=(1024, 1024), margin_frac=0.05):
        self.root_dir = root_dir
        self.image_size = image_size
        self.margin_frac = margin_frac

    ### ---------------------------------------------------------------
    ### NEW: show every RGB frame with its pose on the floorplan
    ### ---------------------------------------------------------------
    def visualize_frame_sequence(self, scene_id, base_map, traj_df, params):
        rgb_dir = os.path.join(self.root_dir, scene_id, "images", "train", "rgb")
        rgb_files = sorted(os.listdir(rgb_dir))

        depth_dir = os.path.join(self.root_dir, scene_id, "images", "train", "depth")
        depth_files = sorted(os.listdir(depth_dir))

        # vignette_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "aria_mmfe", "aria_images", "vignette_aria.png")
        vignette_img = load_aria_vignette()

        print(f"Showing {len(rgb_files)} frames... (ESC to exit)")

        for fname, depth_fname in zip(rgb_files, depth_files):
            # Extract i from fname, e.g. from 'vignette0000060.jpg' → 60
            stem = os.path.splitext(fname)[0]
            i_str = stem.replace('vignette', '')
            try:
                i = int(i_str)
            except ValueError:
                continue  # skip files that don't fit naming pattern

            fpath_rgb = os.path.join(rgb_dir, fname)
            fpath_depth = os.path.join(depth_dir, depth_fname)
            pose_img = overlay_single_pose(base_map, traj_df, params, i)

            rgb = cv2.imread(fpath_rgb)
            rgb = devignette_image_numpy(rgb, vignette_img)
            depth = cv2.imread(fpath_depth, cv2.IMREAD_UNCHANGED)

            if UNDISTORT:
                rgb, pinhole_calib = undistort_image_fisheye(rgb)
                depth, pinhole_calib = undistort_image_fisheye(depth)

                # mask = mask_from_vignette(vignette_img)
                mask = load_aria_vignette(binary=True)
                mask, pinhole_calib = undistort_image_fisheye(mask)
                mask = np.stack([mask, mask, mask], axis=2)
                mask_vis = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                mask_vis = cv2.resize(mask_vis, (800, 600))


            if rgb is None:
                continue

            if depth is None:
                continue

            # --- Rotate RGB 90 degrees clockwise ---
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

            # Resize after rotation
            rgb_vis = cv2.resize(rgb, (800, 600))
            depth_vis = cv2.resize(depth, (800, 600))
            map_vis = resize_with_padding(pose_img, 800, 600)

            image_stack = [rgb_vis, map_vis]
            if SHOW_DEPTH:
                depth_vis = cv2.cvtColor(((depth_vis/depth_vis.max())*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                image_stack.append(depth_vis)

            if SHOW_MASK and UNDISTORT:
                image_stack.append(mask_vis)

            combined = np.hstack(image_stack)

            cv2.imshow("RGB Frame + Pose on Map", combined)
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                break

    def process_scene(self, scene_id):
        print(f"Processing {scene_id}...")
        
        # original_data_path = os.path.join((os.path.dirname(self.root_dir)), "original_data")
        original_data_path = self.root_dir
        points = read_points(original_data_path, scene_id)
        boxes = read_3d_boxes(original_data_path, scene_id)
        traj_df = read_trajectory(self.root_dir, scene_id)
        
        if len(points) == 0:
            print("No points found.")
            return

        # params = get_projection_params(points[:, :2], boxes, self.image_size, self.margin_frac)
        
        base_map, _, params = render_pointcloud_and_boxes_orthographic_cv(
            points, boxes, axis="z", image_size=self.image_size, return_params=True
        )

        # base_map = render_pointcloud(points, params, self.image_size)
        map_with_traj = overlay_trajectory(base_map.copy(), traj_df, params)

        ### Show global map
        cv2.imshow("Full Trajectory Overlay", map_with_traj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ### NEW: show all frames
        self.visualize_frame_sequence(scene_id, base_map, traj_df, params)

if __name__ == "__main__":
    root_directory = "/local/home/xanadon/mmfe/data/aria/SyntheticEnv/original_data" 
    scene_name = "0"

    viz = FloorplanVisualizer(root_directory)
    viz.process_scene(scene_name)
