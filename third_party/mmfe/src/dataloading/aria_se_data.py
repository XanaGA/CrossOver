from __future__ import annotations

import os
import random
import json
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset

# ARIA ASE utilities (top-down renderings)
from aria_mmfe.code_snippets.readers import read_points_file, read_language_file
from aria_mmfe.code_snippets.interpreter import language_to_bboxes
from aria_mmfe.code_snippets.plotters import (
    render_pointcloud_and_boxes_orthographic_cv,
    change_params_resolution,
)
from aria_mmfe.ase_data.ase_utils import read_trajectory
from aria_mmfe.aria_images.aria_cv_tools import (
    get_device_camera_transform,
    pose_from_xyzq,
    undistort_image_fisheye,
    xyzq_from_pose,
    sample_scene_points,
    points_to_image_coords_from_params,
)


class AriaSynthEenvDataset(Dataset):
    """
    PyTorch Dataset for lazy-loading Meta Aria Synthetic Environments (ASE) scenes
    for contrastive learning, modeled after `Structured3DDataset` and `Cubicasa5kDataset`.

    Each item returns a dictionary with two modalities chosen from:
      - "points": orthographic density image of the semi-dense point cloud
      - "wireframe": orthographic wireframe image of language-derived 3D boxes

    You can choose to either generate these modalities on the fly from raw files
    (generate=True) or load them from pre-generated PNG images saved under each
    scene folder (generate=False). In the latter case, images are expected in
    subfolders `ann/` (for wireframe) and `points/` (for point density), with
    filenames keyed by a percentage integer (0..100), e.g. `50.png`.

    Parameters
    - root_dir: absolute path to the ASE dataset root containing scene id folders (e.g., .../SyntheticEnv/original_data/0)
    - scene_ids: optional explicit list of scene IDs to use (integers or strings). If None, auto-discovers numeric folders.
    - scene_ids_file: optional path to a .txt file with one scene id per line. If provided, overrides auto-discovery.
    - image_size: optional (H, W). If set, final image will be resized to this size
    - dual_transform: optional list of callables taking (img0, img1) and returning (img0, img1)
    - modality_pairs: list of tuples defining valid modality pairs. If None, all pairs are valid.
    - generate: if True, generate modalities from raw ASE files; else load from pre-generated images.
    - ortho_axis: projection axis for generation, one of {"x","y","z"}. Default: "z" (top-down).
    - render_size: generation size (H, W) for renderers. Defaults to (1080, 1080).
    """

    def __init__(
        self,
        root_dir: str,
        scene_ids: Optional[Sequence[Union[int, str]]] = None,
        scene_ids_file: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        dual_transform: Optional[Callable] = None,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        generate: bool = False,
        ortho_axis: str = "z",
        render_size: Tuple[int, int] = (1080, 1080),
        n_fpv_images: int = 0,
        fpv_transforms: Optional[List[Callable]] = None,
        load_depth: bool = True,
    ) -> None:
        super().__init__()

        if not os.path.isabs(root_dir):
            root_dir = os.path.abspath(root_dir)

        self.root_dir: str = root_dir
        # Directory for rendered outputs (one level above root_dir)
        parent_dir = os.path.dirname(self.root_dir)
        self.rendered_data: str = os.path.join(parent_dir, "rendered_data")
        os.makedirs(self.rendered_data, exist_ok=True)
        self.image_size: Optional[Tuple[int, int]] = tuple(image_size) if image_size is not None else None
        self.dual_transform: Optional[Callable] = dual_transform
        self.generate: bool = bool(generate)


        self.ortho_axis: str = ortho_axis
        self.render_size: Tuple[int, int] = tuple(render_size)
        self.n_fpv_images: int = int(n_fpv_images)
        self.fpv_transforms: Optional[List[Callable]] = fpv_transforms
        self.load_depth: bool = load_depth

        # Modalities for ASE
        self.modalities = ["points", "wireframe", "window", "wire_points"]
        self.modality_pairs = modality_pairs
        if self.modality_pairs is None:
            self.modality_pairs = [
                (m1, m2) for m1 in self.modalities for m2 in self.modalities if m1 != m2
            ]

        # Resolve scene ids
        if scene_ids_file is not None:
           scene_ids = self._load_ids_from_file(scene_ids_file)
        if scene_ids is None:
           scene_ids = self._discover_scene_ids()
        # Normalize to strings for folder names
        self.scene_ids: List[str] = [str(sid) for sid in scene_ids]
        # self.scene_ids: List[str] = [str(sid) for sid in range(5000)]
        if len(self.scene_ids) == 0:
            raise RuntimeError(f"No ASE scenes found in root_dir={self.root_dir}.")

        # Bind loaders depending on mode
        if self.generate:
            self.load_points = self._generate_points
            self.load_wireframe = self._generate_wireframe
            self.load_window = self._generate_window
            self.load_wire_points = self._generate_wire_points
        else:
            self.load_points = self._load_points_from_image
            self.load_wireframe = self._load_wireframe_from_image
            self.load_window = self._load_window_from_image
            self.load_wire_points = self._load_wire_points_from_image

    def _discover_scene_ids(self) -> List[str]:
        ids: List[str] = []
        if not os.path.isdir(self.root_dir):
            return ids
        for entry in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, entry)
            if os.path.isdir(full_path):
                # Expect numeric folder names like "0", "1", ...
                if entry.isdigit():
                    if self.generate:
                        ids.append(entry)
                    else:
                        ann_dir = os.path.join(full_path, "ann")
                        points_dir = os.path.join(full_path, "dense_points")
                        sim_points_dir = os.path.join(full_path, "sim_points")
                        if os.path.exists(ann_dir) and os.path.exists(points_dir) and os.path.exists(sim_points_dir):
                            ids.append(entry)
        ids.sort(key=lambda x: int(x) if x.isdigit() else x)
        return ids

    def _load_ids_from_file(self, ids_file: str) -> List[str]:
        ids: List[str] = []
        if not os.path.exists(ids_file):
            raise FileNotFoundError(f"Scene IDs file not found: {ids_file}")
        with open(ids_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)
        # validate
        valid: List[str] = []
        for sid in ids:
            scene_dir = os.path.join(self.root_dir, str(sid))
            if os.path.isdir(scene_dir):
                if self.generate:
                    valid.append(str(sid))
                else:
                    ann_dir = os.path.join(scene_dir, "ann")
                    points_dir = os.path.join(scene_dir, "dense_points")
                    sim_points_dir = os.path.join(scene_dir, "sim_points")
                    if os.path.exists(ann_dir) and os.path.exists(points_dir) and os.path.exists(sim_points_dir):
                        valid.append(str(sid))
        if len(valid) == 0:
            raise RuntimeError(f"No valid scene IDs found in file: {ids_file}")
        valid.sort(key=lambda x: int(x) if x.isdigit() else x)
        return valid

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _read_scene_raw(self, scene_id: str):
        scene_path = os.path.join(self.root_dir, scene_id)
        points_path = os.path.join(scene_path, "semidense_points.csv.gz")
        language_path = os.path.join(scene_path, "ase_scene_language.txt")
        points = read_points_file(points_path)
        entities = read_language_file(language_path)
        boxes, scene_aabb = language_to_bboxes(entities)
        return points, boxes

    def _resize_numpy(self, image_np: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        if size_hw is None:
            return image_np
        pil_img = Image.fromarray(image_np)
        height, width = int(size_hw[0]), int(size_hw[1])
        pil_img = pil_img.resize((width, height), resample=Image.BILINEAR)
        return np.array(pil_img)

    def _load_modality(self, scene_id: str, modality_type: str) -> np.ndarray:
        if modality_type == "points":
            return self.load_points(scene_id)
        elif modality_type == "wireframe":
            return self.load_wireframe(scene_id)
        elif modality_type == "window":
            return self.load_window(scene_id)
        elif modality_type == "wire_points":
            return self.load_wire_points(scene_id)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}. Available modalities: {self.modalities}")

    # -------- image-backed loading (generate=False) --------
    def _load_points_from_image(self, scene_id: str) -> np.ndarray:
        scene_path = os.path.join(self.root_dir, scene_id)
        points_path = os.path.join(scene_path, "dense_points", "dense_points.png")
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"Points file not found: {points_path}")
        img = cv2.imread(points_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load points image: {points_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._resize_numpy(img, self.image_size)

    def _load_wireframe_from_image(self, scene_id: str) -> np.ndarray:
        scene_path = os.path.join(self.root_dir, scene_id)
        ann_path = os.path.join(scene_path, "ann", "ann.png")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Wireframe file not found: {ann_path}")
        img = cv2.imread(ann_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load wireframe image: {ann_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._resize_numpy(img, self.image_size)

    def _load_window_from_image(self, scene_id: str) -> np.ndarray:
        scene_path = os.path.join(self.root_dir, scene_id)
        window_path = os.path.join(scene_path, "ann", "window.png")
        if not os.path.exists(window_path):
            raise FileNotFoundError(f"Window file not found: {window_path}")
        img = cv2.imread(window_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load window image: {window_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._resize_numpy(img, self.image_size)

    def _load_wire_points_from_image(self, scene_id: str) -> np.ndarray:
        scene_path = os.path.join(self.root_dir, scene_id)
        path = os.path.join(scene_path, "sim_points", "sim_points.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Wire points file not found: {path}")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load wire points image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._resize_numpy(img, self.image_size)

    # -------- generation from raw ASE --------
    def _generate_points(self, scene_id: str, return_params: bool = False) -> np.ndarray:
        points, boxes = self._read_scene_raw(scene_id)
        if return_params:
            point_img_bgr, _, params = render_pointcloud_and_boxes_orthographic_cv(
                points, boxes, axis=self.ortho_axis, image_size=self.render_size,
                return_params=True,
            )
            img = cv2.cvtColor(point_img_bgr, cv2.COLOR_BGR2RGB)
            image = self._resize_numpy(img, self.image_size)
            return image, params
        else:
            point_img_bgr, _ = render_pointcloud_and_boxes_orthographic_cv(
                points, boxes, axis=self.ortho_axis, image_size=self.render_size,
            )
            img = cv2.cvtColor(point_img_bgr, cv2.COLOR_BGR2RGB)
            image = self._resize_numpy(img, self.image_size)
            return image

    def _generate_wireframe(self, scene_id: str) -> np.ndarray:
        points, boxes = self._read_scene_raw(scene_id)
        _, wire_img_bgr = render_pointcloud_and_boxes_orthographic_cv(
            points, boxes, axis=self.ortho_axis, image_size=self.render_size, window_line_thickness=2
        )
        img = cv2.cvtColor(wire_img_bgr, cv2.COLOR_BGR2RGB)
        return self._resize_numpy(img, self.image_size)

    def _wire_to_points(self, wire_rgb: np.ndarray, points_density_inside_mask: float = 0.2, noise_std_pixels: float = 10.0) -> np.ndarray:
        # Convert to grayscale; background is white, lines are black
        gray = cv2.cvtColor(wire_rgb, cv2.COLOR_RGB2GRAY)
        # Sample points in background (white) areas only (>250 to tolerate AA)
        mask_background = gray < 250
        height, width = gray.shape

        mask_area = int(mask_background.sum())
        if mask_area == 0:
            return np.full((height, width, 3), 255, dtype=np.uint8)

        ys, xs = np.where(mask_background)

        # Smooth spatial density field for non-uniform sampling
        noise_field = np.random.rand(height, width).astype(np.float32)
        sigma = max(1.0, 0.05 * float(min(height, width)))
        # sigma = 10.0
        density_field = cv2.GaussianBlur(noise_field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        min_val = float(density_field.min())
        max_val = float(density_field.max())
        if max_val > min_val:
            density_field = (density_field - min_val) / (max_val - min_val)
        else:
            density_field.fill(1.0)
        gamma = float(np.random.uniform(1.6, 2.5))
        density_field = np.power(density_field + 1e-6, gamma).astype(np.float32)

        masked_weights = np.clip(density_field[ys, xs].astype(np.float64), 0.0, None)
        total_weight = float(masked_weights.sum())
        probs = None if total_weight <= 0.0 or not np.isfinite(total_weight) else (masked_weights / total_weight)
        if probs is not None:
            probs = np.clip(probs, 0.0, 1.0)
            s = float(probs.sum())
            if s == 0.0 or not np.isfinite(s):
                probs = None
            else:
                probs /= s
                if probs.size > 0:
                    residual = 1.0 - float(probs.sum())
                    probs[-1] = max(0.0, probs[-1] + residual)
                    s2 = float(probs.sum())
                    if not (s2 > 0.0 and np.isfinite(s2)):
                        probs = None

        target_points = max(1, min(int(points_density_inside_mask * mask_area), mask_area))
        if target_points == mask_area:
            idx = np.arange(mask_area)
        else:
            idx = np.random.choice(mask_area, size=target_points, replace=False, p=probs)
        y_base = ys[idx].astype(np.float32)
        x_base = xs[idx].astype(np.float32)

        if noise_std_pixels > 0:
            n = y_base.shape[0]
            y_noise = np.random.normal(0.0, noise_std_pixels, size=n).astype(np.float32)
            x_noise = np.random.normal(0.0, noise_std_pixels, size=n).astype(np.float32)
            if n > 0:
                outliers_percentage = np.random.uniform(0.1, 0.3)
                high_std_count = max(1, int(outliers_percentage * n))
                high_idx = np.random.choice(n, size=high_std_count, replace=False)
                scale = np.float32(5)
                y_noise[high_idx] *= scale
                x_noise[high_idx] *= scale
            y_base += y_noise
            x_base += x_noise

        valid = (y_base >= 0) & (y_base < height) & (x_base >= 0) & (x_base < width)
        y_pts = y_base[valid].astype(np.int32)
        x_pts = x_base[valid].astype(np.int32)

        out = np.full((height, width, 3), 255, dtype=np.uint8)
        for x, y in zip(x_pts, y_pts):
            cv2.circle(out, (int(x), int(y)), 1, (0, 0, 0), -1)
        return out

    def _generate_wire_points(self, scene_id: str) -> np.ndarray:
        wire = self._generate_wireframe(scene_id)
        points_img = self._wire_to_points(wire)
        return self._resize_numpy(points_img, self.image_size)

    def _generate_window(self, scene_id: str) -> np.ndarray:
        points, boxes = self._read_scene_raw(scene_id)
        # Emphasize windows/doors by thicker lines
        _, window_img_bgr = render_pointcloud_and_boxes_orthographic_cv(
            points, boxes, axis=self.ortho_axis, image_size=self.render_size, window_line_thickness=5
        )
        img = cv2.cvtColor(window_img_bgr, cv2.COLOR_BGR2RGB)
        return self._resize_numpy(img, self.image_size)


    ###########################################################################################
    #    LOAD FPV IMAGE 
    ###########################################################################################
    def _load_fpv_dict(self, scene_id: str) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with:
          - 'images': list of RGB fpv images randomly sampled from the scene
          - 'poses_floorplan': (N, 2) pixel coordinates of camera centers on the floorplan
          - 'params': projection params used for the mappings
        """
        scene_id_str = str(scene_id)

        # ---- Load projection params (saved during generate_and_save) ----
        params_path = os.path.join(self.rendered_data, scene_id_str, "params.json")
        if not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Projection params not found for scene {scene_id_str}: {params_path}"
            )
        with open(params_path, "r") as f:
            params = json.load(f)

        # ---- Resolve original ASE data root (for RGB + trajectory) ----
        parent_dir = os.path.dirname(self.root_dir)
        candidate = os.path.join(parent_dir, "original_data")
        original_root = candidate if os.path.isdir(candidate) else self.root_dir

        scene_rendered_dir = os.path.join(self.root_dir, scene_id_str) 
        rgb_dir = os.path.join(scene_rendered_dir, "images", "train", "rgb")
        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(
                f"RGB directory not found for scene {scene_id_str}: {rgb_dir}"
            )

        # ---- Load trajectory for this scene ----
        traj_df = read_trajectory(original_root, scene_id_str)

        # ---- Collect RGB frames with valid indices ----
        all_rgb_files = sorted(
            f
            for f in os.listdir(rgb_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        frame_candidates: List[Tuple[str, int]] = []
        for fname in all_rgb_files:
            stem = os.path.splitext(fname)[0]
            i_str = stem.replace("processed", "")
            try:
                frame_idx = int(i_str)
            except ValueError:
                continue
            if 0 <= frame_idx < len(traj_df):
                frame_candidates.append((fname, frame_idx))

        if not frame_candidates:
            raise RuntimeError(
                f"No RGB frames with valid trajectory indices found for scene {scene_id_str}"
            )

        num_fpv = max(1, int(self.n_fpv_images))
        num_fpv = min(num_fpv, len(frame_candidates))
        selected_frames = random.sample(frame_candidates, num_fpv)
        # selected_frames = [frame_candidates[0]]

        # ---- Prepare calibration (device -> camera) ----
        T_device_cam = get_device_camera_transform()

        images: List[np.ndarray] = []
        depths: List[np.ndarray] = []
        cam_centers_world: List[np.ndarray] = []
        cam_quats_world: List[np.ndarray] = []
        cam_thetas: List[float] = []

        for fname, frame_idx in selected_frames:
            img_path = os.path.join(rgb_dir, fname)
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

            if self.load_depth:
                depth_dir = rgb_dir.replace('rendered_data', 'original_data').replace('rgb', 'depth')
                depth_path = os.path.join(depth_dir, fname.replace('processed', 'depth').replace('.jpg', '.png'))
                depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth_raw, pinhole_calib = undistort_image_fisheye(depth_raw) 
                depth_raw = depth_raw / 1000.
                depths.append(depth_raw)

            row = traj_df.iloc[frame_idx]
            pts_xyz = row[["tx_world_device", "ty_world_device", "tz_world_device"]].values
            quat = row[
                ["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
            ].values

            pose_device = pose_from_xyzq(np.asarray(pts_xyz, dtype=np.float64),
                                         np.asarray(quat, dtype=np.float64))
            pose_cam = pose_device @ T_device_cam
            cam_xyz, cam_quat = xyzq_from_pose(pose_cam)
            cam_xyz = np.asarray(cam_xyz, dtype=np.float32)
            cam_quat = np.asarray(cam_quat, dtype=np.float64)

            cam_centers_world.append(cam_xyz)
            cam_quats_world.append(cam_quat.astype(np.float32))

            # Compute 2D orientation theta in floorplan/image coordinates.
            # Mirror the logic from overlay_single_pose: use camera forward [0,0,1].
            r = R.from_quat(cam_quat)
            forward = r.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
            fx, fy = forward[0], forward[1]
            
            dx_px = fx
            dy_px = fy
            theta = float(np.arctan2(dy_px, dx_px))
            theta = theta - np.pi / 2
            cam_thetas.append(theta)

        if not images or not cam_centers_world or not cam_quats_world:
            raise RuntimeError(f"Failed to load FPV images or poses for scene {scene_id_str}")

        cam_centers_world_np = np.stack(cam_centers_world, axis=0)  # (N,3)
        cam_thetas_np = np.asarray(cam_thetas, dtype=np.float32)    # (N,)


        fpv_dict = {
            "images": images,
            "pose_2D_world": {
                "xy": cam_centers_world_np[:, :2].astype(np.float32),
                "theta": cam_thetas_np,
            },
            "params": params,
        }
        if self.load_depth:
            fpv_dict["depths"] = depths
        return fpv_dict



    ###########################################################################################
    #    GETITEM 
    ###########################################################################################

    def __getitem__(self, index: int):
        scene_id = self.scene_ids[index]
        scene_dir = os.path.join(self.root_dir, scene_id)

        # Try different modality pairs until one succeeds
        modality_pairs_shuffled = self.modality_pairs.copy()
        random.shuffle(modality_pairs_shuffled)
        
        m0_image = None
        m1_image = None
        m0_type = None
        m1_type = None
        
        # Strategy 1: Try different modality pairs
        for m0_type_candidate, m1_type_candidate in modality_pairs_shuffled:
            try:
                m0_image = self._load_modality(scene_id, m0_type_candidate)
                m1_image = self._load_modality(scene_id, m1_type_candidate)
                m0_type = m0_type_candidate
                m1_type = m1_type_candidate
                break
            except (FileNotFoundError, Exception):
                continue
        
        # Strategy 2: If even same-modality failed, return black images
        if m0_image is None or m1_image is None:
            # Determine image size: use image_size if set, otherwise use render_size
            if self.image_size is not None:
                height, width = self.image_size
            else:
                height, width = self.render_size
            
            m0_image = np.zeros((height, width, 3), dtype=np.uint8)
            m1_image = np.zeros((height, width, 3), dtype=np.uint8)
            m0_type = "black"
            m1_type = "black"

        if self.dual_transform is not None:
            for t in self.dual_transform:
                m0_image, m1_image = t(m0_image, m1_image)


        return_dict = {
            "modality_0": m0_image,
            "modality_1": m1_image,
            "m0_type": m0_type,
            "m1_type": m1_type,
            "sample_id": scene_id,
            "sample_dir": scene_dir,
        }
        if self.n_fpv_images > 0:
            fpv_dict = self._load_fpv_dict(scene_id)
            if self.fpv_transforms is not None:
                fpv_images = fpv_dict["images"]
                for t in self.fpv_transforms:
                    fpv_images = t(fpv_images)
                fpv_dict["images"] = fpv_images
                if self.load_depth:
                    fpv_dict["depths"] = torch.from_numpy(np.array(fpv_dict["depths"])).float()
            return_dict["fpv_dict"] = fpv_dict 
        return return_dict

    def generate_and_save(self, index: int) -> Dict[str, str]:
        scene_id = self.scene_ids[index]
        scene_dir = os.path.join(self.rendered_data, scene_id)

        ann_dir = os.path.join(scene_dir, "ann")
        points_dir = os.path.join(scene_dir, "dense_points")
        sim_points_dir = os.path.join(scene_dir, "sim_points")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(points_dir, exist_ok=True)
        os.makedirs(sim_points_dir, exist_ok=True)

        # Generate modalities
        points_img, params = self._generate_points(scene_id, return_params=True)
        wire_img = self._generate_wireframe(scene_id)
        try:
            window_img = self._generate_window(scene_id)
        except Exception:
            window_img = None
        try:
            wire_points_img = self._generate_wire_points(scene_id)
        except Exception:
            wire_points_img = None

        # Save projection parameters for this scene
        params_path = os.path.join(scene_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        ann_path = os.path.join(ann_dir, "ann.png")
        points_path = os.path.join(points_dir, "dense_points.png")
        sim_points_path = os.path.join(sim_points_dir, "sim_points.png")
        window_path = os.path.join(ann_dir, "window.png")

        # Save as BGR for OpenCV
        cv2.imwrite(ann_path, cv2.cvtColor(wire_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(points_path, cv2.cvtColor(points_img, cv2.COLOR_RGB2BGR))
        if window_img is not None:
            cv2.imwrite(window_path, cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR))
        if wire_points_img is not None:
            cv2.imwrite(sim_points_path, cv2.cvtColor(wire_points_img, cv2.COLOR_RGB2BGR))

        return {
            "scene_id": scene_id,
            "ann_path": ann_path,
            "points_path": points_path,
            "window_path": window_path if window_img is not None else "",
            "sim_points_path": sim_points_path if wire_points_img is not None else "",
            "ann_dir": ann_dir,
            "dense_points_dir": points_dir,
            "sim_points_dir": sim_points_dir,
        }

    def get_all_sample_modalities(self, index: int):
        scene_id = self.scene_ids[index]

        modalities_dict = {}

        for modality_type in self.modalities:
            modality_dict = {}
            try:
                img = self._load_modality(scene_id, modality_type)
                if self.dual_transform is not None:
                    pil_image = Image.fromarray(img)
                    img = pil_image
                    for t in self.dual_transform:
                        img = t(img)
                key = "0"
                modality_dict[key] = img
            except Exception as e:
                print(
                    f"Warning: Failed to load {modality_type} for scene {scene_id}: {e}"
                )
            modalities_dict[modality_type] = modality_dict

        return modalities_dict


