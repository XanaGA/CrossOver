"""
Utilities and visualizer for Aria synthetic environments.
- Implements vignette loading/mask utilities
- Pose and rotation helpers
- Unprojection for fisheye / pinhole (ASE-aware if available)
- Undistort (ASE-aware if available)
- Test / visualizer using Open3D that shows pointcloud, camera poses and an unprojected depth->rgb view

Notes:
- This code tries to use `projectaria_tools.projects.ase` when available. If not found, it falls back to using dataset pinhole K when possible.
- Depth images in Aria synthetic dataset are uint16 and store depth **along the pixel ray** in mm (not Z-depth). See user notes.

Make sure you have these Python packages installed: numpy, pandas, open3d, opencv-python, scipy, projectaria-tools (optional), torch (optional).

"""

import os
from typing import Union, Tuple
from PIL import Image, ImageOps
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import torch
from projectaria_tools.projects import ase
from projectaria_tools.core import calibration as aria_core_calibration

ARIA_IMAGE_WIDTH = 704
ARIA_IMAGE_HEIGHT = 704
ARIA_FOCAL_LENGTH_X = 297.6375381033778

# ----------------------------- Basic utilities -----------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _to_torch(x, device=None, dtype=None):
    t = torch.from_numpy(np.array(x))
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.type(dtype)
    return t

# ----------------------------- Vignette -----------------------------------

def load_aria_vignette(path=None, rotate=False, as_pil=True, binary=False):
    """ Loads the vignette image from the given path and rotates it if specified.
    Defaults to a sibling path `vignette_aria.png` when path is None.
    """
    if path is None:
        if binary:
            path = os.path.join(os.path.dirname(__file__), 'vignette_aria_binary.png')
        else:
            path = os.path.join(os.path.dirname(__file__), 'vignette_aria.png')

    if as_pil:
        img = Image.open(path)
    else:
        img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Vignette not found at {path}")
    if rotate and not as_pil:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate and as_pil:
        img = img.rotate(90)
    return img

def mask_from_vignette(img, dilation_kernel_size=0, binary=False):
    """ Creates a binary mask (0 where vignette is white 255) from the vignette image.
    Expects uint8 image.
    """
    img = np.asarray(img)
    if img.ndim == 3:
        # Check across channels
        white = np.all(img == 255, axis=2)
    else:
        white = (img == 255)

    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    mask[white] = 0
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    if binary:
        mask = mask > 128
    return mask

def devignette_image_numpy(img: Union[np.ndarray, torch.Tensor], vignette: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Devignette an image using the vignette mask.
    """
    rgb = _to_numpy(img)
    # vignette = _to_numpy(vignette)

    # Convert to 'L' mode (8-bit pixels, black and white)
    vignette = vignette.convert('L')

    # Invert the vignette mask and convert to numpy array
    inverted_mask = ImageOps.invert(vignette)
    inverted_mask_array = np.array(inverted_mask).astype(np.float32)[:, :, None] / 255.0

    # Apply the inverted mask to remove the vignette
    devingetted_array = rgb.astype(np.float32) / inverted_mask_array
    devingetted_array = np.clip(devingetted_array, 0, 255).astype(np.uint8)

    return devingetted_array

# ----------------------------- Pose helpers --------------------------------

def rotate_pose_3d(pose: Union[np.ndarray, torch.Tensor], angle: float, axis: str = 'x') -> Union[np.ndarray, torch.Tensor]:
    """ Rotates a 3D pose (4x4 transformation matrix) by `angle` radians around the pose-local axis `axis` ("x","y","z").
    Returns a matrix of the same type (numpy or torch).
    """
    mat = _to_numpy(pose)
    assert mat.shape == (4, 4), "pose must be 4x4"

    rotvec = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0])
    }[axis.lower()]

    R_delta = R.from_rotvec(rotvec * angle).as_matrix()

    new_R = R_delta @ mat[:3, :3]
    new_mat = mat.copy()
    new_mat[:3, :3] = new_R

    if isinstance(pose, torch.Tensor):
        return _to_torch(new_mat, device=pose.device if hasattr(pose, 'device') else None)
    return new_mat

# ----------------------------- 2D image rotation ---------------------------

def rotate_2d_image_clockwise(img: Union[np.ndarray, torch.Tensor], angle: float) -> Union[np.ndarray, torch.Tensor]:
    """ Rotate a 2D image clockwise by angle degrees. Accepts numpy array (H,W,C) or torch tensor.
    Uses cv2.warpAffine.
    """
    arr = _to_numpy(img)
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # negative for clockwise
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if isinstance(img, torch.Tensor):
        return _to_torch(rotated, device=img.device if hasattr(img, 'device') else None)
    return rotated

# ----------------------------- Pose <-> xyzq --------------------------------

def pose_from_xyzq(xyz: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Convert (3,) xyz and (4,) quaternion (qx,qy,qz,qw) to 4x4 pose matrix.
    """
    if (isinstance(xyz, torch.Tensor) or isinstance(q, torch.Tensor)):
        return pose_from_xyzq_torch(xyz, q)

    t = xyz.reshape(3)
    quat = q.reshape(4)
    norm = np.linalg.norm(quat)
    # Safety Check: Avoid dividing by zero if q is [0,0,0,0]
    if norm < 1e-6:
        # Fallback to identity quaternion [0, 0, 0, 1] or raise error
        quat = np.array([0.0, 0.0, 0.0, 1.0]) 
    else:
        quat = quat / norm  # Normalize
    # scipy Rotation.from_quat expects [x,y,z,w]
    R_mat = R.from_quat(quat).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_mat
    pose[:3, 3] = t
    return pose

import torch

def pose_from_xyzq_torch(xyz: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """ 
    Differentiable PyTorch implementation.
    Input: xyz (3,), q (4,) in (x, y, z, w) format
    """
    # 1. Normalize quaternion to ensure valid rotation
    q = q / torch.norm(q)
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    # 2. Construct Rotation Matrix (from scratch for differentiability)
    # Formula for conversion from x,y,z,w quaternion
    R = torch.stack([
        1 - 2*(y**2 + z**2),  2*(x*y - z*w),      2*(x*z + y*w),
        2*(x*y + z*w),        1 - 2*(x**2 + z**2), 2*(y*z - x*w),
        2*(x*z - y*w),        2*(y*z + x*w),      1 - 2*(x**2 + y**2)
    ]).reshape(3, 3)
    
    # 3. Construct 4x4 Pose
    pose = torch.eye(4, device=xyz.device, dtype=xyz.dtype)
    pose[:3, :3] = R
    pose[:3, 3] = xyz
    
    return pose


def xyzq_from_pose(pose: Union[np.ndarray, torch.Tensor]) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """ Convert 4x4 pose to xyz and quaternion (qx,qy,qz,qw).
    Returns (xyz, quat)
    """
    mat = _to_numpy(pose)
    t = mat[:3, 3].copy()
    rot = R.from_matrix(mat[:3, :3])
    quat = rot.as_quat()  # x,y,z,w
    if isinstance(pose, torch.Tensor):
        return _to_torch(t), _to_torch(quat)
    return t, quat

def get_device_camera_transform():
    ase_calibration = ase.get_ase_rgb_calibration()
    T_device_cam = ase_calibration.get_transform_device_camera().to_matrix()
    return T_device_cam

# ----------------------------- Unprojection --------------------------------

def unproject_2d_points_pinhole(pts: Union[np.ndarray, torch.Tensor], 
                                 pose: Union[np.ndarray, torch.Tensor], 
                                 K: Union[np.ndarray, torch.Tensor],
                                 depth_is_euclidean: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """ Unproject 2D points (u,v,depth).
    - depth_is_euclidean: Set True if input depth is ray length (common in fisheye/Aria). 
                          Set False if input depth is planar Z (standard pinhole).
    """
    pts = _to_numpy(pts)
    K = _to_numpy(K)
    pose = _to_numpy(pose)
    assert pts.shape[1] == 3, "pts must be Nx3 (u,v,depth)"

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = pts[:, 0]
    v = pts[:, 1]
    d = pts[:, 2] / 1000.0  # convert mm -> meters

    # 1. Calculate normalized coordinates (z=1 plane)
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    
    # 2. Convert Euclidean Depth (Ray Length) to Planar Z-Depth if needed
    if depth_is_euclidean:
        # Scale factor = 1 / norm([x_norm, y_norm, 1])
        # This converts the full ray length 'd' into the 'z' component
        scale = 1.0 / np.sqrt(x_norm**2 + y_norm**2 + 1)
        z = d * scale
    else:
        z = d

    # 3. Standard Pinhole Unprojection using Z
    x_cam = x_norm * z
    y_cam = y_norm * z
    xyz_cam = np.stack([x_cam, y_cam, z], axis=1)

    # 4. Transform camera->world
    Rcw = pose[:3, :3]
    tcw = pose[:3, 3]
    world_pts = (Rcw @ xyz_cam.T).T + tcw[None, :]
    return world_pts


def unproject_2d_points_fisheye(pts: Union[np.ndarray, torch.Tensor], 
                                pose: Union[np.ndarray, torch.Tensor], 
                                ase_calibration=None) -> Union[np.ndarray, torch.Tensor]:
    """ Unproject 2D points for ASE fisheye calibration.
    - pts: Nx3 array (u, v, depth_mm_along_ray)
    - ase_calibration: when ASE is available, pass the calibration object returned by ase.get_ase_rgb_calibration()
    - device: ASE device object (if ase_calibration is provided) OR a tuple (width,height,fx) to construct a linear pinhole fallback.

    This function handles the Aria synthetic depth convention: depth is along the pixel ray in mm.
    We compute the unit 3D ray in camera coordinates for pixel (u,v) using the fisheye model (via ASE API if available) and multiply by depth to get the 3D point in camera frame.

    Returns Nx3 world points using the provided pose (4x4 world<-cam).
    """
    pts = _to_numpy(pts)
    pose = _to_numpy(pose)

    rays = []
    depths = []
    us = []
    vs = []
    for (u, v, depth_mm) in pts:
        # convert to numpy ints for ASE calls
        # many ASE calibrations expect (u,v) with origin top-left
        ray = ase_calibration.unproject(np.array([u, v]))  # expected unit vector in camera coords
        if ray is None:
            continue
        ray = ray / np.linalg.norm(ray)
        rays.append(ray)
        depths.append(depth_mm)
        us.append(u)
        vs.append(v)
    rays = np.vstack(rays)
    depths_m = np.array(depths) / 1000.0
    pts_cam = rays * depths_m[:, None]
    Rcw = pose[:3, :3]
    tcw = pose[:3, 3]
    world_pts = (Rcw @ pts_cam.T).T + tcw[None, :]
    return world_pts, np.stack([np.array(us), np.array(vs)], axis=1).astype(np.int16)

# ----------------------------- Undistort fisheye ---------------------------

def undistort_image_fisheye(img: Union[np.ndarray, torch.Tensor], ase_calibration = None) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
    """ Undistort a fisheye RGB image using ASE calibration if available.
    Returns (undistorted_rgb, pinhole_K)

    If ase_calibration is provided and has a method to produce a linear camera calibration, we use it. Otherwise we return the original image and a basic pinhole K computed from image width and a guessed focal length.
    """
    if ase_calibration is None:
        ase_calibration = ase.get_ase_rgb_calibration()
    arr = _to_numpy(img)
    h, w = arr.shape[:2]
    if ase_calibration is not None:
        # try:
        # Attempt to get a linear (pinhole) calibration from ASE
        pinhole_camera_calibration = aria_core_calibration.get_linear_camera_calibration(w, h, ase_calibration.get_focal_lengths()[0])
        # Many ASE helper libraries provide a distortion/rectify method; try common name
        undistorted = aria_core_calibration.distort_by_calibration(arr, pinhole_camera_calibration, ase_calibration)
        # K = np.array([[pinhole_camera_calibration.get_focal_lengths()[0], 0, pinhole_camera_calibration.get_principal_point()[0]],[0,pinhole_camera_calibration.get_focal_lengths()[1],pinhole_camera_calibration.get_principal_point()[1]],[0,0,1]])
        return undistorted, pinhole_camera_calibration

def get_pinhole_matrix_from_ase():
    pinhole_calibration = aria_core_calibration.get_linear_camera_calibration(ARIA_IMAGE_WIDTH, ARIA_IMAGE_HEIGHT, ARIA_FOCAL_LENGTH_X)
    return matrix_from_pinhole_calibration(pinhole_calibration)

def matrix_from_pinhole_calibration(calibration):
    return np.array([[calibration.get_focal_lengths()[0], 0, calibration.get_principal_point()[0]],[0,calibration.get_focal_lengths()[1],calibration.get_principal_point()[1]],[0,0,1]])

def filter_uvs_by_mask(uvs, mask):
    # 1. Handle Input Types (List/Tuple -> Tensor/Array)
    # Note: We store the original type to ensure we return the same type later
    is_original_tensor = False
    is_original_numpy = False
    is_original_list = isinstance(uvs, (list, tuple))

    if is_original_list:
        if isinstance(uvs[0], torch.Tensor):
            uvs = torch.stack(uvs, dim=1)
            is_original_tensor = True # It's now a tensor
        elif isinstance(uvs[0], np.ndarray):
            uvs = np.stack(uvs, axis=1)
            is_original_numpy = True # It's now an array
        else:
            raise ValueError(f"Unsupported type inside list: {type(uvs[0])}")
    elif isinstance(uvs, torch.Tensor):
        is_original_tensor = True
    elif isinstance(uvs, np.ndarray):
        is_original_numpy = True
    
    # 2. Create the Boolean Mask
    # FIX: Ensure indices are Long/Int for indexing, then cast result to Bool
    if is_original_tensor:
        # Ensure coordinates are long (int64) for indexing
        us = uvs[:, 0].long()
        vs = uvs[:, 1].long()
        # mask[...] returns uint8 (0 or 1). Comparison > 0 turns it into Bool (False or True)
        bool_mask = mask[vs, us] > 0
    else:
        # NumPy indexing
        bool_mask = mask[uvs[:, 1], uvs[:, 0]] > 0

    # 3. Apply Filter
    valid_uvs = uvs[bool_mask]

    # 4. Return correct format
    if is_original_tensor:
        return valid_uvs
    elif is_original_numpy:
        return valid_uvs
    elif is_original_list:
        # If input was [(u1, u2...), (v1, v2...)], split columns back to list
        return [valid_uvs[:, 0], valid_uvs[:, 1]]
    else:
        raise ValueError(f"Unsupported type: {type(uvs)}")


# ----------------------------- Projection funtions ---------------------------

def get_projection_params(points_2d, image_size, margin_frac):
    h, w = image_size
    
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    if x_max == x_min: x_max += 1.0
    if y_max == y_min: y_max += 1.0

    span_x = x_max - x_min
    span_y = y_max - y_min
    pad_x = span_x * margin_frac
    pad_y = span_y * margin_frac
    
    x_min_w = x_min - pad_x
    x_max_w = x_max + pad_x
    y_min_w = y_min - pad_y
    y_max_w = y_max + pad_y

    span_x_w = x_max_w - x_min_w
    span_y_w = y_max_w - y_min_w
    scale = min((w - 1) / span_x_w, (h - 1) / span_y_w)

    content_w_px = span_x_w * scale
    content_h_px = span_y_w * scale
    x_offset = (w - content_w_px) * 0.5
    y_offset = (h - content_h_px) * 0.5

    return {
        "scale": scale,
        "x_min_w": x_min_w,
        "y_min_w": y_min_w,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "h": h,
        "w": w
    }

def world_to_pixel(pts_world_2d, params):
    px = params["x_offset"] + (pts_world_2d[:, 0] - params["x_min_w"]) * params["scale"]
    py = params["y_offset"] + (pts_world_2d[:, 1] - params["y_min_w"]) * params["scale"]
    py = (params["h"] - 1) - py 
    if isinstance(pts_world_2d, torch.Tensor):
        return torch.stack([px, py], dim=-1).to(torch.int32)
    else:
        return np.stack([px, py], axis=1).astype(np.int32)


def sample_scene_points(params, num_points: int = 2048) -> np.ndarray:
    """
    Sample 3D points inside the scene region defined by projection `params`.

    The params dict is expected to contain:
        - "scale", "x_min_w", "y_min_w", "z_min_w", "z_max_w",
        - "x_offset", "y_offset", "h", "w"
    as produced by `render_pointcloud_and_boxes_orthographic_cv`.
    """
    scale = float(params["scale"])
    x_min_w = float(params["x_min_w"])
    y_min_w = float(params["y_min_w"])
    z_min_w = float(params.get("z_min_w", 0.0))
    z_max_w = float(params.get("z_max_w", z_min_w))
    x_offset = float(params["x_offset"])
    y_offset = float(params["y_offset"])
    h = float(params["h"])
    w = float(params["w"])

    # Recover world-space extents along x and y from scale and offsets.
    # See `render_pointcloud_and_boxes_orthographic_cv` for the derivation.
    span_x_w = (w - 2.0 * x_offset) / scale
    span_y_w = (h - 2.0 * y_offset) / scale
    x_max_w = x_min_w + span_x_w
    y_max_w = y_min_w + span_y_w

    # Robustness in case z range is degenerate
    if z_max_w <= z_min_w:
        z_max_w = z_min_w + 1e-3

    xs = np.random.uniform(x_min_w, x_max_w, size=num_points)
    ys = np.random.uniform(y_min_w, y_max_w, size=num_points)
    zs = np.random.uniform(z_min_w, z_max_w, size=num_points)

    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def points_to_image_coords_from_params(points_3d: np.ndarray, params) -> np.ndarray:
    """
    Map 3D world points to floorplan image pixel coordinates using precomputed `params`.

    Only the X/Y components are used for projection; Z is ignored.
    """
    pts = np.asarray(points_3d)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"`points_3d` must have shape (N, 3) or (N, >=2); got {pts.shape}")
    pts_xy = pts[:, :2]
    return world_to_pixel(pts_xy, params)


def filter_points_by_fustrum(
    points_world: Union[np.ndarray, torch.Tensor],
    cam_pose: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor],
    image_hw: Tuple[int, int],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Filter 3D **world** points to those lying inside a camera frustum, using PyTorch ops.

    Args:
        points_world: (N, 3) points in world coordinates.
        cam_pose: (4, 4) pose matrix mapping **camera -> world** (world_T_cam).
        K: (3, 3) pinhole intrinsics matrix.
        image_hw: (H, W) of the image plane.

    Returns:
        Subset of `points_world` (in world coordinates) that:
          - are in front of the camera (z_cam > 0)
          - project inside the image bounds [0, W) x [0, H)
        The return type matches the input type (numpy or torch).
    """
    is_tensor = isinstance(points_world, torch.Tensor)

    if is_tensor:
        pts_w = points_world
    else:
        pts_w = torch.as_tensor(points_world, dtype=torch.float32)

    pose_t = torch.as_tensor(cam_pose, dtype=pts_w.dtype, device=pts_w.device).view(4, 4)
    K_t = torch.as_tensor(K, dtype=pts_w.dtype, device=pts_w.device)

    if pts_w.ndim != 2 or pts_w.shape[1] != 3:
        raise ValueError(f"`points_world` must have shape (N, 3); got {pts_w.shape}")
    if pose_t.shape != (4, 4):
        raise ValueError(f"`cam_pose` must have shape (4, 4); got {pose_t.shape}")
    if K_t.shape != (3, 3):
        raise ValueError(f"`K` must have shape (3, 3); got {K_t.shape}")

    h, w = int(image_hw[0]), int(image_hw[1])

    # Decompose world_T_cam to R_wc, t_wc
    R_wc = pose_t[:3, :3]  # world <- cam
    t_wc = pose_t[:3, 3]

    # Transform world points to camera coordinates: pts_cam = R_cw * (pts_w - t_wc)
    R_cw = R_wc.t()
    pts_rel = pts_w - t_wc.unsqueeze(0)  # (N,3)
    pts_cam = (R_cw @ pts_rel.t()).t()   # (N,3)

    # Depth must be positive (in front of camera)
    z = pts_cam[:, 2]
    valid_z = z > 0.0

    # Project using homogeneous coordinates: u = x'/z', v = y'/z'
    cam_pts = pts_cam.t()  # 3 x N
    uvw = K_t @ cam_pts    # 3 x N
    u = uvw[0, :] / uvw[2, :]
    v = uvw[1, :] / uvw[2, :]

    finite = torch.isfinite(u) & torch.isfinite(v)
    in_x = (u >= 0.0) & (u < float(w))
    in_y = (v >= 0.0) & (v < float(h))

    mask = valid_z & finite & in_x & in_y

    filtered_w = pts_w[mask]

    # Preserve input type
    if is_tensor:
        return filtered_w
    return filtered_w.cpu().numpy()