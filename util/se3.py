"""Utilities for handling SE(3) transformations, quaternions, and bounding boxes."""
from typing import Tuple, Dict, Union, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix. Expects q as [w, x, y, z]."""
    # scipy expects [x, y, z, w]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    rot_mat = R.from_quat(q_scipy).as_matrix()
    return rot_mat

def make_M_from_tqs(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Create transformation matrix from translation, quaternion rotation, and scale."""
    # scipy expects [x, y, z, w]; q is [w, x, y, z]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    T = np.eye(4)
    T[0:3, 3] = t
    R_mat = np.eye(4)
    R_mat[0:3, 0:3] = R.from_quat(q_scipy).as_matrix()
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R_mat).dot(S)
    
    return M 

def calc_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate axis-aligned bounding box from points."""
    min_coords = np.min(points, axis = 0)
    max_coords = np.max(points, axis = 0)
    return min_coords, max_coords

def calc_Mbbox(model: Dict[str, Union[Dict, np.ndarray]]) -> np.ndarray:
    """Calculate transformation matrix for model's bounding box."""
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    # rot_obj is [w, x, y, z]; scipy expects [x, y, z, w]
    q_scipy = np.array([rot_obj[1], rot_obj[2], rot_obj[3], rot_obj[0]])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = R.from_quat(q_scipy).as_matrix()
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M

def compose_mat4(
    t: np.ndarray, 
    q: np.ndarray, 
    s: np.ndarray, 
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compose 4x4 transformation matrix from translation, rotation, scale, and optional center.
    Expects q as [w, x, y, z]."""
    q = np.asarray(q)
    # scipy expects [x, y, z, w]; q is [w, x, y, z]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    T = np.eye(4)
    T[0:3, 3] = t
    R_mat = np.eye(4)
    R_mat[0:3, 0:3] = R.from_quat(q_scipy).as_matrix()
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R_mat).dot(S).dot(C)
    return M 

def decompose_mat4(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose 4x4 transformation matrix into translation, quaternion rotation, and scale.
    Returns q as [w, x, y, z] to match the old quaternion library behavior."""
    R_mat = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R_mat[0:3, 0])
    sy = np.linalg.norm(R_mat[0:3, 1])
    sz = np.linalg.norm(R_mat[0:3, 2])

    s = np.array([sx, sy, sz])

    R_mat[:, 0] /= sx
    R_mat[:, 1] /= sy
    R_mat[:, 2] /= sz

    q_obj = R.from_matrix(R_mat[:3, :3])
    q_raw = q_obj.as_quat()  # scipy returns [x, y, z, w]
    q = np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]])  # [w, x, y, z]

    t = M[0:3, 3]
    return t, q, s