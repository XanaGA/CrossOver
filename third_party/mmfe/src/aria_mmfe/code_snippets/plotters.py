# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import numpy as np
import plotly.graph_objects as go
import cv2
from projectaria_tools.projects import ase
from scipy.spatial.transform import Rotation as R
import torch

from aria_mmfe.aria_images.aria_cv_tools import pose_from_xyzq, world_to_pixel, xyzq_from_pose
from aria_mmfe.ase_data.ase_utils import mask_from_raster_lines

from .constants import UNIT_CUBE_LINES_IDXS, UNIT_CUBE_VERTICES
from .interpreter import language_to_bboxes
from .readers import read_language_file, read_points_file, read_trajectory_file


PLOTTING_COLORS = {
    "wall": "#FBFAF5",
    "door": "#F7C59F",
    "window": "#53F4FF",
    "points": "#C7DAE8",
    "trajectory": "#F92A82",
}


# This function plots a wire frame for each language entity bounding box loaded from lang
def plot_box_wireframe(box):
    box_verts = UNIT_CUBE_VERTICES * box["scale"]
    box_verts = (box["rotation"] @ box_verts.T).T
    box_verts = box_verts + box["center"]

    lines_x = []
    lines_y = []
    lines_z = []
    for pair in UNIT_CUBE_LINES_IDXS:
        for idx in pair:
            lines_x.append(box_verts[idx, 0])
            lines_y.append(box_verts[idx, 1])
            lines_z.append(box_verts[idx, 2])
        lines_x.append(None)
        lines_y.append(None)
        lines_z.append(None)

    class_name = box["class"]
    wireframe = go.Scatter3d(
        x=lines_x,
        y=lines_y,
        z=lines_z,
        name=box["id"],
        mode="lines",
        line={
            "color": PLOTTING_COLORS[class_name],
            "width": 10,
        },
    )
    return wireframe


def plot_point_cloud(points, max_points_to_plot=500_000):
    if len(points) > max_points_to_plot:
        print(
            f"The number of points ({len(points)}) exceeds the maximum that can be reliably plotted."
        )
        print(f"Randomly subsampling {max_points_to_plot} points for the plot.")
        sampled = np.random.choice(len(points), max_points_to_plot, replace=False)
        points = points[sampled]
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name="Semi-dense Points",
        marker={
            "size": 1.0,
            "opacity": 0.3,
            "color": PLOTTING_COLORS["points"],
        },
    )


def plot_trajectory(trajectory):
    device_positions = trajectory["ts"]
    return go.Scatter3d(
        x=device_positions[:, 0],
        y=device_positions[:, 1],
        z=device_positions[:, 2],
        name="Device Poses",
        mode="lines+markers",
        marker={
            "size": 3,
            "opacity": 1.0,
            "color": PLOTTING_COLORS["trajectory"],
        },
        line={
            "color": PLOTTING_COLORS["trajectory"],
            "width": 3,
        },
    )


# Main plotting function
def plot_3d_scene(language_path=None, points_path=None, trajectory_path=None):
    traces = []
    if points_path is not None:
        points = read_points_file(points_path)
        traces.append(plot_point_cloud(points))

    if trajectory_path is not None:
        trajectory = read_trajectory_file(trajectory_path)
        traces.append(plot_trajectory(trajectory))

    if language_path is not None:
        entities = read_language_file(language_path)
        boxes, _ = language_to_bboxes(entities)
        for box in boxes:
            traces.append(plot_box_wireframe(box))

    assert traces, "Nothing to visualize."
    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        scene={
            "xaxis": {"showticklabels": False, "title": ""},
            "yaxis": {"showticklabels": False, "title": ""},
            "zaxis": {"showticklabels": False, "title": ""},
        },
    )
    fig.show()


def orthographic_project_points(points, axis="z"):
    """Project a 3D point cloud onto a 2D plane using an orthographic view.

    Args:
        points: numpy array of shape [N, 3]. Columns are x, y, z in world frame.
        axis: which axis to view along. One of {"x", "y", "z"}.

    Returns:
        projected: numpy array of shape [N, 2] with the orthographic projection.
        axes_labels: tuple of strings for the resulting axes (e.g., ("x", "y")).
    """
    assert axis in {"x", "y", "z"}, "axis must be one of {'x','y','z'}"
    if axis == "z":
        idx = (0, 1)
        labels = ("x", "y")
    elif axis == "y":
        idx = (0, 2)
        labels = ("x", "z")
    else:  # axis == "x"
        idx = (1, 2)
        labels = ("y", "z")

    projected = points[:, idx]
    return projected, labels


def plot_point_cloud_orthographic(points, axis="z", bins=1024, colorscale="Viridis"):
    """Display a semi-dense point cloud as a 2D orthographic density image.

    Uses a 2D histogram to render a dense view that scales well to large point counts.

    Args:
        points: numpy array of shape [N, 3].
        axis: viewing axis ("x", "y", or "z"). Defaults to top-down ("z").
        bins: number of bins along each axis for the histogram image.
        colorscale: Plotly colorscale name.

    Returns:
        fig: plotly.graph_objects.Figure with the 2D density image.
    """
    projected, labels = orthographic_project_points(points, axis=axis)
    x = projected[:, 0]
    y = projected[:, 1]

    # Compute bounds for stable framing
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    hist2d = go.Histogram2d(
        x=x,
        y=y,
        xbins={"start": x_min, "end": x_max, "size": (x_max - x_min) / bins if x_max > x_min else 1.0},
        ybins={"start": y_min, "end": y_max, "size": (y_max - y_min) / bins if y_max > y_min else 1.0},
        colorscale=colorscale,
        showscale=True,
        name=f"Ortho density ({labels[0]}-{labels[1]})",
    )

    fig = go.Figure(data=[hist2d])
    fig.update_layout(
        template="plotly_dark",
        xaxis={"title": labels[0], "showticklabels": False},
        yaxis={"title": labels[1], "scaleanchor": "x", "scaleratio": 1, "showticklabels": False},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return fig


def render_point_cloud_orthographic_cv(points, axis="z", image_size=(1024, 1024), clamp_percentile=99.5,):
    """Render a 2D orthographic density image using OpenCV.

    Args:
        points: numpy array of shape [N, 3].
        axis: view axis to collapse ("x", "y", or "z").
        image_size: (height, width) of the output image.
        clamp_percentile: percentile for density clamping for better contrast.
        colormap: OpenCV colormap (e.g., cv2.COLORMAP_VIRIDIS).

    Returns:
        image_bgr: uint8 BGR image suitable for cv2.imshow / cv2.imwrite.
    """
    projected, _ = orthographic_project_points(points, axis=axis)
    h, w = image_size

    x = projected[:, 0]
    y = projected[:, 1]

    # Compute bounds and avoid degenerate ranges
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    # Use a 2D histogram as a density map
    density, x_edges, y_edges = np.histogram2d(
        x, y, bins=[w, h], range=[[x_min, x_max], [y_min, y_max]]
    )
    # histogram2d returns shape (w, h) with axes (x, y); transpose to image (h, w)
    density_img = density.T.astype(np.float32)

    # Optional clamping to improve contrast
    if clamp_percentile is not None:
        vmax = np.percentile(density_img, clamp_percentile)
        if vmax <= 0:
            vmax = np.max(density_img)
        if vmax > 0:
            density_img = np.clip(density_img, 0, vmax)

    # Normalize to 0..255
    max_val = float(np.max(density_img))
    if max_val > 0:
        density_img = (density_img / max_val) * 255.0
    density_img = density_img.astype(np.uint8)

    # Map density to grayscale intensity on white background: higher density -> darker
    intensity_img = 255 - density_img
    image_bgr = cv2.cvtColor(intensity_img, cv2.COLOR_GRAY2BGR)
    return image_bgr


def _compute_box_vertices_world(box):
    """Return 8x3 vertices for a language box in world coordinates."""
    box_verts = UNIT_CUBE_VERTICES * box["scale"]
    box_verts = (box["rotation"] @ box_verts.T).T
    box_verts = box_verts + box["center"]
    return box_verts


def render_box_wireframes_orthographic_cv(boxes, axis="z", image_size=(1024, 1024), thickness=2, margin_frac=0.05):
    """Render 3D wireframe boxes as 2D orthographic lines using OpenCV.

    Args:
        boxes: iterable of language boxes (from language_to_bboxes), each with keys
               center, scale, rotation (3x3), class, id.
        axis: viewing axis to collapse ("x", "y", or "z").
        image_size: (height, width) for output image.
        thickness: line thickness in pixels.
        margin_frac: fraction of the world span to pad on each side (0.05 = 5%).

    Returns:
        image_bgr: uint8 BGR image.
    """
    h, w = image_size

    # Compute projected vertices for all boxes to determine global bounds
    all_proj_points = []
    per_box_proj = []
    for box in boxes:
        verts3d = _compute_box_vertices_world(box)
        verts2d, labels = orthographic_project_points(verts3d, axis=axis)
        per_box_proj.append((box, verts2d))
        all_proj_points.append(verts2d)

    if not per_box_proj:
        return np.zeros((h, w, 3), dtype=np.uint8)

    all_proj = np.vstack(all_proj_points)
    x = all_proj[:, 0]
    y = all_proj[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    # Expand bounds by margin
    span_x = (x_max - x_min) if x_max > x_min else 1.0
    span_y = (y_max - y_min) if y_max > y_min else 1.0
    pad_x = span_x * float(margin_frac)
    pad_y = span_y * float(margin_frac)
    x_min_w = x_min - pad_x
    x_max_w = x_max + pad_x
    y_min_w = y_min - pad_y
    y_max_w = y_max + pad_y

    # Prepare canvas (white background)
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Aspect-preserving scale and center offsets (letterbox)
    span_x_w = (x_max_w - x_min_w)
    span_y_w = (y_max_w - y_min_w)
    scale = min((w - 1) / span_x_w, (h - 1) / span_y_w)
    content_w_px = span_x_w * scale
    content_h_px = span_y_w * scale
    x_offset = (w - content_w_px) * 0.5
    y_offset = (h - content_h_px) * 0.5

    def world_to_pixel(pt2):
        px = x_offset + (pt2[0] - x_min_w) * scale
        py = y_offset + (pt2[1] - y_min_w) * scale
        # Convert to image coords (y down)
        py = (h - 1) - py
        return int(round(px)), int(round(py))

    # Draw each box's wireframe using endpoints only (efficient)
    for box, verts2d in per_box_proj:
        # Use black lines for clarity on white background
        color_bgr = (0, 0, 0)

        for pair in UNIT_CUBE_LINES_IDXS:
            p0 = world_to_pixel(verts2d[pair[0]])
            p1 = world_to_pixel(verts2d[pair[1]])
            cv2.line(img, p0, p1, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)

    # Flip vertically to match expected coordinate orientation
    img = cv2.flip(img, 0)
    return img

def change_params_resolution(params, new_resolution):
    """
    Updates the projection parameters for a new image resolution.
    
    Args:
        params: Original parameters dictionary.
        new_resolution: Tuple of (new_h, new_w).
        
    Returns:
        new_params: Updated dictionary with new scale, offsets, and dimensions.
    """
    new_h, new_w = new_resolution
    new_params = copy.deepcopy(params)
    
    # 1. Recover the original world spans (the size of the bounding box in world units)
    # Based on: x_offset = (w - (span_x_w * scale)) * 0.5
    # We solve for span_x_w:
    old_w = params['w']
    old_h = params['h']
    old_scale = params['scale']
    
    span_x_w = (old_w - 2 * params['x_offset']) / old_scale
    span_y_w = (old_h - 2 * params['y_offset']) / old_scale
    
    # 2. Compute the new aspect-preserving scale
    # Original logic: scale = min((w - 1) / span_x_w, (h - 1) / span_y_w)
    new_scale = torch.min(
        torch.stack([(new_w - 1) / span_x_w, (new_h - 1) / span_y_w]), 
        dim=0
    ).values
    
    # 3. Compute new pixel content sizes
    new_content_w_px = span_x_w * new_scale
    new_content_h_px = span_y_w * new_scale
    
    # 4. Compute new offsets to keep content centered
    new_x_offset = (new_w - new_content_w_px) * 0.5
    new_y_offset = (new_h - new_content_h_px) * 0.5
    
    # 5. Update the dictionary

    if isinstance(params['h'], (np.ndarray, list, torch.Tensor)):
        batch_size = len(params['h'])
        new_params['h'] = torch.full((batch_size,), new_h)
        new_params['w'] = torch.full((batch_size,), new_w)
    else:
        new_params['h'] = new_h
        new_params['w'] = new_w
        
    new_params['scale'] = new_scale
    new_params['x_offset'] = new_x_offset
    new_params['y_offset'] = new_y_offset
    
    return new_params

def render_pointcloud_and_boxes_orthographic_cv(points, boxes, obj_boxes=[], axis="z", image_size=(1024, 1024), 
                                                margin_frac=0.05, clamp_percentile=99.5, line_thickness=2, window_line_thickness=2,
                                                return_params=False):
    """Render aligned orthographic images for pointcloud density and box wireframes.

    Uses the same world-to-pixel mapping and padded bounds so both outputs align.

    Args:
        points: numpy array [N,3] point cloud in world frame.
        boxes: iterable of language boxes (from language_to_bboxes).
        axis: view axis to collapse ("x", "y", or "z").
        image_size: (height, width) of the output images.
        margin_frac: fraction of world span to pad on each side.
        clamp_percentile: percentile for density clamping for contrast.
        line_thickness: pixel thickness for wireframe lines.

    Returns:
        point_img_bgr: uint8 BGR image (white bg, grayscale density where denser=darker).
        wire_img_bgr: uint8 BGR image (white bg, black wireframe lines).
    """
    h, w = image_size

    # If requested, choose best projection axis based on box extent
    if axis == "auto":
        candidate_axes = ["x", "y", "z"]
        best_axis = "z"
        best_area = -1.0
        for ax in candidate_axes:
            ext_pts = []
            for box in boxes:
                verts3d = _compute_box_vertices_world(box)
                v2, _ = orthographic_project_points(verts3d, axis=ax)
                ext_pts.append(v2)
            if len(ext_pts) == 0:
                continue
            allp = np.vstack(ext_pts)
            area = (float(np.max(allp[:, 0]) - np.min(allp[:, 0])) *
                    float(np.max(allp[:, 1]) - np.min(allp[:, 1])))
            if area > best_area:
                best_area = area
                best_axis = ax
        axis = best_axis

    # Compute vertical (height) bounds in world coordinates from boxes if available,
    # otherwise fall back to the point cloud.
    z_min_w = 0.0
    z_max_w = 0.0
    if len(boxes) > 0:
        all_verts = []
        for box in boxes:
            verts3d = _compute_box_vertices_world(box)
            all_verts.append(verts3d)
        all_verts_np = np.vstack(all_verts)
        z_min_w = float(np.min(all_verts_np[:, 2]))
        z_max_w = float(np.max(all_verts_np[:, 2]))
    elif points is not None and points.size > 0:
        z_min_w = float(np.min(points[:, 2]))
        z_max_w = float(np.max(points[:, 2]))

    # Project data to 2D with selected axis
    pts2d, _ = orthographic_project_points(points, axis=axis)
    box_proj_list = []
    is_door_window = [False] * len(boxes)
    for i, box in enumerate(boxes):
        verts3d = _compute_box_vertices_world(box)
        verts2d, _ = orthographic_project_points(verts3d, axis=axis)
        box_proj_list.append((box, verts2d))

        if box["class"] in ["door", "window"]:
            is_door_window[i] = True

    obj_box_proj_list = []
    for box in obj_boxes:
        verts3d = _compute_box_vertices_world(box)
        verts2d, _ = orthographic_project_points(verts3d, axis=axis)
        # if np.max(verts2d[:, 0]) > max_xy[0] or np.min(verts2d[:, 0]) < min_xy[0] or np.max(verts2d[:, 1]) > max_xy[1] or np.min(verts2d[:, 1]) < min_xy[1]:
        #     continue
        obj_box_proj_list.append((box, verts2d))

    # Compute bounds FROM BOXES ONLY (correct size), fallback to points if no boxes
    if len(box_proj_list) > 0:
        all_box_proj = np.vstack([bp[1] for bp in box_proj_list])
    else:
        all_box_proj = pts2d

    x = all_box_proj[:, 0]
    y = all_box_proj[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    # Apply margin
    span_x = (x_max - x_min)
    span_y = (y_max - y_min)
    pad_x = span_x * float(margin_frac)
    pad_y = span_y * float(margin_frac)
    x_min_w = x_min - pad_x
    x_max_w = x_max + pad_x
    y_min_w = y_min - pad_y
    y_max_w = y_max + pad_y

    # Aspect-preserving uniform scale and offsets
    span_x_w = (x_max_w - x_min_w)
    span_y_w = (y_max_w - y_min_w)
    scale = min((w - 1) / span_x_w, (h - 1) / span_y_w)
    content_w_px = span_x_w * scale
    content_h_px = span_y_w * scale
    x_offset = (w - content_w_px) * 0.5
    y_offset = (h - content_h_px) * 0.5

    # World->pixel mapping (vectorized)
    def world_to_pixel_array(pts2):
        px = x_offset + (pts2[:, 0] - x_min_w) * scale
        py = y_offset + (pts2[:, 1] - y_min_w) * scale
        py = (h - 1) - py  # flip y so up is positive
        px = np.rint(px).astype(np.int32)
        py = np.rint(py).astype(np.int32)
        # clip to valid range
        np.clip(px, 0, w - 1, out=px)
        np.clip(py, 0, h - 1, out=py)
        return px, py

    # Filter points to padded bounds, then render density using np.add.at
    density = np.zeros((h, w), dtype=np.float32)
    if pts2d.size > 0:
        in_x = (pts2d[:, 0] >= x_min_w) & (pts2d[:, 0] <= x_max_w)
        in_y = (pts2d[:, 1] >= y_min_w) & (pts2d[:, 1] <= y_max_w)
        in_bounds = in_x & in_y
        pts2d_in = pts2d[in_bounds]
        if pts2d_in.size > 0:
            px, py = world_to_pixel_array(pts2d_in)
            np.add.at(density, (py, px), 1.0)

    # Optional clamping
    if clamp_percentile is not None:
        vmax = float(np.percentile(density, clamp_percentile))
        if vmax <= 0:
            vmax = float(np.max(density))
        if vmax > 0:
            density = np.clip(density, 0.0, vmax)

    # Normalize to 0..255 and invert for darker=denser on white bg
    if density.max() > 0:
        density = (density / float(density.max())) * 255.0
    density_u8 = density.astype(np.uint8)
    intensity_img = 255 - density_u8
    point_img_bgr = cv2.cvtColor(intensity_img, cv2.COLOR_GRAY2BGR)

    # Render wireframes on white background with same mapping
    all_polys_2d = []
    wire_img_bgr = np.full((h, w, 3), 255, dtype=np.uint8)
    for i, (box, verts2d) in enumerate(box_proj_list):
        px, py = world_to_pixel_array(verts2d)
        all_polys_2d.append(np.vstack([px, py]).T)
        # Draw edges
        for pair in UNIT_CUBE_LINES_IDXS:
            p0 = (int(px[pair[0]]), int(py[pair[0]]))
            p1 = (int(px[pair[1]]), int(py[pair[1]]))
            line_thick = line_thickness if not is_door_window[i] else window_line_thickness
            cv2.line(wire_img_bgr, p0, p1, (0, 0, 0), thickness=line_thick, lineType=cv2.LINE_AA)
    
    mask = mask_from_raster_lines(wire_img_bgr)
    for box, verts2d in obj_box_proj_list:
        px, py = world_to_pixel_array(verts2d)
        vertices = np.vstack([px, py]).T
        if (mask[vertices[:, 1], vertices[:, 0]] != 0).all():
            # Draw edges
            for pair in UNIT_CUBE_LINES_IDXS:
                p0 = (int(px[pair[0]]), int(py[pair[0]]))
                p1 = (int(px[pair[1]]), int(py[pair[1]]))
                cv2.line(wire_img_bgr, p0, p1, (0, 0, 0), thickness=line_thickness, lineType=cv2.LINE_AA)

    params = {
        "scale": scale,
        "x_min_w": x_min_w,
        "y_min_w": y_min_w,
        "z_min_w": z_min_w,
        "z_max_w": z_max_w,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "h": h,
        "w": w
    }

    if return_params:
        return point_img_bgr, wire_img_bgr, params
    else:
        return point_img_bgr, wire_img_bgr


def render_pointcloud(points, params, image_size=None):
    if image_size is None:
        h, w = params["h"], params["w"]
    else:
        h, w = image_size
    density = np.zeros((h, w), dtype=np.float32)

    pixels = world_to_pixel(points[:, :2], params)
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & \
            (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    pixels = pixels[valid]

    np.add.at(density, (pixels[:, 1], pixels[:, 0]), 1.0)

    if density.max() > 0:
        vmax = np.percentile(density, 99.5)
        if vmax == 0: vmax = density.max()
        density = np.clip(density, 0, vmax)
        density = (density / vmax) * 255.0

    img_u8 = 255 - density.astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

def overlay_trajectory(img, traj_df, params, apply_device2camera: bool = True):

    if apply_device2camera:
        traj_pts = []
        quats = []
        for i in range(len(traj_df)):
            row = traj_df.iloc[i]
            calibration = ase.get_ase_rgb_calibration()
            T_device_cam = calibration.get_transform_device_camera().to_matrix()
            pts_xyz = row[['tx_world_device', 'ty_world_device', 'tz_world_device']].values
            q = row[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].values
            pose_device = pose_from_xyzq(pts_xyz, q)
            pose_cam = pose_device @ T_device_cam
            pts_cam, q = xyzq_from_pose(pose_cam)
            traj_pts.append(pts_cam)
            quats.append(q)
        traj_pts = np.array(traj_pts)
        quats = np.array(quats)
    else:
        traj_pts = traj_df[['tx_world_device', 'ty_world_device']].values
        quats = traj_df[['qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device']].values

    traj_pixels = world_to_pixel(traj_pts, params)

    cv2.polylines(img, [traj_pixels], isClosed=False, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    r = R.from_quat(quats)
    camera_forward_vector = np.array([0, 0, 1]) 
    view_dirs_world = r.apply(camera_forward_vector)

    arrow_len_px = 15
    subsample = 20 
    
    for i in range(0, len(traj_pixels), subsample):
        start_pt = traj_pixels[i]
        dir_x = view_dirs_world[i, 0]
        dir_y = view_dirs_world[i, 1]
        norm = np.hypot(dir_x, dir_y)
        if norm < 1e-6: continue

        dir_x /= norm
        dir_y /= norm

        end_x = int(start_pt[0] + dir_x * arrow_len_px)
        end_y = int(start_pt[1] - dir_y * arrow_len_px)

        cv2.arrowedLine(img, tuple(start_pt), (end_x, end_y), (255, 0, 0), 1, tipLength=0.3, line_type=cv2.LINE_AA)

    return img

def overlay_points(img, points, params=None, colors=None):
    """
    Overlays points directly onto the image array using vectorized indexing.
    This mimics the logic of render_pointcloud (filtering + array manipulation).
    
    Args:
        img: The source image (BGR).
        points: Numpy array of shape (N, 2) in (x, y) pixel coordinates.
        colors: Optional numpy array of shape (N, 3).
    """
    h, w = img.shape[:2]

    # 1. Round and cast to int (Vectorized)
    if params is not None:
        pixels = world_to_pixel(points, params)
    else:
        pixels = points

    # 2. Boundary Checking (Vectorized - exactly like render_pointcloud)
    # We create a boolean mask for points strictly inside image bounds
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & \
            (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            
    # Apply the mask to keep only valid pixels
    pixels = pixels[valid]

    # 3. Handle Colors and Draw
    # We assign values directly to the image array instead of looping
    if colors is not None:
        # If colors are provided, we must also filter them to match valid pixels
        valid_colors = colors[valid]
        img[pixels[:, 1], pixels[:, 0]] = valid_colors
    else:
        # Default Red: Broadcasts this single color to all valid pixel locations
        img[pixels[:, 1], pixels[:, 0]] = (0, 0, 255)

    return img


def overlay_single_pose(base_img, traj_df, params, idx, apply_device2camera: bool = True):
        img = base_img.copy()
        row = traj_df.iloc[idx]

        if apply_device2camera:
            calibration = ase.get_ase_rgb_calibration()
            T_device_cam = calibration.get_transform_device_camera().to_matrix()
            pts_xyz = row[['tx_world_device', 'ty_world_device', 'tz_world_device']].values
            q = row[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].values
            pose_device = pose_from_xyzq(pts_xyz, q)
            pose_cam = pose_device @ T_device_cam
            pts_cam, q = xyzq_from_pose(pose_cam)
            px, py = pts_cam[0], pts_cam[1]

        else:
            px = row['tx_world_device']
            py = row['ty_world_device']
            q = row[['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].values


        pt = world_to_pixel(np.array([[px, py]]), params)[0]

        r = R.from_quat(q)

        forward = r.apply([0,0,1])
        fx, fy = forward[0], forward[1]
        norm = np.hypot(fx, fy)
        if norm < 1e-6: 
            return img

        fx /= norm
        fy /= norm

        end_x = int(pt[0] + fx * 25)
        end_y = int(pt[1] - fy * 25)

        cv2.circle(img, tuple(pt), 5, (0,0,255), -1)
        cv2.arrowedLine(img, tuple(pt), (end_x, end_y), (0,255,0), 2, tipLength=0.3)

        return img

def overlay_grid(image, params, grid_res=(32, 32), color=(100, 100, 100), thickness=1):
    """
    Overlay a grid on the image.
    
    Args:
        image: Input image (numpy array)
        params: Projection parameters dict with 'scale', 'offset_x', 'offset_y'
        grid_res: Tuple (rows, cols) specifying number of grid cells
        color: Grid line color in BGR format
        thickness: Line thickness in pixels
    
    Returns:
        Image with grid overlay
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    rows, cols = grid_res
    
    # Draw vertical lines
    for i in range(cols + 1):
        x = int(i * w / cols)
        cv2.line(result, (x, 0), (x, h-1), color, thickness)
    
    # Draw horizontal lines
    for i in range(rows + 1):
        y = int(i * h / rows)
        cv2.line(result, (0, y), (w-1, y), color, thickness)
    
    return result