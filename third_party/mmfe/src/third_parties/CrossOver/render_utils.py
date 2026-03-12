from PIL import Image
import cv2
from sklearn.neighbors import NearestNeighbors
import trimesh
import numpy as np
import pyrender

from aria_mmfe.code_snippets.plotters import orthographic_project_points

def load_and_center_mesh(filename: str) -> trimesh.Trimesh:
    """Load a mesh file and center it at the origin."""
    mesh = trimesh.load(filename)  # load file with trimesh
    center = np.min(mesh.vertices, axis=0)+(np.max(mesh.vertices, axis=0)-np.min(mesh.vertices, axis=0))/2
    mesh.vertices -= center
    
    return mesh

def get_camera_zoom(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Calculate camera zoom parameters based on mesh dimensions."""
    min_bounds = mesh.vertices.min(axis=0)
    max_bounds = mesh.vertices.max(axis=0)

    # Compute the extents
    x_extent = max_bounds[0] - min_bounds[0]
    y_extent = max_bounds[1] - min_bounds[1]

    # Adjust the extents based on the aspect ratio of the rendering window
    xmag = x_extent * 0.8
    ymag = y_extent * 0.8
    
    return xmag, ymag

def get_camera(xmag: float, ymag: float) -> tuple[pyrender.OrthographicCamera, np.ndarray]:
    """Create a camera with orthographic projection."""
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    camera_pose = np.zeros((4,4), dtype=np.float32)
    theta=-90
    Rx = np.identity(3)
    Rx[1,1] = np.cos(np.radians(theta))
    Rx[1,2] = -1*np.sin(np.radians(theta))
    Rx[2,1] = np.sin(np.radians(theta))
    Rx[2,2] = np.cos(np.radians(theta))
    camera_pose[3,3] = 1

    camera_pose[0:3, 0:3] = np.dot(Rx, np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]))
    
    return camera, camera_pose

def get_light() -> tuple[pyrender.DirectionalLight, np.ndarray]:
    """Create a directional light source."""
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2) #intensity is important for the final colors
    # set the pose of the directional light
    light_pose = np.zeros((4,4), dtype=np.float32)
    theta=-90
    Rx = np.identity(3)
    Rx[1,1] = np.cos(np.radians(theta))
    Rx[1,2] = -1*np.sin(np.radians(theta))
    Rx[2,1] = np.sin(np.radians(theta))
    Rx[2,2] = np.cos(np.radians(theta))
    light_pose[3,3] = 1

    # # Correct the camera rotation to look down
    light_pose[0:3, 0:3] = np.dot(Rx, np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]))
    
    return light, light_pose

def crop_image(img_np: np.ndarray, pad: int = 50) -> Image.Image:
    """Crops an image to the bounding box of non-white pixels and saves it."""
    # Mask of non-white pixels (we assume white to be [255, 255, 255])
    mask = np.all(img_np != [255, 255, 255], axis=-1)
    
    # Find the bounding box of the non-white pixels
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Crop the image to the bounding box
    image = Image.fromarray(img_np)
    cropped_image = image.crop((x_min - pad, y_min - pad, x_max + pad, y_max + pad))
    
    return np.array(cropped_image)

def render_scene(mesh_filename: str) -> np.ndarray:
    """Render a 3D mesh into a 2D image."""
    scene = pyrender.Scene()
    
    mesh = load_and_center_mesh(mesh_filename)
    xmag, ymag = get_camera_zoom(mesh)
    
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth = False) 
    scene.add(mesh) #adding the mesh
    
    camera, camera_pose = get_camera(xmag, ymag)
    scene.add(camera, pose=camera_pose) # adding the camera with the computed camera pose
    
    light, light_pose = get_light()
    scene.add(light, pose=light_pose) # adding the directional light with the computed pose       

    r = pyrender.OffscreenRenderer(800, 400, 224)
    color, depth = r.render(scene) # it gives a tuple as output, where one is an RGB image and one is a depth image
    
    return color

def render_pointcloud_density(points: np.ndarray,
                              axis: str = "z",
                              resolution: int = 1024,
                              padding: int = 50,
                              point_size: int = 1,
                              rotate: bool = True) -> np.ndarray:
    """
    Render a point cloud as a 2D orthographic density map and apply YOLO-style
    letterbox padding after cropping.
    """

    # ---------------------------------------------------------
    # 1. REMOVE OUTLIERS BEFORE ANY PROCESSING
    # ---------------------------------------------------------
    points = remove_statistical_outliers(points, k=20, std_ratio=2.0)
    # ---------------------------------------------------------

    resolution_inner = resolution - 2 * padding

    if rotate:
        # Rotate 90° clockwise around Z
        R = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        points = points @ R.T

    # Project points
    proj, _ = orthographic_project_points(points, axis=axis)

    # Normalize to 0–1
    # Compute real ranges before normalization
    x = proj[:, 0]
    y = proj[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    w_true = x_max - x_min
    h_true = y_max - y_min

    # Correct aspect ratio canvas size
    resolution_inner = resolution - 2 * padding

    if w_true >= h_true:
        draw_w = resolution_inner
        draw_h = int(resolution_inner * (h_true / w_true))
    else:
        draw_h = resolution_inner
        draw_w = int(resolution_inner * (w_true / h_true))

    # Normalize coords to non-square canvas
    x_norm = (x - x_min) / (w_true + 1e-9)
    y_norm = (y - y_min) / (h_true + 1e-9)

    xi = (x_norm * (draw_w - 1)).astype(np.int32)
    yi = (y_norm * (draw_h - 1)).astype(np.int32)

    # Create NON-SQUARE density image
    img = np.zeros((draw_h, draw_w), dtype=np.float32)

    # Draw points
    half = point_size // 2
    for px, py in zip(xi, yi):
        x_start = max(0, px - half)
        x_end = min(draw_w, px + half + 1)
        y_start = max(0, py - half)
        y_end = min(draw_h, py + half + 1)
        img[draw_h - 1 - y_end:draw_h - y_start, x_start:x_end] += 1.0

    # Normalize density
    img = img / img.max() if img.max() > 0 else img

    # Convert to grayscale (white = empty, black = dense)
    img_color = (255 * (1 - img[..., None].repeat(3, axis=-1))).astype(np.uint8)
    # img_color = 1 - img[..., None].repeat(3, axis=-1)

    # Crop non-zero content
    mask = np.any(img_color != 1, axis=-1)   # note: background is 1
    coords = np.argwhere(mask)
    if coords.size == 0:
        # nothing to letterbox, but still output correctly padded
        return resize_with_padding(img_color, resolution, padding)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    img_cropped = img_color[y_min:y_max, x_min:x_max]

    # Apply YOLO-style resize + padding
    final_img = resize_with_padding(img_cropped, resolution, padding)

    return final_img

def remove_statistical_outliers(points: np.ndarray, k: int = 20, std_ratio: float = 2.0):
    """
    Statistical Outlier Removal (SOR) for point clouds.
    Mimics Open3D/PCL.

    points: (N,3)
    returns: filtered points
    """
    if len(points) < k:
        return points  # not enough points for SOR

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    # skip the 0 distance to itself
    mean_d = distances[:, 1:].mean(axis=1)

    global_mean = mean_d.mean()
    global_std  = mean_d.std()

    threshold = global_mean + std_ratio * global_std
    mask = mean_d < threshold

    return points[mask]

def resize_with_padding(img, resolution, padding, bg_color=(255, 255, 255)):
    """
    Resize an image with aspect ratio preserved and pad to (resolution x resolution).
    """
    # target inner size (after letterboxing but before outer padding)
    target_size = resolution - 2 * padding
    h, w = img.shape[:2]

    # compute scale factor
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize with preserved aspect ratio
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # compute padding for centering inside the target square
    pad_w = target_size - new_w
    pad_h = target_size - new_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    # inner padding (makes square)
    square = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=bg_color
    )

    # outer padding to reach full resolution
    final = cv2.copyMakeBorder(
        square, padding, padding, padding, padding,
        borderType=cv2.BORDER_CONSTANT, value=bg_color
    )

    return final


