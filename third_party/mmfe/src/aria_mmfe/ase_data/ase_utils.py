import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d
import os
import pandas as pd

from aria_mmfe.code_snippets.interpreter import language_to_bboxes
from aria_mmfe.code_snippets.readers import read_language_file

def read_trajectory(root_dir, scene_id):
    traj_path = os.path.join(root_dir, scene_id, "images", "train", "trajectory.csv")
    df = pd.read_csv(traj_path)
    return df

def read_points(root_dir, scene_id):
    points_path = os.path.join(root_dir, scene_id, "semidense_points.csv.gz")
    try:
        df = pd.read_csv(points_path, compression='gzip')
        points = df.iloc[:, 2:5].values
        return points
    except Exception as e:
        print(f"Error reading points: {e}")
        return np.array([])

def read_3d_boxes(root_dir, scene_id):
    language_path = os.path.join(root_dir, scene_id, "ase_scene_language.txt")
    entities = read_language_file(language_path)
    boxes, scene_aabb = language_to_bboxes(entities)
    return boxes

def mask_from_raster_lines(wire_img_bgr):
    # Convert to grayscale
    gray = cv2.cvtColor(wire_img_bgr, cv2.COLOR_BGR2GRAY)

    # Invert colors (lines become white on black)
    inv = cv2.bitwise_not(gray)

    # Threshold to binary (ensure crisp edges)
    _, thresh = cv2.threshold(inv, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty mask
    mask = np.zeros_like(gray)

    # Fill the contours
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Clean small holes or gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    return mask

def cluster_points(points, eps=2.5, min_samples=2):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    pred = model.fit_predict(points)
    return pred

def preprocess_points(points, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0, max_points=50000, scene_aabb=None):
    """
    Preprocess point cloud by downsampling and removing statistical outliers using Open3D.
    
    Args:
        points: numpy array of shape [N, 3] containing 3D point coordinates
        voxel_size: float, size of the voxel grid for downsampling (default: 0.02)
        nb_neighbors: int, number of neighbors for statistical outlier removal (default: 20)
        std_ratio: float, standard deviation ratio for outlier removal (default: 2.0)
        max_points: int, maximum number of points after preprocessing (default: 50000)
        scene_aabb: Optional dict with keys {"min", "max"} giving 3D AABB arrays
        
    Returns:
        numpy array of preprocessed points with shape [M, 3] where M <= max_points
    """
    if len(points) == 0:
        return points
    
    original_count = len(points)
    print(f"Original point cloud has {original_count} points")

    # Optional: crop to scene AABB before any processing
    if scene_aabb is not None and "min" in scene_aabb and "max" in scene_aabb:
        vmin = np.asarray(scene_aabb["min"]).reshape(3)
        vmax = np.asarray(scene_aabb["max"]).reshape(3)
        in_x = (points[:, 0] >= vmin[0]) & (points[:, 0] <= vmax[0])
        in_y = (points[:, 1] >= vmin[1]) & (points[:, 1] <= vmax[1])
        in_z = (points[:, 2] >= vmin[2]) & (points[:, 2] <= vmax[2])
        in_bounds = in_x & in_y & in_z
        cropped_points = points[in_bounds]
        removed = original_count - len(cropped_points)
        print(f"Cropped {removed} points outside scene AABB")
        if len(cropped_points) == 0:
            return cropped_points
        points = cropped_points
    
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Step 1: Downsample using voxel grid
    if voxel_size > 0:
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_count = len(downsampled_pcd.points)
        print(f"Downsampled to {downsampled_count} points (voxel_size={voxel_size})")
    else:
        downsampled_pcd = pcd
        downsampled_count = original_count
        print("No voxel downsampling applied")
    
    # Step 2: Remove statistical outliers
    if len(downsampled_pcd.points) > nb_neighbors:
        cl, ind = downsampled_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, 
            std_ratio=std_ratio
        )
        clean_pcd = downsampled_pcd.select_by_index(ind)
        outlier_count = downsampled_count - len(clean_pcd.points)
        print(f"Removed {outlier_count} statistical outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
    else:
        clean_pcd = downsampled_pcd
        print("Not enough points for statistical outlier removal")
    
    # Step 3: Further downsampling if still too many points
    if len(clean_pcd.points) > max_points:
        # Use random sampling to get exactly max_points
        indices = np.random.choice(len(clean_pcd.points), max_points, replace=False)
        final_pcd = clean_pcd.select_by_index(indices)
        final_count = max_points
        print(f"Random downsampled to {final_count} points")
    else:
        final_pcd = clean_pcd
        final_count = len(final_pcd.points)
        print("No additional downsampling needed")
    
    # Convert back to numpy array
    processed_points = np.asarray(final_pcd.points)
    
    print(f"Final point cloud has {final_count} points ({final_count/original_count*100:.1f}% of original)")
    
    return processed_points

def obj_annotations_from_pointcloud(points, eps=2.5, min_samples=2):
    """
    Generate object annotations from a point cloud in the same format as language_to_bboxes.
    
    Args:
        points: numpy array of shape [N, 3] containing 3D point coordinates
        eps: DBSCAN eps parameter for clustering
        min_samples: DBSCAN min_samples parameter for clustering
        
    Returns:
        List of bounding box dictionaries in the same format as language_to_bboxes output
    """
    if len(points) == 0:
        return []
    
    # Cluster points using DBSCAN
    cluster_labels = cluster_points(points, eps=eps, min_samples=min_samples)
    
    # Get unique cluster IDs (excluding noise points labeled as -1)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  # Remove noise points
    
    bbox_definitions = []
    
    for i, cluster_id in enumerate(unique_clusters):
        # Get points belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_pts = points[cluster_mask]

        if len(cluster_pts) < min_samples:
            continue

        # Axis-aligned bounding box from min/max coordinates
        min_xyz = np.min(cluster_pts, axis=0)
        max_xyz = np.max(cluster_pts, axis=0)
        center = (min_xyz + max_xyz) * 0.5
        scale = (max_xyz - min_xyz)
        rotation = np.eye(3, dtype=float)

        # Create bounding box dictionary matching language_to_bboxes format
        class_name = "object"
        class_label = 99  # generic label for detected objects
        identifier_label = f"{class_name}{i}"

        box = {
            "id": identifier_label,
            "cmd": "make_object",
            "class": class_name,
            "label": class_label,
            "center": center,
            "rotation": rotation,
            "scale": scale,
        }
        bbox_definitions.append(box)
    
    print(f"Generated {len(bbox_definitions)} object annotations from point cloud clustering.")
    return bbox_definitions

if __name__ == "__main__":
    # Test with random points (create a larger point cloud with some outliers)
    np.random.seed(42)  # For reproducible results
    points = np.random.rand(2000, 3) * 10  # Main cluster
    outliers = np.random.rand(50, 3) * 100  # Outliers far from main cluster
    points_with_outliers = np.vstack([points, outliers])
    
    print("=== Testing preprocess_points function ===")
    # Test preprocessing function
    preprocessed_points = preprocess_points(points_with_outliers, voxel_size=0.5, nb_neighbors=20, std_ratio=2.0, max_points=1000)
    print(f"Preprocessed points shape: {preprocessed_points.shape}")
    
    print("\n=== Testing clustering function ===")
    pred = cluster_points(preprocessed_points, eps=2.5, min_samples=50)
    print(f"Clustering result: {len(np.unique(pred))} clusters found")
    
    print("\n=== Testing obj_annotations_from_pointcloud function ===")
    bboxes = obj_annotations_from_pointcloud(preprocessed_points, eps=2.5, min_samples=50)
    print(f"Generated {len(bboxes)} bounding boxes")
    
    # Print details of first bbox if any exist
    if bboxes:
        bbox = bboxes[0]
        print(f"\nFirst bbox details:")
        print(f"  ID: {bbox['id']}")
        print(f"  Class: {bbox['class']}")
        print(f"  Center: {bbox['center']}")
        print(f"  Scale: {bbox['scale']}")
        print(f"  Rotation shape: {bbox['rotation'].shape}")