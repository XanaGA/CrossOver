import json
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

def load_scene_data(mesh_path, segments_path, segments_anno_path):
    """Loads all necessary data for a scene."""
    print("Loading mesh...")
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        print(f"Loaded mesh with {len(vertices)} vertices")
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        return None, None
    
    try:
        with open(segments_path, "r") as f:
            segs = json.load(f)
        # seg_indices = np.array(segs["segIndices"], dtype=np.int32) # Not strictly needed for OBB/density
    except Exception as e:
        print(f"Error loading segments {segments_path}: {e}")
        return None, None

    try:
        with open(segments_anno_path, "r") as f:
            anno = json.load(f)
        seg_groups = anno["segGroups"]
    except Exception as e:
        print(f"Error loading annotations {segments_anno_path}: {e}")
        return None, None

    return vertices, seg_groups

def get_scene_bounds_2d(vertices):
    """Calculates the 2D (XY) bounding box for the entire scene."""
    xy = vertices[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    return min_xy, max_xy

def calculate_density_map(vertices, min_xy, max_xy, res=1024):
    """Rasterizes a 2D log-density map of all vertices with white background and black points."""
    print("Rasterizing density map...")
    xy = vertices[:, :2]
    
    # Normalize to bounding box [0, 1]
    xy_norm = (xy - min_xy) / (max_xy - min_xy)
    
    # Scale to pixel coordinates
    xy_pix = (xy_norm * (res - 1)).astype(np.int32)

    # Initialize with white background (255)
    density = np.full((res, res), 255, dtype=np.float32)
    
    for x, y in xy_pix:
        # Clip coordinates to be within image bounds
        if 0 <= x < res and 0 <= y < res:
            # Mark points by decreasing value (making them darker)
            # We want to represent density as darkness, so more points = lower value
            # The exact value here can be tuned; simply decrementing or setting to 0 works too
            density[y, x] -= 1 # Decrease by 1, more overlap means darker
            
    # Apply log scale for visibility, then normalize to 0-255.
    # We invert the values for visualization: higher density (lower original value) should be black (0).
    # Since we started with 255 and subtracted, the lowest values are where density is highest.
    # To map this to 0 (black) for high density and 255 (white) for low density:
    # 1. Take 255 - current_value to flip the intensity.
    # 2. Apply log scale to the flipped values (this will represent the "count" in a visible way).
    # 3. Normalize this log-scaled count to 0-255.

    # First, ensure no values go below 0 (can happen if many overlaps in one pixel)
    density[density < 0] = 0

    # Calculate actual "counts" per pixel (how much was subtracted from 255)
    counts = 255 - density
    
    # Apply log scale to counts
    log_counts = np.log1p(counts) # log(1 + x)
    
    if log_counts.max() > 0:
        # Normalize log_counts to 0-1, then scale to 0-255 and cast to uint8
        # Higher log_counts (higher density) should be darker (closer to 0)
        normalized_log_counts = log_counts / log_counts.max()
        final_density = (255 * (1 - normalized_log_counts)).astype(np.uint8) # Invert for white background, black foreground
    else:
        final_density = np.full((res, res), 255, dtype=np.uint8) # All white if no points

    return final_density


def calculate_obb_mask(seg_groups, min_xy, max_xy, res=1024, label_filter=None):
    """
    Rasterizes a 2D mask by drawing the 2D footprints of OBBs with white background and black lines.
    
    Args:
        label_filter (str, optional): If provided, only draws OBBs with
                                      this label (e.g., "wall").
                                      If None, draws all OBBs.
    """
    print(f"Rasterizing OBB mask (filter: {label_filter})...")
    # Initialize with white background (255)
    mask = np.full((res, res), 255, dtype=np.uint8)

    for g in seg_groups:
        # Apply label filter
        if label_filter and g.get("label", "").lower() != label_filter:
            continue
        
        obb = g.get("obb")
        if not obb:
            continue

        try:
            center = np.array(obb["centroid"])
            extents = np.array(obb["axesLengths"]) / 2.0
            rot_matrix = np.array(obb["normalizedAxes"]).reshape(3, 3)

            # Find "floor" axes (least aligned with global Z)
            z_axis = np.array([0, 0, 1])
            z_alignment = np.abs(rot_matrix @ z_axis)
            up_axis_index = np.argmax(z_alignment)
            floor_indices = [i for i in range(3) if i != up_axis_index]
            
            axis_A = rot_matrix[floor_indices[0], :]
            axis_B = rot_matrix[floor_indices[1], :]
            extent_A = extents[floor_indices[0]]
            extent_B = extents[floor_indices[1]]

            # Calculate the 4 corners of the OBB's base in 3D
            corners_3d = []
            corners_3d.append(center + extent_A * axis_A + extent_B * axis_B)
            corners_3d.append(center + extent_A * axis_A - extent_B * axis_B)
            corners_3d.append(center - extent_A * axis_A - extent_B * axis_B)
            corners_3d.append(center - extent_A * axis_A + extent_B * axis_B)
            
            # Project to 2D and scale to pixel coordinates
            corners_2d = np.array(corners_3d)[:, :2]
            corners_norm = (corners_2d - min_xy) / (max_xy - min_xy)
            corners_pix = (corners_norm * (res - 1)).astype(np.int32)
            
            # Draw the closed polygon with black color (0)
            cv2.polylines(
                mask, 
                [corners_pix], 
                isClosed=True, 
                color=0, # Black color
                thickness=1
            )

        except Exception as e:
            print(f"Warning: Could not process OBB for group {g.get('id')}: {e}")
            continue
            
    return mask

def save_and_show_results(mask_img, density_img, mask_path, density_path, show=True):
    """Saves the images and optionally displays them with matplotlib."""
    
    # Flip vertically for saving (image origin vs. array origin)
    # The images are already in a [0, 255] range for PIL/matplotlib
    Image.fromarray(mask_img[::-1, :]).save(mask_path)
    Image.fromarray(density_img[::-1, :]).save(density_path)

    print("Saved:")
    print(f" - {mask_path}")
    print(f" - {density_path}")

    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("OBB Mask")
        plt.imshow(mask_img[::-1, :], cmap="gray") # cmap="gray" for 0=black, 255=white
        plt.subplot(1, 2, 2)
        plt.title("2D Orthographic Projection (log density)")
        plt.imshow(density_img[::-1, :], cmap="gray") # cmap="gray" for 0=black, 255=white
        plt.show()

def main():
    """Main script logic."""
    
    # === Configuration ===
    scene_id = "55b2bf8036"
    # scene_id = "00777c41d4"
    base_dir = "/cluster/project/cvg/Shared_datasets/scannetpp_v2/data/"
    save_dir = "/cluster/home/xanadon/mmfe/"
    res = 1024
    
    # --- Set label_filter ---
    # None: Plots all OBBs
    # "wall": Plots only OBBs with the "wall" label
    obb_filter = None 
    
    show_plots = True # Set to False to only save
    
    # === Paths ===
    scene_dir = f"{base_dir}/{scene_id}/scans/"
    mesh_path = scene_dir + "mesh_aligned_0.05.ply"
    segments_path = scene_dir + "segments.json"
    segments_anno_path = scene_dir + "segments_anno.json"
    
    mask_save_path = save_dir + "wall_mask.png"
    density_save_path = save_dir + "floorplan_projection.png"

    # === Load Data ===
    vertices, seg_groups = load_scene_data(mesh_path, segments_path, segments_anno_path)
    if vertices is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Loaded {len(seg_groups)} segment groups")

    # === Calculate Scene Bounds ===
    min_xy, max_xy = get_scene_bounds_2d(vertices)

    # === Generate Images ===
    density_map = calculate_density_map(vertices, min_xy, max_xy, res)
    obb_mask = calculate_obb_mask(seg_groups, min_xy, max_xy, res, label_filter=obb_filter)

    # === Save and Show ===
    save_and_show_results(
        obb_mask, 
        density_map, 
        mask_save_path, 
        density_save_path, 
        show=show_plots
    )

if __name__ == "__main__":
    main()