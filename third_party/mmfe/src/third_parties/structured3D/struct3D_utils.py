import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from shapely.geometry import Polygon, Point
from shapely.plotting import plot_polygon
import io
from PIL import Image
import random
import cv2
import time

semantics_cmap = {
    'living room': '#e6194b',
    'kitchen': '#3cb44b',
    'bedroom': '#ffe119',
    'bathroom': '#0082c8',
    'balcony': '#f58230',
    'corridor': '#911eb4',
    'dining room': '#46f0f0',
    'study': '#f032e6',
    'studio': '#d2f53c',
    'store room': '#fabebe',
    'garden': '#008080',
    'laundry room': '#e6beff',
    'office': '#aa6e28',
    'basement': '#fffac8',
    'garage': '#800000',
    'undefined': '#aaffc3',
    'door': '#808000',
    'window': '#ffd7b4',
    'outwall': '#000000',
    "no_color": "#ffffff"
}

def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners

def convert_lines_to_vertices(lines):
    """Convert line representation to polygon vertices.
    Each line is [start, end] with junction indices.
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) > 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        # find a line that continues from the last vertex
        lineID, juncID = np.where(lines == polygon[-1])
        if len(lineID) == 0:
            # no continuation found → polygon is open/broken
            # save partial polygon if it has enough vertices
            if polygon and len(polygon) >= 3:
                polygons.append(polygon)
            polygon = None
            continue

        # continue polygon
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID[0], 0)

        if vertex in polygon:
            # closed loop
            if len(polygon) >= 3:
                polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    # catch last polygon if loop ended mid-way
    if polygon and len(polygon) >= 3:
        polygons.append(polygon)

    return polygons



def create_polygon_mask(polygons, junctions, shape=(1000, 1000), viz=False):
    """
    Create a binary mask from a list of polygons using OpenCV for speed.
    
    Args:
        polygons (list): List of polygon vertex indices
        junctions (np.ndarray): Array of junction coordinates
        shape (tuple): Shape of the output mask (height, width)
    
    Returns:
        np.ndarray: Binary mask where 1 is inside polygons, 0 is outside
    """
    # Create empty mask
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Get bounds for coordinate transformation
    x_min, x_max = junctions[:, 0].min(), junctions[:, 0].max()
    y_min, y_max = junctions[:, 1].min(), junctions[:, 1].max()
    
    # For each polygon, fill it in the mask
    # skip the last polygon, which is the outerwall
    for polygon_vertices in polygons[:-1]:
        if len(polygon_vertices) < 3:
            print("polygon_vertices", len(polygon_vertices))
            continue
            
        # Get polygon coordinates
        polygon_coords = junctions[polygon_vertices]
        
        # Transform coordinates to image space
        # OpenCV uses (x, y) = (col, row) and origin at top-left
        x_coords = ((polygon_coords[:, 0] - x_min) / (x_max - x_min) * (shape[1] - 1)).astype(np.int32)
        y_coords = ((polygon_coords[:, 1] - y_min) / (y_max - y_min) * (shape[0] - 1)).astype(np.int32)
        
        # Create polygon points for OpenCV
        polygon_points = np.column_stack([x_coords, y_coords])
        
        # Fill the polygon in the mask
        cv2.fillPoly(mask, [polygon_points], 1)

    # Show mask if requested
    if viz:
        # Create a separate figure for the mask
        mask_fig, mask_ax = plt.subplots(figsize=(8, 8))
        mask_ax.imshow(mask, cmap='gray', origin='lower')
        mask_ax.set_title('Binary Mask (White = Inside, Black = Outside)')
        mask_ax.axis('off')
        plt.show()
    
    return mask


def is_bbox_inside_floorplan(bbox, junctions, mask):
    """
    Check if a bounding box is inside the floorplan area.
    
    Args:
        bbox (dict): Bounding box dictionary with 'basis', 'coeffs', 'centroid'
        junctions (np.ndarray): Array of junction coordinates
        mask (np.ndarray): Binary mask of the floorplan
    
    Returns:
        bool: True if bbox is inside floorplan, False otherwise
    """
    # Get corners of the bounding box
    basis = np.array(bbox['basis'])
    coeffs = np.array(bbox['coeffs'])
    centroid = np.array(bbox['centroid'])
    
    corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
    # Take only the first 4 corners (floor level) and 2D coordinates
    corners_2d = corners[[0, 1, 2, 3], :2]
    
    # Get bounds for coordinate transformation (same as in create_polygon_mask)
    x_min, x_max = junctions[:, 0].min(), junctions[:, 0].max()
    y_min, y_max = junctions[:, 1].min(), junctions[:, 1].max()
    
    def world_to_mask_coords(point):
        """Convert world coordinates to mask coordinates"""
        x_idx = int((point[0] - x_min) / (x_max - x_min) * (mask.shape[1] - 1))
        y_idx = int((point[1] - y_min) / (y_max - y_min) * (mask.shape[0] - 1))
        return x_idx, y_idx
    
    def is_point_inside_mask(point):
        """Check if a point is inside the mask"""
        x_idx, y_idx = world_to_mask_coords(point)
        if 0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]:
            return mask[y_idx, x_idx] == 1
        return False
    
    # Check all corners first (fast early exit)
    for corner in corners_2d:
        if not is_point_inside_mask(corner):
            return False
    
    # Check edges by sampling points along them
    # Define the edges of the bounding box (4 edges)
    edges = [
        (corners_2d[0], corners_2d[1]),  # Edge 0-1
        (corners_2d[1], corners_2d[2]),  # Edge 1-2
        (corners_2d[2], corners_2d[3]),  # Edge 2-3
        (corners_2d[3], corners_2d[0])   # Edge 3-0
    ]
    
    # Sample points along each edge
    num_samples = 5  # Adjust based on desired accuracy vs speed
    
    for start_point, end_point in edges:
        # Sample points along the edge
        for i in range(num_samples + 1):  # +1 to include both endpoints
            t = i / num_samples
            # Linear interpolation between start and end points
            sample_point = start_point * (1 - t) + end_point * t
            
            if not is_point_inside_mask(sample_point):
                return False
    
    return True

def scene_to_floorplan_image(scene_path, scene_id, no_color=False, bbox_percentage=1.0, figsize=(10, 10), dpi=100):

    # Load annotations
    with open(os.path.join(scene_path, f"scene_{scene_id:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)
    with open(os.path.join(scene_path, f"scene_{scene_id:05d}", "bbox_3d.json")) as file:
        boxes = json.load(file)

    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})
        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        # keep only proper 2-junction lines
        junction_pairs = [p for p in junction_pairs if len(p) == 2]
        if not junction_pairs:
            continue  # nothing usable for this plane
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

   # --- outer wall polygons ---
    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes).tolist()

        raw_pairs = [
            np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        # filter to valid pairs that lie on the floor (z ≈ 0)
        for pair in raw_pairs:
            if len(pair) != 2:
                continue
            start, end = pair
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    # only try to convert if we actually have edges
    if outerwall_floor:
        outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
        if outerwall_polygon and len(outerwall_polygon[0]) >= 3:
            polygons.append([outerwall_polygon[0], 'outwall'])


    # Get 2D junctions
    junctions_2d = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    
    # Calculate bounds for coordinate transformation
    x_min, x_max = junctions_2d[:, 0].min(), junctions_2d[:, 0].max()
    y_min, y_max = junctions_2d[:, 1].min(), junctions_2d[:, 1].max()

    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    if aspect_ratio > 1:
        width_px = int(figsize[0] * dpi * aspect_ratio)
        height_px = int(figsize[1] * dpi)
        pixels_to_pad = int((width_px - height_px) / 2)
        padding = ((pixels_to_pad, pixels_to_pad), (0, 0), (0, 0))
    else:
        width_px = int(figsize[0] * dpi)
        height_px = int(figsize[1] * dpi * (1 / aspect_ratio))
        pixels_to_pad = int((height_px - width_px) / 2)
        padding = ((0, 0), (pixels_to_pad, pixels_to_pad), (0, 0))

    # Create image with white background
    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    
    def world_to_image_coords(points):
        """Convert world coordinates to image coordinates"""
        x_coords = ((points[:, 0] - x_min) / (x_max - x_min) * (width_px - 1)).astype(np.int32)
        # Flip y-axis to match matplotlib coordinate system (bottom-left origin)
        y_coords = (height_px - 1) - ((points[:, 1] - y_min) / (y_max - y_min) * (height_px - 1)).astype(np.int32)
        return np.column_stack([x_coords, y_coords])
    
    def hex_to_bgr(hex_color):
        """Convert hex color to BGR (OpenCV format)"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # BGR order
    
    # Render polygons using OpenCV
    for polygon, poly_type in polygons[:-1]:
        if len(polygon) < 3:
            continue
            
        # Get polygon coordinates and convert to image space
        poly_coords = junctions_2d[polygon]
        poly_coords_img = world_to_image_coords(poly_coords)
        
        # Get color for this polygon type
        if no_color:
            color = hex_to_bgr(semantics_cmap["no_color"])
        else:
            color = hex_to_bgr(semantics_cmap[poly_type])
        
        # # Fill polygon
        # cv2.fillPoly(img, [poly_coords_img], color)
        
        # Add black border for all polygons
        cv2.polylines(img, [poly_coords_img], True, (0, 0, 0), 1)

    # Render bounding boxes using OpenCV
    if bbox_percentage > 0:
        # Create mask for bbox validation (reuse existing function)
        mask = create_polygon_mask([poly[0] for poly in polygons], junctions_2d, shape=(500, 500))
        
        num_boxes = len(boxes)
        num_to_show = int(num_boxes * bbox_percentage)
        if num_to_show < num_boxes:
            selected_indices = random.sample(range(num_boxes), num_to_show)
            selected_boxes = [boxes[i] for i in selected_indices]
        else:
            selected_boxes = boxes

        # Validate bounding boxes
        valid_boxes = []
        for bbox in selected_boxes:
            if is_bbox_inside_floorplan(bbox, junctions_2d, mask):
                valid_boxes.append(bbox)

        # Render bounding boxes
        for bbox in valid_boxes:
            basis = np.array(bbox['basis'])
            coeffs = np.array(bbox['coeffs'])
            centroid = np.array(bbox['centroid'])
            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            corners_2d = corners[[0, 1, 2, 3], :2]
            
            # Convert to image coordinates
            corners_img = world_to_image_coords(corners_2d)
            
            # Get color for bbox
            if no_color:
                color = hex_to_bgr(semantics_cmap["no_color"])
            else:
                # Generate random color for bboxes
                color = tuple(np.random.randint(0, 256, 3).tolist())
            
            # # Fill bbox polygon
            # cv2.fillPoly(img, [corners_img], color)
            # Add black border
            cv2.polylines(img, [corners_img], True, (0, 0, 0), 1)

    # Convert BGR to RGB (OpenCV uses BGR, but we want RGB output)
    # Pad to square image before converting to RGB
    img = np.pad(img, padding, mode='constant', constant_values=255)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb

def scene_to_lidar_image(scene_path, scene_id, bbox_percentage=1.0, figsize=(10, 10), dpi=100, 
                    points_per_unit_length=0.02, noise_std=110):
    """
    Convert a scene path to a simulated LiDAR scan image as a numpy array.
    
    This function samples points along the contours of the floorplan polygons,
    adds noise to simulate real LiDAR data, and creates a visualization.
    
    Args:
        scene_path (str): Path to the dataset directory
        scene_id (int): Scene ID number
        bbox_percentage (float): Percentage of bounding boxes to display (0.0 to 1.0). 
                               1.0 means all boxes, 0.0 means no boxes.
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): DPI for the image
        points_per_unit_length (float): Density of points to sample along polygon edges
        noise_std (float): Standard deviation of noise to add to points
    
    Returns:
        numpy.ndarray: RGB image as numpy array with shape (H, W, 3)
    """
    
    # Load annotations
    with open(os.path.join(scene_path, f"scene_{scene_id:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)

    with open(os.path.join(scene_path, f"scene_{scene_id:05d}", "bbox_3d.json")) as file:
        boxes = json.load(file)

    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        # keep only proper 2-junction lines
        junction_pairs = [p for p in junction_pairs if len(p) == 2]
        if not junction_pairs:
            continue  # nothing usable for this plane
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

   # --- outer wall polygons ---
    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes).tolist()

        raw_pairs = [
            np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        # filter to valid pairs that lie on the floor (z ≈ 0)
        for pair in raw_pairs:
            if len(pair) != 2:
                continue
            start, end = pair
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    # only try to convert if we actually have edges
    if outerwall_floor:
        outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
        if outerwall_polygon and len(outerwall_polygon[0]) >= 3:
            polygons.append([outerwall_polygon[0], 'outwall'])

    # Get 2D junctions
    junctions_2d = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    
    # Sample points along polygon contours for LiDAR simulation
    lidar_points = []
    
    # Sample points from all polygons (excluding outer wall)
    for polygon_vertices, poly_type in polygons[:-1]:
        if len(polygon_vertices) < 3:
            continue

        # Use appropriate sampling density based on polygon type
        points_per_unit = points_per_unit_length + 0.1 * points_per_unit_length * random.random()
            
        points = sample_points_along_polygon((junctions_2d, polygon_vertices), points_per_unit_length=points_per_unit, noise_std=noise_std)
        if len(points) > 0:
            lidar_points.extend(points)
    # Add points from bounding boxes to simulate furniture/objects
    if bbox_percentage > 0:
        # Create mask for bbox validation
        mask = create_polygon_mask([poly[0] for poly in polygons], junctions_2d, shape=(500, 500))
        
        # Select bounding boxes
        num_boxes = len(boxes)
        num_to_show = int(num_boxes * bbox_percentage)
        
        if num_to_show < num_boxes:
            selected_indices = random.sample(range(num_boxes), num_to_show)
            selected_boxes = [boxes[i] for i in selected_indices]
        else:
            selected_boxes = boxes
        
        # Validate bounding boxes
        valid_boxes = []
        for bbox in selected_boxes:
            if is_bbox_inside_floorplan(bbox, junctions_2d, mask):
                valid_boxes.append(bbox)
        
        # Sample points for each bbox
        for bbox in valid_boxes:
            # Compute corners
            basis = np.array(bbox['basis'])
            coeffs = np.array(bbox['coeffs'])
            centroid = np.array(bbox['centroid'])

            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            corners_2d = corners[[0, 1, 2, 3], :2]
            
            # Calculate edges
            edges = np.roll(corners_2d, -1, axis=0) - corners_2d
            edge_lengths = np.linalg.norm(edges, axis=1)
            bbox_perimeter = np.sum(edge_lengths)
            
            # Generate points
            bbox_points_per_unit = points_per_unit_length * 2  # Double the density for bounding boxes
            total_bbox_points = int(bbox_perimeter * bbox_points_per_unit)
            
            if total_bbox_points > 0:
                # points_per_unit = points_per_unit_length + 0.1 * points_per_unit_length * random.random()
                # bb_points = sample_points_along_polygon(corners_2d, points_per_unit, noise_std=noise_std)
                # lidar_points.extend(bb_points)
                for start_point, edge, edge_length in zip(corners_2d, edges, edge_lengths):
                    # Calculate points for this edge
                    num_points_for_edge = max(1, int((edge_length / bbox_perimeter) * total_bbox_points))
                    
                    if num_points_for_edge == 1:
                        # Single point at the middle of the edge
                        point = start_point + 0.5 * edge
                        noise = np.random.normal(0, noise_std, 2)
                        lidar_points.append(point + noise)
                    else:
                        # Multiple points along the edge
                        t_values = np.linspace(0, 1, num_points_for_edge)
                        points = start_point[None, :] + t_values[:, None] * edge[None, :]
                        
                        # Add noise vectorized
                        noise = np.random.normal(0, noise_std, (num_points_for_edge, 2))
                        points_with_noise = points + noise
                        
                        lidar_points.extend(points_with_noise)
    
    # Convert to numpy array efficiently
    if not lidar_points:
        raise ValueError("No LiDAR points generated")
    
    # Convert to numpy array efficiently
    lidar_points = np.array(lidar_points)
    
    # Calculate bounds for coordinate transformation
    x_min, x_max = junctions_2d[:, 0].min(), junctions_2d[:, 0].max()
    y_min, y_max = junctions_2d[:, 1].min(), junctions_2d[:, 1].max()

    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    if aspect_ratio > 1:
        width_px = int(figsize[0] * dpi * aspect_ratio)
        height_px = int(figsize[1] * dpi)
        pixels_to_pad = int((width_px - height_px) / 2)
        padding = ((pixels_to_pad, pixels_to_pad), (0, 0), (0, 0))
    else:
        width_px = int(figsize[0] * dpi)
        height_px = int(figsize[1] * dpi * (1 / aspect_ratio))
        pixels_to_pad = int((height_px - width_px) / 2)
        padding = ((0, 0), (pixels_to_pad, pixels_to_pad), (0, 0))
    
    # Create image with white background
    img = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    
    def world_to_image_coords(points):
        """Convert world coordinates to image coordinates"""
        x_coords = ((points[:, 0] - x_min) / (x_max - x_min) * (width_px - 1)).astype(np.int32)
        # Flip y-axis to match matplotlib coordinate system (bottom-left origin)
        y_coords = (height_px - 1) - ((points[:, 1] - y_min) / (y_max - y_min) * (height_px - 1)).astype(np.int32)
        return np.column_stack([x_coords, y_coords])
    
    # Convert LiDAR points to image coordinates
    lidar_points_img = world_to_image_coords(lidar_points)
    
    # Filter points that are within image bounds
    valid_mask = (lidar_points_img[:, 0] >= 0) & (lidar_points_img[:, 0] < width_px) & \
                 (lidar_points_img[:, 1] >= 0) & (lidar_points_img[:, 1] < height_px)
    lidar_points_img = lidar_points_img[valid_mask]
    
    # Render LiDAR points using OpenCV
    if len(lidar_points_img) > 0:
        # Draw points as small circles (more efficient than individual point drawing)
        for point in lidar_points_img:
            cv2.circle(img, tuple(point), 1, (0, 0, 0), -1)

    # # Render LiDAR points vectorized
    # if len(lidar_points_img) > 0:
    #    img[lidar_points_img[:, 1], lidar_points_img[:, 0]] = (0, 0, 0)

       # TODO: Add density map
       #unique_coordinates, counts = np.unique(lidar_points_img, return_counts=True, axis=0)
       #unique_coordinates = unique_coordinates.astype(np.int32)

       #density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts
       #density = density / np.max(density)
    
    # Convert BGR to RGB (OpenCV uses BGR, but we want RGB output)
    # Pad to square image before converting to RGB
    img = np.pad(img, padding, mode='constant', constant_values=255)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb

def sample_points_along_polygon(vertices, points_per_unit_length=10, noise_std=70):
    """
    Sample points along the edges of a polygon with added noise.
    The number of points is proportional to the perimeter of the polygon.
    
    Optimized version using vectorized operations.
    
    Args:
        junctions_2d (np.ndarray): Array of 2D junction coordinates
        polygon_vertices (list): List of vertex indices
        points_per_unit_length (float): Number of points per unit length of perimeter
        noise_std (float): Standard deviation of noise to add
        
    Returns:
        np.ndarray: Array of (x, y) coordinates with noise
    """
    if isinstance(vertices, tuple):
        junctions_2d, polygon_vertices = vertices
        vertices = junctions_2d[polygon_vertices]
        
    if len(vertices) < 3:
        return np.array([])
            
    # Close the polygon if not already closed
    if len(vertices) > 0 and not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    
    # Vectorized calculation of edge lengths
    edges = vertices[1:] - vertices[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    
    # Filter out very short edges
    valid_edges = edge_lengths > 1e-6
    if not np.any(valid_edges):
        return np.array([])
    
    edges = edges[valid_edges]
    edge_lengths = edge_lengths[valid_edges]
    start_vertices = vertices[:-1][valid_edges]
    
    # Calculate total perimeter and points
    perimeter = np.sum(edge_lengths)
    total_points = int(perimeter * points_per_unit_length)
    
    if total_points == 0:
        return np.array([])
    
    # Pre-allocate arrays for efficiency
    all_points = []
    
    # Calculate points for each edge using vectorized operations
    for i, (start_vertex, edge, edge_length) in enumerate(zip(start_vertices, edges, edge_lengths)):
        # Calculate number of points for this edge
        num_points_for_edge = max(1, int((edge_length / perimeter) * total_points))
        
        if num_points_for_edge == 1:
            # Single point at the middle of the edge
            t = 0.5
            point = start_vertex + t * edge
            noise = np.random.normal(0, noise_std, 2)
            all_points.append(point + noise)
        else:
            # Multiple points along the edge
            t_values = np.linspace(0, 1, num_points_for_edge)
            points = start_vertex[None, :] + t_values[:, None] * edge[None, :]
            
            # Add noise vectorized
            noise = np.random.normal(0, noise_std, (num_points_for_edge, 2))
            points_with_noise = points + noise
            
            all_points.extend(points_with_noise)
    
    return np.array(all_points)

def scene_to_density_map(scene_path, scene_id):
    """
    Load a density map image as a numpy array, flipped vertically and cropped to content.
    If the image is not square, pad with zeros to make it square.
    
    Args:
        scene_path (str): Path to the dataset directory
        scene_id (int): Scene ID number
        margin_pixels (int): Number of pixels to leave as margin around the content
    
    Returns:
        numpy.ndarray: Cropped and square-padded density map as numpy array
    """
    density_map_path = os.path.join(scene_path, f"scene_{scene_id:05d}", f"density_map_{scene_id:05d}.png")
    density_map = cv2.imread(density_map_path)
    if density_map is not None:
        density_map = 255 - density_map
        density_map = np.flipud(density_map)
        # Pad to square if needed
        h, w = density_map.shape[:2]
        if h != w:
            size = max(h, w)
            # Efficient square padding using np.full and assignment
            if len(density_map.shape) == 3:
                padded = np.full((size, size, density_map.shape[2]), 255, dtype=density_map.dtype)
            else:
                padded = np.full((size, size), 255, dtype=density_map.dtype)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            padded[y_offset:y_offset + h, x_offset:x_offset + w, ...] = density_map
            density_map = padded
        
    #     # Convert to binary image to find content bounds
    #     # Use grayscale if the image has multiple channels
    #     if len(density_map.shape) == 3:
    #         gray_map = cv2.cvtColor(density_map, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray_map = density_map
        
    #     # Create binary image (threshold to separate content from background)
    #     # Assuming white/light areas are content and black areas are background
    #     _, binary_map = cv2.threshold(gray_map, 127, 255, cv2.THRESH_BINARY)
        
    #     # Find the bounding box of non-zero pixels (content)
    #     coords = cv2.findNonZero(binary_map)
    #     if coords is not None:
    #         x, y, w, h = cv2.boundingRect(coords)
            
    #         # Add margin to the bounding box
    #         x_start = max(0, x - margin_pixels)
    #         y_start = max(0, y - margin_pixels)
    #         x_end = min(density_map.shape[1], x + w + margin_pixels)
    #         y_end = min(density_map.shape[0], y + h + margin_pixels)
            
    #         # Crop the density map
    #         density_map = density_map[y_start:y_end, x_start:x_end]
    
    return density_map