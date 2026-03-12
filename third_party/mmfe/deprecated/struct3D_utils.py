import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import random
from shapely.geometry import Polygon

def scene_to_lidar_image_old(scene_path, scene_id, bbox_percentage=1.0, figsize=(10, 10), dpi=100, 
                    points_per_unit_length=0.01, noise_std=70):
    """
    Convert a scene path to a simulated LiDAR scan image as a numpy array.
    
    This function samples points along the contours of the floorplan polygons,
    adds noise to simulate real LiDAR data, and creates a visualization.
    
    Args:
        scene_path (str): Path to the dataset directory
        scene_id (int): Scene ID number
        no_color (bool): If True, use no color for visualization
        bbox_percentage (float): Percentage of bounding boxes to display (0.0 to 1.0). 
                               1.0 means all boxes, 0.0 means no boxes.
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): DPI for the image
    
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
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
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
        mask = create_polygon_mask([poly[0] for poly in polygons], junctions_2d, shape=(500, 500))
        
        # Sample bounding boxes
        num_boxes = len(boxes)
        num_to_show = int(num_boxes * bbox_percentage)
        
        if num_to_show < num_boxes:
            selected_indices = random.sample(range(num_boxes), num_to_show)
            selected_boxes = [boxes[i] for i in selected_indices]
        else:
            selected_boxes = boxes
        
        # Check which boxes are inside the floorplan
        valid_boxes = []
        for bbox in selected_boxes:
            if is_bbox_inside_floorplan(bbox, junctions_2d, mask):
                valid_boxes.append(bbox)
        
        # Sample points along bounding box contours
        for bbox in valid_boxes:
            basis = np.array(bbox['basis'])
            coeffs = np.array(bbox['coeffs'])
            centroid = np.array(bbox['centroid'])

            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            corners_2d = corners[[0, 1, 2, 3], :2]
            
            # Sample points along the bounding box perimeter
            # Use a higher density for bounding boxes (furniture/objects)
            bbox_points_per_unit = points_per_unit_length * 2  # Double the density for bounding boxes
            
            # Convert corners to polygon vertices format for the sampling function
            # The sampling function expects a list of vertex indices, but we have actual coordinates
            # So we'll sample manually along the edges
            edges = np.roll(corners_2d, -1, axis=0) - corners_2d
            edge_lengths = np.linalg.norm(edges, axis=1)
            bbox_perimeter = np.sum(edge_lengths)
            
            # Sample points proportionally to perimeter
            total_bbox_points = int(bbox_perimeter * bbox_points_per_unit)
            
            if total_bbox_points > 0:
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
        # Return empty image if no points
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.text(0, 0, 'No LiDAR points generated', ha='center', va='center')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        plt.close(fig)
        buf.close()
        return img_array
    
    # Convert to numpy array efficiently
    lidar_points = np.array(lidar_points)
    
    # Create the visualization
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot LiDAR points efficiently
    ax.scatter(lidar_points[:, 0], lidar_points[:, 1], c='black', s=1, alpha=0.7)
    
    # Set the plot limits to match the floorplan bounds (using junctions_2d)
    # This ensures both images have the same coordinate system and aspect ratio
    x_min, x_max = junctions_2d[:, 0].min(), junctions_2d[:, 0].max()
    y_min, y_max = junctions_2d[:, 1].min(), junctions_2d[:, 1].max()
    
    # Add small padding to match floorplan behavior
    # x_padding = (x_max - x_min) * 0.05
    # y_padding = (y_max - y_min) * 0.05
    x_padding = 0
    y_padding = 0
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Set equal aspect ratio and remove axes
    ax.set_aspect('equal')
    plt.axis('equal')
    plt.axis('off')
    
    # Convert matplotlib figure to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)
    
    # Convert to PIL Image and then to numpy array
    img = Image.open(buf)
    img_array = np.array(img)
    
    # Close the figure to free memory
    plt.close(fig)
    buf.close()
    
    return img_array


def scene_to_floorplan_image_old(scene_path, scene_id, no_color=False, bbox_percentage=1.0, figsize=(10, 10), dpi=100, profile=False):
    timings = {}
    start_total = time.perf_counter()

    # Load annotations
    t0 = time.perf_counter()
    with open(os.path.join(scene_path, f"scene_{scene_id:05d}", "annotation_3d.json")) as file:
        annos = json.load(file)
    with open(os.path.join(scene_path, f"scene_{scene_id:05d}", "bbox_3d.json")) as file:
        boxes = json.load(file)
    timings["load_json"] = time.perf_counter() - t0

    # extract the floor in each semantic for floorplan visualization
    t0 = time.perf_counter()
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})
        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']
    timings["extract_planes"] = time.perf_counter() - t0

    # extract hole vertices
    t0 = time.perf_counter()
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)
    timings["extract_holes"] = time.perf_counter() - t0

    # junctions on the floor
    t0 = time.perf_counter()
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]
    timings["junctions"] = time.perf_counter() - t0

    # construct each polygon
    t0 = time.perf_counter()
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])
    timings["construct_polygons"] = time.perf_counter() - t0

    # outer wall polygons
    t0 = time.perf_counter()
    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])
    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.append([outerwall_polygon[0], 'outwall'])
    timings["outerwall"] = time.perf_counter() - t0

    # plotting
    t0 = time.perf_counter()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    junctions_2d = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    for (polygon, poly_type) in polygons:
        polygon = np.array(polygon + [polygon[0], ])
        polygon = Polygon(junctions_2d[polygon])
        facecolor = semantics_cmap[poly_type] if not no_color else semantics_cmap["no_color"]
        if poly_type == 'outwall':
            plot_polygon(polygon, ax=ax, add_points=False, facecolor=facecolor, alpha=0, edgecolor='black')
        else:
            plot_polygon(polygon, ax=ax, add_points=False, facecolor=facecolor, alpha=0.5, edgecolor='black')
    timings["plot_polygons"] = time.perf_counter() - t0

    # bounding boxes
    t0 = time.perf_counter()
    if bbox_percentage > 0:
        mask = create_polygon_mask([poly[0] for poly in polygons], junctions_2d, shape=(500, 500))
        num_boxes = len(boxes)
        num_to_show = int(num_boxes * bbox_percentage)
        if num_to_show < num_boxes:
            selected_indices = random.sample(range(num_boxes), num_to_show)
            selected_boxes = [boxes[i] for i in selected_indices]
        else:
            selected_boxes = boxes

        valid_boxes = []
        for bbox in selected_boxes:
            if is_bbox_inside_floorplan(bbox, junctions_2d, mask):
                valid_boxes.append(bbox)

        for bbox in valid_boxes:
            basis = np.array(bbox['basis'])
            coeffs = np.array(bbox['coeffs'])
            centroid = np.array(bbox['centroid'])
            corners = get_corners_of_bb3d_no_index(basis, coeffs, centroid)
            corners = corners[[0, 1, 2, 3, 0], :2]
            polygon = Polygon(corners)
            facecolor = semantics_cmap["no_color"] if no_color else colors.rgb2hex(np.random.rand(3))
            plot_polygon(polygon, ax=ax, add_points=False, facecolor=facecolor, alpha=0.5, edgecolor='black')
    timings["bounding_boxes"] = time.perf_counter() - t0

    # render to numpy
    t0 = time.perf_counter()
    plt.axis('equal')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    buf.close()
    timings["render_to_numpy"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - start_total

    if profile:
        print("\n--- Profiling timings ---")
        for k, v in timings.items():
            print(f"{k:20s}: {v:.4f} sec")

    return img_array