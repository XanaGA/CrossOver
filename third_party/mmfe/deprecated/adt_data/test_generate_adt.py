import numpy as np
import cv2
from pathlib import Path
import random 

from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataProvider,
   AriaDigitalTwinDataPathsProvider,
   bbox3d_to_line_coordinates,
)
from aria_mmfe.code_snippets.plotters import render_polylines_orthographic_cv


if __name__ == "__main__":
    # Use a shorter, more common sequence for a clear example
    # sequence_path = f"./data/aria/Digitaltwin/original_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292"
    sequence_path = f"./data/aria/Digitaltwin/original_data/Apartment_release_clean_seq133_M1292"
    paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    data_paths = paths_provider.get_datapaths()
    print(data_paths)

    print("loading ground truth data...")
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    print("done loading ground truth data")

    # Use dataset time range instead of Aria frames
    start_ns = gt_provider.get_start_time_ns()
    end_ns = gt_provider.get_end_time_ns()
    mid_ns = int((start_ns + end_ns) / 2)
    print(f"Sequence covers {start_ns} → {end_ns}, picking {mid_ns}")

    # Query 3D bounding boxes at mid timestamp
    bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(mid_ns)
    assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
    bboxes3d = bbox3d_with_dt.data()
    print("groundtruth_time - query_time = ", bbox3d_with_dt.dt_ns(), "ns")

    # This will now be a list of all line segments from all boxes
    all_lines = []
    for obj_id in bboxes3d:
        bbox3d = bboxes3d[obj_id]
        aabb = bbox3d.aabb
        
        # This gives 24 points, representing 12 lines
        aabb_coords = bbox3d_to_line_coordinates(aabb)
        obb_coords = np.zeros_like(aabb_coords)

        # Transform all points at once for efficiency
        for i, aabb_pt in enumerate(aabb_coords):
            aabb_pt_homo = np.append(aabb_pt, [1])
            obb_pt = (bbox3d.transform_scene_object.to_matrix() @ aabb_pt_homo)[0:3]
            obb_coords[i] = obb_pt

        # --- THIS IS THE KEY CHANGE ---
        # Reshape the (24, 3) points array into a (12, 2, 3) lines array
        # Each element is now a line segment with a start and end point
        obb_lines = obb_coords.reshape(-1, 2, 3)
        
        # Add each of the 12 line segments to our master list
        all_lines.extend(list(obb_lines))

    # Orthographic projection of OBB polylines
    poly_img = render_polylines_orthographic_cv(
        all_lines, axis="y", image_size=(1080, 1080), margin_frac=0.05, line_thickness=2
    )
    
    # You can also save the image to inspect it
    # cv2.imwrite("floorplan_projection.png", poly_img)
    
    cv2.imshow("ADT OBBs Ortho (top-down)", poly_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
