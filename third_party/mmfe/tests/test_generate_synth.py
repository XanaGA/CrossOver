import cv2
from pathlib import Path
from aria_mmfe.code_snippets.readers import read_points_file, read_language_file, read_points_file_clean
from aria_mmfe.code_snippets.interpreter import language_to_bboxes
from aria_mmfe.code_snippets.plotters import render_pointcloud_and_boxes_orthographic_cv
from aria_mmfe.ase_data.ase_utils import cluster_points, obj_annotations_from_pointcloud, preprocess_points



DATASET_ROOT = "./data/aria/SyntheticEnv/original_data"  # Specify your own dataset path
SCENE_ID = 0  # Select a scene id

if __name__ == "__main__":
    dataset_path = Path(DATASET_ROOT)
    print("Chosen ASE data path: ", dataset_path)
    print(f"Using Scene {SCENE_ID} for these examples")

    scene_path = dataset_path / str(SCENE_ID)


    # Load scene point cloud using read_points_file()
    points_path = scene_path / "semidense_points.csv.gz"
    # points = read_points_file(points_path)
    points = read_points_file_clean(points_path)


    # Load a scene command language using read_language_file()
    language_path = scene_path / "ase_scene_language.txt"
    entities = read_language_file(language_path)

    # Interpret scene commands into 3D Boxes
    entity_boxes, scene_aabb = language_to_bboxes(entities)

    # Preprocess the point cloud (crop to scene AABB, downsample and remove outliers)
    points_preprocessed = preprocess_points(points, voxel_size=0.05, nb_neighbors=20, std_ratio=2.0, max_points=50000, scene_aabb=scene_aabb)

    # # Generate object annotations from point cloud clustering
    # detected_boxes = obj_annotations_from_pointcloud(points_preprocessed, eps=0.15, min_samples=5)
    # n_boxes = len(entity_boxes) + len(detected_boxes)
    n_boxes = len(entity_boxes)
    
    
    print(f"Total boxes to display: {n_boxes} ")
    # detected_boxes = []

    point_img_bgr, wire_img_bgr = render_pointcloud_and_boxes_orthographic_cv(points, entity_boxes, axis="z", image_size=(1080, 1080))
    cv2.imshow("Aria Pointcloud (top-down)", point_img_bgr)
    cv2.imshow("Aria Boxes Ortho (top-down)", wire_img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()