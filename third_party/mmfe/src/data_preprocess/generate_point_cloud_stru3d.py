import argparse
import os
from tqdm import tqdm
from data_preprocess.PointCloudReaderPanorama import PointCloudReaderPanorama


def config():
    a = argparse.ArgumentParser(description='Generate point cloud for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    args = a.parse_args()
    return args

def main(args):
    print("Creating point cloud from perspective views...")
    data_root = args.data_root

    scenes = os.listdir(data_root)
    for scene in tqdm(scenes):
        try:
            print(f"Generating Point Cloud for Scene {scene} ...")
            scene_path = os.path.join(data_root, scene)
            reader = PointCloudReaderPanorama(scene_path, random_level=0, generate_color=True, generate_normal=False)
            save_path = os.path.join(data_root, scene, 'point_cloud.ply')
            reader.export_ply(save_path)
        except Exception as e:
            print(f"Error generating point cloud for Scene {scene}: {e}")
            

if __name__ == "__main__":

    main(config())