import argparse
import json
import os
from tqdm import tqdm
from data_preprocess.stru3d_utils import (generate_density, normalize_annotations, parse_floor_plan_polys, 
                    generate_coco_dict, read_scene_pc, export_density)


### Note: Some scenes have missing/wrong annotations. These are the indices that you should additionally exclude 
### to be consistent with MonteFloor and HEAT:
# invalid_scenes_ids = [76, 183, 335, 491, 663, 681, 703, 728, 865, 936, 985, 986, 1009, 1104, 1155, 1221, 1282, 
#                      1365, 1378, 1635, 1745, 1772, 1774, 1816, 1866, 2037, 2076, 2274, 2334, 2357, 2580, 2665, 
#                      2706, 2713, 2771, 2868, 3156, 3192, 3198, 3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]
invalid_scenes_ids = []

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

def config():
    a = argparse.ArgumentParser(description='Generate coco format data for Structured3D')
    a.add_argument('--data_root', default='Structured3D_panorama', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='coco_stru3d', type=str, help='path to output folder')
    a.add_argument('--width', default=256, type=int, help='width of density map')
    a.add_argument('--height', default=256, type=int, help='height of density map')
    
    args = a.parse_args()
    return args

def main(args):
    data_root = args.data_root

    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    ### begin processing
    scenes = os.listdir(data_root)
    for scene in tqdm(scenes):
        try:
            print(f"Generating Density Map for Scene {scene} ...")
            scene_path = os.path.join(data_root, scene)
            scene_id = scene.split('_')[-1]

            if int(scene_id) in invalid_scenes_ids:
                print('skip {}'.format(scene))
                continue
            
            # load pre-generated point cloud 
            ply_path = os.path.join(scene_path, 'point_cloud.ply')
            points = read_scene_pc(ply_path)
            xyz = points[:, :3]

            ### project point cloud to density map
            density, normalization_dict = generate_density(xyz, width=args.width, height=args.height)
            export_density(density, outFolder, scene_id)
        except Exception as e:
            print(f"Error generating density map for Scene {scene}: {e}")
            # If the point cloud is not generated, remove the density map
            density_path = os.path.join(outFolder, f'density_map_{scene_id}.png')
            if os.path.exists(density_path):
                os.remove(density_path)

if __name__ == "__main__":

    main(config())