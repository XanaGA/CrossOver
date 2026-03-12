import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

from third_parties.structured3D.struct3D_utils import scene_to_floorplan_image, scene_to_lidar_image


def visualize_floorplan(args):
    """visualize floorplan
    """
    # Get the image as numpy array
    # img_array = scene_to_floorplan_image(args.path, args.scene, args.no_color, bbox_percentage=args.bbox_pct)
    img_array = scene_to_lidar_image(args.path, args.scene, args.no_color)
    
    # Display the image
    plt.figure()
    plt.imshow(img_array)
    plt.axis('off')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured3D Floorplan Visualization")
    parser.add_argument("--path", required=True,
                        help="dataset path", metavar="DIR")
    parser.add_argument("--scene", required=True,
                        help="scene id", type=int)
    parser.add_argument("--no_color", action="store_true",
                        help="no color")
    parser.add_argument("--bbox_pct", type=float, default=1.0,
                        help="percentage of bounding boxes to display (0.0 to 1.0)", metavar="FLOAT")
    return parser.parse_args()


def main():
    args = parse_args()

    visualize_floorplan(args)


if __name__ == "__main__":
    main()
