"""
This code is an adaptation that uses Zillow Indoor Dataset for the code base.

Reference: https://github.com/Zillow/Zillow-Indoor-Dataset
You are supposed to replace it in the original code base.
"""

# """CLI script to visualize & validate data for the public-facing Zillow Indoor Dataset (ZInD).
#
# Validation includes:
#  (1) required JSON fields are presented
#  (2) verify non self-intersection of room floor_plan_layouts
#  (3) verify that windows/doors/openings lie on the room layout geometry
#  (4) verify that windows/doors/openings are defined by two points (left/right boundaries)
#  (5) verify that panos_layouts are RGB images with valid FoV ratio (2:1)
#
# Visualization includes:
#  (1) render the top-down floor map projection: merged room floor_plan_layouts,WDO and camera centers
#  (2) render the room floor_plan_layouts and windows/doors/openings on the pano
#
# Example usage (1): Render all layouts on primary and secondary panos.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-layout --visualize-floor-plan \
#  --raw --complete --visible --primary --secondary
#
# Example usage (2): Render all vector layouts using merger (based on raw or complete) and the final redraw layouts.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-floor-plan --redraw --complete --raw
#
# Example usage (3): Render the raster to vector alignments using merger (based on raw or complete) and final redraw.
#  python visualize_zind_cli.py -i <input_folder> -o <output_folder> --visualize-raster --redraw --complete --raw
#

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from floor_plan import FloorPlan
from render import (
    draw_transformed_geometry,
    load_floor_plan_image,
    render_jpg_image,
    render_raster_to_vector_alignment,
    render_room_vertices_on_panos,
    save_rgb_image,
)
from tqdm import tqdm
from utils import Polygon

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)

RENDER_FOLDER = "render_data"


def _apply_transformation_to_polygons(
    polygon_list: List[Polygon], transformation
) -> List[Polygon]:
    """Apply the given transformation to a list of polygons and return the transformed polygons."""
    transformed_polygons: List[Polygon] = []

    for polygon in polygon_list:
        if polygon.num_points == 0:
            continue

        polygon_coords = np.array(
            [(point.x, point.y) for point in polygon.points], dtype=np.float32
        )
        transformed_coords = transformation.apply_inverse(polygon_coords)
        transformed_points = Polygon.list_to_points(transformed_coords.tolist())
        transformed_polygons.append(
            Polygon(
                type=polygon.type,
                points=transformed_points,
                name=polygon.name,
            )
        )

    return transformed_polygons


def validate_and_render(
    zillow_floor_plan: "FloorPlan",
    *,
    input_folder: str,
    output_folder: str,
    args: Dict[str, Any]
):
    """Validate and render various ZInD elements, e.g.
    1. Primary/secondary layout and WDO
    2. Raw/complete/visible layouts
    3. Top-down merger results (draft floor-plan)
    4. Top-down redraw results (final floor-plan)
    5. Raster to vector alignment results.

    :param zillow_floor_plan: ZInD floor plan object.
    :param input_folder: Input folder of the current tour.
    :param output_folder: Folder where the renderings will be saved.
    :param args: Input arguments to the script.

    :return: None
    """
    # Get the types of floor_plan_layouts that we should render.
    geometry_to_visualize = []
    if args.raw:
        geometry_to_visualize.append("raw")
    if args.complete:
        geometry_to_visualize.append("complete")
    if args.visible:
        geometry_to_visualize.append("visible")
    if args.redraw:
        geometry_to_visualize.append("redraw")

    # Get the types of panos_layouts that we should render.
    panos_to_visualize = []
    if args.primary:
        panos_to_visualize.append("primary")
    if args.secondary:
        panos_to_visualize.append("secondary")

    # Render the room shape layouts + WDO on top of the pano textures.
    if args.visualize_layout:
        for geometry_type in geometry_to_visualize:
            if geometry_type == "redraw":
                continue
            for pano_type in panos_to_visualize:
                output_folder_layout = os.path.join(
                    output_folder, "layout", geometry_type, pano_type
                )
                os.makedirs(output_folder_layout, exist_ok=True)
                panos_list = zillow_floor_plan.panos_layouts[geometry_type][pano_type]
                render_room_vertices_on_panos(
                    input_folder=zillow_floor_plan.input_folder,
                    panos_list=panos_list,
                    output_folder=output_folder_layout,
                )

    # Render the top-down draft floor plan, result of the merger stage.
    if args.visualize_floor_plan:
        output_folder_floor_plan = os.path.join(output_folder, "floor_plan")
        os.makedirs(output_folder_floor_plan, exist_ok=True)

        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible":
                continue

            zind_dict = zillow_floor_plan.floor_plan_layouts[geometry_type]

            for floor_id, zind_poly_list in zind_dict.items():
                output_file_name = os.path.join(
                    output_folder_floor_plan,
                    "vector_{}_layout_{}.jpg".format(geometry_type, floor_id),
                )

                transformation = zillow_floor_plan.floor_plan_to_redraw_transformation.get(
                    floor_id
                )

                if transformation is not None:
                    polygons_to_render = _apply_transformation_to_polygons(
                        zind_poly_list, transformation
                    )
                    if not polygons_to_render:
                        polygons_to_render = list(zind_poly_list)
                else:
                    polygons_to_render = list(zind_poly_list)

                render_jpg_image(
                    polygon_list=polygons_to_render, jpg_file_name=output_file_name
                )

    # Render vector geometry on top of the raster floor plan image.
    if args.visualize_raster:
        output_folder_floor_plan_alignment = os.path.join(
            output_folder, "floor_plan_raster_to_vector_alignment"
        )
        os.makedirs(output_folder_floor_plan_alignment, exist_ok=True)

        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible":
                continue

            for (
                floor_id,
                raster_to_vector_transformation,
            ) in zillow_floor_plan.floor_plan_to_redraw_transformation.items():
                floor_plan_image_path = os.path.join(
                    input_folder, zillow_floor_plan.floor_plan_image_path[floor_id]
                )

                zind_poly_list = zillow_floor_plan.floor_plan_layouts[geometry_type][
                    floor_id
                ]

                output_file_name = os.path.join(
                    output_folder_floor_plan_alignment,
                    "raster_to_vector_{}_layout_{}.jpg".format(geometry_type, floor_id),
                )

                render_raster_to_vector_alignment(
                    zind_poly_list,
                    raster_to_vector_transformation,
                    floor_plan_image_path,
                    output_file_name,
                )

    if args.render_mmfe:
        print("Rendering MMFE...")
        output_folder_mmfe = os.path.join(output_folder, "mmfe")
        overlay_folder = os.path.join(output_folder_mmfe, "overlay")
        vector_folder = os.path.join(output_folder_mmfe, "vector")
        os.makedirs(overlay_folder, exist_ok=True)
        os.makedirs(vector_folder, exist_ok=True)

        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible":
                print("Skipping visible geometry")
                continue

            geometry_layouts = zillow_floor_plan.floor_plan_layouts.get(geometry_type)
            if geometry_layouts is None:
                print("Skipping geometry layouts")
                continue

            for (
                floor_id,
                raster_to_vector_transformation,
            ) in zillow_floor_plan.floor_plan_to_redraw_transformation.items():
                if floor_id not in geometry_layouts:
                    print("Skipping floor id not in geometry layouts")
                    continue

                floor_plan_image_path = os.path.join(
                    input_folder, zillow_floor_plan.floor_plan_image_path[floor_id]
                )

                zind_poly_list = geometry_layouts[floor_id]
                try:
                    base_image_rgb = load_floor_plan_image(floor_plan_image_path)
                except FileNotFoundError as ex:
                    LOG.warning(str(ex))
                    continue
                
                print("Drawing overlay image")
                overlay_image = draw_transformed_geometry(
                    base_image_rgb,
                    zind_poly_list,
                    raster_to_vector_transformation,
                )
                overlay_output = os.path.join(
                    overlay_folder,
                    "raster_to_vector_{}_layout_{}.jpg".format(
                        geometry_type, floor_id
                    ),
                )
                print("Saving overlay image")
                save_rgb_image(overlay_image, overlay_output)

                white_canvas = np.full_like(base_image_rgb, 255, dtype=np.uint8)
                vector_only_image = draw_transformed_geometry(
                    white_canvas,
                    zind_poly_list,
                    raster_to_vector_transformation,
                )
                vector_output = os.path.join(
                    vector_folder,
                    "vector_only_{}_layout_{}.jpg".format(geometry_type, floor_id),
                )
                print("Saving vector only image")
                save_rgb_image(vector_only_image, vector_output)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize & validate Zillow Indoor Dataset (ZInD)"
    )

    parser.add_argument(
        "--input",
        "-i",
        help="Input JSON file (or folder with ZInD data)",
        required=True,
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output folder where rendered data will be saved to",
        required=True,
    )

    parser.add_argument(
        "--visualize-layout",
        action="store_true",
        help="Render room vertices and WDO on panoramas.",
    )
    parser.add_argument(
        "--visualize-floor-plan",
        action="store_true",
        help="Render the floor plans as top-down projections with floor plan layouts and WDO elements.",
    )

    parser.add_argument(
        "--visualize-raster",
        action="store_true",
        help="Render the vector floor plan (draft or final) on the raster floor plan image.",
    )
    parser.add_argument(
        "--render-mmfe",
        action="store_true",
        help=(
            "Render transformed floor plan geometry both over the raster image and on a white background "
            "for MMFE evaluation."
        ),
    )

    parser.add_argument(
        "--max-tours", default=float("inf"), help="Max tours to process."
    )

    parser.add_argument(
        "--primary", action="store_true", help="Visualize primary panoramas."
    )
    parser.add_argument(
        "--secondary", action="store_true", help="Visualize secondary panoramas."
    )

    parser.add_argument("--raw", action="store_true", help="Visualize raw layout.")
    parser.add_argument(
        "--complete", action="store_true", help="Visualize complete layout."
    )
    parser.add_argument(
        "--visible", action="store_true", help="Visualize visible layout."
    )

    parser.add_argument(
        "--redraw", action="store_true", help="Visualize 2D redraw geometry."
    )

    parser.add_argument(
        "--debug", "-d", action="store_true", help="Set log level to DEBUG"
    )

    args = parser.parse_args()

    if args.debug:
        LOG.setLevel(logging.DEBUG)

    input = args.input

    # Useful to debug, by restricting the number of tours to process.
    max_tours_to_process = int(args.max_tours) if args.max_tours != float("inf") else float("inf")

    # Collect all the feasible input JSON files.
    input_files_list = [input]
    if Path(input).is_dir():
        input_files_list = sorted(Path(input).glob("**/zind_data.json"))

    num_failed = 0
    num_success = 0
    failed_tours = []
    for input_file in tqdm(input_files_list, desc="Validating ZInD data"):
        # Try loading and validating the file.
        try:
            zillow_floor_plan = FloorPlan(input_file)

            current_input_folder = os.path.join(str(Path(input_file).parent))
            current_output_folder = os.path.join(
                args.output, RENDER_FOLDER, str(Path(input_file).parent.stem)
            )
            os.makedirs(current_output_folder, exist_ok=True)

            validate_and_render(
                zillow_floor_plan,
                input_folder=current_input_folder,
                output_folder=current_output_folder,
                args=args,
            )
            num_success += 1

            if num_success >= max_tours_to_process:
                LOG.info("Max tours to process reached {}".format(num_success))
                break
        except Exception as ex:
            failed_tours.append(str(Path(input_file).parent.stem))
            num_failed += 1
            track = traceback.format_exc()
            LOG.warning("Error validating {}: {}".format(input_file, str(ex)))
            LOG.debug(track)
            continue

    if num_failed > 0:
        LOG.warning("Failed to validate: {}".format(num_failed))

        LOG.debug("Failed_tours: {}".format(failed_tours))
    else:
        LOG.info("All ZInD validated successfully")


if __name__ == "__main__":
    main()
