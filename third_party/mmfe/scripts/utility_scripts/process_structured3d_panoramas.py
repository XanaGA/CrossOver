#!/usr/bin/env python3
"""
Script to download and process Structured3D panoramas one by one to extract density maps.

This script:
1. Downloads one panorama zip file at a time from the Structured3D dataset
2. Extracts the panorama data
3. Processes it using the existing scripts to generate density maps
4. Moves density maps to the correct output directory
5. Removes the downloaded data to save disk space
6. Continues with the next panorama

Usage:
    python process_structured3d_panoramas.py --start_idx 0 --num_scenes 10 --temp_dir /tmp/structured3d
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import ssl
import zipfile
from pathlib import Path

# List of all panorama download URLs
PANORAMA_URLS = [
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_00.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_01.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_02.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_03.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_04.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_05.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_06.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_07.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_08.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_09.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_10.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_11.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_12.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_13.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_14.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_15.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_16.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_17.zip"
]

# Invalid scene IDs to skip (from generate_coco_stru3d.py)
# INVALID_SCENES = {76, 183, 335, 491, 663, 681, 703, 728, 865, 936, 985, 986, 1009, 1104, 1155, 1221, 1282,
#                   1365, 1378, 1635, 1745, 1772, 1774, 1816, 1866, 2037, 2076, 2274, 2334, 2357, 2580, 2665,
#                   2706, 2713, 2771, 2868, 3156, 3192, 3198, 3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496}

INVALID_SCENES = set()


def download_and_extract_zip(url, extract_path, download = False):
    """Download and extract a zip file to the specified path using wget and unzip."""

    print(f"Downloading {url}...")

    # Save the zip file in data/tmp_zip directory
    os.makedirs('data/tmp/structured3d_zips', exist_ok=True)
    tmp_zip_path = os.path.join('data/tmp/structured3d_zips', os.path.basename(url))

    try:
        # Download the zip file using wget with --no-check-certificate
        if download:
            result = subprocess.run(
                ['wget', '--no-check-certificate', '-O', tmp_zip_path, url]
            )
            if result.returncode != 0:
                print(f"Error downloading {url}: {result.stderr}")
                return False

        elif not os.path.exists(tmp_zip_path):
            print(f"Zip file {tmp_zip_path} does not exist")
            return False

        extract_path = os.path.dirname(extract_path)
        print(f"Extracting to {extract_path}...")

        # Extract the zip file using unzip
        result = subprocess.run(
            ['unzip', tmp_zip_path, '-d', extract_path]
        )

        print("Download and extraction completed.")

        return True

    except Exception as e:
        print(f"Error extracting {tmp_zip_path}: {e}")
        return False

    #finally:
        # Clean up the temporary zip file
        # if os.path.exists(tmp_zip_path):
        #     os.unlink(tmp_zip_path)

def move_density_maps(temp_output, target_base_dir):
    """Move density maps to the correct directory structure."""
    density_maps_moved = 0

    # Process all PNG files in the temp output directory
    for filename in os.listdir(temp_output):
        if filename.endswith('.png'):
            # Extract scene ID from filename (e.g., '00000.png' -> '00000')
            scene_id_str = filename.replace('.png', '')
            scene_id = int(scene_id_str)

            # Skip invalid scenes
            if scene_id in INVALID_SCENES:
                print(f"Skipping invalid scene {scene_id}")
                continue

            # Create target directory path
            target_scene_dir = os.path.join(target_base_dir, 'Structured3D', f'scene_{scene_id_str.zfill(5)}')
            os.makedirs(target_scene_dir, exist_ok=True)

            # Move the density map
            source_path = os.path.join(temp_output, filename)
            target_filename = f'density_map_{scene_id_str.zfill(5)}.png'
            target_path = os.path.join(target_scene_dir, target_filename)

            shutil.move(source_path, target_path)
            density_maps_moved += 1
            print(f"Moved density map to {target_path}")

    return density_maps_moved


def process_single_zip(url, temp_base_dir, script_dir, target_base_dir, part_idx, width, height):
    """Process a single panorama zip file."""
    print(f"\n{'='*60}")
    print(f"Processing zip {part_idx}: {url}")
    print(f"{'='*60}")

    # Create temporary directory for this panorama
    temp_dir = os.path.join(temp_base_dir, f'Structured3D')
    os.makedirs(temp_dir, exist_ok=True)


    script1 = os.path.join(script_dir, "generate_point_cloud_stru3d.py")
    script2 = os.path.join(script_dir, "generate_coco_stru3d.py")

    try:
        # Download and extract the zip file
        success = download_and_extract_zip(url, temp_dir)
        if not success:
            print(f"Failed to download and extract zip file to {temp_dir}")
            return -1

        print(f"Downloaded and extracted zip file to {temp_dir}")
        
        # Run point cloud generation
        subprocess.run([
            sys.executable, script1,
            f"--data_root={temp_dir}"
        ])

        # Create temporary output directory for density maps
        temp_output = tempfile.mkdtemp()
        print(f"Temporary output directory: {temp_output}")

        subprocess.run([
            sys.executable, script2,
            f"--data_root={temp_dir}",
            f"--output={temp_output}",
            f"--width={width}",
            f"--height={height}"
        ])


        # Move density maps to target location
        density_maps_moved = move_density_maps(temp_output, target_base_dir)

        # Clean up temporary output directory
        shutil.rmtree(temp_output)

        print(f"Successfully processed {density_maps_moved} density maps from panorama part {part_idx}")
        return density_maps_moved

    except Exception as e:
        print(f"Error processing panorama part {part_idx}: {e}")
        return -1

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)



def main():
    parser = argparse.ArgumentParser(description='Process Structured3D panoramas to extract density maps')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index in the panorama URL list (default: 0)')
    parser.add_argument('--num_zips', type=int, default=len(PANORAMA_URLS),
                       help=f'Number of panoramas to process (default: {len(PANORAMA_URLS)})')
    parser.add_argument('--temp_dir', type=str, default='/tmp/structured3d',
                       help='Temporary directory for downloads and processing (default: /tmp/structured3d)')
    parser.add_argument('--target_dir', type=str,
                       default='./data/structure3D/Structured3D_bbox',
                       help='Target directory for density maps')
    parser.add_argument('--script_dir', type=str,
                       default='./src/data_preprocess',
                       help='Directory containing the processing scripts')
    parser.add_argument('--width', type=int, default=256,
                       help='Width of density maps (default: 256)')
    parser.add_argument('--height', type=int, default=256,
                       help='Height of density maps (default: 256)')

    args = parser.parse_args()


    # Validate arguments
    if args.start_idx < 0 or args.start_idx >= len(PANORAMA_URLS):
        print(f"Error: start_idx must be between 0 and {len(PANORAMA_URLS)-1}")
        return 1

    end_idx = min(args.start_idx + args.num_zips, len(PANORAMA_URLS))

    # Create temporary directory
    os.makedirs(args.temp_dir, exist_ok=True)

    print("Structured3D Panorama Processing Script")
    print(f"Processing panoramas {args.start_idx} to {end_idx-1}")
    print(f"Density map dimensions: {args.width}x{args.height}")
    print(f"Temporary directory: {args.temp_dir}")
    print(f"Target directory: {args.target_dir}")
    print(f"Script directory: {args.script_dir}")
    print()

    total_density_maps = 0

    # Process each panorama
    for i in range(args.start_idx, end_idx):
        url = PANORAMA_URLS[i]
        density_maps = process_single_zip(url, args.temp_dir, args.script_dir, args.target_dir, i, args.width, args.height)
        if density_maps == -1:
            print(f"Failed to process zip {url}")
            continue
        total_density_maps += density_maps

    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Total density maps processed: {total_density_maps}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
