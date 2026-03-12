import shutil
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Reorganize ZIND dataset images and generate partition files")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="outputs/render_data",
        help="Source directory containing scene folders (default: outputs/render_data)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="zind",
        help="Main target directory for output (default: zind)"
    )
    parser.add_argument(
        "--partition-file",
        type=str,
        default="zind_partition.json",
        help="Path to partition JSON file (default: zind_partition.json)"
    )
    return parser.parse_args()

args = parse_args()
SOURCE_BASE_DIR = Path(args.source_dir)
MAIN_TARGET_DIR = Path(args.target_dir)
PARTITION_JSON_FILE = Path(args.partition_file)

# All images will be copied into this subfolder
IMAGE_TARGET_BASE_DIR = MAIN_TARGET_DIR / "rendered_data"
# ---------------------

# Regex to find the floor number, e.g., "floor_01.jpg" -> "01"
floor_pattern = re.compile(r"floor_(\d+)\.jpg$")

# This dict will store the mapping, e.g.:
# { "0001": {"0001_f1", "0001_f2"}, "0000": {"0000_f1"} }
discovered_scene_floors = defaultdict(set)

# Ensure the main target directories exist
IMAGE_TARGET_BASE_DIR.mkdir(parents=True, exist_ok=True)

print(f"🚀 Starting image reorganization and partition generation...")
print(f"Image Source:      {SOURCE_BASE_DIR.resolve()}")
print(f"Image Destination: {IMAGE_TARGET_BASE_DIR.resolve()}")
print(f"Partition File:    {PARTITION_JSON_FILE.resolve()}")
print(f"Output Splits:     {MAIN_TARGET_DIR.resolve()}")

if not SOURCE_BASE_DIR.exists():
    print(f"❌ ERROR: Source directory not found: {SOURCE_BASE_DIR}")
    exit()

if not PARTITION_JSON_FILE.exists():
    print(f"❌ ERROR: Partition file not found: {PARTITION_JSON_FILE}")
    exit()

# --- 1. Copy Images and Discover Scene/Floor Combinations ---

print("\nProcessing and copying images...")
copied_count = 0
scene_count = 0

# Iterate over scene directories (e.g., '0000', '0001')
for scene_path in SOURCE_BASE_DIR.iterdir():
    if not scene_path.is_dir():
        continue

    scene_id = scene_path.name  # e.g., '0000'
    scene_count += 1

    search_subdirs = ["overlay", "vector"]
    
    for subdir_name in search_subdirs:
        current_search_path = scene_path / subdir_name
        
        if not current_search_path.exists():
            continue
            
        for image_path in current_search_path.glob("*.jpg"):
            filename = image_path.name
            match = floor_pattern.search(filename)
            
            if match:
                floor_num_str = match.group(1) # '01'
                floor_id = str(int(floor_num_str)) # '1'
                
                # Create the new target directory name, e.g., '0000_f1'
                target_scene_name = f"{scene_id}_f{floor_id}"
                
                # --- This is the key step ---
                # Add the discovered name to our tracking dictionary
                discovered_scene_floors[scene_id].add(target_scene_name)
                # -----------------------------
                
                # Create the full path for the new scene directory
                target_scene_dir = IMAGE_TARGET_BASE_DIR / target_scene_name
                target_scene_dir.mkdir(parents=True, exist_ok=True)
                
                # Define and copy to the final destination file path
                dest_file_path = target_scene_dir / filename
                shutil.copy(image_path, dest_file_path)
                copied_count += 1

print(f"✅ Image copy complete. Processed {scene_count} scenes and copied {copied_count} files.")

# --- 2. Generate Partition Files (train.txt, val.txt, test.txt) ---

print("\nGenerating partition files...")

# Load the partition data
try:
    with open(PARTITION_JSON_FILE, 'r') as f:
        partitions = json.load(f)
except json.JSONDecodeError:
    print(f"❌ ERROR: Could not parse {PARTITION_JSON_FILE}. Is it valid JSON?")
    exit()

missing_scenes = set()

# Iterate over 'train', 'val', and 'test'
for split_name, scene_id_list in partitions.items():
    output_filename = MAIN_TARGET_DIR / f"{split_name}.txt"
    final_folders_for_split = []
    
    # For each scene ID in the list (e.g., "0001")
    for scene_id in scene_id_list:
        # Check if we found this scene during our file copy
        if scene_id in discovered_scene_floors:
            # Add all its floors (e.g., "0001_f1", "0001_f2") to the list
            final_folders_for_split.extend(list(discovered_scene_floors[scene_id]))
        else:
            # Keep track of scenes in the JSON that we couldn't find
            missing_scenes.add(scene_id)
            
    # Sort the list alphabetically for consistent output
    final_folders_for_split.sort()
    
    # Write the list to the file (e.g., zind/train.txt)
    with open(output_filename, 'w') as f:
        for folder_name in final_folders_for_split:
            f.write(f"{folder_name}\n")
            
    print(f"Wrote {len(final_folders_for_split)} entries to {output_filename}")

if missing_scenes:
    print(f"\n⚠️ Warning: The following {len(missing_scenes)} scene IDs from "
          f"{PARTITION_JSON_FILE} were not found in the {SOURCE_BASE_DIR} directory:")
    print(f"{', '.join(sorted(list(missing_scenes)))}")

print("\n✅ All tasks complete.")