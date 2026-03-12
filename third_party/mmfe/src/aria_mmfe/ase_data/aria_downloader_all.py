#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import json
import os
import ssl
import urllib.request
import shutil
from zipfile import ZipFile

from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

SCENES_PER_CHUNK = 10


def urllib_tqdm_hook(t):
    """Progress bar hook for urllib downloads."""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def ASEIdsParser(string):
    """Handler for list of scene ids. Handles lists of integers and ranges."""
    try:
        ids = string.split(",")
        ids = [
            (
                list(range(int(x.split("-")[0]), int(x.split("-")[1]) + 1))
                if "-" in x
                else [int(x)]
            )
            for x in ids
        ]
        ids = [item for sublist in ids for item in sublist]
        return sorted(list(set(ids)))
    except Exception as e:
        print("Error: ", e)
        raise argparse.ArgumentTypeError(
            "Scene ids must be comma separated integers or ranges. For example: 1,2,3-5,6"
        )


def load_metadata(path):
    """Load the metadata file downloaded from the ASE website."""
    with open(path) as f:
        return json.load(f)


def safe_mkdir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def extract_scene_data(scene_folder, scene_id, set_type, output_root, separation):
    """Move rgb/, depth/, trajectory.csv, semidense_points.csv.gz and ase_scene_language.txt."""

    # 1. Images + trajectory destination
    dst_images_base = os.path.join(output_root, str(scene_id), "images", set_type)
    safe_mkdir(dst_images_base)

    # Move rgb directory (with separation)
    rgb_src = os.path.join(scene_folder, "rgb")
    if os.path.isdir(rgb_src):
        rgb_dst = os.path.join(dst_images_base, "rgb")
        safe_mkdir(rgb_dst)

        files = sorted(os.listdir(rgb_src))

        for i, f in enumerate(files):
            if i % separation != 0:
                continue
            src_file = os.path.join(rgb_src, f)
            dst_file = os.path.join(rgb_dst, f)
            shutil.move(src_file, dst_file)

    # Move depth directory (with separation)
    depth_src = os.path.join(scene_folder, "depth")
    if os.path.isdir(depth_src):
        depth_dst = os.path.join(dst_images_base, "depth")
        safe_mkdir(depth_dst)
        print(f"Moving depth directory for scene {scene_id}")
        print(f"Depth folder: {depth_src}")

        files = sorted(os.listdir(depth_src))

        for i, f in enumerate(files):
            if i % separation != 0:
                continue
            src_file = os.path.join(depth_src, f)
            dst_file = os.path.join(depth_dst, f)
            shutil.move(src_file, dst_file)
    else:
        print(f"No depth directory found for scene {scene_id}")
        print(f"Depth folder: {depth_src}")

    # Move trajectory.csv
    traj_src = os.path.join(scene_folder, "trajectory.csv")
    if os.path.exists(traj_src):
        shutil.move(traj_src, os.path.join(dst_images_base, "trajectory.csv"))

    # --------------------------------------------------------
    # 2. semidense_points.csv.gz + ase_scene_language.txt
    #    → go to ../original_data/<scene_id>/
    # --------------------------------------------------------
    original_root = os.path.join(
        os.path.dirname(output_root), str(scene_id)
    )
    safe_mkdir(original_root)

    semidense_src = os.path.join(scene_folder, "semidense_points.csv.gz")
    if os.path.exists(semidense_src):
        shutil.move(
            semidense_src, os.path.join(original_root, "semidense_points.csv.gz")
        )

    language_src = os.path.join(scene_folder, "ase_scene_language.txt")
    if os.path.exists(language_src):
        shutil.move(
            language_src, os.path.join(original_root, "ase_scene_language.txt")
        )


def cleanup_folder(path):
    """Remove folder if it exists."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def process_chunk(set_type, chunk_id, metadata, output_dir, target_scene_ids, separation):
    """Download → unzip → extract needed data → delete chunk."""
    chunk_filename = f"{set_type}_chunk_{chunk_id:07}.zip"
    print(f"\nProcessing chunk {chunk_filename} ...")

    # Find metadata entry
    entry = next((m for m in metadata if m["filename"] == chunk_filename), None)
    if entry is None:
        print(f"  Warning: No metadata entry for {chunk_filename}")
        return

    set_dir = os.path.join(output_dir, set_type)
    safe_mkdir(set_dir)

    zip_path = os.path.join(set_dir, chunk_filename)

    # -----------------------
    # 1. DOWNLOAD THE ZIP
    # -----------------------
    print("  Downloading...")
    with tqdm(unit="B", unit_scale=True, desc="  Progress") as t:
        urllib.request.urlretrieve(
            entry["cdn"],
            zip_path,
            reporthook=urllib_tqdm_hook(t)
        )

    # SHA1 check
    with open(zip_path, "rb") as f:
        sha_local = hashlib.sha1(f.read()).hexdigest()
    if sha_local != entry["sha"]:
        print(f"  ❌ SHA1 mismatch! Expected: {entry['sha']}, got: {sha_local}")
        print("  Deleting corrupt file.")
        os.remove(zip_path)
        return
    print("  ✔ SHA1 correct")

    # -----------------------
    # 2. UNZIP
    # -----------------------
    print("  Extracting...")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(set_dir)

    os.remove(zip_path)   # delete zip immediately

    # -----------------------
    # 3. PROCESS SCENES
    # -----------------------
    print("  Processing extracted scenes...")
    for name in os.listdir(set_dir):
        scene_path = os.path.join(set_dir, name)
        if not os.path.isdir(scene_path):
            continue

        try:
            scene_id = int(name)
        except ValueError:
            continue

        if scene_id in target_scene_ids:
            print(f"    ✔ Keeping scene {scene_id}")
            extract_scene_data(scene_path, scene_id, set_type, output_dir, separation)

        # Remove the extracted scene folder regardless
        cleanup_folder(scene_path)

    print("  ✔ Finished chunk")


def main(cdn_file, output_dir, scene_ids, sets, separation):
    """Main function to download and organize all Aria Synthetic Environment data."""
    metadata = load_metadata(cdn_file)

    for set_type in sets:
        print(f"\n===== Processing split: {set_type.upper()} =====")
        safe_mkdir(os.path.join(output_dir, set_type))

        # Determine which chunks contain our scenes
        chunks = sorted({sid // SCENES_PER_CHUNK for sid in scene_ids})
        print(f"Chunks needed: {chunks}")

        for chunk_id in chunks:
            process_chunk(
                set_type=set_type,
                chunk_id=chunk_id,
                metadata=metadata,
                output_dir=output_dir,
                target_scene_ids=set(scene_ids),
                separation=separation
            )

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Aria Synthetic Environments Downloader (All)",
        description="Downloads and organizes Aria Synthetic Environments as defined in CDN json file.",
    )
    parser.add_argument(
        "--cdn-file",
        help="Input file listing the CDN urls, downloaded from ASE website.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory",
        required=True,
    )
    parser.add_argument(
        "--scene-ids",
        help="Scene ids to download.",
        required=True,
        type=ASEIdsParser,
    )
    parser.add_argument(
        "--set",
        help="The type of scenes to download. If not specified, downloads both train and test.",
        choices=["train", "test"],
        required=False,
    )
    parser.add_argument(
        "--separation",
        type=int,
        default=1,
        help="Move only 1 out of every N rgb/depth images (default = 1 = keep all)",
    )

    args = parser.parse_args()

    sets = ["train", "test"] if args.set is None else [args.set]

    main(
        cdn_file=args.cdn_file,
        output_dir=args.output_dir,
        scene_ids=args.scene_ids,
        sets=sets,
        separation=args.separation,
    )
