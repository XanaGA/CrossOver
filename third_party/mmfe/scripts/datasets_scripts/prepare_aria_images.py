#!/usr/bin/env python3
"""Prepare Aria Synthetic Environment RGB images.

This script walks an `original_folder` containing ASE RGB images
(e.g. `data/aria/SyntheticEnv/original_data/<scene_id>/images/train/rgb/vignette0000000.jpg`)
and writes processed images to `dest_folder` preserving the same
relative folder structure (e.g. `data/aria/SyntheticEnv/rendered_data/<scene_id>/images/train/rgb/vignette0000000.jpg`).

Processing consists of:
  1. Devignetting using the Aria vignette mask.
  2. Undistorting fisheye images into a pinhole view (using ASE calibration
     via `undistort_image_fisheye`).

Examples of the processing pipeline can be found in:
  - `tests/test_aria_3D.py`
  - `tests/test_aria_images.py`

Usage
-----

    python scripts/datasets_scripts/prepare_aria_images.py \
        --original-folder data/aria/SyntheticEnv/original_data \
        --dest-folder     data/aria/SyntheticEnv/rendered_data

"""

import argparse
import os
import sys
from typing import Iterable, List

import cv2
from tqdm import tqdm

# Make sure we can import the project modules when running this script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from aria_mmfe.aria_images.aria_cv_tools import (  # type: ignore  # noqa: E402
    devignette_image_numpy,
    load_aria_vignette,
    undistort_image_fisheye,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIGNETTE_PREFIX = "vignette"
PROCESSED_PREFIX = "processed"


def iter_image_files(
    root: str,
    prefix: str | None = VIGNETTE_PREFIX,
    allowed_scenes: set[str] | None = None,
) -> Iterable[str]:
    """Yield full paths to candidate image files under ``root``.

    Parameters
    ----------
    root:
        Root directory containing scene subfolders.
    prefix:
        Optional filename prefix filter (e.g. ``"vignette"``). If ``None``,
        all supported image files are returned.
    allowed_scenes:
        Optional set of scene folder names (first-level subfolders under ``root``)
        to restrict processing. If ``None``, all scenes are included.
    """

    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            scene_id = None
        else:
            scene_id = rel_dir.split(os.sep)[0]

        if allowed_scenes is not None and (scene_id is None or scene_id not in allowed_scenes):
            continue

        for fname in filenames:
            f_lower = fname.lower()
            if not f_lower.endswith(IMAGE_EXTENSIONS):
                continue
            if prefix is not None and not fname.startswith(prefix):
                continue
            yield os.path.join(dirpath, fname)


def make_processed_filename(src_name: str) -> str:
    """
    Create an output filename of the form ``processed0000000.jpg`` from an input
    filename such as ``vignette0000000.jpg``.
    """
    stem, _ext = os.path.splitext(src_name)
    if stem.startswith(VIGNETTE_PREFIX):
        suffix = stem[len(VIGNETTE_PREFIX) :]
    else:
        suffix = stem
    return f"{PROCESSED_PREFIX}{suffix}.jpg"


def process_single_image(
    src_path: str,
    dst_path: str,
    vignette_img,
    do_undistort: bool = True,
) -> bool:
    """Process a single RGB image and save it.

    Returns ``True`` if the image was processed and saved, ``False`` otherwise.
    """

    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Failed to read image: {src_path}")
        return False

    # Devignette: see tests/test_aria_images.py for reference
    try:
        img = devignette_image_numpy(img, vignette_img)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Devignetting failed for {src_path}: {e}")
        return False

    # Undistort fisheye into pinhole image
    if do_undistort:
        try:
            img, _ = undistort_image_fisheye(img)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[WARN] Undistortion failed for {src_path}: {e}. Saving devignetted image only.")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not cv2.imwrite(dst_path, img):  # pragma: no cover - cv2 returns bool
        print(f"[WARN] Failed to write image: {dst_path}")
        return False

    return True


def prepare_aria_images(
    original_folder: str,
    dest_folder: str,
    prefix: str | None = "vignette",
    overwrite: bool = False,
    undistort: bool = True,
    max_scenes: int | None = None,
) -> None:
    """Process all RGB images from ``original_folder`` into ``dest_folder``.

    The directory tree under ``original_folder`` is mirrored under ``dest_folder``.
    """

    original_folder = os.path.abspath(original_folder)
    dest_folder = os.path.abspath(dest_folder)

    if not os.path.isdir(original_folder):
        raise FileNotFoundError(f"original_folder does not exist or is not a directory: {original_folder}")

    os.makedirs(dest_folder, exist_ok=True)

    # Determine which scene folders (top-level subdirectories) to process
    allowed_scenes: set[str] | None = None
    if max_scenes is not None and max_scenes > 0:
        scene_dirs = [
            d
            for d in os.listdir(original_folder)
            if os.path.isdir(os.path.join(original_folder, d)) and d.isdigit()
        ]
        scene_dirs.sort(key=lambda x: int(x))
        scene_subset = scene_dirs[:max_scenes]
        allowed_scenes = set(scene_subset)
        print(f"[INFO] Limiting to {len(scene_subset)} scenes: {', '.join(scene_subset)}")

    # Load vignette image once and reuse
    vignette_img = load_aria_vignette()

    image_paths: List[str] = list(
        iter_image_files(original_folder, prefix=prefix, allowed_scenes=allowed_scenes)
    )
    if not image_paths:
        print(f"[INFO] No images found under {original_folder} with prefix={prefix!r}.")
        return

    print(f"[INFO] Found {len(image_paths)} images to process.")
    print(f"[INFO] Original folder: {original_folder}")
    print(f"[INFO] Destination folder: {dest_folder}")
    print(f"[INFO] Undistort: {undistort}")

    processed = 0
    skipped = 0

    for src_path in tqdm(image_paths, desc="Processing images"):
        rel_path = os.path.relpath(src_path, original_folder)
        rel_dir, src_name = os.path.split(rel_path)
        dst_name = make_processed_filename(src_name)
        dst_rel_path = os.path.join(rel_dir, dst_name) if rel_dir != "." else dst_name
        dst_path = os.path.join(dest_folder, dst_rel_path)

        if not overwrite and os.path.exists(dst_path):
            skipped += 1
            continue

        if process_single_image(src_path, dst_path, vignette_img, do_undistort=undistort):
            processed += 1

    print(f"[INFO] Done. Processed={processed}, Skipped (existing)={skipped}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Devignette and undistort Aria SyntheticEnv RGB images.")
    parser.add_argument(
        "--original-folder",
        type=str,
        required=True,
        help="Root folder containing original ASE RGB images (e.g. data/aria/SyntheticEnv/original_data)",
    )
    parser.add_argument(
        "--dest-folder",
        type=str,
        required=True,
        help="Destination root folder where processed images will be saved (e.g. data/aria/SyntheticEnv/rendered_data)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="vignette",
        help="Only process files whose names start with this prefix (default: 'vignette'). Use '' or '--prefix None' to disable.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="If set, only devignetting is applied (no undistortion).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in dest_folder (default: skip existing).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scene folders (top-level subdirectories) to process. "
        "If omitted, all scenes are processed.",
    )

    args = parser.parse_args()

    # Allow explicit "None" to disable prefix filtering
    if args.prefix is not None and str(args.prefix).lower() == "none":
        args.prefix = None

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    prepare_aria_images(
        original_folder=cli_args.original_folder,
        dest_folder=cli_args.dest_folder,
        prefix=cli_args.prefix,
        overwrite=cli_args.overwrite,
        undistort=not cli_args.no_undistort,
        max_scenes=cli_args.max_scenes,
    )
