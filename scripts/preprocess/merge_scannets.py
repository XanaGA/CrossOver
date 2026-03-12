from pathlib import Path
import shutil

src_root = Path("scannet_data/scannet/rendered")
dst_root = Path("tmp/preprocess_feats/Scannet/scans")

# filename mapping
name_map = {
    "density_large": "density_large",
    "density": "density",
    "density_mesh_noisy": "density_mesh_noisy",
    "density_mesh": "density_mesh",
    "floorplan": "mmfe_floorplan",
}

for scene_dir in src_root.iterdir():
    if not scene_dir.is_dir():
        continue

    scene = scene_dir.name
    dst_scene = dst_root / scene

    if not dst_scene.exists():
        print(f"Skipping {scene} (destination missing)")
        continue

    for img in scene_dir.glob("*.png"):
        name = img.stem  # e.g. scene_scene0000_00_density_large
        parts = name.split("_")

        suffix = "_".join(parts[3:])  # density_large, floorplan, etc.

        if suffix not in name_map:
            print(f"Skipping unknown file: {img}")
            continue

        new_name = name_map[suffix] + ".png"
        dst_path = dst_scene / new_name

        shutil.move(str(img), dst_path)
        print(f"{img} -> {dst_path}")