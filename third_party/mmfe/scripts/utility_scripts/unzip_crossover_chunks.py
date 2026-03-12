import os
import zipfile
import shutil
import stat

# Paths (edit as needed)
CHUNKS_DIR = "data/scannet_zip"
OUTPUT_DIR = "data/scannet"
TEMP_EXTRACT_DIR = "data/tmp_extract"

FILES_OF_INTEREST = ["data3D.npz", "floor+obj.ply"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

chunk_files = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".zip")])

def make_writable(path):
    """Ensure a file is writable (u+rw)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except Exception as e:
        print(f"  WARNING: could not chmod {path}: {e}")

for chunk in chunk_files:
    zip_path = os.path.join(CHUNKS_DIR, chunk)
    print(f"\n=== Processing {zip_path} ===")

    # Reset temp folder
    if os.path.exists(TEMP_EXTRACT_DIR):
        shutil.rmtree(TEMP_EXTRACT_DIR)
    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

    # Extract chunk
    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(TEMP_EXTRACT_DIR)

    # Process each scene
    for scene_name in os.listdir(TEMP_EXTRACT_DIR):
        scene_path = os.path.join(TEMP_EXTRACT_DIR, scene_name)
        if not os.path.isdir(scene_path):
            continue

        out_scene_dir = os.path.join(OUTPUT_DIR, scene_name)
        os.makedirs(out_scene_dir, exist_ok=True)

        # Copy files of interest
        for filename in FILES_OF_INTEREST:
            src_file = os.path.join(scene_path, filename)
            dst_file = os.path.join(out_scene_dir, filename)

            if os.path.exists(src_file):
                # Remove destination file if exists (and fix permissions)
                if os.path.exists(dst_file):
                    try:
                        make_writable(dst_file)
                        os.remove(dst_file)
                    except Exception as e:
                        print(f"  ERROR: can't remove {dst_file}: {e}")
                        continue

                shutil.copy(src_file, dst_file)
                make_writable(dst_file)

                print(f"  Copied {filename} for {scene_name}")
            else:
                print(f"  WARNING: missing {filename} in {scene_name}")

    print("Cleaning extracted chunk...\n")
    shutil.rmtree(TEMP_EXTRACT_DIR)

print("\nAll chunks processed successfully.")
