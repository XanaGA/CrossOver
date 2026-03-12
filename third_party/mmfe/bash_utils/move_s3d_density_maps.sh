#!/bin/bash

# Paths
SCENES_DIR="data/structure3D/Structured3D_bbox/Structured3D"
DENSITY_DIR="/local/home/xanadon/Downloads/stru3d/all_density_maps"

# Loop over all density map PNG files
for density_file in "$DENSITY_DIR"/*.png; do
    # Extract just the numeric ID from the filename (e.g., 00000 from 00000.png)
    base_name=$(basename "$density_file" .png)
    
    # Create the corresponding scene folder name (e.g., scene_00000)
    scene_folder="$SCENES_DIR/scene_${base_name}"
    
    # Check if the scene folder exists
    if [ -d "$scene_folder" ]; then
        # Copy and rename the density map into the scene folder
        cp "$density_file" "$scene_folder/density_map_${base_name}.png"
        echo "Copied $density_file → $scene_folder/density_map_${base_name}.png"
    else
        echo "Warning: $scene_folder does not exist. Skipping."
    fi
done
