find data/structure3D/Structured3D_annotation_3d/Structured3D/ -type f -name annotation_3d.json | while read file; do
  scene_dir=$(basename $(dirname "$file"))
  dest_dir="data/structure3D/Structured3D_bbox/Structured3D/$scene_dir"
  mkdir -p "$dest_dir"
  cp "$file" "$dest_dir/"
done
