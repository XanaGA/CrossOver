import os
import re

target_dir = "/home/xavi/.cache/torch/hub/facebookresearch_dinov3_main"
patch_line = "from __future__ import annotations\n"

for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                continue

            # --- Fix 1: Add __future__ annotations ---
            lines = content.splitlines(keepends=True)
            if not any("from __future__ import annotations" in line for line in lines[:3]):
                insert_pos = 1 if lines[0].startswith("#!") else 0
                lines.insert(insert_pos, patch_line)
                content = "".join(lines)
                print(f"Added future annotations: {file_path}")

            # --- Fix 2: Remove kw_only=True (Python 3.10+ only) ---
            if "kw_only=True" in content:
                # This regex handles 'kw_only=True' and cleans up surrounding commas/spaces
                # It looks for (kw_only=True), (, kw_only=True), or (kw_only=True, )
                content = re.sub(r',\s*kw_only=True', '', content)
                content = re.sub(r'kw_only=True\s*,', '', content)
                content = re.sub(r'kw_only=True', '', content)
                print(f"Removed kw_only: {file_path}")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

print("\nPatching complete. Try running your evaluation again.")