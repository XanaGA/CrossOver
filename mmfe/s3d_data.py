from __future__ import annotations

import os
import re
import json
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

# Use the utilities that return numpy images
from third_parties.structured3D.struct3D_utils import (
    scene_to_density_map,
    scene_to_floorplan_image,
    scene_to_lidar_image,
)

class Structured3DDataset(Dataset):
    """
    PyTorch Dataset for lazy-loading Structured3D samples for contrastive learning.

    Each item returns a dictionary with two modalities: floorplan and lidar views.

    Parameters
    - root_dir: absolute path to the Structured3D dataset root containing `scene_XXXXX` folders
    - scene_ids: optional explicit list of scene IDs to use (integers like 1, 2, ...). If None, auto-discovers.
    - scene_ids_file: optional path to a .txt file with one scene per line. Lines may
      be integers (e.g., "42") or names like "scene_00042". If provided, overrides auto-discovery.
    - no_color: passed through to the utility to render grayscale/no-color
    - bbox_percentage: fraction of 3D boxes to render (0.0..1.0)
    - image_size: optional (H, W). If set, final image will be resized to this size
    - dpi: rendering DPI, forwarded to the utility
    - dual_transform: optional callable taking (img0, img1) and returning (img0, img1)
      with identical random params applied to both (use for paired random augs)
    - modality_pairs: list of tuples defining valid modality pairs. If None, all pairs are valid.
    - generate: if True, generate floorplan and lidar from scene data. If False, load from pre-generated PNG files.
    - furniture_pct: fixed furniture percentage to use (0.0 to 1.0). If None, random percentage is used.
    """

    def __init__(
        self,
        root_dir: str,
        scene_ids: Optional[Sequence[int]] = None,
        scene_ids_file: Optional[str] = None,
        no_color: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
        dpi: int = 100,
        dual_transform: Optional[Callable] = None,
        to_tensor: bool = True,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        generate: bool = False,
        furniture_pct: Union[float, List[float]] = None,
    ) -> None:
        super().__init__()

        if not os.path.isabs(root_dir):
            # Prefer absolute paths for robustness
            root_dir = os.path.abspath(root_dir)

        self.root_dir: str = root_dir
        self.no_color: bool = no_color
        self.image_size: Optional[Tuple[int, int]] = tuple(image_size) if image_size is not None else None
        self.dpi: int = int(dpi)
        self.dual_transform: Optional[Callable] = dual_transform
        self.generate: bool = bool(generate)
        self.furniture_pct:  Union[float, List[float]] = furniture_pct
        if isinstance(furniture_pct, float):
            self.furniture_pct = [furniture_pct]

        if generate:
            self.load_floorplan = self._generate_floorplan
            self.load_lidar = self._generate_lidar
        else:
            self.load_floorplan = self._load_floorplan_from_image
            self.load_lidar = self._load_lidar_from_image
            if furniture_pct is None:
                self.furniture_pct = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Modality configuration for S3D
        self.modalities = ["floorplan", "lidar", "density_map"] 
        self.modality_pairs = modality_pairs
        
        # Create valid modality pairs if not provided
        if self.modality_pairs is None:
            self.modality_pairs = [(m1, m2) for m1 in self.modalities 
                                  for m2 in self.modalities if m1 != m2]

        # Resolve scene IDs from file, explicit list, or discovery
        if scene_ids_file is not None:
            scene_ids = self._load_ids_from_file(scene_ids_file)
        if scene_ids is None:
            scene_ids = self._discover_scene_ids()
        self.scene_ids: List[int] = list(scene_ids)

        if len(self.scene_ids) == 0:
            raise RuntimeError(f"No scenes found in root_dir={self.root_dir}. Expected folders like 'scene_00001'.")

    def _discover_scene_ids(self) -> List[int]:
        """
        Find all `scene_XXXXX` folders inside `self.root_dir` and return their integer IDs.
        """
        discovered_ids: List[int] = []
        if not os.path.isdir(self.root_dir):
            return discovered_ids

        for entry in os.listdir(self.root_dir):
            full_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(full_path):
                continue

            match = re.match(r"scene_(\d{5})$", entry)
            if match:
                try:
                    scene_id = int(match.group(1))
                    if self.generate:
                        # When generating, just need the scene directory
                        discovered_ids.append(scene_id)
                    else:
                        # When loading from images, need ann/ and points/ directories
                        ann_dir = os.path.join(full_path, "ann")
                        points_dir = os.path.join(full_path, "points")
                        if os.path.exists(ann_dir) and os.path.exists(points_dir):
                            discovered_ids.append(scene_id)
                except ValueError:
                    continue

        discovered_ids.sort()
        return discovered_ids

    def _load_ids_from_file(self, ids_file: str) -> List[int]:
        """
        Load scene IDs from a split file.
        Supports:
          - Text files (.txt): one per line, e.g., "42" or "scene_00042" (ignores comments '#')
          - JSON files (.json): COCO-style with key 'images' and entries containing 'file_name'
            like "00042.png". The numeric stem is used as the scene ID.
        """
        ids: List[int] = []
        if not os.path.exists(ids_file):
            raise FileNotFoundError(f"Scene IDs file not found: {ids_file}")

        _, ext = os.path.splitext(ids_file.lower())
        if ext == '.json':
            with open(ids_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON split file '{ids_file}': {e}")
            images = data.get('images', [])
            for img in images:
                fname = img.get('file_name', '')
                # extract numeric stem before extension, e.g., 00042.png -> 42
                stem = os.path.splitext(os.path.basename(fname))[0]
                try:
                    ids.append(int(stem))
                except ValueError:
                    # Also try matching 5-digit numbers anywhere in name
                    m = re.search(r"(\d{5})", fname)
                    if m:
                        ids.append(int(m.group(1)))
            # Ensure uniqueness
            ids = list(sorted(set(ids)))
        else:
            raise ValueError(f"Structured3D scene IDs must be .json, but got {ids_file}")
            
        # Validate folders exist
        valid_ids: List[int] = []
        for sid in ids:
            scene_dir = os.path.join(self.root_dir, f"scene_{sid:05d}")
            if os.path.isdir(scene_dir):
                if self.generate:
                    # When generating, just need the scene directory
                    valid_ids.append(sid)
                else:
                    # When loading from images, need ann/ and points/ directories
                    ann_dir = os.path.join(scene_dir, "ann")
                    points_dir = os.path.join(scene_dir, "points")
                    if os.path.exists(ann_dir) and os.path.exists(points_dir):
                        valid_ids.append(sid)
        if len(valid_ids) == 0:
            raise RuntimeError(f"No valid scene IDs found in file: {ids_file}")
        valid_ids.sort()
        return valid_ids

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _load_modality(self, scene_id: int, modality_type: str, furniture_pct: float = None) -> np.ndarray:
        """
        Load a specific modality for a given scene.
        
        Args:
            scene_id: The scene ID
            modality_type: Type of modality to load ("floorplan", "lidar", or "density_map")
            furniture_pct: Furniture percentage for file selection (used when generate=False)
            
        Returns:
            numpy array of the loaded modality
        """
        if modality_type == "floorplan":
            return self.load_floorplan(scene_id, furniture_pct)
        elif modality_type == "lidar":
            return self.load_lidar(scene_id, furniture_pct)
        elif modality_type == "density_map":
            return scene_to_density_map(
                scene_path=self.root_dir,
                scene_id=scene_id
            )
        else:
            raise ValueError(f"Unknown modality type: {modality_type}. Available modalities: {self.modalities}")

    def _load_floorplan_from_image(self, scene_id: int, furniture_pct: float) -> np.ndarray:
        """
        Load floorplan from a pre-generated PNG file.
        
        Args:
            scene_id: The scene ID to load floorplan for
            furniture_pct: Furniture percentage (0.0 to 1.0) to determine which file to load
            
        Returns:
            numpy array of the loaded floorplan image
        """
        scene_path = os.path.join(self.root_dir, f"scene_{scene_id:05d}")
        ann_path = os.path.join(scene_path, "ann", f"{int(furniture_pct*100)}.png")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Floorplan file not found: {ann_path}")
        
        # Load the image using cv2
        image = cv2.imread(ann_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load floorplan image: {ann_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_lidar_from_image(self, scene_id: int, furniture_pct: float) -> np.ndarray:
        """
        Load LiDAR visualization from a pre-generated PNG file.
        
        Args:
            scene_id: The scene ID to load LiDAR for
            furniture_pct: Furniture percentage (0.0 to 1.0) to determine which file to load
            
        Returns:
            numpy array of the loaded LiDAR image
        """
        scene_path = os.path.join(self.root_dir, f"scene_{scene_id:05d}")
        points_path = os.path.join(scene_path, "points", f"{int(furniture_pct*100)}.png")
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"LiDAR file not found: {points_path}")
        
        # Load the image using cv2
        image = cv2.imread(points_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load LiDAR image: {points_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _generate_floorplan(self, scene_id: int, furniture_pct: float) -> np.ndarray:
        """
        Generate floorplan from scene data.
        """
        bbox_percentage = furniture_pct if furniture_pct is not None else random.uniform(0.0, 1.0)
        return scene_to_floorplan_image(
            scene_path=self.root_dir,
            scene_id=scene_id,
            no_color=self.no_color,
            bbox_percentage=bbox_percentage,
            dpi=self.dpi,
        )

    def _generate_lidar(self, scene_id: int, furniture_pct: float) -> np.ndarray:
        """
        Generate LiDAR visualization from scene data.
        """
        bbox_percentage = furniture_pct if furniture_pct is not None else random.uniform(0.0, 1.0)
        return scene_to_lidar_image(
            scene_path=self.root_dir,
            scene_id=scene_id,
            bbox_percentage=bbox_percentage,
            dpi=self.dpi,
        )

    @staticmethod
    def _resize_numpy(image_np: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        # Convert to PIL for reliable resizing (expects width, height)
        pil_img = Image.fromarray(image_np)
        height, width = int(size_hw[0]), int(size_hw[1])
        pil_img = pil_img.resize((width, height), resample=Image.BILINEAR)
        return np.array(pil_img)

    def __getitem__(self, index: int):
        scene_id = self.scene_ids[index]
        scene_dir = os.path.join(self.root_dir, f"scene_{scene_id:05d}")

        # Randomly select a modality pair
        import random
        m0_type, m1_type = random.choice(self.modality_pairs)
        
        # Load both modalities
        if self.furniture_pct is None:
            furniture_pct = random.uniform(0.0, 1.0)
        else:
            furniture_pct = np.random.choice(self.furniture_pct)
        m0_image = self._load_modality(scene_id, m0_type, furniture_pct)
        m1_image = self._load_modality(scene_id, m1_type, furniture_pct)

        if m0_image is None:
            m0_image = self._load_modality(scene_id, "floorplan", 0.0)
        if m1_image is None:
            m1_image = self._load_modality(scene_id, "floorplan", 0.0)
        

        # Apply transforms if specified
        if self.dual_transform is not None:
            for t in self.dual_transform:
                m0_image, m1_image = t(m0_image, m1_image)

        return {
            "modality_0": m0_image,
            "modality_1": m1_image,
            "m0_type": m0_type,
            "m1_type": m1_type,
            "sample_id": str(scene_id),
            "sample_dir": scene_dir,
        }

    def generate_and_save(self, index: int, furniture_pct: float) -> Dict[str, str]:
        """
        Generate and save floorplan and lidar images for a given scene and furniture percentage.
        
        Args:
            index: Index of the scene in the dataset
            furniture_pct: Furniture percentage (0.0 to 1.0) to use for generation
            
        Returns:
            Dictionary with paths to the saved files
        """
        scene_id = self.scene_ids[index]
        scene_dir = os.path.join(self.root_dir, f"scene_{scene_id:05d}")
        print("Saving scene_id: ", scene_id)
        
        # Create directories if they don't exist
        ann_dir = os.path.join(scene_dir, "ann")
        points_dir = os.path.join(scene_dir, "points")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(points_dir, exist_ok=True)
        
        # Generate floorplan (annotations)
        floorplan_image = scene_to_floorplan_image(
            scene_path=self.root_dir,
            scene_id=scene_id,
            no_color=self.no_color,
            bbox_percentage=furniture_pct,
            dpi=self.dpi,
        )
        
        # Generate lidar visualization (points)
        lidar_image = scene_to_lidar_image(
            scene_path=self.root_dir,
            scene_id=scene_id,
            bbox_percentage=furniture_pct,
            dpi=self.dpi,
        )
        
        # Define file paths
        ann_path = os.path.join(ann_dir, f"{int(furniture_pct*100)}.png")
        points_path = os.path.join(points_dir, f"{int(furniture_pct*100)}.png")
        
        # Save images
        # Convert RGB to BGR for OpenCV
        floorplan_bgr = cv2.cvtColor(floorplan_image, cv2.COLOR_RGB2BGR)
        lidar_bgr = cv2.cvtColor(lidar_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(ann_path, floorplan_bgr)
        cv2.imwrite(points_path, lidar_bgr)
        
        return {
            "scene_id": scene_id,
            "furniture_pct": furniture_pct,
            "ann_path": ann_path,
            "points_path": points_path,
            "ann_dir": ann_dir,
            "points_dir": points_dir
        }

    def get_all_sample_modalities(self, index: int):
        """
        Get all modalities for a given scene in self.modalities.
        Return a dictionary with the modality type as the key and the modality image as the value.
        Loads all furniture percentages from self.furniture_pct with different keys.
        """
        scene_id = self.scene_ids[index]
        
        # Get all furniture percentages to load
        if self.furniture_pct is None:
            furniture_pcts = [random.uniform(0.0, 1.0)]
        else:
            furniture_pcts = self.furniture_pct
        
        modalities_dict = {}
        
        # Load all modalities for this scene with all furniture percentages
        for modality_type in self.modalities:
            modality_dict = {}
            
            for furniture_pct in furniture_pcts:
                try:
                    modality_image = self._load_modality(scene_id, modality_type, furniture_pct)
                    
                    # Apply transforms if specified
                    if self.dual_transform is not None:
                        # Convert to PIL for transforms
                        pil_image = Image.fromarray(modality_image)
                        modality_image = pil_image
                        for t in self.dual_transform:
                            modality_image = t(modality_image)
                    
                    # Use furniture percentage as key (convert to int for cleaner keys)
                    furniture_key = f"{int(furniture_pct*100)}"
                    modality_dict[furniture_key] = modality_image
                    
                except Exception as e:
                    print(f"Warning: Failed to load {modality_type} with furniture_pct {furniture_pct} for scene {scene_id}: {e}")
                    # Skip this furniture percentage if it fails to load
                    continue
            
            modalities_dict[modality_type] = modality_dict
        
        return modalities_dict
