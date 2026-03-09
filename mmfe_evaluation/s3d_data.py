"""
Structured3D dataset for CrossOver evaluation.
Loads pre-generated modality images (floorplan, lidar, density_map) from disk.
"""

from __future__ import annotations

import os
import re
import json
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset

try:
    from third_party.structured3D.struct3D_utils import scene_to_density_map
    _HAS_DENSITY_MAP_GEN = True
except ImportError:
    _HAS_DENSITY_MAP_GEN = False


class Structured3DDataset(Dataset):
    """
    PyTorch Dataset for Structured3D scenes.

    Loads pre-generated image pairs from disk for cross-modal evaluation.
    Modalities: floorplan (ann/), lidar (points/), density_map.

    Parameters
    ----------
    root_dir : str
        Absolute path to the Structured3D root with ``scene_XXXXX`` folders.
    scene_ids_file : str, optional
        Path to a ``.json`` split file (COCO-style with ``images`` key).
    image_size : tuple, optional
        Not used directly (resizing is handled by dual_transform).
    dual_transform : list of callables, optional
        Paired transforms applied to (modality_0, modality_1).
    modality_pairs : list of (str, str), optional
        Which modality pairs to sample. Defaults to all non-identical pairs.
    furniture_pct : float or list of float, optional
        Furniture percentages for file selection. Defaults to [0.0, 0.25, 0.5, 0.75, 1.0].
    """

    MODALITIES = ("floorplan", "lidar", "density_map")

    def __init__(
        self,
        root_dir: str,
        scene_ids_file: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        dual_transform: Optional[List[Callable]] = None,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        furniture_pct: Union[float, List[float], None] = None,
    ) -> None:
        super().__init__()

        self.root_dir = os.path.abspath(root_dir)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.dual_transform = dual_transform
        self.modalities = list(self.MODALITIES)

        if modality_pairs is None:
            self.modality_pairs = [
                (a, b) for a in self.modalities for b in self.modalities if a != b
            ]
        else:
            self.modality_pairs = list(modality_pairs)

        if furniture_pct is None:
            self.furniture_pct = [0.0, 0.25, 0.5, 0.75, 1.0]
        elif isinstance(furniture_pct, (int, float)):
            self.furniture_pct = [float(furniture_pct)]
        else:
            self.furniture_pct = [float(p) for p in furniture_pct]

        if scene_ids_file is not None:
            self.scene_ids = self._load_ids_from_file(scene_ids_file)
        else:
            self.scene_ids = self._discover_scene_ids()

        if len(self.scene_ids) == 0:
            raise RuntimeError(
                f"No valid scenes found in {self.root_dir}. "
                "Expected folders like scene_00001 with ann/ and points/ subdirectories."
            )

    def _discover_scene_ids(self) -> List[int]:
        ids: List[int] = []
        if not os.path.isdir(self.root_dir):
            return ids
        for entry in sorted(os.listdir(self.root_dir)):
            m = re.match(r"scene_(\d{5})$", entry)
            if m and os.path.isdir(os.path.join(self.root_dir, entry)):
                scene_dir = os.path.join(self.root_dir, entry)
                if os.path.isdir(os.path.join(scene_dir, "ann")) and os.path.isdir(
                    os.path.join(scene_dir, "points")
                ):
                    ids.append(int(m.group(1)))
        return ids

    def _load_ids_from_file(self, path: str) -> List[int]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")

        ext = os.path.splitext(path.lower())[1]
        if ext == ".json":
            with open(path) as f:
                data = json.load(f)
            raw_ids: List[int] = []
            for img in data.get("images", []):
                stem = os.path.splitext(os.path.basename(img.get("file_name", "")))[0]
                try:
                    raw_ids.append(int(stem))
                except ValueError:
                    m = re.search(r"(\d{5})", img.get("file_name", ""))
                    if m:
                        raw_ids.append(int(m.group(1)))
            raw_ids = sorted(set(raw_ids))
        elif ext == ".txt":
            raw_ids = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    m = re.search(r"(\d+)", line)
                    if m:
                        raw_ids.append(int(m.group(1)))
            raw_ids = sorted(set(raw_ids))
        else:
            raise ValueError(f"Unsupported split file format: {ext}")

        valid = [
            sid
            for sid in raw_ids
            if os.path.isdir(os.path.join(self.root_dir, f"scene_{sid:05d}"))
        ]
        if not valid:
            raise RuntimeError(f"No valid scene IDs found from file: {path}")
        return valid

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_modality(
        self, scene_id: int, modality: str, furniture_pct: float
    ) -> np.ndarray:
        scene_dir = os.path.join(self.root_dir, f"scene_{scene_id:05d}")
        pct_key = int(furniture_pct * 100)

        if modality == "floorplan":
            return self._load_image(os.path.join(scene_dir, "ann", f"{pct_key}.png"))
        elif modality == "lidar":
            return self._load_image(os.path.join(scene_dir, "points", f"{pct_key}.png"))
        elif modality == "density_map":
            pre_gen = os.path.join(scene_dir, "density_map.png")
            if os.path.exists(pre_gen):
                return self._load_image(pre_gen)
            if _HAS_DENSITY_MAP_GEN:
                return scene_to_density_map(
                    scene_path=self.root_dir, scene_id=scene_id
                )
            raise FileNotFoundError(
                f"No density_map.png in {scene_dir} and generation library not available."
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def __getitem__(self, index: int):
        scene_id = self.scene_ids[index]
        furn_pct = float(np.random.choice(self.furniture_pct))

        pairs = list(self.modality_pairs)
        random.shuffle(pairs)

        for m0_type, m1_type in pairs:
            try:
                m0_img = self._load_modality(scene_id, m0_type, furn_pct)
            except FileNotFoundError:
                continue
            try:
                m1_img = self._load_modality(scene_id, m1_type, furn_pct)
            except FileNotFoundError:
                continue

            if self.dual_transform is not None:
                for t in self.dual_transform:
                    m0_img, m1_img = t(m0_img, m1_img)

            return {
                "modality_0": m0_img,
                "modality_1": m1_img,
                "m0_type": m0_type,
                "m1_type": m1_type,
                "sample_id": str(scene_id),
                "sample_dir": os.path.join(self.root_dir, f"scene_{scene_id:05d}"),
            }

        return None
