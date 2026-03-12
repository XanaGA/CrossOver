import os
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class ZillowDataset(Dataset):
    """
    PyTorch Dataset for loading Zillow Indoor Dataset (ZInD) floorplan renderings for
    multi-modality contrastive learning.

    Each item returns two modalities selected from:
        - raster_to_vector
        - vector_only
        - points (generated from vector_only drawings)

    Parameters
    ----------
    root_dir:
        Absolute path to the dataset root containing per-floor folders (e.g., 0000_f1).
    sample_ids:
        Optional explicit list of sample IDs (folder names). If None, auto-discovers.
    sample_ids_file:
        Optional path to a text file (train.txt / val.txt) with one sample ID per line.
        If provided it overrides auto-discovery.
    image_size:
        Optional (height, width). If provided, images are resized to this spatial size.
    dual_transform:
        Optional iterable of callables applied sequentially to both modalities.
    modality_pairs:
        Optional list of (mod_0, mod_1) tuples describing the valid modality pairings.
        If None, defaults to pairing every modality with the others (ordered pairs).
    generate:
        If True, generate the `points` modality on the fly from the `vector_only` modality.
        If False, expect `points_floor_*.jpg` files on disk and load them.
    """

    MODALITIES = ("raster_to_vector", "vector_only", "points")

    def __init__(
        self,
        root_dir: str,
        sample_ids: Optional[Sequence[str]] = None,
        sample_ids_file: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        dual_transform: Optional[Iterable[Callable]] = None,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        generate: bool = False,
    ) -> None:
        super().__init__()

        if not os.path.isabs(root_dir):
            root_dir = os.path.abspath(root_dir)

        self.root_dir: str = root_dir
        self.image_size: Optional[Tuple[int, int]] = (
            tuple(image_size) if image_size is not None else None
        )
        self.dual_transform: Optional[Iterable[Callable]] = dual_transform
        self.generate: bool = bool(generate)

        self.modalities: Tuple[str, ...] = self.MODALITIES
        self.modality_pairs: List[Tuple[str, str]] = (
            modality_pairs
            if modality_pairs is not None
            else self._default_modality_pairs()
        )

        if sample_ids_file is not None:
            sample_ids = self._load_ids_from_file(sample_ids_file)
        if sample_ids is None:
            sample_ids = self._discover_sample_ids()

        self.sample_ids: List[str] = list(sample_ids)
        if len(self.sample_ids) == 0:
            raise RuntimeError(f"No samples found in root_dir={self.root_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        sample_id = self.sample_ids[index]
        sample_dir = os.path.join(self.root_dir, sample_id)

        m0_type, m1_type = random.choice(self.modality_pairs)

        m0_image = self._load_modality(sample_dir, sample_id, m0_type)
        m1_image = self._load_modality(sample_dir, sample_id, m1_type)

        if self.dual_transform is not None:
            for transform in self.dual_transform:
                m0_image, m1_image = transform(m0_image, m1_image)

        return {
            "modality_0": m0_image,
            "modality_1": m1_image,
            "m0_type": m0_type,
            "m1_type": m1_type,
            "sample_id": sample_id,
            "sample_dir": sample_dir,
        }

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _default_modality_pairs(self) -> List[Tuple[str, str]]:
        """
        Default modality pairs: all ordered combinations of distinct modalities.
        """
        pairs: List[Tuple[str, str]] = []
        for m0 in self.MODALITIES:
            for m1 in self.MODALITIES:
                if m0 == m1:
                    continue
                pairs.append((m0, m1))
        return pairs

    def _discover_sample_ids(self) -> List[str]:
        """
        Auto-discover sample IDs by scanning folders under `root_dir`.
        Only directories containing every required modality file are returned.
        """
        if not os.path.isdir(self.root_dir):
            return []

        sample_ids: List[str] = []
        for entry in sorted(os.listdir(self.root_dir)):
            entry_dir = os.path.join(self.root_dir, entry)
            if not os.path.isdir(entry_dir):
                continue
            if self._has_all_modalities(entry_dir, entry):
                sample_ids.append(entry)

        return sample_ids

    def _load_ids_from_file(self, ids_file: str) -> List[str]:
        """
        Load sample IDs from a text file where each line contains one ID.
        Empty lines and comments ('#') are ignored.
        """
        if not os.path.exists(ids_file):
            raise FileNotFoundError(f"Sample IDs file not found: {ids_file}")

        ids: List[str] = []
        with open(ids_file, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)

        valid_ids = [
            sample_id
            for sample_id in ids
            if self._has_all_modalities(os.path.join(self.root_dir, sample_id), sample_id)
        ]

        if len(valid_ids) == 0:
            raise RuntimeError(f"No valid sample IDs found in file: {ids_file}")

        return sorted(valid_ids)

    def _has_all_modalities(self, sample_dir: str, sample_id: str) -> bool:
        """
        Verify that the sample directory contains all required modality files.
        """
        for modality in self.modalities:
            if modality == "points" and self.generate:
                # When generating points, no on-disk file required
                continue
            path = self._resolve_modality_path(sample_dir, sample_id, modality)
            if not os.path.exists(path):
                return False
        return True

    def _load_modality(self, sample_dir: str, sample_id: str, modality: str) -> np.ndarray:
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality '{modality}'. Available: {self.modalities}")

        if modality == "points" and self.generate:
            return self._generate_points(sample_dir, sample_id)

        image_path = self._resolve_modality_path(sample_dir, sample_id, modality)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Modality '{modality}' not found at path: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size is not None:
            height, width = self.image_size
            image = cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)

        return image

    def _resolve_modality_path(self, sample_dir: str, sample_id: str, modality: str) -> str:
        filename = self._build_filename(sample_dir, sample_id, modality)
        return os.path.join(sample_dir, filename)

    def _build_filename(self, sample_dir: str, sample_id: str, modality: str) -> str:
        """
        Construct the filename for the given modality.

        Expected sample IDs follow the convention: `<scene_id>_f<floor_index>`, e.g., `1236_f2`.
        Files follow the pattern: `{modality}_redraw_layout_floor_{floor_index:02d}.jpg`
        """
        try:
            scene_part, floor_part = sample_id.split("_f")
        except ValueError as exc:
            raise ValueError(
                f"Sample ID '{sample_id}' does not match '<scene>_f<floor>' pattern."
            ) from exc

        if not floor_part:
            raise ValueError(f"Sample ID '{sample_id}' missing floor index.")

        try:
            floor_index = int(floor_part)
        except ValueError as exc:
            raise ValueError(
                f"Floor component '{floor_part}' in sample ID '{sample_id}' is not an integer."
            ) from exc

        floor_suffix = f"floor_{floor_index:02d}"

        if modality == "raster_to_vector":
            prefix = "raster_to_vector_redraw_layout_"
            return f"{prefix}{floor_suffix}.jpg"
        elif modality == "vector_only":
            prefix = "vector_only_redraw_layout_"
            return f"{prefix}{floor_suffix}.jpg"
        elif modality == "points":
            prefix = "points_"
            # Prefer JPG, fallback to PNG if present
            jpg_name = f"{prefix}{floor_suffix}.jpg"
            png_name = f"{prefix}{floor_suffix}.png"
            jpg_path = os.path.join(sample_dir, jpg_name)
            if os.path.exists(jpg_path):
                return jpg_name
            png_path = os.path.join(sample_dir, png_name)
            if os.path.exists(png_path):
                return png_name
            # Default to JPG naming if neither exists (will raise upstream)
            return jpg_name
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    # ------------------------------------------------------------------
    # Points generation helpers
    # ------------------------------------------------------------------
    def _generate_points(self, sample_dir: str, sample_id: str) -> np.ndarray:
        """
        Generate the points modality from the vector_only modality.
        """
        vector_image = self._load_image_without_resize(sample_dir, sample_id, "vector_only")
        points = self._binary_to_points(vector_image)

        if self.image_size is not None:
            height, width = self.image_size
            points = cv2.resize(points, (int(width), int(height)), interpolation=cv2.INTER_AREA)

        return points

    def _load_image_without_resize(self, sample_dir: str, sample_id: str, modality: str) -> np.ndarray:
        """
        Load an image modality without resizing. Used internally for generation.
        """
        if modality not in ("raster_to_vector", "vector_only"):
            raise ValueError("Only raster_to_vector and vector_only can be loaded raw.")

        image_path = self._resolve_modality_path(sample_dir, sample_id, modality)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _binary_to_points(
        self,
        binary_rgb: np.ndarray,
        points_density_inside_mask: float = 0.2,
        noise_std_pixels: float = 10.0,
    ) -> np.ndarray:
        """
        Sample points from the vector drawing to create a point-cloud-like rendering.

        Adapted from SwissDwellings `_binary_to_points`, with the binary mask derived
        from the vector_only modality.
        """
        gray = cv2.cvtColor(binary_rgb, cv2.COLOR_RGB2GRAY)
        # Foreground: dark pixels (vector drawing lines)
        mask_foreground = gray < 250
        height, width = gray.shape

        mask_area = int(mask_foreground.sum())
        if mask_area == 0:
            return np.full((height, width, 3), 255, dtype=np.uint8)

        ys, xs = np.where(mask_foreground)

        # Smooth spatial density field for non-uniform sampling
        noise_field = np.random.rand(height, width).astype(np.float32)
        sigma = max(0.01, 0.01 * float(min(height, width)))
        density_field = cv2.GaussianBlur(noise_field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        min_val = float(density_field.min())
        max_val = float(density_field.max())
        if max_val > min_val:
            density_field = (density_field - min_val) / (max_val - min_val)
        else:
            density_field.fill(1.0)
        gamma = float(np.random.uniform(1.6, 2.5))
        density_field = np.power(density_field + 1e-6, gamma).astype(np.float32)

        masked_weights = np.clip(density_field[ys, xs].astype(np.float64), 0.0, None)
        total_weight = float(masked_weights.sum())
        probs = None if total_weight <= 0.0 or not np.isfinite(total_weight) else (masked_weights / total_weight)
        if probs is not None:
            probs = np.clip(probs, 0.0, 1.0)
            s = float(probs.sum())
            if s == 0.0 or not np.isfinite(s):
                probs = None
            else:
                probs /= s
                if probs.size > 0:
                    residual = 1.0 - float(probs.sum())
                    probs[-1] = max(0.0, probs[-1] + residual)
                    s2 = float(probs.sum())
                    if not (s2 > 0.0 and np.isfinite(s2)):
                        probs = None

        target_points = max(1, min(int(points_density_inside_mask * mask_area), mask_area))
        if target_points == mask_area:
            idx = np.arange(mask_area)
        else:
            idx = np.random.choice(mask_area, size=target_points, replace=False, p=probs)
        y_base = ys[idx].astype(np.float32)
        x_base = xs[idx].astype(np.float32)

        if noise_std_pixels > 0:
            n = y_base.shape[0]
            y_noise = np.random.normal(0.0, noise_std_pixels, size=n).astype(np.float32)
            x_noise = np.random.normal(0.0, noise_std_pixels, size=n).astype(np.float32)
            if n > 0:
                outliers_percentage = np.random.uniform(0.05, 0.1)
                high_std_count = max(1, int(outliers_percentage * n))
                high_idx = np.random.choice(n, size=high_std_count, replace=False)
                scale = np.float32(5)
                y_noise[high_idx] *= scale
                x_noise[high_idx] *= scale
            y_base += y_noise
            x_base += x_noise

        valid = (y_base >= 0) & (y_base < height) & (x_base >= 0) & (x_base < width)
        y_pts = y_base[valid].astype(np.int32)
        x_pts = x_base[valid].astype(np.int32)

        out = np.full((height, width, 3), 255, dtype=np.uint8)
        for x, y in zip(x_pts, y_pts):
            cv2.circle(out, (int(x), int(y)), 1, (0, 0, 0), -1)
        return out

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def generate_and_save(self, index: int, overwrite: bool = False) -> Dict[str, str]:
        """
        Generate (if needed) and save the points modality for the specified sample.
        """
        sample_id = self.sample_ids[index]
        sample_dir = os.path.join(self.root_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        points_filename = self._build_filename(sample_dir, sample_id, "points")
        points_path = os.path.join(sample_dir, points_filename)

        if not overwrite and os.path.exists(points_path):
            return {
                "sample_id": sample_id,
                "points_path": points_path,
                "generated": False,
            }

        points_image = self._generate_points(sample_dir, sample_id)
        points_bgr = cv2.cvtColor(points_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(points_path, points_bgr)

        return {
            "sample_id": sample_id,
            "points_path": points_path,
            "generated": True,
        }

    def get_all_sample_modalities(self, index: int) -> Dict[str, np.ndarray]:
        """
        Load all modalities for a given sample index.
        """
        sample_id = self.sample_ids[index]
        sample_dir = os.path.join(self.root_dir, sample_id)
        modalities_dict: Dict[str, np.ndarray] = {}
        for modality in self.modalities:
            try:
                modalities_dict[modality] = self._load_modality(sample_dir, sample_id, modality)
            except Exception as exc:
                print(f"Warning: Failed to load {modality} for sample {sample_id}: {exc}")
                continue
        return modalities_dict


