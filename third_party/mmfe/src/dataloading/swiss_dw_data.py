from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import cv2

from torch.utils.data import Dataset

from third_parties.SwissDwellings.constants import ROOM_NAMES, CMAP_ROOMTYPE
from third_parties.SwissDwellings.utils import colorize_floorplan


class SwissDwellingsDataset(Dataset):
    """
    PyTorch Dataset for SwissDwellings with two modalities: "binary" and "colored".

    Data layout expectation (see tests/test_swissdwellings.py example):
      root_dir/
        struct_in/    -> files like '{id}.npy' (stack[..., 0] is structural components)
        full_out/     -> files like '{id}.npy' (stack[..., 0] is room classes)

    When generate=True, images are created on-the-fly from the .npy stacks.
    When generate=False, images are loaded from cached PNGs:
      root_dir/
        binary/       -> '{id}.png'
        colored/      -> '{id}.png'

    Parameters
    - root_dir: absolute path to split directory (e.g., data/.../train)
    - sample_ids: optional explicit list of integer IDs to use. If None, auto-discovers from 'struct_in'.
    - sample_ids_file: optional path to a .txt file (one id per line). If provided, overrides auto-discovery.
    - image_size: optional (H, W). If set, final image will be resized to this size.
    - dual_transform: optional list of transforms taking (img0, img1) and returning (img0, img1)
    - modality_pairs: optional list of tuples of modalities. If None, all pairs are valid.
    - generate: if True, generate from .npy files; else, load cached PNGs from 'binary/' and 'colored/'.
    """

    def __init__(
        self,
        root_dir: str,
        sample_ids: Optional[Sequence[int]] = None,
        sample_ids_file: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        dual_transform: Optional[Callable] = None,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        generate: bool = False,
    ) -> None:
        super().__init__()

        if not os.path.isabs(root_dir):
            root_dir = os.path.abspath(root_dir)

        self.root_dir: str = root_dir
        self.struct_in_dir: str = os.path.join(self.root_dir, "struct_in")
        self.full_out_dir: str = os.path.join(self.root_dir, "full_out")
        self.binary_cache_dir: str = os.path.join(self.root_dir, "binary")
        self.colored_cache_dir: str = os.path.join(self.root_dir, "colored")
        self.points_cache_dir: str = os.path.join(self.root_dir, "points")

        self.image_size: Optional[Tuple[int, int]] = tuple(image_size) if image_size is not None else None
        self.dual_transform: Optional[Callable] = dual_transform
        self.generate: bool = bool(generate)

        # Modality configuration
        self.modalities: List[str] = ["binary", "colored", "sampled_points"]
        self.modality_pairs = modality_pairs
        if self.modality_pairs is None:
            self.modality_pairs = [
                (m1, m2) for m1 in self.modalities for m2 in self.modalities if m1 != m2
            ]

        if sample_ids_file is not None:
            sample_ids = self._load_ids_from_file(sample_ids_file)
        if sample_ids is None:
            sample_ids = self._discover_sample_ids()
        self.sample_ids: List[int] = list(sample_ids)

        if len(self.sample_ids) == 0:
            raise RuntimeError(
                f"No samples found in root_dir={self.root_dir}. Expected 'struct_in' and 'full_out' directories."
            )

        if self.generate:
            self.load_binary = self._generate_binary
            self.load_colored = self._generate_colored
            self.load_points = self._generate_sampled_points
        else:
            self.load_binary = self._load_binary_from_image
            self.load_colored = self._load_colored_from_image
            self.load_points = self._load_points_from_image

    def _discover_sample_ids(self) -> List[int]:
        """
        Discover sample IDs by listing files.
        When generate=True: lists files in 'struct_in' directory (.npy files).
        When generate=False: lists files in 'binary' directory (.png files).
        Only IDs that have all required files are kept.
        """
        discovered: List[int] = []
        
        if self.generate:
            if not os.path.isdir(self.struct_in_dir):
                return discovered

            struct_files = [f for f in os.listdir(self.struct_in_dir) if f.endswith(".npy")]
            for fname in struct_files:
                stem = os.path.splitext(fname)[0]
                if not re.fullmatch(r"\d+", stem):
                    continue
                sid = int(stem)
                # validate presence in full_out
                full_path = os.path.join(self.full_out_dir, f"{sid}.npy")
                if os.path.exists(full_path):
                    discovered.append(sid)
        else:
            if not os.path.isdir(self.binary_cache_dir):
                return discovered

            binary_files = [f for f in os.listdir(self.binary_cache_dir) if f.endswith(".png")]
            for fname in binary_files:
                stem = os.path.splitext(fname)[0]
                if not re.fullmatch(r"\d+", stem):
                    continue
                sid = int(stem)
                # validate presence in colored and points directories
                colored_path = os.path.join(self.colored_cache_dir, f"{sid}.png")
                points_path = os.path.join(self.points_cache_dir, f"{sid}.png")
                if os.path.exists(colored_path) and os.path.exists(points_path):
                    discovered.append(sid)

        discovered.sort()
        return discovered

    def _load_ids_from_file(self, ids_file: str) -> List[int]:
        """
        Load integer sample IDs from a text file (.txt), one per line.
        Lines starting with '#' and empty lines are ignored.
        """

        if not os.path.exists(ids_file):
            raise FileNotFoundError(f"Sample IDs file not found: {ids_file}")
        ids: List[int] = []
        with open(ids_file, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    ids.append(int(line))
                except ValueError:
                    # Try extracting digits anywhere in the line
                    m = re.search(r"(\d+)", line)
                    if m:
                        ids.append(int(m.group(1)))

        if self.generate:
            # Validate existence
            valid: List[int] = []
            for sid in ids:
                struct_path = os.path.join(self.struct_in_dir, f"{sid}.npy")
                full_path = os.path.join(self.full_out_dir, f"{sid}.npy")
                if os.path.exists(struct_path) and os.path.exists(full_path):
                    valid.append(sid)
            if len(valid) == 0:
                raise RuntimeError(f"No valid sample IDs found in file: {ids_file}")
            valid.sort()

        else:
            # Validate existence in cache directories (binary, colored, points)
            valid: List[int] = []
            for sid in ids:
                binary_path = os.path.join(self.binary_cache_dir, f"{sid}.png")
                colored_path = os.path.join(self.colored_cache_dir, f"{sid}.png")
                points_path = os.path.join(self.points_cache_dir, f"{sid}.png")
                if os.path.exists(binary_path) and os.path.exists(colored_path) and os.path.exists(points_path):
                    valid.append(sid)
            if len(valid) == 0:
                raise RuntimeError(f"No valid sample IDs found in file: {ids_file}")
            valid.sort()

        return valid

    def __len__(self) -> int:
        return len(self.sample_ids)

    @staticmethod
    def _resize_numpy(image_np: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        pil_img = Image.fromarray(image_np)
        height, width = int(size_hw[0]), int(size_hw[1])
        pil_img = pil_img.resize((width, height), resample=Image.BILINEAR)
        return np.array(pil_img)

    def _read_struct_channel(self, sample_id: int) -> np.ndarray:
        """
        Load structural components stack and return channel 0 as uint8 array.
        Path: struct_in/{id}.npy
        """
        path = os.path.join(self.struct_in_dir, f"{sample_id}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Structural components file not found: {path}")
        stack = np.load(path)
        struct = stack[..., 0].astype(np.uint8)
        return struct

    def _read_room_classes(self, sample_id: int) -> np.ndarray:
        """
        Load room classes stack and return channel 0 as uint8 integer labels.
        Path: full_out/{id}.npy
        """
        path = os.path.join(self.full_out_dir, f"{sample_id}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Room classes file not found: {path}")
        stack = np.load(path)
        classes = stack[..., 0].astype(np.uint8)
        return classes

    def _generate_binary(self, sample_id: int) -> np.ndarray:
        """
        Generate binary modality as a 3-channel RGB uint8 image from struct_in channel 0.
        Foreground pixels are black (0), background white (255).
        """
        struct = self._read_struct_channel(sample_id)
        # Normalize to 0/255
        mask = (struct > 0).astype(np.uint8) * 255
        img = np.stack([mask, mask, mask], axis=-1)
        if self.image_size is not None:
            img = self._resize_numpy(img, self.image_size)
        return img

    def _generate_colored(self, sample_id: int) -> np.ndarray:
        """
        Generate colored modality by mapping room class indices through the ROOM_NAMES colormap.
        Returns 3-channel RGB uint8 image.
        """
        room_labels = self._read_room_classes(sample_id)
        # Build classes list as in tests: enumerate(ROOM_NAMES)
        class_mapping = {cat: index for index, cat in enumerate(ROOM_NAMES)}
        classes = list(map(class_mapping.get, ROOM_NAMES))
        colored = colorize_floorplan(room_labels, classes=classes, cmap=CMAP_ROOMTYPE).astype(np.uint8)
        if self.image_size is not None:
            colored = self._resize_numpy(colored, self.image_size)
        return colored

    def _binary_to_points(self, binary_rgb: np.ndarray, points_density_inside_mask: float = 0.2, noise_std_pixels: float = 10.0) -> np.ndarray:
        """
        Sample points from the binary image's foreground (black lines) to create a point-cloud-like rendering.
        Mirrors the sampling strategy used in Aria's _wire_to_points.
        """
        gray = cv2.cvtColor(binary_rgb, cv2.COLOR_RGB2GRAY)
        # Foreground are dark pixels (lines): threshold to select near-black
        mask_foreground = gray < 250
        height, width = gray.shape

        mask_area = int(mask_foreground.sum())
        if mask_area == 0:
            return np.full((height, width, 3), 255, dtype=np.uint8)

        ys, xs = np.where(mask_foreground)

        # Smooth spatial density field for non-uniform sampling
        noise_field = np.random.rand(height, width).astype(np.float32)
        sigma = max(0.01, 0.005 * float(min(height, width)))
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

    def _generate_sampled_points(self, sample_id: int) -> np.ndarray:
        binary_img = self._generate_binary(sample_id)
        points_img = self._binary_to_points(binary_img)
        if self.image_size is not None:
            points_img = self._resize_numpy(points_img, self.image_size)
        return points_img

    def _load_binary_from_image(self, sample_id: int) -> np.ndarray:
        path = os.path.join(self.binary_cache_dir, f"{sample_id}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Binary image not found: {path}")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load binary image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            image = self._resize_numpy(image, self.image_size)
        return image

    def _load_colored_from_image(self, sample_id: int) -> np.ndarray:
        path = os.path.join(self.colored_cache_dir, f"{sample_id}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Colored image not found: {path}")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load colored image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            image = self._resize_numpy(image, self.image_size)
        return image

    def _load_points_from_image(self, sample_id: int) -> np.ndarray:
        path = os.path.join(self.points_cache_dir, f"{sample_id}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sampled points image not found: {path}")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load sampled points image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_size is not None:
            image = self._resize_numpy(image, self.image_size)
        return image

    def _load_modality(self, sample_id: int, modality_type: str) -> np.ndarray:
        if modality_type == "binary":
            return self.load_binary(sample_id)
        elif modality_type == "colored":
            return self.load_colored(sample_id)
        elif modality_type == "sampled_points":
            return self.load_points(sample_id)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}. Available: {self.modalities}")

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]

        # Randomly select a modality pair
        import random
        m0_type, m1_type = random.choice(self.modality_pairs)

        # Load both modalities
        m0_image = self._load_modality(sample_id, m0_type)
        m1_image = self._load_modality(sample_id, m1_type)

        # Apply transforms if specified
        if self.dual_transform is not None:
            for t in self.dual_transform:
                m0_image, m1_image = t(m0_image, m1_image)

        return {
            "modality_0": m0_image,
            "modality_1": m1_image,
            "m0_type": m0_type,
            "m1_type": m1_type,
            "sample_id": str(sample_id),
            "sample_dir": self.root_dir,
        }

    def generate_and_save(self, index: int) -> Dict[str, str]:
        """
        Generate and save both modalities for a given sample ID into cache directories.
        Returns paths to the saved files.
        """

        def replace_all_with_all_render(path: str) -> str:
            """Safely replace the folder named 'all' with 'all_render' in a path."""
            parts = path.split(os.sep)
            parts = [p if p != 'all' else 'all_render' for p in parts]
            return os.sep.join(parts)

        sample_id = self.sample_ids[index]

        save_binary_dir  = replace_all_with_all_render(self.binary_cache_dir)
        save_colored_dir = replace_all_with_all_render(self.colored_cache_dir)
        save_points_dir  = replace_all_with_all_render(self.points_cache_dir)

        os.makedirs(save_binary_dir, exist_ok=True)
        os.makedirs(save_colored_dir, exist_ok=True)
        os.makedirs(save_points_dir, exist_ok=True)

        binary_img = self._generate_binary(sample_id)
        colored_img = self._generate_colored(sample_id)
        points_img = self._binary_to_points(binary_img)
        if self.image_size is not None:
            points_img = self._resize_numpy(points_img, self.image_size)

        binary_path = os.path.join(save_binary_dir, f"{sample_id}.png")
        colored_path = os.path.join(save_colored_dir, f"{sample_id}.png")
        points_path = os.path.join(save_points_dir, f"{sample_id}.png")

        # Save using OpenCV (expects BGR)
        cv2.imwrite(binary_path, cv2.cvtColor(binary_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(colored_path, cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(points_path, cv2.cvtColor(points_img, cv2.COLOR_RGB2BGR))

        return {
            "sample_id": sample_id,
            "binary_path": binary_path,
            "colored_path": colored_path,
            "binary_dir": self.binary_cache_dir,
            "colored_dir": self.colored_cache_dir,
            "points_path": points_path,
            "points_dir": self.points_cache_dir,
        }

    def get_all_sample_modalities(self, index: int) -> Dict[str, np.ndarray]:
        """
        Load both modalities for a given sample index and return a dict.
        """
        sample_id = self.sample_ids[index]
        modalities_dict: Dict[str, np.ndarray] = {}
        for modality_type in self.modalities:
            try:
                modalities_dict[modality_type] = self._load_modality(sample_id, modality_type)
            except Exception as e:
                print(f"Warning: Failed to load {modality_type} for sample {sample_id}: {e}")
                continue
        return modalities_dict


