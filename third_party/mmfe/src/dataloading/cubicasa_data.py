import os
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
from IPython.display import Image as IPythonImage
import io
import cairosvg

from torch.utils.data import Dataset


# Import CubiCasa5k utilities
from third_parties.CubiCasa5k.floortrans.loaders.house import House


class Cubicasa5kDataset(Dataset):
    """
    PyTorch Dataset for loading CubiCasa5k samples for contrastive learning.
    
    Each item returns a dictionary with two randomly selected modalities for contrastive learning.
    
    Parameters
    - root_dir: absolute path to the CubiCasa5k dataset root containing sample folders
    - sample_ids: optional explicit list of sample IDs to use. If None, auto-discovers.
    - sample_ids_file: optional path to a .txt file with one ID per line. Lines may
      start/end with '/', and should reference folders relative to dataset root
      (e.g., "/colorful/10052/" or "colorful/10052"). If provided, overrides auto-discovery.
    - image_size: optional (H, W). If set, final image will be resized to this size
    - use_original_size: if True, use F1_original.png instead of F1_scaled.png
    - transform: optional callable applied to the images
    - to_tensor: if True (default), convert images to torch.Tensor CHW float32 in [0,1] after transform
    - modalities: list of modality types to choose from. Default: ["drawing", "gt_svg_annotations", "lidar_points"]
    - modality_pairs: list of tuples defining valid modality pairs. If None, all pairs are valid.
    - generate: if True, generate gt_svg_annotations and lidar_points from SVG data. If False, load from pre-generated PNG files.
    """

    def __init__(
        self,
        root_dir: str,
        sample_ids: Optional[Sequence[str]] = None,
        sample_ids_file: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        use_original_size: bool = False,
        dual_transform: Optional[Callable] = None,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        generate: bool = False,
        furniture_pct: float = None,
    ) -> None:
        super().__init__()

        if not os.path.isabs(root_dir):
            # Prefer absolute paths for robustness
            root_dir = os.path.abspath(root_dir)

        self.root_dir: str = root_dir
        self.use_original_size: bool = bool(use_original_size)
        self.dual_transform: Optional[Callable] = dual_transform
        self.image_size: Optional[Tuple[int, int]] = tuple(image_size) if image_size is not None else None
        self.generate: bool = bool(generate)
        self.furniture_pct:  Union[float, List[float]] = furniture_pct
        if isinstance(furniture_pct, float):
            self.furniture_pct = [furniture_pct]

        if generate:
            self.load_anns = self._generate_anns
            self.load_points = self._generate_points
        else:
            self.load_anns = self._load_anns_from_image
            self.load_points = self._load_points_from_image
            if furniture_pct is None:
                self.furniture_pct = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Modality configuration
        self.modalities = ["drawing", "gt_svg_annotations", "lidar_points"]
        self.modality_pairs = modality_pairs
        
        # Validate modalities
        if len(self.modalities) < 2:
            raise ValueError("At least 2 modalities are required for contrastive learning")
        
        # Create valid modality pairs if not provided
        if self.modality_pairs is None:
            self.modality_pairs = [(m1, m2) for m1 in self.modalities 
                                  for m2 in self.modalities if m1 != m2]

        # File naming conventions
        self.image_file_name = '/F1_original.png' if self.use_original_size else '/F1_scaled.png'
        self.svg_file_name = '/model.svg'

        # Resolve sample IDs from file, explicit list, or discovery
        if sample_ids_file is not None:
            sample_ids = self._load_ids_from_file(sample_ids_file)
        if sample_ids is None:
            sample_ids = self._discover_sample_ids()
        self.sample_ids: List[str] = list(sample_ids)

        if len(self.sample_ids) == 0:
            raise RuntimeError(f"No samples found in root_dir={self.root_dir}.")

    def _discover_sample_ids(self) -> List[str]:
        """
        Find all sample folders inside `root_dir` and return their IDs.
        Handles the three subdirectories: colorful, high_quality, and high_quality_architectural.
        """
        discovered_ids: List[str] = []
        if not os.path.isdir(self.root_dir):
            return discovered_ids

        # Expected subdirectories in CubiCasa5k
        subdirs = ["colorful", "high_quality", "high_quality_architectural"]
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            for entry in os.listdir(subdir_path):
                full_path = os.path.join(subdir_path, entry)
                if not os.path.isdir(full_path):
                    continue

                # Check if this folder contains the required files
                image_path = os.path.join(full_path, self.image_file_name.lstrip('/'))
                svg_path = os.path.join(full_path, self.svg_file_name.lstrip('/'))
                
                # Check required files based on generate mode
                if self.generate:
                    # When generating, need SVG file
                    if os.path.exists(image_path) and os.path.exists(svg_path):
                        discovered_ids.append(os.path.join(subdir, entry))
                else:
                    # When loading from images, need points/ and ann/ directories
                    points_dir = os.path.join(full_path, "points")
                    ann_dir = os.path.join(full_path, "ann")
                    if os.path.exists(image_path) and os.path.exists(points_dir) and os.path.exists(ann_dir):
                        discovered_ids.append(os.path.join(subdir, entry))

        discovered_ids.sort()
        return discovered_ids

    def _load_ids_from_file(self, ids_file: str) -> List[str]:
        """
        Load sample IDs from a text file. Accepts lines like:
          /colorful/10052/
          colorful/10052
        Empty lines and lines starting with '#' are ignored.
        """
        ids: List[str] = []
        try:
            with open(ids_file, 'r') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Normalize: remove leading/trailing slashes
                    normalized = line.strip('/')
                    # Ensure it contains a subdir component and id
                    if normalized:
                        ids.append(normalized)
        except FileNotFoundError:
            raise FileNotFoundError(f"Sample IDs file not found: {ids_file}")
        # Validate that folders exist and contain required files
        valid_ids: List[str] = []
        for sid in ids:
            full_path = os.path.join(self.root_dir, sid)
            image_path = os.path.join(full_path, self.image_file_name.lstrip('/'))
            svg_path = os.path.join(full_path, self.svg_file_name.lstrip('/'))
            
            # Check required files based on generate mode
            if self.generate:
                # When generating, need SVG file
                if os.path.exists(full_path) and os.path.exists(image_path) and os.path.exists(svg_path):
                    valid_ids.append(sid)
            else:
                # When loading from images, need points/ and ann/ directories
                points_dir = os.path.join(full_path, "points")
                ann_dir = os.path.join(full_path, "ann")
                if os.path.exists(full_path) and os.path.exists(image_path) and os.path.exists(points_dir) and os.path.exists(ann_dir):
                    valid_ids.append(sid)
        if len(valid_ids) == 0:
            raise RuntimeError(f"No valid sample IDs found in file: {ids_file}")
        valid_ids.sort()
        return valid_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_floorplan_image(self, sample_id: str) -> np.ndarray:
        """
        Load the original floorplan image for a given sample.
        """
        image_path = os.path.join(self.root_dir, sample_id, self.image_file_name.lstrip('/'))
        # Use cv2.imread with IMREAD_UNCHANGED and check result directly
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found or failed to load: {image_path}")
        # OpenCV loads as BGR, convert to RGB in-place for efficiency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _read_svg_data(self, sample_id: str) -> np.ndarray:
        """
        Read SVG data for a given sample.
        """
        svg_path = os.path.join(self.root_dir, sample_id, self.svg_file_name.lstrip('/'))
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        
        image_path = os.path.join(self.root_dir, sample_id, self.image_file_name.lstrip('/'))
        image_height, image_width = cv2.imread(image_path).shape[:2]
        
        house = House(svg_path, image_height, image_width)

        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        
        return label

    def _remove_icons(self, label: np.ndarray, icon_percentage) -> np.ndarray:
        """
        Remove icons from the label.
        """
        num_icons = np.unique(label[1]).shape[0]
        num_icons_to_remove = int((num_icons-1) * (1-icon_percentage))
        removed_icons = random.sample(range(1, num_icons), num_icons_to_remove)
        for icon in removed_icons:
            label[1][label[1] == icon] = 0
        return label

    def _ann_to_mask(self, label: np.ndarray) -> np.ndarray:
        """
        Load SVG annotations and convert to rasterized image.
        
        Args:
            sample_id: The sample ID to load annotations for
            icons_to_keep: List of icon IDs to keep. If None, all icons are kept.
        """
        svg_image = 255 * ((label[0] != 2) & (label[1] == 0)).numpy().astype(np.uint8)
        # svg_image = 255 * (label[0] == 11).numpy().astype(np.uint8)
        svg_image = np.stack([svg_image]*3, axis=-1)
        return svg_image


    def _load_modality(self, sample_id: str, modality_type: str, label: np.ndarray, furniture_pct: float = None) -> np.ndarray:
        """
        Load a specific modality for a given sample.
        
        Args:
            sample_id: The sample ID (can include subdirectory path)
            modality_type: Type of modality to load ("drawing", "gt_svg_annotations", or "lidar_points")
            label: SVG label data (used when generate=True)
            furniture_pct: Furniture percentage for file selection (used when generate=False)
            
        Returns:
            numpy array of the loaded modality
        """
        if modality_type == "drawing":
            return self._load_floorplan_image(sample_id)
        elif modality_type == "gt_svg_annotations":
            return self.load_anns(sample_id, furniture_pct, label)
        elif modality_type == "lidar_points":
            return self.load_points(sample_id, furniture_pct, label)
        else:
            raise ValueError(f"Unknown modality type: {modality_type}. Available modalities: {self.modalities}")
    
    # def _load_svg_gt(self, sample_id: str) -> np.ndarray:
    #     """
    #     Load SVG annotations and convert to rasterized image.
    #     """
    #     svg_path = os.path.join(self.root_dir, sample_id, self.svg_file_name.lstrip('/'))
    #     if not os.path.exists(svg_path):
    #         raise FileNotFoundError(f"SVG file not found: {svg_path}")

    #     # Match the rasterization size to the underlying floorplan image
    #     image_path = os.path.join(self.root_dir, sample_id, self.image_file_name.lstrip('/'))
    #     if not os.path.exists(image_path):
    #         raise FileNotFoundError(f"Image file not found: {image_path}")
    #     image_height, image_width = cv2.imread(image_path).shape[:2]

    #     # Rasterize SVG to the exact image dimensions
    #     img_bytes = cairosvg.svg2png(
    #         url=svg_path,
    #         output_width=image_width,
    #         output_height=image_height,
    #     )

    #     # Convert to PIL and composite on white background to remove transparency
    #     pil_img = Image.open(io.BytesIO(img_bytes))

    #     # Convert to NumPy (H, W, 3) aligned with the floorplan image
    #     svg_image = np.array(pil_img)
    #     return svg_image

    def _ann_to_points(
        self,
        label: np.ndarray,
        points_density_inside_mask: float = 0.02,
        noise_std_pixels: float = 10.0
    ) -> np.ndarray:
        """
        Simulate a LiDAR points modality from the binary SVG mask.

        Efficiently samples random pixels inside the floorplan mask and adds small
        Gaussian noise before rasterizing as a points image.

        Args:
            sample_id: Dataset sample identifier
            points_density_inside_mask: Fraction of mask pixels to sample (0..1)
            noise_std_pixels: Standard deviation of jitter (in pixels)
            point_color: RGB color for points

        Returns:
            Numpy uint8 image (H, W, 3) with points drawn on white background
        """
        # Build mask from SVG annotations (3-channel uint8 where mask is 255)
        svg_ann = self._ann_to_mask(label)

        mask = svg_ann[:, :, 0] == 0
        height, width = mask.shape

        # Early exit with blank if mask empty
        mask_area = int(mask.sum())
        if mask_area == 0:
            return np.full((height, width, 3), 255, dtype=np.uint8)

        # Compute how many points to sample (cap to mask size)
        # Density is a fraction of mask pixels; clamp to [1, mask_area]
        target_points = int(points_density_inside_mask * mask_area)
        target_points = max(1, min(target_points, mask_area))

        # Vectorized sampling of coordinates inside the mask
        ys, xs = np.where(mask)

        # Build a spatially-varying density map by blurring random noise
        # This produces low-frequency variations in density over the mask
        noise_field = np.random.rand(height, width).astype(np.float32)
        # Choose a blur sigma relative to image size to control smoothness
        sigma = max(1.0, 0.08 * float(min(height, width)))
        density_field = cv2.GaussianBlur(noise_field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Normalize to [0,1]
        min_val = float(density_field.min())
        max_val = float(density_field.max())
        if max_val > min_val:
            density_field = (density_field - min_val) / (max_val - min_val)
        else:
            density_field.fill(1.0)
        # Randomly adjust contrast to vary density extremes
        gamma = float(np.random.uniform(1.6, 2.5))
        density_field = np.power(density_field + 1e-6, gamma).astype(np.float32)
        # Masked densities and convert to sampling probabilities
        masked_weights = density_field[ys, xs].astype(np.float64)
        # Ensure non-negative weights then normalize
        masked_weights = np.clip(masked_weights, 0.0, None)
        total_weight = float(masked_weights.sum())
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            # Fallback to uniform if something went wrong
            probs = None
        else:
            probs = masked_weights / total_weight
            # Numerical stability: clip and renormalize, then force exact sum==1
            probs = np.clip(probs, 0.0, 1.0)
            s = float(probs.sum())
            if s == 0.0 or not np.isfinite(s):
                probs = None
            else:
                probs /= s
                # Force exact sum to 1.0 by adjusting last element
                if probs.size > 0:
                    residual = 1.0 - float(probs.sum())
                    probs[-1] = max(0.0, probs[-1] + residual)
                    # Final guard: renormalize if needed
                    s2 = float(probs.sum())
                    if s2 > 0.0 and np.isfinite(s2):
                        probs /= s2
                    else:
                        probs = None

        # Sample indices with probability proportional to local density
        if target_points == mask_area:
            base_coords_idx = np.arange(mask_area)
        else:
            base_coords_idx = np.random.choice(mask_area, size=target_points, replace=False, p=probs)
        y_base = ys[base_coords_idx].astype(np.float32)
        x_base = xs[base_coords_idx].astype(np.float32)

        # Add Gaussian noise; 25% of points use a slightly higher std (e.g., 1.2x)
        if noise_std_pixels > 0:
            num_points = y_base.shape[0]
            # Base noise
            y_noise = np.random.normal(0.0, noise_std_pixels, size=num_points).astype(np.float32)
            x_noise = np.random.normal(0.0, noise_std_pixels, size=num_points).astype(np.float32)
            # Indices for higher std subset
            if num_points > 0:
                outliers_percentage = np.random.uniform(0.1, 0.3)
                high_std_count = max(1, int(outliers_percentage * num_points))
                high_idx = np.random.choice(num_points, size=high_std_count, replace=False)
                scale = np.float32(5)
                y_noise[high_idx] *= scale
                x_noise[high_idx] *= scale
            y_base += y_noise
            x_base += x_noise

        # Filter points that are within image bounds
        valid_mask = (y_base >= 0) & (y_base < height) & \
                     (x_base >= 0) & (x_base < width)
        y_pts = y_base[valid_mask].astype(np.int32)
        x_pts = x_base[valid_mask].astype(np.int32)


        # Rasterize points to an image (white background)
        img = np.full((height, width, 3), 255, dtype=np.uint8)

        if y_pts.size > 0:
            # Draw points as small circles (more efficient than individual point drawing)
            for point in zip(x_pts, y_pts):
                cv2.circle(img, tuple(point), 1, (0, 0, 0), -1)

        return img

    def _generate_anns(self, unused_sample_id: str, furniture_pct: float, label: np.ndarray) -> np.ndarray:
        """
        Load SVG annotations.
        """
        icon_percentage = furniture_pct if furniture_pct is not None else random.uniform(0.0, 1.0)
        label = self._remove_icons(label, icon_percentage)
        return self._ann_to_mask(label)

    def _generate_points(self, unused_sample_id: str, furniture_pct: float, label: np.ndarray) -> np.ndarray:
        """
        Generate LiDAR points.
        """
        icon_percentage = furniture_pct if furniture_pct is not None else random.uniform(0.0, 1.0)
        label = self._remove_icons(label, icon_percentage)
        return self._ann_to_points(label)

    def _load_points_from_image(self, sample_id: str, furniture_pct: float, unused_label: np.ndarray) -> np.ndarray:
        """
        Load LiDAR points from a pre-generated PNG file.
        
        Args:
            sample_id: The sample ID to load points for
            furniture_pct: Furniture percentage (0.0 to 1.0) to determine which file to load
            
        Returns:
            numpy array of the loaded points image
        """
        points_path = os.path.join(self.root_dir, sample_id, "points", f"f{int(furniture_pct*100)}.png")
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"Points file not found: {points_path}")
        
        # Load the image using cv2
        image = cv2.imread(points_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load points image: {points_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_anns_from_image(self, sample_id: str, furniture_pct: float, unused_label: np.ndarray) -> np.ndarray:
        """
        Load SVG annotations from a pre-generated PNG file.
        
        Args:
            sample_id: The sample ID to load annotations for
            furniture_pct: Furniture percentage (0.0 to 1.0) to determine which file to load
            
        Returns:
            numpy array of the loaded annotations image
        """
        ann_path = os.path.join(self.root_dir, sample_id, "ann", f"f{int(furniture_pct*100)}.png")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")
        
        # Load the image using cv2
        image = cv2.imread(ann_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load annotations image: {ann_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, index: int):
        timings = {}
        t0 = time.time()

        sample_id = self.sample_ids[index]
        sample_dir = os.path.join(self.root_dir, sample_id)
        timings['sample_id_and_dir'] = time.time() - t0

        # Randomly select a modality pair
        t1 = time.time()
        m0_type, m1_type = random.choice(self.modality_pairs)
        timings['modality_pair_selection'] = time.time() - t1

        # Load both modalities
        # Choose what icons to keep (same for both modalities)
        svg_data = None
        if self.furniture_pct is None:
            furniture_pct = random.uniform(0.0, 1.0)
        else:
            furniture_pct = np.random.choice(self.furniture_pct)
        if self.generate and (m0_type in ["gt_svg_annotations", "lidar_points"] or m1_type in ["gt_svg_annotations", "lidar_points"]):
            svg_data = self._read_svg_data(sample_id)

        m0_image = self._load_modality(sample_id, m0_type, svg_data, furniture_pct)
        m1_image = self._load_modality(sample_id, m1_type, svg_data, furniture_pct)


        # Apply transforms if specified
        t4 = time.time()
        if self.dual_transform is not None:
            for t in self.dual_transform:
                m0_image, m1_image = t(m0_image, m1_image)

        timings['transform_modalities'] = time.time() - t4

        timings['total'] = time.time() - t0

        # Optionally, print timings for debugging (comment out if not needed)
        # print(f"Timings Sample {sample_id}:")
        # for k, v in timings.items():
        #     print(f"{k}: {v:.3f}s")
        
        return {
            "modality_0": m0_image,
            "modality_1": m1_image,
            "m0_type": m0_type,
            "m1_type": m1_type,
            "sample_id": sample_id,
            "sample_dir": sample_dir,
        }

    def generate_and_save(self, index: int, furniture_pct: float) -> Dict[str, str]:
        """
        Generate and save annotations and points images for a given sample and furniture percentage.
        
        Args:
            index: Index of the sample in the dataset
            furniture_pct: Furniture percentage (0.0 to 1.0) to use for generation
            
        Returns:
            Dictionary with paths to the saved files
        """
        sample_id = self.sample_ids[index]
        sample_dir = os.path.join(self.root_dir, sample_id)
        
        # Create directories if they don't exist
        ann_dir = os.path.join(sample_dir, "ann")
        points_dir = os.path.join(sample_dir, "points")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(points_dir, exist_ok=True)
        
        # Read SVG data for generation
        svg_data = self._read_svg_data(sample_id)
        
        # Generate annotations (gt_svg_annotations)
        label_ann = self._remove_icons(svg_data, furniture_pct)
        ann_image = self._ann_to_mask(label_ann)
        
        # Generate points (lidar_points)
        points_image = self._ann_to_points(label_ann)
        
        # Define file paths
        ann_path = os.path.join(ann_dir, f"f{int(furniture_pct*100)}.png")
        points_path = os.path.join(points_dir, f"f{int(furniture_pct*100)}.png")
        
        # Save images
        # Convert RGB to BGR for OpenCV
        ann_bgr = cv2.cvtColor(ann_image, cv2.COLOR_RGB2BGR)
        points_bgr = cv2.cvtColor(points_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(ann_path, ann_bgr)
        cv2.imwrite(points_path, points_bgr)
        
        return {
            "sample_id": sample_id,
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
        sample_id = self.sample_ids[index]
        
        # Get all furniture percentages to load
        if self.furniture_pct is None:
            furniture_pcts = [random.uniform(0.0, 1.0)]
        else:
            furniture_pcts = self.furniture_pct
        
        # Load SVG data if needed for generation
        svg_data = None
        if self.generate and any(mod in ["gt_svg_annotations", "lidar_points"] for mod in self.modalities):
            svg_data = self._read_svg_data(sample_id)
        
        modalities_dict = {}
        
        # Load all modalities for this scene with all furniture percentages
        for modality_type in self.modalities:
            modality_dict = {}
            
            for furniture_pct in furniture_pcts:
                try:
                    modality_image = self._load_modality(sample_id, modality_type, svg_data, furniture_pct)
                    
                    # Apply transforms if specified
                    if self.dual_transform is not None:
                        # Convert to PIL for transforms
                        pil_image = Image.fromarray(modality_image, mode='RGB')
                        modality_image = pil_image
                        for t in self.dual_transform:
                            modality_image = t(modality_image)
                    
                    # Use furniture percentage as key (convert to int for cleaner keys)
                    furniture_key = f"{int(furniture_pct*100)}"
                    modality_dict[furniture_key] = modality_image
                    
                except Exception as e:
                    print(f"Warning: Failed to load {modality_type} with furniture_pct {furniture_pct} for sample {sample_id}: {e}")
                    # Skip this furniture percentage if it fails to load
                    continue
            
            modalities_dict[modality_type] = modality_dict
        
        return modalities_dict