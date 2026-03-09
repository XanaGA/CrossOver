from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import cv2
import torch
from torch.utils.data import Dataset

from dataloading.dual_transforms import PairNormalize, PairRandomAffine

# Local datasets
from .cubicasa_data import Cubicasa5kDataset
from .s3d_data import Structured3DDataset
from .aria_se_data import AriaSynthEenvDataset
from .swiss_dw_data import SwissDwellingsDataset
from .scannet_data import ScannetDataset
from .zillow_data import ZillowDataset  # type: ignore[import-error]
import numpy as np


class _DatasetRegistry:
    """
    Simple registry to map string keys to Dataset classes.

    Add new datasets by calling `register` or extending the default mapping below.
    """

    def __init__(self) -> None:
        self._name_to_cls: Dict[str, Type[Dataset]] = {}

    def register(self, name: str, dataset_cls: Type[Dataset]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Dataset registry name must be non-empty")
        if not issubclass(dataset_cls, Dataset):
            raise TypeError("dataset_cls must be a subclass of torch.utils.data.Dataset")
        self._name_to_cls[key] = dataset_cls

    def get(self, name: str) -> Type[Dataset]:
        key = name.strip().lower()
        if key not in self._name_to_cls:
            raise KeyError(f"Dataset type '{name}' is not registered. Registered: {sorted(self._name_to_cls.keys())}")
        return self._name_to_cls[key]

    def registered(self) -> Tuple[str, ...]:
        return tuple(sorted(self._name_to_cls.keys()))


_REGISTRY = _DatasetRegistry()
_REGISTRY.register("cubicasa5k", Cubicasa5kDataset)
_REGISTRY.register("structured3d", Structured3DDataset)
_REGISTRY.register("aria_synthenv", AriaSynthEenvDataset)
_REGISTRY.register("swiss_dwellings", SwissDwellingsDataset)
_REGISTRY.register("scannet", ScannetDataset)
_REGISTRY.register("zillow", ZillowDataset)


class UnifiedDataset(Dataset):
    """
    Dataset wrapper that unifies multiple underlying datasets under a single iterable.

    You can pass either:
    - `datasets`: an explicit list of instantiated `Dataset` objects, or
    - `dataset_configs`: a list of configs, each like `{ "type": str, "args": dict }`,
      where `type` is a key registered in the internal registry and `args` are kwargs
      for that dataset constructor.

    Returns items from each child dataset sequentially (concatenated). The returned
    sample dict is augmented with `source_dataset` to identify the origin.

    Example:
        UnifiedDataset(
            dataset_configs=[
                {
                    "type": "cubicasa5k",
                    "args": {"root_dir": "/data/CubiCasa5k", "image_size": (256, 256)},
                },
                {
                    "type": "structured3d",
                    "args": {"root_dir": "/data/Structured3D", "image_size": (256, 256)},
                },
            ]
        )
    """

    def __init__(
        self,
        datasets: Optional[Sequence[Dataset]] = None,
        dataset_configs: Optional[Sequence[Dict[str, Any]]] = None,
        common_transform: Optional[Callable] = None,
        invertible_transform: Optional[Callable] = None,
        text_description: bool = False,
    ) -> None:
        super().__init__()

        t = common_transform[-1] if common_transform is not None else None
        if isinstance(t, PairNormalize):
            m = t.mean if isinstance(t.mean, torch.Tensor) else torch.tensor(t.mean)
            std = t.std if isinstance(t.std, torch.Tensor) else torch.tensor(t.std)
            self.norm_filler = ((1-m) / std)
        else:
            self.norm_filler = None
        
        

        self.common_transform: Optional[Callable] = common_transform
        self.invertible_transform: Optional[Callable] = invertible_transform
        self.text_description: bool = bool(text_description)

        if datasets is None and dataset_configs is None:
            raise ValueError("Provide either `datasets` or `dataset_configs`.")

        built_datasets: List[Tuple[str, Dataset]] = []

        if datasets is not None:
            for ds in datasets:
                if not isinstance(ds, Dataset):
                    raise TypeError("All elements in `datasets` must be instances of torch.utils.data.Dataset")
                # Try to infer a readable name from the class
                built_datasets.append((ds.__class__.__name__, ds))

        if dataset_configs is not None:
            for cfg in dataset_configs:
                if not isinstance(cfg, dict):
                    raise TypeError("Each dataset config must be a dict with keys 'type' and 'args'.")
                if "type" not in cfg:
                    raise KeyError("Dataset config missing required key 'type'.")
                ds_type = cfg["type"]
                ds_args = cfg.get("args", {}) or {}
                ds_cls = _REGISTRY.get(ds_type)
                ds_instance = ds_cls(**ds_args)
                # Use the registry key as the dataset name for consistency
                built_datasets.append((ds_type.strip().lower(), ds_instance))

        if len(built_datasets) == 0:
            raise ValueError("No datasets were provided or constructed.")

        self._datasets: List[Tuple[str, Dataset]] = built_datasets

        # Precompute cumulative lengths for index routing
        self._lengths: List[int] = [len(ds) for _, ds in self._datasets]
        self._cumulative: List[int] = list(itertools.accumulate(self._lengths))
        self._total_len: int = sum(self._lengths)

    def __len__(self) -> int:
        return self._total_len

    def _locate_dataset(self, global_index: int) -> Tuple[int, int]:
        if global_index < 0 or global_index >= self._total_len:
            raise IndexError("UnifiedDataset index out of range")
        # Find first cumulative length greater than index
        # Equivalent to bisect
        running = 0
        for i, length in enumerate(self._lengths):
            next_running = running + length
            if global_index < next_running:
                return i, global_index - running
            running = next_running
        # Should never reach here
        raise IndexError("UnifiedDataset index routing failed")

    def _furniture_pct_to_text(self, pct: Optional[float]) -> str:
        if pct is None:
            return "partially furnished"
        # Clamp to [0,1]
        p = max(0.0, min(1.0, float(pct)))
        # Map to natural language buckets
        if p == 0.0:
            return "empty"
        if p < 0.5:
            return "barely furnished"
        if p < 0.75:
            return "partially furnished"
        if p < 1.0:
            return "heavily furnished"
        return "fully furnished"

    def _modality_template(self,mod_type: str) -> str:
        mt = (mod_type or "").lower()
        if mt in ("drawing"):
            return "An architect drawing of a {furn} apartment."
        if mt in ("gt_svg_annotations", "svg", "annotations", "floorplan"):
            return "A 2D digital floorplan of a {furn} apartment. White background, black lines represent walls and furniture."
        if mt in ("lidar_points", "lidar", "points"):
            return "A orthographic view of a 2D pointcloud scanned {furn} apartment. White background, black points represent walls and furniture.."
        if mt in ("density_map", "density"):
            return "A density map of a {furn} apartment."
        return "An image of a {furn} apartment."

    def __getitem__(self, index: int):
        ds_idx, local_index = self._locate_dataset(index)
        source_name, dataset = self._datasets[ds_idx]
        item = dataset[local_index]
        # print(f"Getting item {index} from {source_name}")
        # print(f"Item: {item}")
        # Add origin marker; do not overwrite if child already provided one
        if isinstance(item, dict) and ("source_dataset" not in item):
            item["source_dataset"] = source_name

        if self.norm_filler is not None:
            item["norm_filler"] = self.norm_filler

        # Apply Common Transformations
        if self.common_transform is not None:
            m0 = item["modality_0"]
            m1 = item["modality_1"]
            og_m0, og_m1 = None, None
            for t in self.common_transform:
                if isinstance(t, PairRandomAffine):
                    og_m0, og_m1 = m0, m1
                    m0, m1, params = t(m0, m1, return_transform=True)
                    item["transform_params"] = params
                else:
                    m0, m1 = t(m0, m1)
                    if og_m0 is not None:
                        og_m0, og_m1 = t(og_m0, og_m1)

            item["modality_0"] = m0
            item["modality_1"] = m1

            if og_m0 is not None:
                item["original_modality_0"] = og_m0
                item["original_modality_1"] = og_m1


        # Add Inversible Transformations
        if self.invertible_transform is not None and len(self.invertible_transform) > 0:
            m0 = item["modality_0"]
            m1 = item["modality_1"]

            m0, m1, params = self.invertible_transform[0](m0, m1, return_transform=True)

            item["modality_0_noise"] = m0
            item["modality_1_noise"] = m1
            item["noise_params"] = params

        # Add text description if requested
        if self.text_description and isinstance(item, dict):
            raise NotImplementedError("Text description is not implemented for aria_synthenv")
            # m0_type = item.get("m0_type", None)
            # m1_type = item.get("m1_type", None)

            # furn_pct0 = item.get("furniture_pct", None)
            # furn_pct1 = item.get("furniture_pct", None) 
            # furn_text0 = "fully furnished" if (m0_type == "density_map") or (m0_type == "drawing") else self._furniture_pct_to_text(furn_pct0)
            # furn_text1 = "fully furnished" if (m1_type == "density_map") or (m1_type == "drawing") else self._furniture_pct_to_text(furn_pct1)

            # # Prefer modality_0's type for the single description
            # template0 = self._modality_template(m0_type if m0_type is not None else "")
            # template1 = self._modality_template(m1_type if m1_type is not None else "")
            # item["m0_description"] = template0.format(furn=furn_text0)
            # item["m1_description"] = template1.format(furn=furn_text1)

        return item

    def generate_and_save(self, index: int, furniture_pct: float) -> Dict[str, Any]:
        """
        Generate and save images for a given index and furniture percentage.
        
        Args:
            index: Global index in the unified dataset
            furniture_pct: Furniture percentage (0.0 to 1.0)
            
        Returns:
            Dictionary with generation results
        """
        # Locate the specific dataset and local index
        ds_idx, local_index = self._locate_dataset(index)
        source_name, dataset = self._datasets[ds_idx]
        
        # Call the appropriate generate_and_save method
        if hasattr(dataset, 'generate_and_save'):
            if source_name in ("aria_synthenv", "swiss_dwellings", "zillow"):
                # These datasets' generate_and_save do not take furniture_pct
                result = dataset.generate_and_save(local_index)
            else:
                result = dataset.generate_and_save(local_index, furniture_pct)
            result["source_dataset"] = source_name
            result["global_index"] = index
            result["local_index"] = local_index
            return result
        else:
            raise AttributeError(f"Dataset {source_name} does not have generate_and_save method")

    def get_all_sample_modalities(self, index: int):
        """
        Get all modalities for a given scene in self.modalities.
        Return a dictionary with the modality type as the key and the modality image as the value.
        Loads all furniture percentages from self.furniture_pct with different keys.
        """
        ds_idx, local_index = self._locate_dataset(index)
        source_name, dataset = self._datasets[ds_idx]
        if hasattr(dataset, 'get_all_sample_modalities'):
            result = dataset.get_all_sample_modalities(local_index)
            result["source_dataset"] = source_name
            result["global_index"] = index
            result["local_index"] = local_index
            return result
        else:
            raise AttributeError(f"Dataset {source_name} does not have get_all_sample_modalities method")


__all__ = [
    "UnifiedDataset",
    "register_dataset",
]


