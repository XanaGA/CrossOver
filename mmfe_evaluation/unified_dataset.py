"""
Simplified UnifiedDataset that wraps one or more sub-datasets under a single
iterable.  Designed for CrossOver evaluation on Structured3D but works with
any Dataset whose __getitem__ returns a dict.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from .s3d_data import Structured3DDataset

_REGISTRY = {
    "structured3d": Structured3DDataset,
}


class UnifiedDataset(Dataset):
    """
    Concatenating wrapper around one or more datasets.

    Accepts either pre-built ``datasets`` or declarative ``dataset_configs``
    (list of dicts with ``type`` and ``args`` keys).

    Example
    -------
    >>> UnifiedDataset(dataset_configs=[{
    ...     "type": "structured3d",
    ...     "args": {"root_dir": "/data/Structured3D"},
    ... }])
    """

    def __init__(
        self,
        datasets: Optional[Sequence[Dataset]] = None,
        dataset_configs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()

        if datasets is None and dataset_configs is None:
            raise ValueError("Provide either `datasets` or `dataset_configs`.")

        built: List[Tuple[str, Dataset]] = []

        if datasets is not None:
            for ds in datasets:
                built.append((ds.__class__.__name__, ds))

        if dataset_configs is not None:
            for cfg in dataset_configs:
                ds_type = cfg["type"].strip().lower()
                ds_args = cfg.get("args", {}) or {}
                cls = _REGISTRY.get(ds_type)
                if cls is None:
                    raise KeyError(
                        f"Unknown dataset type '{ds_type}'. "
                        f"Registered: {sorted(_REGISTRY)}"
                    )
                built.append((ds_type, cls(**ds_args)))

        if not built:
            raise ValueError("No datasets were provided or constructed.")

        self._datasets = built
        self._lengths = [len(ds) for _, ds in self._datasets]
        self._cumulative = list(itertools.accumulate(self._lengths))
        self._total_len = sum(self._lengths)

    def __len__(self) -> int:
        return self._total_len

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= self._total_len:
            raise IndexError("UnifiedDataset index out of range")
        running = 0
        for i, length in enumerate(self._lengths):
            if index < running + length:
                return i, index - running
            running += length
        raise IndexError("UnifiedDataset index routing failed")

    def __getitem__(self, index: int):
        ds_idx, local_idx = self._locate(index)
        name, dataset = self._datasets[ds_idx]
        item = dataset[local_idx]
        if isinstance(item, dict) and "source_dataset" not in item:
            item["source_dataset"] = name
        return item
