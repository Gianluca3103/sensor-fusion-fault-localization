"""Loading and dataset adapter for coarse reconstruction samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Fault_Localization_Model.sample_utils import InvalidSampleError
from PFS_Radar.radar_data import radar_cache_path

from .fault_selector import FaultSelectorConfig
from .fault_selector_cache import (
    load_selector_cache,
    selector_cache_path,
)


def load_bev_triplet(
    sample_path: str | Path,
    radar_root: str | Path,
) -> dict[str, object]:
    """Load one aligned, normalized BEV triplet from existing artifacts."""

    sample_path = Path(sample_path)
    radar_root = Path(radar_root)
    try:
        with np.load(sample_path, allow_pickle=False) as sample:
            clean = np.asarray(sample["clean_rgb"])
            faulty = np.asarray(sample["faulty_rgb"])
            metadata = json.loads(str(sample["metadata_json"]))
            observability = (
                np.asarray(sample["observability_confidence"], dtype=np.float32)
                if "observability_confidence" in sample.files
                else None
            )
        radar_path = radar_cache_path(radar_root, metadata)
        with np.load(radar_path, allow_pickle=False) as radar_cache:
            radar = np.asarray(radar_cache["radar_bev"], dtype=np.float32)
    except InvalidSampleError:
        raise
    except Exception as exc:
        raise InvalidSampleError(
            f"Cannot load aligned BEV triplet for {sample_path}: {exc}"
        ) from exc

    item = {
        "clean_bev": torch.from_numpy(
            clean.astype(np.float32).transpose(2, 0, 1) / 255.0
        ),
        "radar_bev": torch.from_numpy(radar),
        "faulty_bev": torch.from_numpy(
            faulty.astype(np.float32).transpose(2, 0, 1) / 255.0
        ),
        "sample_path": str(sample_path),
    }
    if observability is not None:
        if observability.shape != clean.shape[:2]:
            raise InvalidSampleError(
                "observability_confidence must align with the LiDAR BEV; "
                f"got {observability.shape} and {clean.shape[:2]}"
            )
        item["observability_confidence"] = torch.from_numpy(observability)[None]
    return item


class CoarseReconstructionDataset(Dataset):
    """Load aligned BEVs and precomputed Stage-I oracle masks."""

    def __init__(
        self,
        sample_paths,
        radar_root: str | Path,
        *,
        data_root: str | Path,
        selector_config: FaultSelectorConfig | None = None,
    ):
        self.sample_paths = tuple(Path(path) for path in sample_paths)
        if not self.sample_paths:
            raise FileNotFoundError("No coarse reconstruction samples were provided")
        self.radar_root = Path(radar_root)
        self.data_root = Path(data_root)
        self.selector_config = selector_config or FaultSelectorConfig()

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = load_bev_triplet(
            self.sample_paths[index],
            self.radar_root,
        )
        cache_path = selector_cache_path(
            self.sample_paths[index],
            self.data_root,
        )
        cached = load_selector_cache(
            cache_path,
            self.selector_config,
        )
        item.update(
            {
                "reconstruction_mask": torch.from_numpy(
                    cached["reconstruction_mask"]
                )[None],
                "healthy_context_mask": torch.from_numpy(
                    cached["healthy_context_mask"]
                )[None],
                "halo_mask": torch.from_numpy(
                    cached["halo_mask"]
                )[None],
            }
        )
        return item
