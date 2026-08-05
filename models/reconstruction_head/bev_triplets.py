
"""Validated loading of aligned clean-LiDAR, radar, and faulty-LiDAR BEVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from Fault_Localization_Model.sample_utils import (
    InvalidSampleError,
    validate_heatmap_array,
    validate_radar_array,
    validate_rgb_array,
)
from Fault_Localization_Model.model_blocks import resize_reliability_map
from PFS_Radar.radar_data import radar_cache_path
from .encoders import mask_unreliable_lidar


def _resize(tensor: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tensor.shape[-2:] == size:
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def load_bev_triplet(
    sample_path: str | Path,
    radar_root: str | Path,
    *,
    resize_hw: tuple[int, int] | None = (320, 320),
) -> dict[str, object]:
    """Load one aligned, normalized BEV triplet from existing pipeline artifacts."""

    sample_path = Path(sample_path)
    radar_root = Path(radar_root)
    try:
        with np.load(sample_path, allow_pickle=False) as sample:
            required = {
                "clean_rgb",
                "faulty_rgb",
                "fault_heatmap",
                "reliability_map",
                "faulty_counts",
                "added_faulty_counts",
                "missing_faulty_counts",
                "moved_faulty_counts",
                "metadata_json",
            }
            missing = required - set(sample.files)
            if missing:
                raise KeyError(f"missing arrays: {', '.join(sorted(missing))}")
            clean = validate_rgb_array(
                sample["clean_rgb"], name="clean_rgb", path=sample_path
            )
            faulty = validate_rgb_array(
                sample["faulty_rgb"], name="faulty_rgb", path=sample_path
            )
            fault_heatmap = validate_heatmap_array(
                sample["fault_heatmap"], path=sample_path
            )
            reliability_map = validate_heatmap_array(
                sample["reliability_map"], path=sample_path
            )
            evidence_counts = {
                name: np.asarray(sample[name], dtype=np.float32)
                for name in (
                    "added_faulty_counts",
                    "missing_faulty_counts",
                    "moved_faulty_counts",
                    "faulty_counts",
                )
            }
            metadata_json = str(sample["metadata_json"])
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise TypeError("metadata_json must decode to an object")
        radar_path = radar_cache_path(radar_root, metadata)
        with np.load(radar_path, allow_pickle=False) as radar_cache:
            if "radar_bev" not in radar_cache:
                raise KeyError("radar_bev is missing")
            radar = validate_radar_array(radar_cache["radar_bev"], path=radar_path)
    except InvalidSampleError:
        raise
    except Exception as exc:
        raise InvalidSampleError(
            f"Cannot load aligned BEV triplet for {sample_path}: {exc}"
        ) from exc

    if clean.shape != faulty.shape:
        raise InvalidSampleError(
            f"clean_rgb and faulty_rgb in {sample_path} have different shapes: "
            f"{clean.shape} and {faulty.shape}"
        )
    output_size = tuple(resize_hw) if resize_hw is not None else None
    clean_tensor = torch.from_numpy(
        clean.astype(np.float32).transpose(2, 0, 1) / 255.0
    )
    faulty_tensor = torch.from_numpy(
        faulty.astype(np.float32).transpose(2, 0, 1) / 255.0
    )
    radar_tensor = torch.from_numpy(radar.astype(np.float32))
    heatmap_tensor = torch.from_numpy(fault_heatmap.astype(np.float32))[None]
    reliability_tensor = torch.from_numpy(reliability_map)[None]
    evidence_tensors = {
        name: torch.from_numpy(values)[None]
        for name, values in evidence_counts.items()
    }
    if output_size is not None:
        heatmap_tensor = resize_reliability_map(
            heatmap_tensor[None], output_size
        )[0]
        reliability_tensor = resize_reliability_map(
            reliability_tensor[None], output_size
        )[0]
        evidence_tensors = {
            name: F.interpolate(
                tensor[None], size=output_size, mode="nearest"
            )[0]
            for name, tensor in evidence_tensors.items()
        }
    resized_clean = _resize(clean_tensor, output_size)
    resized_faulty = _resize(faulty_tensor, output_size)
    lidar_trusted = mask_unreliable_lidar(
        resized_faulty[None], reliability_tensor[None]
    )[0]
    return {
        "clean_bev": resized_clean,
        "radar_bev": _resize(radar_tensor, output_size),
        "faulty_bev": resized_faulty,
        "lidar_trusted_bev": lidar_trusted,
        "fault_heatmap": heatmap_tensor,
        "reliability_map": reliability_tensor,
        **evidence_tensors,
        "metadata_json": metadata_json,
        "sample_path": str(sample_path),
        "radar_path": str(radar_path),
    }


class BEVTripletDataset(Dataset):
    """Dataset exposing the aligned inputs required by stage two."""

    def __init__(
        self,
        sample_paths,
        radar_root: str | Path,
        *,
        resize_hw: tuple[int, int] | None = (320, 320),
    ):
        self.sample_paths = tuple(Path(path) for path in sample_paths)
        if not self.sample_paths:
            raise FileNotFoundError("No BEV triplet samples were provided")
        self.radar_root = Path(radar_root)
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        return load_bev_triplet(
            self.sample_paths[index],
            self.radar_root,
            resize_hw=self.resize_hw,
        )
