from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from PFS_Radar.radar_data import radar_cache_path
from Fault_Localization_Model.model_blocks import resize_reliability_map
from Fault_Localization_Model.sample_utils import (
    InvalidSampleError,
    validate_heatmap_array,
    validate_radar_array,
    validate_rgb_array,
)


def _resize(tensor, size, mode):
    if size is None or tensor.shape[-2:] == tuple(size):
        return tensor
    options = {"size": tuple(size), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        options["align_corners"] = False
    return F.interpolate(tensor[None], **options)[0]


def _load_sample(path: Path, radar_root: Path, resize_hw, include_clean):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"faulty_rgb", "fault_heatmap", "metadata_json"}
            if include_clean:
                required.add("clean_rgb")
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"missing arrays: {', '.join(sorted(missing))}")

            faulty_rgb = validate_rgb_array(
                data["faulty_rgb"], name="faulty_rgb", path=path
            )
            target_array = validate_heatmap_array(data["fault_heatmap"], path=path)
            metadata_json = str(data["metadata_json"])
            clean_rgb = (
                validate_rgb_array(data["clean_rgb"], name="clean_rgb", path=path)
                if include_clean
                else None
            )
    except InvalidSampleError:
        raise
    except Exception as exc:
        raise InvalidSampleError(f"Cannot load reliability sample {path}: {exc}") from exc

    if clean_rgb is not None and clean_rgb.shape != faulty_rgb.shape:
        raise InvalidSampleError(
            f"faulty_rgb and clean_rgb in {path} have different shapes: "
            f"{faulty_rgb.shape} and {clean_rgb.shape}"
        )
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise InvalidSampleError(f"Invalid metadata_json in {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise InvalidSampleError(f"metadata_json in {path} must decode to an object")

    cache_path = radar_cache_path(radar_root, metadata)
    if not cache_path.exists():
        raise FileNotFoundError(f"Radar cache missing for {path}: {cache_path}")
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "radar_bev" not in data:
                raise KeyError("radar_bev is missing")
            radar = validate_radar_array(data["radar_bev"], path=cache_path)
    except InvalidSampleError:
        raise
    except Exception as exc:
        raise InvalidSampleError(f"Cannot load radar cache {cache_path}: {exc}") from exc

    faulty = faulty_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    target = target_array.astype(np.float32)[None]
    clean = (
        clean_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        if clean_rgb is not None
        else None
    )
    radar = radar.astype(np.float32)
    faulty_tensor = _resize(torch.from_numpy(faulty), resize_hw, "bilinear")
    radar_tensor = _resize(torch.from_numpy(radar), resize_hw, "bilinear")
    target_tensor = resize_reliability_map(
        torch.from_numpy(target)[None],
        resize_hw,
    )[0]
    clean_tensor = (
        _resize(torch.from_numpy(clean), resize_hw, "bilinear")
        if clean is not None
        else None
    )
    return faulty_tensor, radar_tensor, clean_tensor, target_tensor, metadata_json


class _BaseRadarDataset(Dataset):
    def __init__(self, paths, radar_root: Path, resize_hw=(320, 320)):
        self.paths = list(paths)
        self.radar_root = Path(radar_root)
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        if not self.paths:
            raise FileNotFoundError("No reliability-map samples were provided")

    def __len__(self):
        return len(self.paths)


class RadarReliabilityDataset(_BaseRadarDataset):
    """Training dataset with the clean LiDAR feature-stabilization reference."""

    def __getitem__(self, index):
        return _load_sample(
            self.paths[index],
            self.radar_root,
            self.resize_hw,
            include_clean=True,
        )


class RadarEvaluationDataset(_BaseRadarDataset):
    """Evaluation dataset without the training-only clean LiDAR tensor."""

    def __getitem__(self, index):
        faulty, radar, _, target, metadata_json = _load_sample(
            self.paths[index],
            self.radar_root,
            self.resize_hw,
            include_clean=False,
        )
        return faulty, radar, target, metadata_json
