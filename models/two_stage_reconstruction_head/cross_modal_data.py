"""Full-scan VoD inputs for radar-to-LiDAR representation learning.

The released radar stacks remain in radar coordinates. This loader aligns all
seven radar fields to the current LiDAR frame, preserving time and Doppler.
Clean LiDAR is opened only when ``include_clean`` is enabled for training.
The optional radar height gate uses only the faulty LiDAR observed by the model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_lidar,
    load_vod_radar,
    load_vod_radar_to_lidar,
)
from .voxelization.inputs import read_sample_metadata


def observed_lidar_height_mask(
    radar_in_lidar: np.ndarray, observed_lidar: np.ndarray,
) -> np.ndarray:
    """Keep radar within the observed LiDAR's inclusive per-frame Z envelope.

    Both point sets must already be in the current LiDAR coordinate system.
    If a fault leaves fewer than two LiDAR returns, no useful envelope exists;
    retain radar so complete LiDAR loss can still be reconstructed from it.
    """
    radar = np.asarray(radar_in_lidar)
    lidar = np.asarray(observed_lidar)
    if radar.ndim != 2 or radar.shape[1] < 3:
        raise ValueError("Radar must be a [N,3+] array in LiDAR coordinates")
    if lidar.ndim != 2 or lidar.shape[1] < 3:
        raise ValueError("Observed LiDAR must be a [N,3+] array")
    if not np.isfinite(radar[:, :3]).all() or not np.isfinite(lidar[:, :3]).all():
        raise ValueError("Radar and observed LiDAR XYZ must be finite")
    if len(lidar) < 2:
        return np.ones(len(radar), dtype=bool)
    lowest = float(lidar[:, 2].min())
    highest = float(lidar[:, 2].max())
    return (radar[:, 2] >= lowest) & (radar[:, 2] <= highest)


class CrossModalVoDDataset(Dataset):
    """Pair existing full-scan faulty samples with verified raw VoD radar."""

    def __init__(
        self,
        sample_paths: list[str | Path],
        vod_root: str | Path,
        *,
        radar_variant: str = "radar_20frames_verified_doppler_radial",
        include_clean: bool = False,
        radar_height_filter: bool = False,
    ) -> None:
        self.paths = [Path(path) for path in sample_paths]
        self.include_clean = include_clean
        self.radar_height_filter = radar_height_filter
        identifiers: dict[str, set[str]] = defaultdict(set)
        self.keys = []
        for path in self.paths:
            metadata = read_sample_metadata(path)
            if str(metadata.get("dataset", "")).strip().lower() not in {
                "view-of-delft", "view of delft", "vod"
            }:
                raise ValueError(f"Expected a VoD full-scan artifact: {path}")
            split = str(metadata.get("split", ""))
            frame_id = str(metadata.get("frame_id", "")).zfill(5)
            if split not in {"train", "val", "test"} or not frame_id.isdigit():
                raise ValueError(f"Invalid split or frame ID in {path}")
            if not metadata.get("range_view_full_scan", False):
                raise ValueError(f"Cross-modal encoding requires a full scan: {path}")
            key = (split, frame_id)
            self.keys.append(key)
            identifiers[split].add(frame_id)
        self.frames = {}
        for split, frame_ids in identifiers.items():
            frames = discover_vod_frames(
                vod_root, split, radar_variant=radar_variant,
                frame_ids=sorted(frame_ids),
            )
            self.frames.update({(split, frame.frame_id): frame for frame in frames})

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        frame = self.frames[self.keys[index]]
        with np.load(path, allow_pickle=False) as archive:
            if "faulty_lidar_points" not in archive.files:
                raise ValueError(f"No faulty LiDAR in {path}")
            faulty = np.asarray(archive["faulty_lidar_points"][:, :4], dtype=np.float32)
        if faulty.ndim != 2 or faulty.shape[1] != 4 or not np.isfinite(faulty).all():
            raise ValueError(f"Malformed faulty LiDAR in {path}")
        radar = load_vod_radar(frame.radar_path)
        lidar_from_radar = load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path,
        )
        radar = align_radar_to_lidar(radar, lidar_from_radar)
        if self.radar_height_filter:
            radar = radar[observed_lidar_height_mask(radar, faulty)]
        result = {
            "radar": torch.from_numpy(radar),
            "observed_lidar": torch.from_numpy(faulty),
            "split": frame.split,
            "frame_id": frame.frame_id,
            "sample_path": str(path),
        }
        if self.include_clean:
            result["clean_lidar"] = torch.from_numpy(load_vod_lidar(frame.lidar_path))
        return result


def collate_cross_modal(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if len({"clean_lidar" in item for item in batch}) != 1:
        raise ValueError("Cannot mix teacher and inference examples")
    result = {
        "split": [item["split"] for item in batch],
        "frame_id": [item["frame_id"] for item in batch],
        "sample_path": [item["sample_path"] for item in batch],
    }
    for name in ("radar", "observed_lidar", "clean_lidar"):
        if name not in batch[0]:
            continue
        maximum = max(len(item[name]) for item in batch)
        columns = batch[0][name].shape[-1]
        values = torch.zeros(len(batch), maximum, columns, dtype=torch.float32)
        valid = torch.zeros(len(batch), maximum, dtype=torch.bool)
        for index, item in enumerate(batch):
            count = len(item[name])
            values[index, :count] = item[name]
            valid[index, :count] = True
        result[name] = values
        result[f"{name}_valid"] = valid
    return result
