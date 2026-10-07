"""Pair official VoD radar/clean frames with the exact cached faulty scan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1


def within_fault_region(xyz: np.ndarray, region: dict) -> np.ndarray:
    """The cache injected faults only after this range and BEV crop."""
    if region.get("full_scan") is True:
        return np.isfinite(xyz).all(axis=1)
    r = np.linalg.norm(xyz, axis=1)
    x0, x1 = region["x_range"]
    y0, y1 = region["y_range"]
    return ((r >= region["min_range_m"]) & (r <= region["max_range_m"])
            & (xyz[:, 0] >= x0) & (xyz[:, 0] < x1)
            & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1))


def fault_region_from_metadata(metadata: dict, path: str | Path) -> dict:
    """Accept cropped BEV artifacts and explicitly marked full-scan artifacts.

    A missing point_filter alone cannot establish that the faulty scan is
    complete. The range-view cache intentionally removes this field after
    regenerating faults on the entire original LiDAR scan.
    """
    region = metadata.get("point_filter")
    if region is not None:
        if not isinstance(region, dict) or not all(k in region for k in
               ("x_range", "y_range", "min_range_m", "max_range_m")):
            raise ValueError(f"Invalid cache point_filter metadata: {path}")
        return region
    if metadata.get("range_view_full_scan") is True:
        return {"full_scan": True}
    raise ValueError(f"Cannot determine faulty LiDAR spatial coverage: {path}")


class PairedFaultDataset(Dataset):
    def __init__(self, vod_root: str | Path, samples_root: str | Path, split: str,
                 *, radar_variant: str, fault_pattern: str = "*", frame_ids: list[str] | None = None):
        self.base = VoDStage1Dataset(vod_root, split, radar_variant=radar_variant,
                                     frame_ids=frame_ids, include_clean=True)
        root = Path(samples_root) / split
        self.paths = []
        for frame in self.base.frames:
            matches = sorted(root.glob(f"{int(frame.frame_id):05d}_{fault_pattern}.npz"))
            if len(matches) != 1:
                raise ValueError(f"Expected one faulty cache sample for {split}/{frame.frame_id}; "
                                 f"found {len(matches)} in {root}. Select --fault-pattern if needed.")
            self.paths.append(matches[0])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as archive:
            points = np.asarray(archive["faulty_lidar_points"], dtype=np.float32)[:, :4]
            meta = json.loads(str(archive["metadata_json"].item()))
        if (str(meta.get("frame_id", "")).zfill(5) != sample["frame_id"]
                or meta.get("split") != sample["split"]):
            raise ValueError(f"Fault cache/frame mismatch: {path}")
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError(f"Invalid faulty XYZI: {path}")
        region = fault_region_from_metadata(meta, path)
        clean = sample["clean_lidar"].numpy()
        sample["clean_lidar"] = torch.from_numpy(clean[within_fault_region(clean[:, :3], region)].copy())
        if not bool(within_fault_region(points[:, :3], region).all()):
            raise ValueError(f"Faulty points escaped cache point_filter: {path}")
        sample["faulty_lidar"] = torch.from_numpy(points.copy())
        sample["fault_region"] = region
        sample["fault_kind"] = meta.get("fault", "unknown")
        return sample


def collate_paired(batch: list[dict]) -> dict:
    result = collate_stage1(batch)
    maximum = max(len(row["faulty_lidar"]) for row in batch)
    faulty = torch.zeros((len(batch), maximum, 4), dtype=torch.float32)
    valid = torch.zeros((len(batch), maximum), dtype=torch.bool)
    for i, row in enumerate(batch):
        count = len(row["faulty_lidar"])
        faulty[i, :count] = row["faulty_lidar"]
        valid[i, :count] = True
    result.update(faulty_lidar=faulty, faulty_lidar_valid=valid,
                  fault_region=[row["fault_region"] for row in batch],
                  fault_kind=[row["fault_kind"] for row in batch])
    return result
