"""Build whole-scan range-view examples from existing reconstruction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..voxelization.inputs import load_aligned_point_inputs, load_clean_lidar_from_metadata
from .geometry import RangeGeometry, RangeProjection, project_lidar
from .radar import project_aligned_radar
from .targets import RangeTargets, build_range_targets


TARGET_KEYS = ("add", "add_range_m", "delete", "delete_valid", "clean_valid", "clean_range_m")


@dataclass(frozen=True)
class RangeSample:
    features: np.ndarray
    targets: RangeTargets
    faulty_projection: RangeProjection
    clean_projection: RangeProjection
    radar_features: np.ndarray
    faulty_points: np.ndarray
    clean_points: np.ndarray
    faulty_source_ids: np.ndarray
    metadata: dict
    sample_path: Path

    def tensors(self) -> dict[str, torch.Tensor]:
        result = {"features": torch.from_numpy(self.features)}
        for key in TARGET_KEYS:
            result[key] = torch.from_numpy(np.asarray(getattr(self.targets, key), dtype=np.float32))
        return result


def load_range_sample(
    sample_path: str | Path,
    radar_root: str | Path,
    geometry: RangeGeometry,
    *,
    fault_map_root: str | Path | None = None,
    range_tolerance_m: float = 0.2,
    point_tolerance_m: float = 0.05,
    forward_only: bool = True,
) -> RangeSample:
    sample_path = Path(sample_path)
    aligned = load_aligned_point_inputs(sample_path, radar_root, lidar_source="faulty")
    if not aligned.metadata.get("range_view_full_scan", False):
        raise ValueError(
            f"{sample_path} is a legacy cropped artifact. Generate full-scan range-view "
            "artifacts before training; otherwise reconstruction remains BEV-limited."
        )
    clean = load_clean_lidar_from_metadata(aligned.metadata)
    with np.load(sample_path, allow_pickle=False) as archive:
        if "faulty_source_ids" not in archive.files:
            raise ValueError(f"{sample_path} has no provenance IDs; regenerate the artifact")
        source_ids = np.asarray(archive["faulty_source_ids"], dtype=np.int64)
    faulty = aligned.lidar_points
    radar = aligned.radar_points
    if forward_only:
        # This is a sensor-FOV selection, not a fault-selector repair box.
        # Reindex raw-clean provenance after the same front-half selection.
        keep_clean = clean[:, 0] >= 0
        clean_index = np.full(len(clean), -1, dtype=np.int64)
        clean_index[keep_clean] = np.arange(int(keep_clean.sum()))
        keep_faulty = faulty[:, 0] >= 0
        faulty = faulty[keep_faulty]
        source_ids = source_ids[keep_faulty]
        original_id = source_ids >= 0
        source_ids[original_id] = clean_index[source_ids[original_id]]
        clean = clean[keep_clean]
        radar = radar[radar[:, 0] >= 0]
    faulty_projection = project_lidar(faulty, geometry)
    clean_projection = project_lidar(clean, geometry)
    radar_features = project_aligned_radar(radar, geometry)
    targets = build_range_targets(
        faulty_projection, clean_projection, faulty, clean, source_ids,
        range_tolerance_m=range_tolerance_m,
        point_tolerance_m=point_tolerance_m,
    )
    fault_map = np.zeros(geometry.shape, dtype=np.float32)
    if fault_map_root is not None:
        predicted_path = Path(fault_map_root) / sample_path.parent.name / sample_path.name
        with np.load(predicted_path, allow_pickle=False) as archive:
            # Never use artifact fault_heatmap/reliability_map: they depend on
            # the clean target. Only an independent predictor may write this.
            if "fault_probability_range_view" not in archive.files:
                raise ValueError(f"{predicted_path} lacks predicted range-view fault probability")
            fault_map = np.asarray(archive["fault_probability_range_view"], dtype=np.float32)
        if fault_map.shape != geometry.shape or not np.isfinite(fault_map).all():
            raise ValueError("predicted fault map must be finite and match sensor geometry")
    dataset_name = str(aligned.metadata.get("dataset", "")).strip().lower()
    # VoD's fourth LiDAR field is reflectivity; the HeRCULES Aeva loader's
    # fourth field is radial velocity. Do not silently call that intensity.
    lidar_reflectivity = (
        np.tanh(faulty_projection.reflectivity)
        if dataset_name in {"view-of-delft", "view of delft", "vod"}
        else np.zeros(geometry.shape, dtype=np.float32)
    )
    features = np.concatenate((
        faulty_projection.range_m[None] / geometry.max_range_m,
        faulty_projection.valid[None].astype(np.float32),
        lidar_reflectivity[None],
        radar_features,
        fault_map[None],
    ), axis=0).astype(np.float32)
    return RangeSample(features, targets, faulty_projection, clean_projection,
                       radar_features, faulty, clean, source_ids,
                       aligned.metadata, sample_path)


class RangeViewDataset(Dataset):
    def __init__(self, paths: list[Path], radar_root: Path, geometry: RangeGeometry,
                 *, fault_map_root: Path | None = None,
                 range_tolerance_m: float = 0.2, point_tolerance_m: float = 0.05,
                 forward_only: bool = True) -> None:
        self.paths = paths
        self.radar_root = radar_root
        self.geometry = geometry
        self.fault_map_root = fault_map_root
        self.range_tolerance_m = range_tolerance_m
        self.point_tolerance_m = point_tolerance_m
        self.forward_only = forward_only

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return load_range_sample(
            self.paths[index], self.radar_root, self.geometry,
            fault_map_root=self.fault_map_root,
            range_tolerance_m=self.range_tolerance_m,
            point_tolerance_m=self.point_tolerance_m,
            forward_only=self.forward_only,
        ).tensors()
