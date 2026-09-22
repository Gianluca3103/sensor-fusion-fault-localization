"""Load aligned raw point inputs from current reconstruction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.data_injection_utils import filter_pointcloud
from Fault_Localization_Model.vod_dataset.vod_io import (
    VOD_LIDAR_FIELDS,
    load_vod_lidar,
)


LIDAR_FIELDS = tuple(VOD_LIDAR_FIELDS)
RADAR_CACHE_FIELDS = (
    "x",
    "y",
    "z",
    "rcs",
    "compensated_radial_velocity",
)


@dataclass(frozen=True)
class AlignedPointInputs:
    lidar_points: np.ndarray
    radar_points: np.ndarray
    lidar_feature_names: tuple[str, ...]
    radar_feature_names: tuple[str, ...]
    metadata: dict
    sample_path: Path
    radar_path: Path


def read_sample_metadata(sample_path: str | Path) -> dict:
    with np.load(sample_path, allow_pickle=False) as sample:
        return json.loads(str(sample["metadata_json"].item()))


def radar_cache_path(radar_root: str | Path, metadata: dict) -> Path:
    """Resolve the existing aligned radar cache without importing PyTorch."""
    split = str(metadata.get("split", "")).strip()
    frame_id = str(
        metadata.get("frame_id", metadata.get("radar_index", ""))
    ).strip()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid artifact split {split!r}")
    if not frame_id.isdigit():
        raise ValueError(f"Frame ID must be numeric, got {frame_id!r}")
    return Path(radar_root) / split / f"{int(frame_id):05d}.npz"


def load_clean_lidar_from_metadata(metadata: dict) -> np.ndarray:
    """Load the clean LiDAR target referenced by one reconstruction artifact.

    VoD and HeRCULES store their raw LiDAR in different binary record layouts,
    but both generator variants persist the dataset name and absolute source
    path in ``metadata_json``.  Keeping the choice here ensures every 3D
    consumer uses the same dataset-aware decoder.
    """
    source = Path(str(metadata.get("source_relative_path", "")))
    if not source.is_file():
        raise FileNotFoundError(
            f"Clean source LiDAR is unavailable: {source}. Use lidar_source='faulty' "
            "or make the dataset source path accessible."
        )
    dataset = str(metadata.get("dataset", "")).strip().lower()
    if dataset in {"view-of-delft", "view of delft", "vod"}:
        points = load_vod_lidar(source).astype(np.float32, copy=False)
    elif dataset == "hercules":
        # Keep VoD-only preprocessing free of the SciPy dependency used by
        # HeRCULES pose interpolation.
        from Fault_Localization_Model.hercules_dataset import load_hercules_lidar

        points = load_hercules_lidar(source).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported dataset in metadata: {metadata.get('dataset')!r}")

    point_filter = metadata.get("point_filter")
    if point_filter is None:
        # Legacy non-3D artifacts are returned unchanged for existing callers.
        return points
    try:
        minimum = float(point_filter["min_range_m"])
        maximum = float(point_filter["max_range_m"])
        x_min, x_max = map(float, point_filter["x_range"])
        y_min, y_max = map(float, point_filter["y_range"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid point_filter in reconstruction metadata") from error
    _, range_mask = filter_pointcloud(points, minimum, maximum, return_mask=True)
    within_bev = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
    )
    return points[range_mask & within_bev]


def load_aligned_point_inputs(
    sample_path: str | Path,
    radar_root: str | Path,
    *,
    lidar_source: str = "clean",
) -> AlignedPointInputs:
    """Load LiDAR and radar already expressed in the LiDAR coordinate frame.

    ``lidar_source='clean'`` follows the source path stored by the generator;
    ``'faulty'`` loads the injected points embedded in the sample artifact.
    Radar points come from the maintained aligned cache and therefore exactly
    match the five fields currently consumed by the reconstruction models.
    """
    sample_path = Path(sample_path)
    if lidar_source not in {"clean", "faulty"}:
        raise ValueError("lidar_source must be 'clean' or 'faulty'")
    with np.load(sample_path, allow_pickle=False) as sample:
        metadata = json.loads(str(sample["metadata_json"].item()))
        if lidar_source == "faulty":
            if "faulty_lidar_points" not in sample.files:
                raise KeyError(f"{sample_path} has no faulty_lidar_points")
            lidar = np.asarray(sample["faulty_lidar_points"], dtype=np.float32)
        else:
            lidar = None
    if lidar is None:
        lidar = load_clean_lidar_from_metadata(metadata)
    if lidar.ndim != 2 or lidar.shape[1] != len(LIDAR_FIELDS):
        raise ValueError(f"LiDAR points must have shape [N,4], got {lidar.shape}")

    radar_path = radar_cache_path(radar_root, metadata)
    with np.load(radar_path, allow_pickle=False) as radar_cache:
        if "radar_points" not in radar_cache.files:
            raise KeyError(f"{radar_path} has no radar_points")
        radar = np.asarray(radar_cache["radar_points"], dtype=np.float32)
    if radar.ndim != 2 or radar.shape[1] != len(RADAR_CACHE_FIELDS):
        raise ValueError(f"Radar points must have shape [N,5], got {radar.shape}")
    return AlignedPointInputs(
        lidar_points=lidar,
        radar_points=radar,
        lidar_feature_names=LIDAR_FIELDS,
        radar_feature_names=RADAR_CACHE_FIELDS,
        metadata=metadata,
        sample_path=sample_path,
        radar_path=radar_path,
    )


def discover_sample_paths(
    data_root: str | Path,
    split: str,
    *,
    limit: int | None = None,
    seed: int = 0,
    unique_frames: bool = True,
) -> list[Path]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    paths = sorted((Path(data_root) / split).glob("*.npz"))
    if unique_frames:
        unique: dict[str, Path] = {}
        for path in paths:
            metadata = read_sample_metadata(path)
            frame_id = str(metadata.get("frame_id", path.stem))
            unique.setdefault(frame_id, path)
        paths = list(unique.values())
    if limit is not None and len(paths) > limit:
        generator = np.random.default_rng(seed)
        chosen = np.sort(generator.choice(len(paths), size=limit, replace=False))
        paths = [paths[int(index)] for index in chosen]
    return paths
