"""Geometry-specific, resumable cache of projected range-view training tensors."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..voxelization.inputs import radar_cache_path, read_sample_metadata
from .data import TARGET_KEYS, load_range_sample
from .geometry import RangeGeometry


CACHE_VERSION = 1
CACHE_KEYS = ("features",) + TARGET_KEYS
BASE_CACHE_KEYS = CACHE_KEYS[:-3]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cache_settings(geometry: RangeGeometry, *, forward_only: bool = True,
                   radar_floor_band_m: float = 0.0,
                   filter_radar_by_lidar_min: bool = True,
                   require_lidar_intensity: bool = False,
                   include_object_targets: bool = False,
                   radar_region_row_radius: int = 8,
                   radar_region_col_radius: int = 32,
                   fault_map_root: Path | None = None) -> dict:
    """Only preprocessing settings belong here; ray directions are appended at load time."""
    if fault_map_root is not None:
        raise ValueError("Projected cache does not support external fault maps")
    settings = {
        "version": CACHE_VERSION, "geometry": asdict(geometry),
        "forward_only": bool(forward_only),
        "radar_floor_band_m": float(radar_floor_band_m),
        "filter_radar_by_lidar_min": bool(filter_radar_by_lidar_min),
        "require_lidar_intensity": bool(require_lidar_intensity),
    }
    if include_object_targets:
        if radar_region_row_radius < 0 or radar_region_col_radius < 0:
            raise ValueError("radar region radii must be nonnegative")
        settings.update(version=3, include_object_targets=True,
                        radar_region_row_radius=int(radar_region_row_radius),
                        radar_region_col_radius=int(radar_region_col_radius))
    return settings


def _file_stamp(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def source_signature(sample_path: Path, radar_root: Path, *,
                     include_object_targets: bool = False) -> str:
    metadata = read_sample_metadata(sample_path)
    sources = [sample_path, Path(str(metadata["source_relative_path"])),
               radar_cache_path(radar_root, metadata)]
    if include_object_targets:
        from .object_targets import vod_label_paths
        sources.extend(vod_label_paths(metadata))
    return _digest([_file_stamp(path) for path in sources])


def cached_path(root: Path, sample_path: Path) -> Path:
    return root / sample_path.parent.name / sample_path.name


def write_cached_sample(sample_path: Path, radar_root: Path, geometry: RangeGeometry,
                        cache_root: Path, settings: dict, *, resume: bool = True) -> bool:
    """Write one sample atomically; return False when an existing valid entry was reused."""
    output = cached_path(cache_root, sample_path)
    signature = source_signature(sample_path, radar_root,
                                 include_object_targets=settings.get("include_object_targets", False))
    settings_hash = _digest(settings)
    if resume and output.is_file():
        try:
            with np.load(output, allow_pickle=False) as archive:
                if (str(archive["source_signature"].item()) == signature
                        and str(archive["settings_hash"].item()) == settings_hash
                        and all(key in archive.files for key in (
                            CACHE_KEYS if settings.get("include_object_targets") else BASE_CACHE_KEYS))):
                    return False
        except (OSError, ValueError, KeyError, EOFError):
            pass
    sample = load_range_sample(
        sample_path, radar_root, geometry,
        forward_only=settings["forward_only"],
        radar_floor_band_m=settings["radar_floor_band_m"],
        filter_radar_by_lidar_min=settings["filter_radar_by_lidar_min"],
        require_lidar_intensity=settings["require_lidar_intensity"],
        use_ray_encoding=False,
        include_object_targets=settings.get("include_object_targets", False),
        radar_region_row_radius=settings.get("radar_region_row_radius", 8),
        radar_region_col_radius=settings.get("radar_region_col_radius", 32),
    )
    tensors = sample.tensors()
    if tensors["features"].shape[0] != 10:
        raise ValueError("Cache expects ten base feature channels")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, **{key: tensors[key].numpy() for key in CACHE_KEYS},
                source_signature=np.asarray(signature), settings_hash=np.asarray(settings_hash),
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def cache_manifest(paths: list[Path], radar_root: Path, settings: dict) -> dict:
    return {
        "version": settings["version"],
        "settings": settings,
        "settings_hash": _digest(settings),
        "samples": {f"{path.parent.name}/{path.name}": source_signature(
            path, radar_root, include_object_targets=settings.get("include_object_targets", False))
                    for path in paths},
    }


def validate_cache(root: Path, paths: list[Path], radar_root: Path, settings: dict) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing range-view cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = cache_manifest(paths, radar_root, settings)
    if (manifest.get("version") != settings["version"]
            or manifest.get("settings_hash") != expected["settings_hash"]
            or any(manifest.get("samples", {}).get(name) != signature
                   for name, signature in expected["samples"].items())):
        raise ValueError("Range-view cache has different geometry, inputs, or preprocessing settings; rebuild it")
    missing = [path for path in paths if not cached_path(root, path).is_file()]
    if missing:
        raise FileNotFoundError(f"Range-view cache is incomplete: {len(missing)} entries missing; first: {missing[0]}")


def load_cached_tensors(root: Path, sample_path: Path, geometry: RangeGeometry, *,
                        use_ray_encoding: bool = False) -> dict[str, torch.Tensor]:
    with np.load(cached_path(root, sample_path), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key], dtype=np.float32)
                  for key in CACHE_KEYS if key in archive.files}
    for key in BASE_CACHE_KEYS:
        if key not in arrays:
            raise KeyError(f"Cached sample lacks {key}: {sample_path}")
    arrays.setdefault("object_class", np.zeros(geometry.shape, dtype=np.float32))
    arrays.setdefault("radar_region", np.ones(geometry.shape, dtype=np.float32))
    arrays.setdefault("ground_mask", np.zeros(geometry.shape, dtype=np.float32))
    if arrays["features"].shape != (10, *geometry.shape):
        raise ValueError(f"Cached feature shape does not match geometry: {sample_path}")
    if any(arrays[key].shape != geometry.shape for key in TARGET_KEYS):
        raise ValueError(f"Cached target shape does not match geometry: {sample_path}")
    if use_ray_encoding:
        arrays["features"] = np.concatenate((
            arrays["features"], np.moveaxis(geometry.ray_directions(), -1, 0)
        ), axis=0)
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()}
