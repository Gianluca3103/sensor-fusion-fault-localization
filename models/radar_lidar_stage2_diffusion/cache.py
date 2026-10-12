"""Versioned, per-frame cache of deterministic Stage-II preprocessing.

The cache contains training labels, but the deployed conditioning path reads
only radar-derived sources, candidates, and faulty-LiDAR coverage.
"""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2.candidate_domain import CandidateDomain, make_candidates
from models.radar_lidar_stage2.stage1_conditioning import physical_neighbor_map
from models.radar_lidar_stage2.voxel_target import VoxelTargets
from models.radar_lidar_stage2_residual.coverage import FaultyCoverage, faulty_coverage
from models.radar_lidar_stage2_residual.targets import ResidualTargets, make_residual_targets


CACHE_VERSION = 2


def preprocessing_code_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "models" / "radar_lidar_stage1").glob("*.py"))
    files += [root / path for path in (
        "models/radar_lidar_stage2/candidate_domain.py",
        "models/radar_lidar_stage2/stage1_conditioning.py",
        "models/radar_lidar_stage2/voxel_target.py",
        "models/radar_lidar_stage2_residual/coverage.py",
        "models/radar_lidar_stage2_residual/data.py",
        "models/radar_lidar_stage2_residual/targets.py",
        "models/radar_lidar_stage2_diffusion/cache.py",
    )]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(*, stage1_checkpoint: str | Path, vod_root: str | Path,
               fault_samples_root: str | Path, radar_variant: str,
               fault_pattern: str, config, compression: str | None = None) -> dict:
    result = {
        "version": CACHE_VERSION,
        "preprocessing_code_sha256": preprocessing_code_sha256(),
        "stage1_checkpoint_sha256": file_sha256(stage1_checkpoint),
        "vod_root": str(Path(vod_root).resolve()),
        "fault_samples_root": str(Path(fault_samples_root).resolve()),
        "radar_variant": radar_variant,
        "fault_pattern": fault_pattern,
        "stage2_config": json.loads(json.dumps(config.residual.as_dict())),
    }
    if compression is not None:
        result["compression"] = compression
    return result


def _array(value: torch.Tensor, dtype=None) -> np.ndarray:
    array = value.detach().cpu().numpy()
    return np.asarray(array, dtype=dtype) if dtype is not None else array


@torch.no_grad()
def prepare_frame(evidence, grid: VoxelGrid, batch: dict, config, *,
                  include_validation_clouds: bool = True) -> dict[str, np.ndarray]:
    """Compute all fixed geometry, supervision, and four local neighbor maps."""
    if len(batch["frame_id"]) != 1:
        raise ValueError("Cache entries must contain exactly one frame")
    base = config.residual
    domain = make_candidates(evidence, grid, confidence_threshold=base.confidence_threshold,
                             expansion_zyx=base.expansion_zyx,
                             max_sites=base.max_candidate_sites)
    coverage = faulty_coverage(domain, batch["faulty_lidar"],
                               batch["faulty_lidar_valid"], batch["fault_region"],
                               ray_tolerance_m=base.ray_tolerance_m)
    target = make_residual_targets(domain, batch["clean_lidar"],
                                   batch["clean_lidar_valid"], coverage,
                                   free_ray_tolerance_m=base.free_ray_tolerance_m)
    clean = target.clean
    result = {
        "frame_id": np.asarray(batch["frame_id"][0]),
        "coordinates": _array(domain.coordinates, np.int32),
        "confidence": _array(domain.confidence, np.float32),
        "counts_json": np.asarray(json.dumps(domain.counts)),
        "coverage_features": _array(coverage.features, np.float32),
        "measured_voxel": _array(coverage.measured_voxel, np.bool_),
        "measured_ray": _array(coverage.measured_ray, np.bool_),
        "in_fault_region": _array(coverage.in_fault_region, np.bool_),
        "target_addition": _array(target.addition, np.bool_),
        "target_observed": _array(target.observed, np.bool_),
        "target_known_free": _array(target.known_free, np.bool_),
        "clean_occupied": _array(clean.occupied, np.bool_),
        "clean_known_free": _array(clean.known_free, np.bool_),
        "clean_offsets": _array(clean.offsets_normalized, np.float32),
        "clean_centroids": _array(clean.clean_centroid_xyz, np.float32),
        "clean_point_count": _array(clean.clean_point_count, np.int64),
        "clean_points_in_grid": np.asarray(clean.clean_points_in_grid, np.int64),
        "clean_points_in_candidates": np.asarray(clean.clean_points_in_candidates, np.int64),
        "point_centroid_errors": _array(clean.point_centroid_errors_m, np.float32),
    }
    if include_validation_clouds:
        result["clean_lidar"] = _array(batch["clean_lidar"][0, batch["clean_lidar_valid"][0]], np.float32)
        result["faulty_lidar"] = _array(batch["faulty_lidar"][0, batch["faulty_lidar_valid"][0]], np.float32)
    dest_xyz, dest_batch = domain.centers_xyz_numpy(), domain.batch_numpy()
    for i in range(4):
        sites = evidence.features[f"s{i+1}"]
        source_xyz = _array(sites.centers_xyz(grid), np.float32)
        source_batch = _array(sites.coords[:, 0], np.int64)
        index, _ = physical_neighbor_map(
            dest_xyz, dest_batch, source_xyz, source_batch,
            base.query_radii_m[i], base.max_neighbors,
        )
        result[f"source_{i}"] = _array(sites.features, np.float32)
        result[f"source_center_{i}"] = source_xyz
        if len(source_xyz) < np.iinfo(np.uint16).max:
            result[f"neighbor_index_{i}"] = np.where(index < 0, np.iinfo(np.uint16).max,
                                                       index).astype(np.uint16)
        else:
            result[f"neighbor_index_{i}"] = index.astype(np.int32)
    return result


def save_frame(path: Path, arrays: dict[str, np.ndarray], compression: str) -> None:
    if compression not in {"fast", "none"}:
        raise ValueError("Compression must be fast or none")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    method = zipfile.ZIP_DEFLATED if compression == "fast" else zipfile.ZIP_STORED
    with zipfile.ZipFile(temporary, "w", compression=method, compresslevel=1) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


class CachedFrameDataset(Dataset):
    def __init__(self, root: str | Path, split: str, expected: dict):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError("Stage-II preprocessing cache is incomplete")
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Stage-II cache differs on {key}; rebuild separately")
        self.frame_ids = manifest["splits"][split]
        self.grid = VoxelGrid(**{key: tuple(value) for key, value in manifest["grid"].items()})
        self.split = split

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        frame_id = self.frame_ids[index]
        path = self.root / self.split / f"{frame_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            row = {name: archive[name] for name in archive.files}
        if str(row["frame_id"].item()) != frame_id:
            raise ValueError(f"Cached frame ID differs: {path}")
        if self.split == "val" and ("clean_lidar" not in row or "faulty_lidar" not in row):
            raise ValueError(f"Validation cloud data missing from cache: {path}")
        return row


def _cat(rows: list[dict], key: str, dtype=None) -> torch.Tensor:
    arrays = [row[key] for row in rows]
    value = np.concatenate(arrays, axis=0)
    return torch.from_numpy(value.astype(dtype, copy=False) if dtype is not None else value)


def collate_cached(rows: list[dict], grid: VoxelGrid) -> dict:
    coordinates = _cat(rows, "coordinates", np.int64)
    start = 0
    for batch_id, row in enumerate(rows):
        amount = len(row["coordinates"])
        coordinates[start:start+amount, 0] = batch_id
        start += amount
    counts: dict = {}
    for row in rows:
        for key, value in json.loads(str(row["counts_json"].item())).items():
            counts[key] = counts.get(key, 0) + value if isinstance(value, (int, float)) else value
    sources, source_centers, indices = [], [], []
    for scale in range(4):
        source_rows = [row[f"source_{scale}"] for row in rows]
        sources.append(_cat(rows, f"source_{scale}", np.float32))
        source_centers.append(_cat(rows, f"source_center_{scale}", np.float32))
        index_rows = []
        source_offset = 0
        for row, source in zip(rows, source_rows):
            stored = row[f"neighbor_index_{scale}"]
            local = stored.astype(np.int64)
            if stored.dtype == np.uint16:
                local[local == np.iinfo(np.uint16).max] = -1
            index_rows.append(np.where(local >= 0, local + source_offset, -1))
            source_offset += len(source)
        indices.append(torch.from_numpy(np.concatenate(index_rows, axis=0)))
    cache = {
        "coordinates": coordinates,
        "confidence": _cat(rows, "confidence", np.float32),
        "counts": counts,
        "grid": grid,
        "coverage_features": _cat(rows, "coverage_features", np.float32),
        "measured_voxel": _cat(rows, "measured_voxel", np.bool_),
        "measured_ray": _cat(rows, "measured_ray", np.bool_),
        "in_fault_region": _cat(rows, "in_fault_region", np.bool_),
        "target_addition": _cat(rows, "target_addition", np.bool_),
        "target_observed": _cat(rows, "target_observed", np.bool_),
        "target_known_free": _cat(rows, "target_known_free", np.bool_),
        "clean_occupied": _cat(rows, "clean_occupied", np.bool_),
        "clean_known_free": _cat(rows, "clean_known_free", np.bool_),
        "clean_offsets": _cat(rows, "clean_offsets", np.float32),
        "clean_centroids": _cat(rows, "clean_centroids", np.float32),
        "clean_point_count": _cat(rows, "clean_point_count", np.int64),
        "clean_points_in_grid": sum(int(row["clean_points_in_grid"]) for row in rows),
        "clean_points_in_candidates": sum(int(row["clean_points_in_candidates"]) for row in rows),
        "point_centroid_errors": _cat(rows, "point_centroid_errors", np.float32),
        "sources": sources,
        "source_centers": source_centers,
        "indices": indices,
    }
    def padded(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = max(len(row.get(key, ())) for row in rows)
        points = torch.zeros((len(rows), maximum, 4), dtype=torch.float32)
        valid = torch.zeros((len(rows), maximum), dtype=torch.bool)
        for i, row in enumerate(rows):
            value = row.get(key)
            length = 0 if value is None else len(value)
            if length:
                points[i, :length] = torch.from_numpy(value)
            valid[i, :length] = True
        return points, valid
    clean, clean_valid = padded("clean_lidar")
    faulty, faulty_valid = padded("faulty_lidar")
    return {"cache": cache, "frame_id": [str(row["frame_id"].item()) for row in rows],
            "clean_lidar": clean, "clean_lidar_valid": clean_valid,
            "faulty_lidar": faulty, "faulty_lidar_valid": faulty_valid}


def cached_loader(args, split: str, limit: int, shuffle: bool, expected: dict):
    dataset = CachedFrameDataset(args.preprocess_cache_root, split, expected)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    grid = dataset.dataset.grid if isinstance(dataset, Subset) else dataset.grid
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers,
                      collate_fn=partial(collate_cached, grid=grid),
                      pin_memory=True, persistent_workers=args.num_workers > 0)


def restore_cached(batch: dict, model) -> tuple:
    data = batch["cache"]
    domain = CandidateDomain(data["coordinates"], data["confidence"],
                             data["grid"], data["counts"])
    coverage = FaultyCoverage(data["coverage_features"], data["measured_voxel"],
                              data["measured_ray"], data["in_fault_region"])
    clean = VoxelTargets(data["clean_occupied"], data["clean_known_free"],
                         data["clean_offsets"], data["clean_centroids"],
                         data["clean_point_count"], data["clean_points_in_grid"],
                         data["clean_points_in_candidates"], data["point_centroid_errors"])
    target = ResidualTargets(data["target_addition"], data["target_observed"],
                             data["target_known_free"], clean.offsets_normalized, clean)
    return model.prepare_cached(domain, coverage, data["sources"],
                                data["indices"], None, len(batch["frame_id"]),
                                source_centers=data["source_centers"]), target
