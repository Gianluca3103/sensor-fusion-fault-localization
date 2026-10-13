"""Reusable, per-frame geometric supervision for bottleneck Stage I.

The sparse encoder's coordinates depend on sensor coordinates and its fixed
convolution geometry, not learned weights. Cache only local neighbor indices,
distances, occupied/free labels and counts. Learned features and support scores
are recomputed at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch

from .config import Stage1Config, VoxelGrid
from .correspondence import Neighborhood, local_neighbors
from .sparse import SparseSites


_SCHEMA = "stage1_static_geometry_v1"


@dataclass(frozen=True)
class SupportLabels:
    occupied: torch.Tensor
    known_free: torch.Tensor
    clean_points_in_candidates: int
    clean_points_in_grid: int


class Stage1StaticGeometryCache:
    """Build on first use; reuse across shuffled batches and resumed runs."""

    def __init__(self, root: Path, config: Stage1Config, frames) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.support_grid = VoxelGrid(
            config.grid.minimum_xyz, config.grid.maximum_xyz,
            tuple(size * config.support_stride for size in config.grid.size_xyz))
        self.frames = {}
        for frame in frames:
            key = (frame.split, frame.frame_id)
            if key in self.frames:
                raise ValueError(f"Duplicate VoD frame in static cache: {key}")
            self.frames[key] = frame
        self._signatures = {}
        self._verified = set()

    def _path(self, split: str, frame_id: str) -> Path:
        if split not in ("train", "val") or not frame_id.isdigit():
            raise ValueError("Static cache requires a VoD split and numeric frame ID")
        return self.root / split / f"{frame_id}.pt"

    def _signature(self, split: str, frame_id: str) -> str:
        key = (split, frame_id)
        if key not in self._signatures:
            frame = self.frames[key]
            files = []
            for path in (frame.radar_path, frame.lidar_path,
                         frame.radar_calibration_path, frame.lidar_calibration_path):
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            payload = (_SCHEMA, self.config.as_dict(), frame.radar_variant, files)
            self._signatures[key] = sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return self._signatures[key]

    def _load(self, split: str, frame_id: str):
        path = self._path(split, frame_id)
        if not path.is_file():
            return None
        entry = torch.load(path, map_location="cpu", weights_only=True)
        if entry.get("signature") != self._signature(split, frame_id):
            return None
        return entry

    def has_entry(self, split: str, frame_id: str) -> bool:
        """Return whether a reusable entry exists for this frame and config."""
        return self._load(split, frame_id) is not None

    def _save(self, split: str, frame_id: str, entry) -> None:
        path = self._path(split, frame_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(entry, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _sample_slices(coordinates: np.ndarray, batch_size: int):
        batch = coordinates[:, 0]
        return [(int(np.searchsorted(batch, index, side="left")),
                 int(np.searchsorted(batch, index, side="right")))
                for index in range(batch_size)]

    @staticmethod
    def _local_coordinates(coordinates: torch.Tensor, start: int, stop: int):
        local = coordinates[start:stop].clone()
        local[:, 0] = 0
        return local

    def _build_batch(self, frame_ids, split, radar: SparseSites, lidar: SparseSites,
                     support: SparseSites, clean_lidar: torch.Tensor,
                     clean_valid: torch.Tensor):
        from models.radar_lidar_stage2.candidate_domain import CandidateDomain
        from models.radar_lidar_stage2.voxel_target import make_targets

        neighbors = local_neighbors(radar, lidar, self.config.grid,
                                    self.config.attention_radii_m[-1],
                                    self.config.max_neighbors)
        domain = CandidateDomain(support.coords, support.features[:, 0].detach(),
                                 self.support_grid, {})
        targets = make_targets(domain, clean_lidar, clean_valid)

        radar_cpu = radar.coords.detach().cpu()
        lidar_cpu = lidar.coords.detach().cpu()
        support_cpu = support.coords.detach().cpu()
        radar_slices = self._sample_slices(radar_cpu.numpy(), len(frame_ids))
        lidar_slices = self._sample_slices(lidar_cpu.numpy(), len(frame_ids))
        support_slices = self._sample_slices(support_cpu.numpy(), len(frame_ids))
        indices_cpu = neighbors.indices.detach().cpu()
        distances_cpu = neighbors.distances_m.detach().cpu()
        occupied_cpu = targets.occupied.detach().cpu()
        free_cpu = targets.known_free.detach().cpu()
        represented_cpu = targets.clean_point_count.detach().cpu()

        # make_targets reports a batch total. Preserve its exact in-grid rule
        # while obtaining the per-frame counts needed after reshuffling.
        minimum = clean_lidar.new_tensor(self.support_grid.minimum_xyz)
        size = clean_lidar.new_tensor(self.support_grid.size_xyz)
        shape = clean_lidar.new_tensor(self.support_grid.shape_zyx[::-1], dtype=torch.long)
        xyz = torch.floor((clean_lidar[..., :3] - minimum) / size).long()
        inside = clean_valid & ((xyz >= 0) & (xyz < shape)).all(-1)
        total_per_frame = inside.sum(1).detach().cpu().tolist()

        for index, frame_id in enumerate(frame_ids):
            rs, re = radar_slices[index]
            ls, le = lidar_slices[index]
            ss, se = support_slices[index]
            local_indices = indices_cpu[rs:re].clone()
            valid = local_indices >= 0
            local_indices[valid] -= ls
            entry = {
                "signature": self._signature(split, frame_id),
                "radar_coords": self._local_coordinates(radar_cpu, rs, re),
                "lidar_coords": self._local_coordinates(lidar_cpu, ls, le),
                "neighbor_indices": local_indices,
                "neighbor_distances_m": distances_cpu[rs:re].clone(),
                "occupied": occupied_cpu[ss:se].clone(),
                "known_free": free_cpu[ss:se].clone(),
                "clean_points_in_candidates": int(represented_cpu[ss:se].sum()),
                "clean_points_in_grid": int(total_per_frame[index]),
            }
            self._save(split, frame_id, entry)
        return neighbors, SupportLabels(
            targets.occupied, targets.known_free,
            targets.clean_points_in_candidates, targets.clean_points_in_grid)

    def _verify_once(self, entries, frame_ids, split, radar: SparseSites,
                     lidar: SparseSites, support: SparseSites) -> None:
        unseen = [(split, frame_id) for frame_id in frame_ids
                  if (split, frame_id) not in self._verified]
        if not unseen:
            return
        device = radar.coords.device
        def assembled(name):
            pieces = []
            for batch, entry in enumerate(entries):
                coordinates = entry[name].clone()
                coordinates[:, 0] = batch
                pieces.append(coordinates)
            return torch.cat(pieces).to(device=device, dtype=torch.long)
        if (not torch.equal(radar.coords, assembled("radar_coords")) or
                not torch.equal(lidar.coords, assembled("lidar_coords")) or
                len(support.coords) != sum(len(entry["occupied"]) for entry in entries)):
            raise RuntimeError("Stage-I static cache coordinates differ from current encoder output")
        self._verified.update(unseen)

    def get_batch(self, frame_ids: list[str], split: str,
                  radar: SparseSites, lidar: SparseSites, support: SparseSites,
                  clean_lidar: torch.Tensor, clean_valid: torch.Tensor):
        if not frame_ids or clean_lidar.shape[0] != len(frame_ids):
            raise ValueError("Frame IDs and batch size differ")
        entries = [self._load(split, frame_id) for frame_id in frame_ids]
        if any(entry is None for entry in entries):
            neighbors, labels = self._build_batch(
                frame_ids, split, radar, lidar, support, clean_lidar, clean_valid)
            return neighbors, labels, 0, len(frame_ids)

        self._verify_once(entries, frame_ids, split, radar, lidar, support)
        device = radar.coords.device
        offset = 0
        indices, distances, occupied, free = [], [], [], []
        for entry in entries:
            local = entry["neighbor_indices"].to(device, non_blocking=True)
            indices.append(torch.where(local >= 0, local + offset, local))
            distances.append(entry["neighbor_distances_m"].to(device, non_blocking=True))
            occupied.append(entry["occupied"].to(device, non_blocking=True))
            free.append(entry["known_free"].to(device, non_blocking=True))
            offset += len(entry["lidar_coords"])
        labels = SupportLabels(
            torch.cat(occupied), torch.cat(free),
            sum(entry["clean_points_in_candidates"] for entry in entries),
            sum(entry["clean_points_in_grid"] for entry in entries))
        return Neighborhood(torch.cat(indices), torch.cat(distances)), labels, len(entries), 0
