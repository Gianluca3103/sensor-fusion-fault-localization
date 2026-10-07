"""Bounded sparse candidate coordinates from Stage-I radar-only confidence.

Candidate coordinates mean 'worth considering', never 'occupied LiDAR'.
No clean or faulty LiDAR data are accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import decode_keys, encode_keys


def voxel_centers_xyz(coordinates: torch.Tensor, grid: VoxelGrid, *,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Compute fine-grid centers without float32 error at distant cell indices."""
    xyz = coordinates[:, [3, 2, 1]].to(torch.float64)
    origin = torch.tensor(grid.minimum_xyz, dtype=torch.float64, device=coordinates.device)
    size = torch.tensor(grid.size_xyz, dtype=torch.float64, device=coordinates.device)
    return (origin + (xyz + 0.5) * size).to(dtype)


@dataclass(frozen=True)
class CandidateDomain:
    coordinates: torch.Tensor  # sorted unique [N,4] in (batch,z,y,x) order
    confidence: torch.Tensor  # strongest supporting Stage-I seed, [N]
    grid: VoxelGrid
    counts: dict[str, int | float | str]

    @property
    def centers_xyz(self) -> torch.Tensor:
        return voxel_centers_xyz(self.coordinates, self.grid)


def make_candidates(output: Stage1Output, grid: VoxelGrid, *,
                    confidence_threshold: float = 0.25,
                    expansion_zyx: tuple[int, int, int] = (1, 1, 1),
                    max_sites: int = 100_000) -> CandidateDomain:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be in [0,1]")
    if len(expansion_zyx) != 3 or any(not isinstance(v, int) or v < 0 for v in expansion_zyx):
        raise ValueError("Expansion radius must contain three non-negative voxel counts")
    if max_sites < 1:
        raise ValueError("max_sites must be positive")
    fine = output.features["s1"]
    confidence = output.confidence
    if fine.stride != 1 or confidence.stride != 1:
        raise ValueError("Stage-I S1 and confidence must use fine-grid stride 1")
    if fine.shape_zyx != grid.shape_zyx or confidence.shape_zyx != grid.shape_zyx:
        raise ValueError("Stage-I grid and Stage-II candidate grid differ")
    if not torch.equal(fine.coords, confidence.coords):
        raise ValueError("Stage-I confidence must align with S1 coordinates")
    if output.surface is None:
        # Legacy Stage-I checkpoints have scores only at radar-occupied cells.
        seed_coords = fine.coords
        scores = confidence.features.squeeze(-1)
        source = "radar_voxel"
    else:
        surface = output.surface
        if (surface.xyz.ndim != 3 or surface.xyz.shape[0] != len(fine.coords)
                or surface.xyz.shape[2] != 3 or surface.score.shape != surface.xyz.shape[:2]):
            raise ValueError("Stage-I surface proposals have invalid shape")
        xyz = surface.xyz.reshape(-1, 3)
        scores = surface.score.reshape(-1)
        if not bool(torch.isfinite(xyz).all()):
            raise ValueError("Stage-I surface proposals must be finite")
        minimum = torch.tensor(grid.minimum_xyz, dtype=torch.float64, device=xyz.device)
        size = torch.tensor(grid.size_xyz, dtype=torch.float64, device=xyz.device)
        index_xyz = torch.floor((xyz.to(torch.float64)-minimum)/size).long()
        seed_coords = torch.cat((fine.coords[:,0,None].repeat_interleave(surface.xyz.shape[1],dim=0),
                                 index_xyz[:,[2,1,0]]),dim=1)
        source = "predicted_lidar_surface"
    if scores.shape != (len(seed_coords),) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Stage-I proposal confidence has invalid shape or values")
    if len(scores) and (bool((scores < 0).any()) or bool((scores > 1).any())):
        raise ValueError("Stage-I proposal confidence must lie in [0,1]")
    shape = torch.tensor(grid.shape_zyx, device=fine.coords.device)
    inside = ((seed_coords[:,1:] >= 0) & (seed_coords[:,1:] < shape)).all(-1)
    chosen = torch.nonzero((scores > confidence_threshold) & inside, as_tuple=False).flatten()
    offsets = torch.tensor(list(product(*(range(-r, r + 1) for r in expansion_zyx))),
                           dtype=torch.long, device=fine.coords.device)
    if max_sites < len(offsets):
        raise ValueError("max_sites is smaller than one expanded neighborhood")
    selected_total = len(chosen)
    chosen = chosen[torch.argsort(scores[chosen], descending=True, stable=True)]
    # Bound temporary expansion memory to four times the final site budget.
    # Count *unique* sites, so overlapping neighborhoods do not cause an
    # unnecessarily severe seed cap. Binary search keeps the strongest seeds.
    max_proposals = max_sites * 4
    chosen = chosen[:max_proposals // len(offsets)]
    def expanded_keys(seed_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        seeds = seed_coords[chosen[:seed_count]].long()
        expanded = seeds[:, None, :].expand(-1, len(offsets), -1).clone()
        expanded[:, :, 1:] += offsets[None]
        valid = ((expanded[:, :, 1:] >= 0) & (expanded[:, :, 1:] < shape)).all(-1)
        return encode_keys(expanded[valid], grid.shape_zyx), scores[chosen[:seed_count], None].expand(-1, len(offsets))[valid]

    raw_keys, seed_scores = expanded_keys(len(chosen))
    if len(raw_keys) and len(torch.unique(raw_keys)) > max_sites:
        low, high = 0, len(chosen)
        while low + 1 < high:
            middle = (low + high) // 2
            trial_keys, _ = expanded_keys(middle)
            if len(torch.unique(trial_keys)) <= max_sites:
                low = middle
            else:
                high = middle
        chosen = chosen[:low]
        raw_keys, seed_scores = expanded_keys(len(chosen))
    if len(raw_keys):
        keys, inverse = torch.unique(raw_keys, sorted=True, return_inverse=True)
        coordinates = decode_keys(keys, grid.shape_zyx)
        propagated = scores.new_full((len(keys),), -torch.inf)
        propagated.scatter_reduce_(0, inverse, seed_scores, reduce="amax", include_self=True)
    else:
        coordinates = fine.coords.new_empty((0, 4))
        propagated = scores.new_empty(0)
    counts = {"initial_stage1_sites": len(fine.coords),
              "initial_surface_proposals": len(seed_coords),
              "seed_source": source,
              "candidate_sites_after_confidence": selected_total,
              "selected_seeds_after_cap": len(chosen),
              "expanded_candidate_sites": len(coordinates),
              "candidate_expansion_ratio": len(coordinates) / max(len(chosen), 1),
              "cap_applied": int(selected_total > len(chosen))}
    return CandidateDomain(coordinates, propagated, grid, counts)
