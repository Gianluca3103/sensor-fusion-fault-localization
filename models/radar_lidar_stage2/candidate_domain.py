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


@dataclass(frozen=True)
class CandidateDomain:
    coordinates: torch.Tensor  # sorted unique [N,4] in (batch,z,y,x) order
    confidence: torch.Tensor  # strongest supporting Stage-I seed, [N]
    grid: VoxelGrid
    counts: dict[str, int | float]

    @property
    def centers_xyz(self) -> torch.Tensor:
        return self.grid.centers_xyz(self.coordinates)


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
    scores = confidence.features.squeeze(-1)
    if scores.shape != (len(fine.coords),) or not bool(torch.isfinite(scores).all()):
        raise ValueError("Stage-I confidence has invalid shape or values")
    if len(scores) and (bool((scores < 0).any()) or bool((scores > 1).any())):
        raise ValueError("Stage-I confidence must lie in [0,1]")
    chosen = torch.nonzero(scores > confidence_threshold, as_tuple=False).flatten()
    offsets = torch.tensor(list(product(*(range(-r, r + 1) for r in expansion_zyx))),
                           dtype=torch.long, device=fine.coords.device)
    # Limit seed count before expansion, so even zero-overlap neighborhoods
    # cannot exceed max_sites. Prefer strongest evidence deterministically.
    seed_cap = max_sites // len(offsets)
    if seed_cap < 1:
        raise ValueError("max_sites is smaller than one expanded neighborhood")
    selected_total = len(chosen)
    if len(chosen) > seed_cap:
        chosen = chosen[torch.argsort(scores[chosen], descending=True, stable=True)[:seed_cap]]
    seeds = fine.coords[chosen].long()
    expanded = seeds[:, None, :].expand(-1, len(offsets), -1).clone()
    expanded[:, :, 1:] += offsets[None]
    shape = torch.tensor(grid.shape_zyx, device=expanded.device)
    valid = ((expanded[:, :, 1:] >= 0) & (expanded[:, :, 1:] < shape)).all(-1)
    flat = expanded[valid]
    seed_scores = scores[chosen, None].expand(-1, len(offsets))[valid]
    if len(flat):
        keys, inverse = torch.unique(encode_keys(flat, grid.shape_zyx),
                                     sorted=True, return_inverse=True)
        coordinates = decode_keys(keys, grid.shape_zyx)
        propagated = scores.new_full((len(keys),), -torch.inf)
        propagated.scatter_reduce_(0, inverse, seed_scores, reduce="amax", include_self=True)
    else:
        coordinates = fine.coords.new_empty((0, 4))
        propagated = scores.new_empty(0)
    counts = {"initial_stage1_sites": len(fine.coords),
              "candidate_sites_after_confidence": selected_total,
              "selected_seeds_after_cap": len(chosen),
              "expanded_candidate_sites": len(coordinates),
              "candidate_expansion_ratio": len(coordinates) / max(len(chosen), 1),
              "cap_applied": int(selected_total > len(chosen))}
    return CandidateDomain(coordinates, propagated, grid, counts)
