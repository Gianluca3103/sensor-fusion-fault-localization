"""Faulty-only candidate context and conservative measured-ray suppression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.sparse import encode_keys, voxelize
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from .data import within_fault_region


@dataclass(frozen=True)
class FaultyCoverage:
    features: torch.Tensor  # occupied voxel, local counts, distance, intensity, ray and region masks
    measured_voxel: torch.Tensor
    measured_ray: torch.Tensor
    in_fault_region: torch.Tensor

    @property
    def blocked(self) -> torch.Tensor:
        return self.measured_voxel | self.measured_ray

    @property
    def may_add(self) -> torch.Tensor:
        return self.in_fault_region & ~self.blocked


def faulty_coverage(domain: CandidateDomain, faulty: torch.Tensor, valid: torch.Tensor,
                    regions: list[dict], *, ray_tolerance_m: float = 0.10) -> FaultyCoverage:
    """Only faulty LiDAR is used. Never derive the observed mask from clean LiDAR.

    A surviving first return suppresses additions on the same measured ray,
    including space both before and behind it. The tight perpendicular
    tolerance avoids treating broad nearby angular regions as observed.
    """
    if faulty.device != domain.coordinates.device or len(regions) != faulty.shape[0]:
        raise ValueError("Faulty LiDAR, candidates and region metadata must share a batch")
    if ray_tolerance_m <= 0:
        raise ValueError("ray_tolerance_m must be positive")
    n = len(domain.coordinates)
    device = faulty.device
    features = faulty.new_zeros((n, 7))
    measured_voxel = torch.zeros(n, dtype=torch.bool, device=device)
    measured_ray = torch.zeros_like(measured_voxel)
    in_region = torch.zeros_like(measured_voxel)
    if n == 0:
        return FaultyCoverage(features, measured_voxel, measured_ray, in_region)
    coords, _, _, _ = voxelize(faulty, valid, domain.grid)
    keys = encode_keys(domain.coordinates, domain.grid.shape_zyx)
    fault_keys = encode_keys(coords, domain.grid.shape_zyx)
    if len(fault_keys):
        position = torch.searchsorted(keys, fault_keys)
        matched = (position < n) & (keys[position.clamp(max=n-1)] == fault_keys)
        measured_voxel[position[matched]] = True
    centers = domain.centers_xyz.detach().cpu().numpy()
    batches = domain.coordinates[:, 0].detach().cpu().numpy()
    pts = faulty.detach().cpu().numpy()
    masks = valid.detach().cpu().numpy()
    rays = np.zeros(n, dtype=np.bool_)
    local = np.zeros((n, 7), dtype=np.float32)
    eligible = np.zeros(n, dtype=np.bool_)
    for batch in range(len(regions)):
        idx = np.flatnonzero(batches == batch)
        if not len(idx):
            continue
        query = centers[idx]
        eligible[idx] = within_fault_region(query, regions[batch])
        observed = pts[batch, masks[batch]]
        if not len(observed):
            continue
        xyz = observed[:, :3]
        tree = cKDTree(xyz)
        distance8, nearest8 = tree.query(query, k=min(8, len(xyz)), workers=1)
        distance8 = np.asarray(distance8).reshape(len(query), -1)
        nearest8 = np.asarray(nearest8).reshape(len(query), -1)
        near, nearest = distance8[:, 0], nearest8[:, 0]
        local[idx, 1] = np.log1p((distance8 <= .4).sum(axis=1))
        local[idx, 2] = np.log1p((distance8 <= 1.0).sum(axis=1))
        local[idx, 3] = np.minimum(near, 3.0) / 3.0
        local[idx, 4] = observed[nearest, 3]
        length = np.linalg.norm(xyz, axis=1)
        usable = length > 1e-3
        qlength = np.linalg.norm(query, axis=1)
        qgood = qlength > 1e-3
        if not usable.any() or not qgood.any():
            continue
        dirs = xyz[usable] / length[usable, None]
        chord, _ = cKDTree(dirs).query(query[qgood] / qlength[qgood, None], k=1, workers=1)
        # Endpoint offsets and arbitrary farther returns are irrelevant here:
        # this test asks only whether a faulty first-return ray was measured.
        rays[idx[qgood]] = qlength[qgood] * chord <= ray_tolerance_m
    local[:, 0] = measured_voxel.detach().cpu().numpy().astype(np.float32)
    local[:, 5] = rays.astype(np.float32)
    local[:, 6] = eligible.astype(np.float32)
    features = torch.as_tensor(local, device=device, dtype=faulty.dtype)
    measured_ray = torch.as_tensor(rays, device=device)
    in_region = torch.as_tensor(eligible, device=device)
    return FaultyCoverage(features, measured_voxel, measured_ray, in_region)
