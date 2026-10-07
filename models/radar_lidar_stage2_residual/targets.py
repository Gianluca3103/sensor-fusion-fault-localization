"""Additional clean geometry only, with measured and visible-free negatives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.voxel_target import VoxelTargets, make_targets
from .coverage import FaultyCoverage


@dataclass(frozen=True)
class ResidualTargets:
    addition: torch.Tensor
    observed: torch.Tensor
    known_free: torch.Tensor
    offsets_normalized: torch.Tensor
    clean: VoxelTargets

    @property
    def supervised(self) -> torch.Tensor:
        return self.addition | self.observed | self.known_free


def make_residual_targets(domain: CandidateDomain, clean_lidar: torch.Tensor,
                          clean_valid: torch.Tensor, coverage: FaultyCoverage,
                          *, free_ray_tolerance_m: float) -> ResidualTargets:
    clean = make_targets(domain, clean_lidar, clean_valid,
                         free_ray_tolerance_m=free_ray_tolerance_m)
    if len(clean.occupied) != len(coverage.measured_ray):
        raise ValueError("Clean targets and faulty coverage differ in candidate count")
    addition = clean.occupied & coverage.may_add
    observed = coverage.blocked & coverage.in_fault_region
    known_free = clean.known_free & coverage.may_add
    return ResidualTargets(addition, observed, known_free,
                           clean.offsets_normalized, clean)
