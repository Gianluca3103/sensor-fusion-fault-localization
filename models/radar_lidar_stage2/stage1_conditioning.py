"""Local physical XYZ queries of all deployed radar-only Stage-I scales."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from models.radar_lidar_stage1.model import Stage1Output
from .candidate_domain import CandidateDomain
from .config import Stage2Config


def physical_neighbor_map(candidate_xyz: np.ndarray, candidate_batch: np.ndarray,
                          source_xyz: np.ndarray, source_batch: np.ndarray,
                          radius_m: float, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """Return capped local source indices and metric distances, never cross frames."""
    idx = np.full((len(candidate_xyz), max_neighbors), -1, dtype=np.int64)
    dist = np.full_like(idx, np.inf, dtype=np.float32)
    for batch in np.unique(candidate_batch):
        dest = np.flatnonzero(candidate_batch == batch)
        src = np.flatnonzero(source_batch == batch)
        if not len(dest) or not len(src):
            continue
        d, local = cKDTree(source_xyz[src]).query(candidate_xyz[dest], k=max_neighbors,
                                                    distance_upper_bound=radius_m, workers=1)
        d = np.asarray(d).reshape(len(dest), max_neighbors)
        local = np.asarray(local).reshape(len(dest), max_neighbors)
        good = local < len(src)
        idx[dest] = np.where(good, src[np.minimum(local, len(src)-1)], -1)
        dist[dest] = d
    return idx, dist


class Stage1Conditioning(nn.Module):
    def __init__(self, stage1_channels: tuple[int, ...], config: Stage2Config):
        super().__init__()
        if len(stage1_channels) != 4:
            raise ValueError("Stage II needs Stage-I S1–S4")
        self.config = config
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(ch, config.conditioning_dim),
                                                       nn.LayerNorm(config.conditioning_dim), nn.SiLU())
                                         for ch in stage1_channels)
        self.fuse = nn.Sequential(nn.Linear(4 * (config.conditioning_dim + 1) + 4,
                                            config.channels[0]),
                                  nn.LayerNorm(config.channels[0]), nn.SiLU())

    def forward(self, stage1: Stage1Output, domain: CandidateDomain, *,
                ablation: str = "real", replacement_stage1: Stage1Output | None = None) -> torch.Tensor:
        if ablation not in {"real", "zero", "shuffle", "wrong_sample", "no_confidence", "s1_only", "s4_only"}:
            raise ValueError(ablation)
        if (ablation == "wrong_sample") != (replacement_stage1 is not None):
            raise ValueError("wrong_sample requires exactly one replacement Stage-I output")
        n = len(domain.coordinates)
        if not n:
            return self.fuse[0].weight.new_empty((0, self.config.channels[0]))
        center = domain.centers_xyz
        dest_xyz = center.detach().cpu().numpy()
        dest_batch = domain.coordinates[:, 0].detach().cpu().numpy()
        parts = []
        for i in range(4):
            sites = (replacement_stage1 or stage1).features[f"s{i+1}"]
            if sites.stride != 2 ** i:
                raise ValueError(f"Stage-I s{i+1} stride is inconsistent")
            if sites.shape_zyx != domain.grid.scale_shape(sites.stride):
                raise ValueError(f"Stage-I s{i+1} physical grid differs from Stage II")
            src_xyz = sites.centers_xyz(domain.grid).detach().cpu().numpy()
            src_batch = sites.coords[:, 0].detach().cpu().numpy()
            idx_np, dist_np = physical_neighbor_map(dest_xyz, dest_batch, src_xyz, src_batch,
                                                     self.config.query_radii_m[i], self.config.max_neighbors)
            index = torch.as_tensor(idx_np, device=sites.features.device)
            distance = torch.as_tensor(dist_np, device=sites.features.device)
            valid = index >= 0
            if len(sites.features):
                source = sites.features.detach()
                if ablation == "zero" or (ablation == "s1_only" and i != 0) or (ablation == "s4_only" and i != 3):
                    source = torch.zeros_like(source)
                elif ablation == "shuffle":
                    source = source.flip(0)
                weights = torch.where(valid, 1 / (distance + 0.05), 0.)
                weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
                pooled = (source[index.clamp_min(0)] * weights[..., None]).sum(1)
            else:
                pooled = center.new_zeros((n, self.projections[i][0].in_features))
            embedding = self.projections[i](pooled)
            support = valid.any(1).to(center.dtype)[:, None]
            if ablation == "zero":
                support = torch.zeros_like(support)
            parts += [embedding, support]
        confidence = domain.confidence.detach()
        if ablation in {"zero", "no_confidence"}:
            confidence = torch.zeros_like(confidence)
        # Relative physical position is weak context, never a substitute for radar.
        xyz = ((center - center.new_tensor(domain.grid.minimum_xyz)) /
               center.new_tensor(tuple(b-a for a,b in zip(domain.grid.minimum_xyz, domain.grid.maximum_xyz))))
        parts += [confidence[:, None], xyz]
        return self.fuse(torch.cat(parts, dim=1))
