"""Local physical XYZ queries of the deployed Stage-I bottleneck or legacy scales."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.bottleneck_support import BottleneckSupportOutput
from .candidate_domain import CandidateDomain
from .config import Stage2Config
from .cpu_query import query_workers


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
                                                    distance_upper_bound=radius_m,
                                                    workers=query_workers(len(dest)))
        d = np.asarray(d).reshape(len(dest), max_neighbors)
        local = np.asarray(local).reshape(len(dest), max_neighbors)
        good = local < len(src)
        idx[dest] = np.where(good, src[np.minimum(local, len(src)-1)], -1)
        dist[dest] = d
    return idx, dist


class Stage1Conditioning(nn.Module):
    def __init__(self, stage1_channels: tuple[int, ...], config: Stage2Config):
        super().__init__()
        if len(stage1_channels) not in (1, 4):
            raise ValueError("Stage II needs either the bottleneck or legacy S1–S4")
        self.config = config
        self.source_count = len(stage1_channels)
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(ch, config.conditioning_dim),
                                                       nn.LayerNorm(config.conditioning_dim), nn.SiLU())
                                         for ch in stage1_channels)
        self.fuse = nn.Sequential(nn.Linear(self.source_count * (config.conditioning_dim + 1) + 4,
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
        if isinstance(stage1, BottleneckSupportOutput) != (self.source_count == 1):
            raise ValueError("Stage-I output and conditioner checkpoint use different interfaces")
        dest_xyz = domain.centers_xyz_numpy()
        dest_batch = domain.batch_numpy()
        sources, indices, distances = [], [], []
        for i in range(self.source_count):
            evidence = replacement_stage1 or stage1
            sites = (evidence.bottleneck if self.source_count == 1 else evidence.features[f"s{i+1}"])
            scale_index = 3 if self.source_count == 1 else i
            if sites.stride != 2 ** scale_index:
                raise ValueError(f"Stage-I s{i+1} stride is inconsistent")
            if sites.shape_zyx != domain.grid.scale_shape(sites.stride):
                raise ValueError(f"Stage-I s{i+1} physical grid differs from Stage II")
            src_xyz = sites.centers_xyz(domain.grid).detach().cpu().numpy()
            src_batch = sites.coords[:, 0].detach().cpu().numpy()
            idx_np, dist_np = physical_neighbor_map(dest_xyz, dest_batch, src_xyz, src_batch,
                                                     self.config.query_radii_m[scale_index], self.config.max_neighbors)
            sources.append(sites.features)
            indices.append(torch.as_tensor(idx_np, device=sites.features.device))
            distances.append(torch.as_tensor(dist_np, device=sites.features.device))
        return self.forward_cached(domain, sources, indices, distances,
                                   ablation="real" if ablation == "wrong_sample" else ablation)

    def forward_cached(self, domain: CandidateDomain, sources: list[torch.Tensor],
                       indices: list[torch.Tensor], distances: list[torch.Tensor] | None, *,
                       source_centers: list[torch.Tensor] | None = None,
                       ablation: str = "real") -> torch.Tensor:
        """Project cached neighbor maps with the current trainable Stage-II weights."""
        if ablation not in {"real", "zero", "shuffle", "no_confidence", "s1_only", "s4_only"}:
            raise ValueError(ablation)
        n = len(domain.coordinates)
        if not n:
            return self.fuse[0].weight.new_empty((0, self.config.channels[0]))
        if (len(sources) != self.source_count or len(indices) != self.source_count or
                (distances is None and (source_centers is None or len(source_centers) != self.source_count)) or
                (distances is not None and len(distances) != self.source_count)):
            raise ValueError("Cached Stage-I source count differs from the conditioner")
        center = domain.centers_xyz
        parts = []
        for i, (source, index) in enumerate(zip(sources, indices)):
            if (index.shape != (n, self.config.max_neighbors) or source.ndim != 2
                    or source.shape[1] != self.projections[i][0].in_features):
                raise ValueError("Cached Stage-I neighbor shapes differ from the model")
            index = index.long()
            valid = index >= 0
            if distances is None:
                source_xyz = source_centers[i]
                if source_xyz.shape != (len(source), 3):
                    raise ValueError("Cached Stage-I source positions differ from features")
                if len(source):
                    gathered = source_xyz.index_select(0, index.clamp_min(0).reshape(-1))
                    gathered = gathered.reshape(n, index.shape[1], 3)
                    distance = torch.linalg.vector_norm(gathered - center[:, None, :], dim=-1)
                    distance = torch.where(valid, distance, torch.inf)
                else:
                    distance = center.new_full(index.shape, torch.inf)
            else:
                distance = distances[i]
                if distance.shape != index.shape:
                    raise ValueError("Cached Stage-I distance shape differs from neighbors")
            linear, norm, activation = self.projections[i]
            if len(source):
                source = source.detach()
                if ablation == "zero" or (ablation == "s1_only" and i != 0) or (ablation == "s4_only" and i != self.source_count-1):
                    source = torch.zeros_like(source)
                elif ablation == "shuffle":
                    source = source.flip(0)
                # Stage I and the neighbor map are frozen. Pool raw features
                # without autograd, then project each candidate. The linear
                # operation commutes with weighted pooling, but this order
                # avoids scattering millions of candidate gradients back to
                # the much smaller Stage-I source set during backward.
                with torch.no_grad():
                    weights = torch.where(valid, 1 / (distance + 0.05), 0.)
                    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
                    pooled = source.new_zeros((n, source.shape[1]))
                    for neighbor in range(index.shape[1]):
                        pooled.add_(source.index_select(0, index[:, neighbor].clamp_min(0))
                                    * weights[:, neighbor, None])
                projected = F.linear(pooled, linear.weight, bias=None)
            else:
                projected = center.new_zeros((n, linear.out_features))
            if linear.bias is not None:
                projected = projected + linear.bias
            embedding = activation(norm(projected))
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
