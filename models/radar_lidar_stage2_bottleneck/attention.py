"""One local, coordinate-aware fusion operation at the Stage-II bottleneck."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2.stage1_conditioning import physical_neighbor_map


def bottleneck_neighbors(query_coords: torch.Tensor, query_stride: int,
                         source_xyz: torch.Tensor, source_batch: torch.Tensor,
                         grid: VoxelGrid, radius_m: float, maximum: int,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Find source voxels in the same frame and return metric relative XYZ."""
    if query_coords.ndim != 2 or query_coords.shape[1] != 4:
        raise ValueError("Expected BZYX bottleneck coordinates")
    if source_xyz.shape != (len(source_batch), 3):
        raise ValueError("Stage-I feature coordinates and batch IDs differ")
    query_xyz = grid.centers_xyz(query_coords, stride=query_stride)
    idx, _ = physical_neighbor_map(
        query_xyz.detach().cpu().numpy(), query_coords[:, 0].detach().cpu().numpy(),
        source_xyz.detach().cpu().numpy(), source_batch.detach().cpu().numpy(),
        radius_m, maximum)
    index = torch.as_tensor(idx, dtype=torch.long, device=device)
    if len(source_xyz):
        centers = source_xyz.to(device=device, dtype=torch.float32)
        gathered = centers.index_select(0, index.clamp_min(0).reshape(-1))
        relative = gathered.reshape(len(index), maximum, 3) - query_xyz.to(device)[:, None]
        relative = torch.where(index[..., None] >= 0, relative, 0.)
    else:
        relative = torch.zeros((len(index), maximum, 3), device=device)
    return index, relative


class LocalBottleneckFusion(nn.Module):
    """Trainable attention or a spatially matched concatenation ablation."""

    def __init__(self, query_dim: int, source_dim: int, *, heads: int,
                 mode: str, radius_m: float, max_neighbors: int):
        super().__init__()
        if query_dim % heads or source_dim < 1 or mode not in {"cross_attention", "concat"}:
            raise ValueError("Invalid bottleneck fusion dimensions or mode")
        self.query_dim = query_dim
        self.source_dim = source_dim
        self.heads = heads
        self.mode = mode
        self.radius_m = radius_m
        self.max_neighbors = max_neighbors
        self.query_norm = nn.LayerNorm(query_dim)
        if mode == "cross_attention":
            self.q = nn.Linear(query_dim, query_dim)
            self.k = nn.Linear(source_dim, query_dim)
            self.v = nn.Linear(source_dim, query_dim)
            self.position = nn.Sequential(nn.Linear(3, query_dim), nn.SiLU(),
                                          nn.Linear(query_dim, heads))
            self.output = nn.Linear(query_dim, query_dim)
        else:
            self.output = nn.Sequential(nn.Linear(query_dim + source_dim + 3, query_dim),
                                        nn.SiLU(), nn.Linear(query_dim, query_dim))
        self.gate = nn.Parameter(torch.zeros(()))
        self.output_norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, query_coords: torch.Tensor,
                source: torch.Tensor, source_xyz: torch.Tensor,
                source_batch: torch.Tensor, grid: VoxelGrid, query_stride: int,
                *, neighbors: tuple[torch.Tensor, torch.Tensor] | None = None,
                cache: dict | None = None) -> torch.Tensor:
        if query.shape != (len(query_coords), self.query_dim):
            raise ValueError("Bottleneck query shape differs from sparse coordinates")
        if source.shape != (len(source_xyz), self.source_dim):
            raise ValueError("Stage-I source features and centers differ")
        if neighbors is None:
            neighbors = bottleneck_neighbors(query_coords, query_stride, source_xyz,
                                             source_batch, grid, self.radius_m,
                                             self.max_neighbors, query.device)
        if cache is not None and "bottleneck_neighbors" not in cache:
            cache["bottleneck_neighbors"] = neighbors
        index, relative = neighbors
        if index.shape != (len(query), self.max_neighbors) or relative.shape != (len(query), self.max_neighbors, 3):
            raise ValueError("Bottleneck neighbor map has the wrong shape")
        valid = index >= 0
        if not len(query) or not len(source):
            return query
        # Stage I is frozen, including when its features came from a disk cache.
        selected = source.detach().index_select(0, index.clamp_min(0).reshape(-1))
        selected = selected.reshape(len(query), self.max_neighbors, self.source_dim)
        normalized = self.query_norm(query)
        if self.mode == "cross_attention":
            head_dim = self.query_dim // self.heads
            q = self.q(normalized).reshape(len(query), self.heads, head_dim)
            k = self.k(selected).reshape(len(query), self.max_neighbors, self.heads, head_dim)
            v = self.v(selected).reshape(len(query), self.max_neighbors, self.heads, head_dim)
            logits = (q[:, None] * k).sum(-1) / math.sqrt(head_dim)
            logits = logits + self.position(relative.to(query.dtype) / self.radius_m)
            logits = logits.masked_fill(~valid[..., None], -1e4)
            weights = torch.softmax(logits.float(), dim=1).to(query.dtype)
            weights = weights * valid[..., None]
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            update = self.output((weights[..., None] * v).sum(1).reshape(len(query), self.query_dim))
        else:
            distance = torch.linalg.vector_norm(relative, dim=-1)
            weights = torch.where(valid, 1 / (distance + 0.05), 0.)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            pooled = (selected * weights[..., None]).sum(1)
            position = (relative * weights[..., None]).sum(1) / self.radius_m
            update = self.output(torch.cat((normalized, pooled, position.to(query.dtype)), dim=1))
        # No radar neighbor means no trainable cross-modal update at that site.
        update = update * valid.any(1)[:, None]
        fused = self.output_norm(query + torch.sigmoid(self.gate) * update)
        return torch.where(valid.any(1)[:, None], fused, query)
