"""Local radar/LiDAR attention on calibrated LiDAR ray-depth candidates.

This is a reconstruction blueprint, not a point generator. Inference uses only
radar and surviving LiDAR. Clean LiDAR can be encoded as a training teacher but
never participates in proposals, attention, gating, or first-return logits.
Process the scan in overlapping ray tiles to bound the neighborhood search.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .cross_modal_encoders import EncodedGrid, EncoderGrid, RadarLidarEncoders
from .range_view.geometry import RangeGeometry
from .ray_depth_queries import RayDepthQueries, propose_ray_depth_queries


@dataclass
class RayDepthBlueprint:
    queries: RayDepthQueries
    features: torch.Tensor              # [B,Q,D,C]
    evidence_weights: torch.Tensor      # [B,Q,D,3]: radar, observed, null
    first_return_logits: torch.Tensor   # [B,Q,D+1], final index means no return
    depth_residual_m: torch.Tensor       # [B,Q,D], bounded candidate correction
    clean_teacher: EncodedGrid | None = None

    @property
    def first_return_probabilities(self) -> torch.Tensor:
        """Uncalibrated model probabilities; calibrate on held-out data."""
        return self.first_return_logits.softmax(-1)

    def predicted_first_return(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return selected metric ranges and a mask; no-return ranges are NaN."""
        slots = self.queries.depths_m.shape[-1]
        selected = self.first_return_logits.argmax(-1)
        exists = selected != slots
        index = selected.clamp_max(slots - 1)[..., None]
        depth = (self.queries.depths_m + self.depth_residual_m).gather(-1, index).squeeze(-1)
        return depth.masked_fill(~exists, torch.nan), exists


def _grid_centres(encoded: EncodedGrid, grid: EncoderGrid, scene: int):
    features, zyx = encoded.active_tokens(occupied_only=True)[scene]
    if not len(zyx):
        return features, features.new_empty((0, 3))
    xyz = zyx[:, [2, 1, 0]].to(features.dtype)
    centre = ((xyz + 0.5) * features.new_tensor(grid.voxel_size_xyz)
              + features.new_tensor(grid.minimum_xyz))
    return features, centre


def _point_tokens(encoded: EncodedGrid, grid: EncoderGrid, scene: int):
    valid = encoded.point_valid[scene]
    xyz = encoded.point_xyz[scene, valid]
    fine = encoded.point_features[scene, valid]
    if not len(xyz):
        return fine, xyz
    minimum = xyz.new_tensor(grid.minimum_xyz)
    size = xyz.new_tensor(grid.voxel_size_xyz)
    index = torch.floor((xyz - minimum) / size).long()
    zyx_shape = grid.shape_zyx
    index = torch.stack((index[:, 2].clamp(0, zyx_shape[0] - 1),
                         index[:, 1].clamp(0, zyx_shape[1] - 1),
                         index[:, 0].clamp(0, zyx_shape[2] - 1)), -1)
    context = encoded.features[scene, :, index[:, 0], index[:, 1], index[:, 2]].T
    return fine + context, xyz


class _LocalCrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, neighbors: int, *, radar: bool,
                 coarse: bool, chunk_size: int) -> None:
        super().__init__()
        self.heads, self.neighbors = heads, neighbors
        self.radar, self.coarse, self.chunk_size = radar, coarse, chunk_size
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        # Relative XYZ, along-ray offset, perpendicular distance, normalized
        # RCS/Doppler/scan age, and query-to-radar Doppler agreement.
        self.bias = nn.Sequential(nn.Linear(10, width), nn.SiLU(),
                                  nn.Linear(width, heads))
        self.uncertainty = nn.Linear(width + 4, 2) if radar else None
        self.expected_doppler = nn.Linear(width, 1) if radar else None
        self.output = nn.Linear(width, width)

    def forward(self, query: torch.Tensor, direction: torch.Tensor,
                depth: torch.Tensor, source: torch.Tensor,
                source_xyz: torch.Tensor, metadata: torch.Tensor):
        count, width = query.shape
        result = torch.zeros_like(query)
        available = torch.zeros(count, dtype=torch.bool, device=query.device)
        if not len(source):
            return result, available
        head_width = width // self.heads
        keys = self.key(source).reshape(-1, self.heads, head_width)
        values = self.value(source).reshape(-1, self.heads, head_width)
        if self.radar:
            # Direction-dependent uncertainty grows with range. Learned scales
            # are bounded so the network cannot trivially attend globally.
            adjustment = self.uncertainty(torch.cat((source, metadata), -1)).tanh()
            long_scale = (2.0 if self.coarse else 1.0) * adjustment[:, 0].exp()
            angular_scale = (0.12 if self.coarse else 0.06) * adjustment[:, 1].exp()
        else:
            long_scale = source.new_full((len(source),), 2.5 if self.coarse else 1.0)
            angular_scale = source.new_full((len(source),), 0.06 if self.coarse else 0.025)
        source_range = source_xyz.norm(dim=-1)
        for start in range(0, count, self.chunk_size):
            end = min(start + self.chunk_size, count)
            ray = direction[start:end]
            along = ray @ source_xyz.T
            lateral = (source_xyz.square().sum(-1)[None, :] - along.square()).clamp_min(0).sqrt()
            offset = along - depth[start:end, None]
            lateral_scale = 0.35 + source_range[None, :] * angular_scale[None, :]
            metric = (offset / long_scale[None, :]).square() + (lateral / lateral_scale).square()
            k = min(self.neighbors, len(source))
            best, indices = metric.topk(k, dim=-1, largest=False)
            mask = best <= 9.0
            local_available = mask.any(-1)
            available[start:end] = local_available
            picked_xyz = source_xyz[indices]
            origin = ray[:, None, :] * depth[start:end, None, None]
            delta = (picked_xyz - origin) / 10.0
            attrs = metadata[indices]
            if self.radar:
                assert self.expected_doppler is not None
                doppler_agreement = (attrs[..., 2:3] -
                    self.expected_doppler(query[start:end]).tanh()[:, None, :])
            else:
                doppler_agreement = attrs[..., :1] * 0
            relative = torch.cat((delta, offset.gather(1, indices)[..., None] / 10,
                                  lateral.gather(1, indices)[..., None] / 10,
                                  attrs, doppler_agreement), -1)
            q = self.query(query[start:end]).reshape(-1, self.heads, head_width)
            dot = (q[:, None] * keys[indices]).sum(-1) / math.sqrt(head_width)
            logits = dot + self.bias(relative) - 0.25 * best[..., None]
            logits = logits.masked_fill(~mask[..., None], -1e4)
            weights = logits.softmax(1) * mask[..., None]
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            attended = (weights[..., None] * values[indices]).sum(1).reshape(-1, width)
            result[start:end] = self.output(attended) * local_available[:, None]
        return result, available


class _RayNeighborhoodAttention(nn.Module):
    def __init__(self, width: int, heads: int, neighbors: int,
                 chunk_size: int) -> None:
        super().__init__()
        self.heads, self.neighbors, self.chunk_size = heads, neighbors, chunk_size
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, values: torch.Tensor, rows: torch.Tensor,
                cols: torch.Tensor, depths: torch.Tensor, valid: torch.Tensor,
                azimuth_bins: int, wrap: bool) -> torch.Tensor:
        count, width = values.shape
        result = torch.zeros_like(values)
        h = self.heads
        keys = self.k(values).reshape(count, h, width // h)
        contents = self.v(values).reshape(count, h, width // h)
        for start in range(0, count, self.chunk_size):
            end = min(start + self.chunk_size, count)
            dr = (rows[start:end, None] - rows[None, :]).abs()
            dc = (cols[start:end, None] - cols[None, :]).abs()
            if wrap:
                dc = torch.minimum(dc, azimuth_bins - dc)
            dd = (depths[start:end, None] - depths[None, :]).abs()
            eligible = (dr <= 1) & (dc <= 2) & (dd <= 4.0) & valid[None, :]
            distance = dr.float().square() + 0.25 * dc.float().square() + 0.1 * dd.square()
            distance = distance.masked_fill(~eligible, torch.inf)
            nearest, index = distance.topk(min(self.neighbors, count), -1, largest=False)
            mask = torch.isfinite(nearest)
            query = self.q(values[start:end]).reshape(-1, h, width // h)
            logits = ((query[:, None] * keys[index]).sum(-1) / math.sqrt(width // h)
                      - 0.1 * nearest.clamp_max(100)[..., None])
            logits = logits.masked_fill(~mask[..., None], -1e4)
            weights = logits.softmax(1) * mask[..., None]
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            result[start:end] = self.output(
                (weights[..., None] * contents[index]).sum(1).reshape(-1, width)
            ) * (mask.any(-1) & valid[start:end])[:, None]
        return result


class _FusionBlock(nn.Module):
    def __init__(self, width: int, heads: int, neighbors: int,
                 chunk_size: int) -> None:
        super().__init__()
        self.radar = nn.ModuleList(_LocalCrossAttention(width, heads, neighbors,
            radar=True, coarse=coarse, chunk_size=chunk_size) for coarse in (False, True))
        self.lidar = nn.ModuleList(_LocalCrossAttention(width, heads, neighbors,
            radar=False, coarse=coarse, chunk_size=chunk_size) for coarse in (False, True))
        self.scale_gate = nn.ModuleList(nn.Linear(width, 2) for _ in range(2))
        self.evidence_gate = nn.Linear(width * 3, 3)
        self.null_feature = nn.Parameter(torch.zeros(width))
        self.norm = nn.LayerNorm(width)
        self.exchange = _RayNeighborhoodAttention(width, heads, neighbors, chunk_size)
        self.exchange_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(),
                                          nn.Linear(width * 2, width))
        self.final_norm = nn.LayerNorm(width)

    def forward(self, feature, directions, depths, rows, cols, valid,
                radar_sources, lidar_sources, azimuth_bins, wrap):
        branches, branch_valid = [], []
        for modality, sources, scale_index in ((self.radar, radar_sources, 0),
                                                (self.lidar, lidar_sources, 1)):
            scale_features, scale_available = [], []
            for attention, (tokens, xyz, attrs) in zip(modality, sources):
                attended, available = attention(feature, directions, depths,
                                                tokens, xyz, attrs)
                scale_features.append(attended)
                scale_available.append(available)
            available = torch.stack(scale_available, -1)
            logits = self.scale_gate[scale_index](feature).masked_fill(~available, -1e4)
            weight = logits.softmax(-1) * available
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            branches.append((torch.stack(scale_features, -2) * weight[..., None]).sum(-2))
            branch_valid.append(available.any(-1))
        evidence = torch.stack(branch_valid + [torch.ones_like(valid)], -1)
        weights = self.evidence_gate(torch.cat((feature, *branches), -1))
        weights = weights.masked_fill(~evidence, -1e4).softmax(-1)
        fused = (weights[:, 0:1] * branches[0] + weights[:, 1:2] * branches[1]
                 + weights[:, 2:3] * self.null_feature)
        feature = self.norm(feature + fused)
        feature = self.exchange_norm(feature + self.exchange(
            feature, rows, cols, depths, valid, azimuth_bins, wrap))
        feature = self.final_norm(feature + self.feed_forward(feature))
        return feature * valid[:, None], weights


class RayDepthCrossAttention(nn.Module):
    """Two-scale, two-block ray-query fusion with an explicit null path."""

    def __init__(self, grid: EncoderGrid = EncoderGrid(), width: int = 32,
                 heads: int = 4, neighbors: int = 16, chunk_size: int = 64,
                 max_candidates: int = 2048, history_scans: int = 20) -> None:
        super().__init__()
        if (width < 8 or heads < 1 or width % heads or neighbors < 1 or
                chunk_size < 1 or max_candidates < 1 or history_scans < 1):
            raise ValueError("Invalid attention width, heads, neighbors or chunk size")
        self.grid, self.width = grid, width
        self.max_candidates, self.history_scans = max_candidates, history_scans
        self.query_embedding = nn.Sequential(nn.Linear(8, width), nn.LayerNorm(width),
                                             nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(_FusionBlock(width, heads, neighbors, chunk_size)
                                    for _ in range(2))
        self.return_head = nn.Linear(width, 1)
        self.depth_head = nn.Linear(width, 1)
        self.no_return_head = nn.Linear(width, 1)

    def forward(self, geometry: RangeGeometry, queries: RayDepthQueries,
                radar: EncodedGrid, observed_lidar: EncodedGrid,
                raw_radar: torch.Tensor) -> RayDepthBlueprint:
        batch, rays, slots = queries.depths_m.shape
        count = rays * slots
        if count > self.max_candidates:
            raise ValueError("Ray tile is too large; reduce its size or raise max_candidates")
        if radar.features.shape[0] != batch or observed_lidar.features.shape[0] != batch:
            raise ValueError("Encoded sensors and rays must have the same batch size")
        if raw_radar.shape[:2] != radar.point_valid.shape or raw_radar.shape[-1] != 7:
            raise ValueError("Raw radar must match the encoded [B,N,7] points")
        output, gates = [], []
        for scene in range(batch):
            direction = queries.directions[scene, :, None, :].expand(-1, slots, -1).reshape(-1, 3)
            depth = queries.depths_m[scene].reshape(-1)
            valid = queries.valid[scene].reshape(-1)
            source_id = queries.source[scene].reshape(-1)
            qinput = torch.cat((direction, torch.log1p(depth[:, None]) / 5,
                                F.one_hot(source_id, 3).to(depth.dtype),
                                (depth / geometry.max_range_m)[:, None]), -1)
            feature = self.query_embedding(qinput) * valid[:, None]
            rf, rxyz = _point_tokens(radar, self.grid, scene)
            raw_metadata = raw_radar[scene, radar.point_valid[scene], 3:7]
            rm = torch.cat((
                torch.tanh(raw_metadata[:, :1] / 30),
                torch.tanh(raw_metadata[:, 1:3] / 20),
                raw_metadata[:, 3:4] / max(self.history_scans - 1, 1),
            ), -1)
            rc, rcxyz = _grid_centres(radar, self.grid, scene)
            lf, lxyz = _point_tokens(observed_lidar, self.grid, scene)
            lc, lcxyz = _grid_centres(observed_lidar, self.grid, scene)
            radar_sources = ((rf, rxyz, rm), (rc, rcxyz, rc.new_zeros((len(rc), 4))))
            lidar_sources = ((lf, lxyz, lf.new_zeros((len(lf), 4))),
                             (lc, lcxyz, lc.new_zeros((len(lc), 4))))
            rows = queries.rows[scene, :, None].expand(-1, slots).reshape(-1)
            cols = queries.cols[scene, :, None].expand(-1, slots).reshape(-1)
            for block in self.blocks:
                feature, weight = block(feature, direction, depth, rows, cols,
                    valid, radar_sources, lidar_sources, geometry.azimuth_bins,
                    abs(geometry.azimuth_span_rad - 2 * math.pi) < 1e-6)
            output.append(feature.reshape(rays, slots, self.width))
            gates.append(weight.reshape(rays, slots, 3))
        output = torch.stack(output)
        gates = torch.stack(gates)
        candidate_logits = self.return_head(output).squeeze(-1)
        # The null path is operational, not merely a visualization: a ray
        # candidate with no local sensor support cannot claim a return.
        supported = gates[..., :2].sum(-1) > 0
        candidate_logits = candidate_logits.masked_fill(~(queries.valid & supported), -1e4)
        residual = 3.0 * self.depth_head(output).squeeze(-1).tanh()
        residual = residual.clamp(
            min=geometry.min_range_m - queries.depths_m,
            max=geometry.max_range_m - queries.depths_m,
        )
        # Null logit pools only valid candidates and remains available even if
        # every observed and radar measurement is absent.
        pooled = (output * queries.valid[..., None]).sum(2) / queries.valid.sum(2).clamp_min(1)[..., None]
        no_return = self.no_return_head(pooled)
        return RayDepthBlueprint(queries, output, gates,
                                 torch.cat((candidate_logits, no_return), -1), residual)


class RayDepthBlueprintModel(nn.Module):
    """End-to-end encoder + query + fusion entry point; clean is teacher only."""

    def __init__(self, geometry: RangeGeometry,
                 grid: EncoderGrid = EncoderGrid(), width: int = 32,
                 history_scans: int = 20, **attention_kwargs) -> None:
        super().__init__()
        self.geometry = geometry
        self.encoders = RadarLidarEncoders(grid, width, history_scans)
        self.fusion = RayDepthCrossAttention(grid, width,
                                             history_scans=history_scans,
                                             **attention_kwargs)

    def forward(self, radar: torch.Tensor, radar_valid: torch.Tensor,
                observed_lidar: torch.Tensor, observed_valid: torch.Tensor,
                rows: torch.Tensor, cols: torch.Tensor, *,
                clean_lidar: torch.Tensor | None = None,
                clean_valid: torch.Tensor | None = None) -> RayDepthBlueprint:
        encoded = self.encoders(radar, radar_valid, observed_lidar, observed_valid,
                                clean_lidar=clean_lidar, clean_valid=clean_valid)
        queries = propose_ray_depth_queries(self.geometry, rows, cols, radar,
            radar_valid, observed_lidar, observed_valid)
        blueprint = self.fusion(self.geometry, queries, encoded["radar"],
                                encoded["observed_lidar"], radar)
        blueprint.clean_teacher = encoded.get("clean_teacher")
        return blueprint
