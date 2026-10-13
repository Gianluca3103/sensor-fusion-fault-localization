"""Radar-only inference interface trained with a clean-LiDAR bottleneck teacher.

The public output contains exactly two sparse tensors: LiDAR-aware radar
bottleneck features and a scored 3D support field. Neither is a point cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .config import Stage1Config, VoxelGrid
from .correspondence import LocalCorrespondence, local_neighbors
from .pfe_spconv import PFEPointVoxelEncoder, PFESpconvBackbone
from .sparse import SparseSites, encode_keys
from .static_geometry_cache import Stage1StaticGeometryCache


@dataclass(frozen=True)
class BottleneckSupportOutput:
    bottleneck: SparseSites  # sparse stride-8 [N4, attention_dim]
    confidence_region: SparseSites  # sparse stride-2 [M, 1], scores in [0, 1]


class RadarOnlyBottleneckEncoder(nn.Module):
    """Infer LiDAR-aware features and support using radar, without clean LiDAR."""

    def __init__(self, config: Stage1Config):
        super().__init__()
        if not config.bottleneck_support_only or config.encoder_backend != "pfe_spconv":
            raise ValueError("The bottleneck-support interface requires the PFE sparse encoder")
        self.config = config
        self.point = PFEPointVoxelEncoder(config.grid, "radar")
        self.backbone = PFESpconvBackbone(config.channels)
        self.student = nn.Sequential(nn.Linear(config.channels[-1], config.attention_dim),
                                     nn.LayerNorm(config.attention_dim), nn.SiLU(),
                                     nn.Linear(config.attention_dim, config.attention_dim))
        self.support_head = nn.Sequential(
            nn.Linear(config.attention_dim + config.channels[1] + 4, config.channels[-1]),
            nn.LayerNorm(config.channels[-1]), nn.SiLU(),
            nn.Linear(config.channels[-1], 1))
        self.confidence_calibrated = False
        self.confidence_trained = True

    def encode(self, radar_points: torch.Tensor, radar_valid: torch.Tensor, *,
               defer_diagnostics: bool = False, check_finite: bool = True):
        sites, _ = self.point(radar_points, radar_valid,
                              defer_diagnostics=defer_diagnostics, check_finite=check_finite)
        levels, _ = self.backbone(sites, defer_diagnostics=defer_diagnostics)
        coarse = levels[-1]
        lidar_aware = coarse.replace_features(F.normalize(self.student(coarse.features), dim=-1))
        support, logits = self._support_field(lidar_aware, levels[1])
        return BottleneckSupportOutput(lidar_aware, support), levels, logits

    def forward(self, radar_points: torch.Tensor, radar_valid: torch.Tensor, *,
                defer_diagnostics: bool = False, check_finite: bool = True) -> BottleneckSupportOutput:
        output, _, _ = self.encode(radar_points, radar_valid,
                                   defer_diagnostics=defer_diagnostics, check_finite=check_finite)
        return output

    def _support_field(self, bottleneck: SparseSites, fine: SparseSites):
        """Decode every stride-8 cell to a scored stride-2 lattice.

        The lattice is a set of candidate *locations*, not a radar sphere or a
        positive label. The head must learn which cells receive confidence.
        """
        if bottleneck.stride != 8 or fine.stride != 2:
            raise ValueError("Expected stride-8 bottleneck and stride-2 radar skip")
        offsets = torch.tensor(list(product(range(4), repeat=3)), device=bottleneck.coords.device,
                               dtype=bottleneck.coords.dtype)
        n = len(bottleneck.coords)
        shape = self.config.grid.scale_shape(2)
        coordinates = bottleneck.coords[:, None, :].expand(n, 64, 4).clone()
        coordinates[:, :, 1:] = bottleneck.coords[:, None, 1:] * 4 + offsets[None]
        coordinates = coordinates.reshape(-1, 4)
        if not len(coordinates):
            empty = SparseSites._from_validated(coordinates, bottleneck.features.new_empty((0, 1)),
                                                shape, 2)
            return empty, bottleneck.features.new_empty((0, 1))
        order = torch.argsort(encode_keys(coordinates, shape), stable=True)
        coordinates = coordinates[order]
        coarse_features = bottleneck.features.repeat_interleave(64, dim=0)[order]
        relative = offsets.to(coarse_features.dtype).repeat(n, 1)[order]
        relative = (relative + .5) / 4 - .5
        keys = encode_keys(fine.coords, shape)
        queries = encode_keys(coordinates, shape)
        if len(keys):
            location = torch.searchsorted(keys, queries)
            matched = (location < len(keys)) & (keys[location.clamp(max=len(keys)-1)] == queries)
            skip = coarse_features.new_zeros((len(queries), fine.features.shape[1]))
            skip[matched] = fine.features[location[matched]]
        else:
            matched = torch.zeros(len(queries), dtype=torch.bool, device=queries.device)
            skip = coarse_features.new_zeros((len(queries), fine.features.shape[1]))
        logits = self.support_head(torch.cat((coarse_features, skip, relative,
                                              matched[:, None].to(coarse_features.dtype)), dim=-1))
        support = SparseSites._from_validated(coordinates, torch.sigmoid(logits), shape, 2)
        return support, logits


class BottleneckSupportStage1(nn.Module):
    """Training wrapper; only the radar-only submodule is exported."""

    def __init__(self, config: Stage1Config):
        super().__init__()
        if not config.bottleneck_support_only:
            raise ValueError("Use the bottleneck-support config")
        self.config = config
        self.radar_only = RadarOnlyBottleneckEncoder(config)
        self.lidar_point = PFEPointVoxelEncoder(config.grid, "lidar")
        self.lidar_backbone = PFESpconvBackbone(config.channels)
        self.correspondence = LocalCorrespondence(config.channels[-1], config.channels[-1],
                                                   config.attention_dim)

    def forward_radar(self, radar_points, radar_valid) -> BottleneckSupportOutput:
        return self.radar_only(radar_points, radar_valid)

    def forward_train(self, radar_points, radar_valid, clean_lidar, clean_valid, *,
                      defer_diagnostics: bool = False, trusted_inputs: bool = False,
                      static_cache: Stage1StaticGeometryCache | None = None,
                      frame_ids: list[str] | None = None, split: str | None = None):
        output, radar_levels, support_logits = self.radar_only.encode(
            radar_points, radar_valid, defer_diagnostics=defer_diagnostics,
            check_finite=not trusted_inputs)
        lidar_sites, _ = self.lidar_point(clean_lidar, clean_valid,
                                           defer_diagnostics=defer_diagnostics,
                                           check_finite=not trusted_inputs)
        lidar_levels, _ = self.lidar_backbone(lidar_sites,
                                               defer_diagnostics=defer_diagnostics)
        radar_coarse, lidar_coarse = radar_levels[-1], lidar_levels[-1]
        support = output.confidence_region
        if static_cache is None:
            from models.radar_lidar_stage2.candidate_domain import CandidateDomain
            from models.radar_lidar_stage2.voxel_target import make_targets
            neighbors = local_neighbors(radar_coarse, lidar_coarse, self.config.grid,
                                        self.config.attention_radii_m[-1], self.config.max_neighbors)
            coarse_grid = VoxelGrid(
                self.config.grid.minimum_xyz, self.config.grid.maximum_xyz,
                tuple(v * self.config.support_stride for v in self.config.grid.size_xyz))
            domain = CandidateDomain(support.coords, support.features[:, 0].detach(), coarse_grid, {})
            targets = make_targets(domain, clean_lidar, clean_valid)
            cache_hits = cache_builds = 0
        else:
            if frame_ids is None or split is None:
                raise ValueError("Static cache requires frame IDs and split")
            neighbors, targets, cache_hits, cache_builds = static_cache.get_batch(
                frame_ids, split, radar_coarse, lidar_coarse, support,
                clean_lidar, clean_valid)
        attention = self.correspondence(
            radar_coarse, lidar_coarse, self.config.grid, neighbors,
            self.config.attention_radii_m[-1], self.config.positive_radii_m[-1],
            self.config.temperature, self.config.negative_strategy,
            self.config.num_negatives, defer_diagnostics=defer_diagnostics)
        zero = output.bottleneck.features.sum() * 0
        positive = attention["positive_mask"]
        if len(lidar_coarse.coords):
            lidar_embedding = F.normalize(self.correspondence.key_value(lidar_coarse.features), dim=-1)
            selected = lidar_embedding[neighbors.indices.clamp_min(0)]
            weights = attention["weights"] * positive
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            target = F.normalize((selected * weights[:, :, None]).sum(1), dim=-1).detach()
            valid = positive.any(-1)
            error = 1 - F.cosine_similarity(output.bottleneck.features, target)
            distill = (error * valid).sum() / valid.sum().clamp_min(1)
        else:
            distill = zero

        # The clean scan supplies positives and measured free space. Unobserved
        # sites (including those behind a first return) are never labeled free.
        positives, free = targets.occupied, targets.known_free
        unknown = ~(positives | free)
        scores = support_logits[:, 0]
        positive_loss = (F.softplus(-scores).masked_fill(~positives, 0).sum() /
                         positives.sum().clamp_min(1))
        free_loss = (F.softplus(scores).masked_fill(~free, 0).sum() /
                     free.sum().clamp_min(1))
        abstention = (support.features[:, 0].masked_fill(~unknown, 0).sum() /
                      unknown.sum().clamp_min(1))
        support_loss = positive_loss + free_loss + self.config.support_abstention_weight * abstention
        total = (self.config.correspondence_weight * attention["loss"] +
                 self.config.bottleneck_distill_weight * distill +
                 self.config.support_loss_weight * support_loss)
        losses = {"loss/corr": attention["loss"], "loss/distill": distill,
                  "loss/support": support_loss, "loss/total": total}
        diagnostics = {"output": output, "occupied": positives, "known_free": free,
                       "positive_queries": attention["positive_queries"],
                       "support_sites": len(support.coords),
                       "clean_points_in_support": targets.clean_points_in_candidates,
                       "clean_points_in_grid": targets.clean_points_in_grid,
                       "cache_hits": cache_hits, "cache_builds": cache_builds}
        return losses, diagnostics

    def radar_only_state_dict(self) -> dict:
        return {"config": self.config.as_dict(), "radar_only": self.radar_only.state_dict(),
                "confidence_calibrated": False, "confidence_trained": True}
