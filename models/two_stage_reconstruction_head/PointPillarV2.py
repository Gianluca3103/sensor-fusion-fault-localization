"""Neighbor-aware PointPillars ablation for coarse BEV reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn

from .pointpillars import (
    BEVGridGeometry,
    PointPillarsEncoder,
    PointPillarsOutput,
)


@dataclass(frozen=True)
class PointPillarsV2Config:
    """Configuration for neighbor-aware pillar feature aggregation."""

    enabled: bool = False
    output_channels: int = 64
    max_points_per_pillar: int = 100
    max_pillars: int | None = None
    lidar_use_reflectivity: bool = True
    radar_use_power: bool = True
    radar_use_radial_velocity: bool = True
    neighbor_enabled: bool = True
    neighbor_radius_m: float = 0.4
    neighbor_max_neighbors: int = 16
    neighbor_initial_scale: float = 0.1

    @property
    def lidar_raw_channels(self) -> int:
        return 4 if self.lidar_use_reflectivity else 3

    @property
    def radar_raw_channels(self) -> int:
        return (
            3
            + int(self.radar_use_power)
            + int(self.radar_use_radial_velocity)
        )

    @property
    def lidar_decorated_channels(self) -> int:
        return self.lidar_raw_channels + 5

    @property
    def radar_decorated_channels(self) -> int:
        return self.radar_raw_channels + 5

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("pointpillars_v2.enabled must be boolean")
        if self.output_channels < 1:
            raise ValueError("pointpillars_v2.output_channels must be positive")
        if self.max_points_per_pillar < 1:
            raise ValueError(
                "pointpillars_v2.max_points_per_pillar must be positive"
            )
        if self.max_pillars is not None and self.max_pillars < 1:
            raise ValueError(
                "pointpillars_v2.max_pillars must be positive or null"
            )
        for name in (
            "lidar_use_reflectivity",
            "radar_use_power",
            "radar_use_radial_velocity",
            "neighbor_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"pointpillars_v2.{name} must be boolean")
        if (
            isinstance(self.neighbor_radius_m, bool)
            or not math.isfinite(float(self.neighbor_radius_m))
            or self.neighbor_radius_m <= 0
        ):
            raise ValueError(
                "pointpillars_v2.neighbor_radius_m must be finite and positive"
            )
        if (
            isinstance(self.neighbor_max_neighbors, bool)
            or self.neighbor_max_neighbors < 1
        ):
            raise ValueError(
                "pointpillars_v2.neighbor_max_neighbors must be at least 1"
            )
        if (
            isinstance(self.neighbor_initial_scale, bool)
            or not math.isfinite(float(self.neighbor_initial_scale))
            or self.neighbor_initial_scale < 0
        ):
            raise ValueError(
                "pointpillars_v2.neighbor_initial_scale must be finite and "
                "non-negative"
            )

    def to_dict(self) -> dict:
        return asdict(self)


class NeighborAwarePillarEnhancer(nn.Module):
    """Mix bounded nearby occupied-pillar features before dense scatter.

    Candidate lookup uses flattened sparse BEV coordinates and ``searchsorted``.
    It allocates ``P x K`` indices, where ``K`` is the fixed number of grid
    offsets inside the metric radius, rather than an unrestricted ``P x P``
    distance matrix. ``max_neighbors`` caps those valid offsets afterward; for
    0.20 m square cells and the default 0.40 m radius, K is only 12, so the
    default cap of 16 does not discard any neighbor. The learnable residual
    starts at 0.1 by default so V2 is initially close to the baseline.
    """

    def __init__(
        self,
        geometry: BEVGridGeometry,
        channels: int,
        *,
        radius_m: float = 0.4,
        max_neighbors: int = 16,
        initial_scale: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__()
        geometry.validate()
        if channels < 1:
            raise ValueError("channels must be positive")
        if not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("radius_m must be finite and positive")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be at least 1")
        if not math.isfinite(initial_scale) or initial_scale < 0:
            raise ValueError("initial_scale must be finite and non-negative")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")

        self.geometry = geometry
        self.channels = int(channels)
        self.radius_m = float(radius_m)
        self.max_neighbors = int(max_neighbors)
        self.enabled = enabled
        self.update = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.neighbor_scale = nn.Parameter(
            torch.tensor(float(initial_scale), dtype=torch.float32)
        )

        offsets = self._metric_offsets()
        self.register_buffer(
            "row_offsets",
            torch.tensor([offset[0] for offset in offsets], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "col_offsets",
            torch.tensor([offset[1] for offset in offsets], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "offset_distances",
            torch.tensor([offset[2] for offset in offsets], dtype=torch.float32),
            persistent=False,
        )

    def _metric_offsets(self) -> list[tuple[int, int, float]]:
        max_row = math.floor(self.radius_m / self.geometry.pillar_size_x)
        max_col = math.floor(self.radius_m / self.geometry.pillar_size_y)
        offsets = []
        for row_offset in range(-max_row, max_row + 1):
            for col_offset in range(-max_col, max_col + 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                distance = math.hypot(
                    row_offset * self.geometry.pillar_size_x,
                    col_offset * self.geometry.pillar_size_y,
                )
                if distance <= self.radius_m + 1.0e-9:
                    offsets.append((row_offset, col_offset, distance))
        offsets.sort(key=lambda item: (item[2], item[0], item[1]))
        return offsets

    def _neighbor_indices(
        self,
        pillar_batches: torch.Tensor,
        pillar_rows: torch.Tensor,
        pillar_cols: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pillar_count = len(pillar_rows)
        offset_count = len(self.row_offsets)
        if pillar_count == 0 or offset_count == 0:
            shape = (pillar_count, 0)
            return (
                torch.empty(shape, dtype=torch.long, device=pillar_rows.device),
                torch.empty(shape, dtype=torch.bool, device=pillar_rows.device),
            )

        cells = self.geometry.height * self.geometry.width
        keys = (
            pillar_batches * cells
            + pillar_rows * self.geometry.width
            + pillar_cols
        )
        sorted_keys, sorted_indices = torch.sort(keys)
        candidate_rows = pillar_rows[:, None] + self.row_offsets[None, :]
        candidate_cols = pillar_cols[:, None] + self.col_offsets[None, :]
        in_bounds = (
            (candidate_rows >= 0)
            & (candidate_rows < self.geometry.height)
            & (candidate_cols >= 0)
            & (candidate_cols < self.geometry.width)
        )
        candidate_keys = (
            pillar_batches[:, None] * cells
            + candidate_rows * self.geometry.width
            + candidate_cols
        )
        lookup = torch.searchsorted(
            sorted_keys,
            candidate_keys.flatten(),
        ).reshape_as(candidate_keys)
        safe_lookup = lookup.clamp_max(pillar_count - 1)
        valid = in_bounds & (lookup < pillar_count)
        valid &= sorted_keys[safe_lookup] == candidate_keys
        neighbor_indices = sorted_indices[safe_lookup]

        keep_count = min(self.max_neighbors, offset_count)
        if keep_count == offset_count:
            return neighbor_indices, valid
        if self.training:
            scores = torch.rand(
                valid.shape,
                device=valid.device,
                dtype=torch.float32,
            ).masked_fill(~valid, -1.0)
            chosen = scores.topk(keep_count, dim=1).indices
        else:
            distances = self.offset_distances[None, :].expand_as(valid)
            scores = distances.masked_fill(~valid, torch.inf)
            chosen = scores.topk(
                keep_count,
                dim=1,
                largest=False,
            ).indices
        return (
            neighbor_indices.gather(1, chosen),
            valid.gather(1, chosen),
        )

    @staticmethod
    def _statistics(
        neighbor_counts: torch.Tensor,
        pillar_batches: torch.Tensor,
        batch_size: int,
        neighbor_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pillars_per_sample = torch.bincount(
            pillar_batches,
            minlength=batch_size,
        )
        neighbor_sum = torch.zeros(
            batch_size,
            dtype=torch.float32,
            device=pillar_batches.device,
        )
        if len(neighbor_counts):
            neighbor_sum.index_add_(
                0,
                pillar_batches,
                neighbor_counts.to(torch.float32),
            )
        average = neighbor_sum / pillars_per_sample.clamp_min(1).to(torch.float32)
        maximum = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=pillar_batches.device,
        )
        if len(neighbor_counts):
            maximum.scatter_reduce_(
                0,
                pillar_batches,
                neighbor_counts,
                reduce="amax",
                include_self=True,
            )
        neighborless = torch.bincount(
            pillar_batches[neighbor_counts == 0],
            minlength=batch_size,
        )
        fraction = neighborless.to(torch.float32) / pillars_per_sample.clamp_min(1)
        return {
            "average_neighbors_per_pillar": average,
            "maximum_neighbors_per_pillar": maximum,
            "pillars_with_no_neighbors": neighborless,
            "neighborless_pillar_fraction": fraction,
            "neighbor_scale": neighbor_scale.detach().to(torch.float32).expand(
                batch_size
            ),
        }

    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_batches: torch.Tensor,
        pillar_rows: torch.Tensor,
        pillar_cols: torch.Tensor,
        *,
        batch_size: int | None = None,
        return_statistics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pillar_features.ndim != 2 or pillar_features.shape[1] != self.channels:
            raise ValueError(
                f"pillar_features must have shape [P,{self.channels}]"
            )
        pillar_count = len(pillar_features)
        for name, coordinates in (
            ("pillar_batches", pillar_batches),
            ("pillar_rows", pillar_rows),
            ("pillar_cols", pillar_cols),
        ):
            if coordinates.shape != (pillar_count,) or coordinates.dtype != torch.long:
                raise ValueError(f"{name} must have shape [P] and dtype long")
            if coordinates.device != pillar_features.device:
                raise ValueError(f"{name} must share the pillar feature device")
        if batch_size is None:
            batch_size = (
                int(pillar_batches.max().item()) + 1 if pillar_count else 0
            )

        neighbor_indices, valid_neighbors = self._neighbor_indices(
            pillar_batches,
            pillar_rows,
            pillar_cols,
        )
        neighbor_counts = valid_neighbors.sum(dim=1)
        if not self.enabled or pillar_count == 0 or not valid_neighbors.shape[1]:
            enhanced = pillar_features
        else:
            gathered = pillar_features[neighbor_indices]
            mask = valid_neighbors.unsqueeze(-1)
            neighbor_sum = (gathered * mask).sum(dim=1)
            neighbor_mean = neighbor_sum / neighbor_counts.clamp_min(1).to(
                pillar_features.dtype
            ).unsqueeze(1)
            minimum = torch.finfo(pillar_features.dtype).min
            neighbor_max = gathered.masked_fill(~mask, minimum).amax(dim=1)
            neighbor_max = torch.where(
                neighbor_counts[:, None] > 0,
                neighbor_max,
                torch.zeros_like(neighbor_max),
            )
            neighborhood_update = self.update(
                torch.cat(
                    (pillar_features, neighbor_mean, neighbor_max),
                    dim=1,
                )
            )
            neighborhood_update = neighborhood_update * (
                neighbor_counts[:, None] > 0
            )
            enhanced = (
                pillar_features
                + self.neighbor_scale.to(pillar_features.dtype)
                * neighborhood_update
            )

        if not return_statistics:
            return enhanced
        statistics = self._statistics(
            neighbor_counts,
            pillar_batches,
            batch_size,
            self.neighbor_scale,
        )
        return enhanced, statistics


class PointPillarsEncoderV2(PointPillarsEncoder):
    """Baseline PFN plus neighbor aggregation before unchanged BEV scatter."""

    def __init__(
        self,
        geometry: BEVGridGeometry,
        *,
        raw_channels: int,
        output_channels: int,
        max_points_per_pillar: int,
        max_pillars: int | None,
        neighbor_enabled: bool = True,
        neighbor_radius_m: float = 0.4,
        neighbor_max_neighbors: int = 16,
        neighbor_initial_scale: float = 0.1,
    ):
        super().__init__(
            geometry,
            raw_channels=raw_channels,
            output_channels=output_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
        )
        self.neighbor_enhancer = NeighborAwarePillarEnhancer(
            geometry,
            output_channels,
            radius_m=neighbor_radius_m,
            max_neighbors=neighbor_max_neighbors,
            initial_scale=neighbor_initial_scale,
            enabled=neighbor_enabled,
        )

    def forward(
        self,
        point_clouds: Sequence[torch.Tensor],
        *,
        return_sparse: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | PointPillarsOutput:
        if not point_clouds:
            raise ValueError("point_clouds must contain at least one sample")
        (
            features,
            point_to_pillar,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            statistics,
        ) = self._pillarize_batch(point_clouds)
        total_pillars = len(pillar_rows)
        if total_pillars:
            base_pillar_features = self.feature_net(
                features,
                point_to_pillar,
                total_pillars,
            )
        else:
            base_pillar_features = point_clouds[0].new_empty(
                (0, self.feature_net.output_channels)
            )
        enhanced_pillar_features, neighbor_statistics = self.neighbor_enhancer(
            base_pillar_features,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            batch_size=len(point_clouds),
            return_statistics=True,
        )
        statistics.update(neighbor_statistics)
        dense_features = self.scatter.forward_batch(
            enhanced_pillar_features,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            len(point_clouds),
        )
        if not return_sparse:
            return dense_features, statistics

        sparse_coordinates = torch.stack(
            (pillar_batches, pillar_rows, pillar_cols),
            dim=1,
        )
        return PointPillarsOutput(
            dense_features=dense_features,
            sparse_features=enhanced_pillar_features,
            sparse_coordinates=sparse_coordinates,
            statistics=statistics,
        )
