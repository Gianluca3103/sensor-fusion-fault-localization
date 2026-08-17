"""PointPillars-style sensor encoders for coarse BEV reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class BEVGridGeometry:
    """Physical geometry shared by generated BEVs, masks, and pillars."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: int = 320
    width: int = 320

    @property
    def pillar_size_x(self) -> float:
        return (self.x_max - self.x_min) / self.height

    @property
    def pillar_size_y(self) -> float:
        return (self.y_max - self.y_min) / self.width

    def validate(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("BEV grid height and width must be positive")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("BEV grid bounds must be increasing")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PointPillarsConfig:
    """Configuration for the representation ablation's two pillar encoders."""

    enabled: bool = False
    output_channels: int = 64
    max_points_per_pillar: int = 100
    max_pillars: int | None = 12000
    lidar_use_reflectivity: bool = True
    radar_use_power: bool = True
    radar_use_radial_velocity: bool = True

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
            raise ValueError("pointpillars.enabled must be boolean")
        if self.output_channels < 1:
            raise ValueError("pointpillars.output_channels must be positive")
        if self.max_points_per_pillar < 1:
            raise ValueError("pointpillars.max_points_per_pillar must be positive")
        if self.max_pillars is not None and self.max_pillars < 1:
            raise ValueError(
                "pointpillars.max_pillars must be positive or null"
            )
        for name in (
            "lidar_use_reflectivity",
            "radar_use_power",
            "radar_use_radial_velocity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"pointpillars.{name} must be boolean")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PillarizedPointCloud:
    """Decorated points and their dense-canvas pillar locations."""

    features: torch.Tensor
    point_to_pillar: torch.Tensor
    pillar_rows: torch.Tensor
    pillar_cols: torch.Tensor
    statistics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PointPillarsOutput:
    """Dense and sparse views of one batched PointPillars encoding."""

    dense_features: torch.Tensor
    sparse_features: torch.Tensor
    sparse_coordinates: torch.Tensor
    statistics: dict[str, torch.Tensor]


class Pillarizer(nn.Module):
    """Assign one variable-sized point cloud to the reconstruction BEV grid."""

    def __init__(
        self,
        geometry: BEVGridGeometry,
        *,
        raw_channels: int,
        max_points_per_pillar: int,
        max_pillars: int | None,
    ):
        super().__init__()
        geometry.validate()
        if raw_channels < 3:
            raise ValueError("raw_channels must include at least XYZ")
        self.geometry = geometry
        self.raw_channels = int(raw_channels)
        self.max_points_per_pillar = int(max_points_per_pillar)
        self.max_pillars = (
            None if max_pillars is None else int(max_pillars)
        )

    def grid_indices(
        self, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map XYZ to the exact row/column convention used by metric_to_grid."""

        finite = torch.isfinite(xyz[:, :3]).all(dim=1)
        valid = (
            finite
            & (xyz[:, 0] >= self.geometry.x_min)
            & (xyz[:, 0] < self.geometry.x_max)
            & (xyz[:, 1] >= self.geometry.y_min)
            & (xyz[:, 1] < self.geometry.y_max)
        )
        rows_from_bottom = torch.floor(
            (xyz[:, 0] - self.geometry.x_min) / self.geometry.pillar_size_x
        ).to(torch.long)
        cols = torch.floor(
            (xyz[:, 1] - self.geometry.y_min) / self.geometry.pillar_size_y
        ).to(torch.long)
        rows = self.geometry.height - 1 - rows_from_bottom
        valid &= (
            (rows >= 0)
            & (rows < self.geometry.height)
            & (cols >= 0)
            & (cols < self.geometry.width)
        )
        return rows, cols, valid

    def _select_pillars(
        self,
        inverse: torch.Tensor,
        pillar_count: int,
    ) -> torch.Tensor:
        if self.max_pillars is None or pillar_count <= self.max_pillars:
            return torch.ones(pillar_count, dtype=torch.bool, device=inverse.device)
        selected = torch.zeros(pillar_count, dtype=torch.bool, device=inverse.device)
        if self.training:
            indices = torch.randperm(pillar_count, device=inverse.device)[
                : self.max_pillars
            ]
        else:
            indices = torch.arange(self.max_pillars, device=inverse.device)
        selected[indices] = True
        return selected

    def forward(self, points: torch.Tensor) -> PillarizedPointCloud:
        if points.ndim != 2 or points.shape[1] != self.raw_channels:
            raise ValueError(
                f"points must have shape [N,{self.raw_channels}], got "
                f"{tuple(points.shape)}"
            )
        rows, cols, valid = self.grid_indices(points[:, :3])
        points = points[valid]
        rows = rows[valid]
        cols = cols[valid]
        raw_point_count = len(points)
        if raw_point_count == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=points.device)
            empty_features = points.new_empty((0, self.raw_channels + 5))
            scalar_zero = torch.zeros((), dtype=torch.long, device=points.device)
            return PillarizedPointCloud(
                features=empty_features,
                point_to_pillar=empty_long,
                pillar_rows=empty_long,
                pillar_cols=empty_long,
                statistics={
                    "raw_points": scalar_zero,
                    "nonempty_pillars": scalar_zero,
                    "available_nonempty_pillars": scalar_zero,
                    "retained_points": scalar_zero,
                    "average_points_per_pillar": points.new_zeros(()),
                    "maximum_points_per_pillar": scalar_zero,
                    "empty_pillar_fraction": points.new_ones(()),
                },
            )

        flat = rows * self.geometry.width + cols
        unique_flat, inverse, counts = torch.unique(
            flat,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        available_pillars = len(unique_flat)
        selected_pillars = self._select_pillars(inverse, available_pillars)
        point_selected = selected_pillars[inverse]
        points = points[point_selected]
        flat = flat[point_selected]

        order = torch.argsort(flat, stable=True)
        points = points[order]
        flat = flat[order]
        _, sorted_counts = torch.unique_consecutive(
            flat, return_counts=True
        )
        starts = torch.cumsum(sorted_counts, dim=0) - sorted_counts
        rank = torch.arange(len(points), device=points.device) - torch.repeat_interleave(
            starts, sorted_counts
        )
        keep = rank < self.max_points_per_pillar
        points = points[keep]
        flat = flat[keep]

        pillar_flat, point_to_pillar, retained_counts = torch.unique_consecutive(
            flat,
            return_inverse=True,
            return_counts=True,
        )
        pillar_count = len(pillar_flat)
        xyz = points[:, :3]
        cluster_sum = xyz.new_zeros((pillar_count, 3))
        cluster_sum.index_add_(0, point_to_pillar, xyz)
        cluster_mean = cluster_sum / retained_counts.to(xyz.dtype).unsqueeze(1)
        cluster_offsets = xyz - cluster_mean[point_to_pillar]

        pillar_rows = torch.div(
            pillar_flat, self.geometry.width, rounding_mode="floor"
        )
        pillar_cols = pillar_flat.remainder(self.geometry.width)
        rows_from_bottom = self.geometry.height - 1 - pillar_rows
        center_x = self.geometry.x_min + (
            rows_from_bottom.to(xyz.dtype) + 0.5
        ) * self.geometry.pillar_size_x
        center_y = self.geometry.y_min + (
            pillar_cols.to(xyz.dtype) + 0.5
        ) * self.geometry.pillar_size_y
        center_offsets = torch.stack(
            (
                xyz[:, 0] - center_x[point_to_pillar],
                xyz[:, 1] - center_y[point_to_pillar],
            ),
            dim=1,
        )
        decorated = torch.cat((points, cluster_offsets, center_offsets), dim=1)
        statistics = {
            "raw_points": torch.as_tensor(
                raw_point_count, dtype=torch.long, device=points.device
            ),
            "nonempty_pillars": torch.as_tensor(
                pillar_count, dtype=torch.long, device=points.device
            ),
            "available_nonempty_pillars": torch.as_tensor(
                available_pillars, dtype=torch.long, device=points.device
            ),
            "retained_points": torch.as_tensor(
                len(points), dtype=torch.long, device=points.device
            ),
            "average_points_per_pillar": retained_counts.to(torch.float32).mean(),
            "maximum_points_per_pillar": counts.max(),
            "empty_pillar_fraction": points.new_tensor(
                1.0 - pillar_count / (self.geometry.height * self.geometry.width)
            ),
        }
        return PillarizedPointCloud(
            features=decorated,
            point_to_pillar=point_to_pillar,
            pillar_rows=pillar_rows,
            pillar_cols=pillar_cols,
            statistics=statistics,
        )


class PillarFeatureNet(nn.Module):
    """Point-wise projection and max pooling for one sensor's pillars."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.linear = nn.Linear(input_channels, output_channels, bias=False)
        self.normalization = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.output_channels = int(output_channels)

    def forward(
        self,
        features: torch.Tensor,
        point_to_pillar: torch.Tensor,
        pillar_count: int,
    ) -> torch.Tensor:
        if pillar_count == 0:
            return features.new_empty((0, self.output_channels))
        encoded = self.activation(self.normalization(self.linear(features)))
        pooled = encoded.new_full(
            (pillar_count, self.output_channels),
            -torch.inf,
        )
        pooled.scatter_reduce_(
            0,
            point_to_pillar[:, None].expand(-1, self.output_channels),
            encoded,
            reduce="amax",
            include_self=True,
        )
        return pooled


class PillarScatter(nn.Module):
    """Scatter sparse pillar features into the aligned dense pseudo-image."""

    def __init__(self, geometry: BEVGridGeometry, channels: int):
        super().__init__()
        self.geometry = geometry
        self.channels = int(channels)

    def forward(
        self,
        pillar_features: torch.Tensor,
        pillar_rows: torch.Tensor,
        pillar_cols: torch.Tensor,
    ) -> torch.Tensor:
        canvas = pillar_features.new_zeros(
            (self.channels, self.geometry.height * self.geometry.width)
        )
        if len(pillar_features):
            flat = pillar_rows * self.geometry.width + pillar_cols
            canvas.index_copy_(1, flat, pillar_features.transpose(0, 1))
        return canvas.reshape(
            self.channels, self.geometry.height, self.geometry.width
        )

    def forward_batch(
        self,
        pillar_features: torch.Tensor,
        pillar_batches: torch.Tensor,
        pillar_rows: torch.Tensor,
        pillar_cols: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Scatter every sample with one device-side indexed write."""

        dense = pillar_features.new_zeros(
            (
                batch_size,
                self.channels,
                self.geometry.height * self.geometry.width,
            )
        )
        if len(pillar_features):
            flat = pillar_rows * self.geometry.width + pillar_cols
            dense[pillar_batches, :, flat] = pillar_features
        return dense.reshape(
            batch_size,
            self.channels,
            self.geometry.height,
            self.geometry.width,
        )


class PointPillarsEncoder(nn.Module):
    """Encode a batch of variable-sized point clouds as dense pseudo-BEVs."""

    def __init__(
        self,
        geometry: BEVGridGeometry,
        *,
        raw_channels: int,
        output_channels: int,
        max_points_per_pillar: int,
        max_pillars: int | None,
    ):
        super().__init__()
        self.pillarizer = Pillarizer(
            geometry,
            raw_channels=raw_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
        )
        self.feature_net = PillarFeatureNet(raw_channels + 5, output_channels)
        self.scatter = PillarScatter(geometry, output_channels)

    def _select_batched_pillars(
        self,
        pillar_batches: torch.Tensor,
        available_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Select at most the configured pillar count without sample loops."""

        pillar_count = len(pillar_batches)
        if pillar_count == 0:
            return torch.empty(
                0, dtype=torch.bool, device=pillar_batches.device
            )
        if self.pillarizer.max_pillars is None:
            return torch.ones(
                pillar_count, dtype=torch.bool, device=pillar_batches.device
            )
        if self.training:
            randomized = torch.argsort(
                torch.rand(pillar_count, device=pillar_batches.device),
                stable=True,
            )
            grouped = torch.argsort(
                pillar_batches[randomized], stable=True
            )
            candidates = randomized[grouped]
        else:
            candidates = torch.arange(
                pillar_count, device=pillar_batches.device
            )
        starts = torch.cumsum(available_per_sample, dim=0) - available_per_sample
        ranks = torch.arange(
            pillar_count, device=pillar_batches.device
        ) - torch.repeat_interleave(starts, available_per_sample)
        chosen = candidates[
            ranks < self.pillarizer.max_pillars
        ]
        selected = torch.zeros(
            pillar_count, dtype=torch.bool, device=pillar_batches.device
        )
        selected[chosen] = True
        return selected

    def _pillarize_batch(
        self,
        point_clouds: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Pillarize a complete sensor batch with batched CPU/CUDA kernels."""

        batch_size = len(point_clouds)
        reference = point_clouds[0]
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != self.pillarizer.raw_channels:
                raise ValueError(
                    "Each point cloud must have shape "
                    f"[N,{self.pillarizer.raw_channels}], got "
                    f"{tuple(points.shape)}"
                )
            if points.device != reference.device or points.dtype != reference.dtype:
                raise ValueError(
                    "All point clouds in a batch must share device and dtype"
                )
        lengths = torch.as_tensor(
            [len(points) for points in point_clouds],
            dtype=torch.long,
            device=reference.device,
        )
        all_points = torch.cat(point_clouds, dim=0)
        point_batches = torch.repeat_interleave(
            torch.arange(batch_size, device=reference.device), lengths
        )
        rows, cols, valid = self.pillarizer.grid_indices(all_points[:, :3])
        points = all_points[valid]
        point_batches = point_batches[valid]
        rows = rows[valid]
        cols = cols[valid]
        raw_points = torch.bincount(point_batches, minlength=batch_size)
        cells = (
            self.pillarizer.geometry.height
            * self.pillarizer.geometry.width
        )

        if not len(points):
            empty_long = torch.empty(
                0, dtype=torch.long, device=reference.device
            )
            empty_features = reference.new_empty(
                (0, self.pillarizer.raw_channels + 5)
            )
            zeros_long = torch.zeros(
                batch_size, dtype=torch.long, device=reference.device
            )
            zeros_float = reference.new_zeros(batch_size)
            return (
                empty_features,
                empty_long,
                empty_long,
                empty_long,
                empty_long,
                {
                    "raw_points": zeros_long,
                    "nonempty_pillars": zeros_long,
                    "available_nonempty_pillars": zeros_long,
                    "retained_points": zeros_long,
                    "average_points_per_pillar": zeros_float,
                    "maximum_points_per_pillar": zeros_long,
                    "empty_pillar_fraction": reference.new_ones(batch_size),
                },
            )

        local_flat = rows * self.pillarizer.geometry.width + cols
        global_flat = point_batches * cells + local_flat
        available_flat, inverse, available_counts = torch.unique(
            global_flat,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        available_batches = torch.div(
            available_flat, cells, rounding_mode="floor"
        )
        available_per_sample = torch.bincount(
            available_batches, minlength=batch_size
        )
        maximum_points = torch.zeros(
            batch_size, dtype=torch.long, device=reference.device
        )
        maximum_points.scatter_reduce_(
            0,
            available_batches,
            available_counts,
            reduce="amax",
            include_self=True,
        )
        selected_pillars = self._select_batched_pillars(
            available_batches, available_per_sample
        )
        selected_points = selected_pillars[inverse]
        points = points[selected_points]
        global_flat = global_flat[selected_points]

        order = torch.argsort(global_flat, stable=True)
        points = points[order]
        global_flat = global_flat[order]
        _, counts_before_cap = torch.unique_consecutive(
            global_flat, return_counts=True
        )
        starts = torch.cumsum(counts_before_cap, dim=0) - counts_before_cap
        point_rank = torch.arange(
            len(points), device=reference.device
        ) - torch.repeat_interleave(starts, counts_before_cap)
        keep = point_rank < self.pillarizer.max_points_per_pillar
        points = points[keep]
        global_flat = global_flat[keep]

        pillar_flat, point_to_pillar, retained_counts = torch.unique_consecutive(
            global_flat,
            return_inverse=True,
            return_counts=True,
        )
        pillar_count = len(pillar_flat)
        pillar_batches = torch.div(
            pillar_flat, cells, rounding_mode="floor"
        )
        pillar_local_flat = pillar_flat.remainder(cells)
        pillar_rows = torch.div(
            pillar_local_flat,
            self.pillarizer.geometry.width,
            rounding_mode="floor",
        )
        pillar_cols = pillar_local_flat.remainder(
            self.pillarizer.geometry.width
        )

        xyz = points[:, :3]
        cluster_sum = xyz.new_zeros((pillar_count, 3))
        cluster_sum.index_add_(0, point_to_pillar, xyz)
        cluster_mean = cluster_sum / retained_counts.to(xyz.dtype).unsqueeze(1)
        cluster_offsets = xyz - cluster_mean[point_to_pillar]
        rows_from_bottom = self.pillarizer.geometry.height - 1 - pillar_rows
        center_x = self.pillarizer.geometry.x_min + (
            rows_from_bottom.to(xyz.dtype) + 0.5
        ) * self.pillarizer.geometry.pillar_size_x
        center_y = self.pillarizer.geometry.y_min + (
            pillar_cols.to(xyz.dtype) + 0.5
        ) * self.pillarizer.geometry.pillar_size_y
        center_offsets = torch.stack(
            (
                xyz[:, 0] - center_x[point_to_pillar],
                xyz[:, 1] - center_y[point_to_pillar],
            ),
            dim=1,
        )
        decorated = torch.cat(
            (points, cluster_offsets, center_offsets), dim=1
        )

        nonempty = torch.bincount(pillar_batches, minlength=batch_size)
        retained = torch.bincount(
            pillar_batches[point_to_pillar], minlength=batch_size
        )
        average = retained.to(torch.float32) / nonempty.clamp_min(1)
        statistics = {
            "raw_points": raw_points,
            "nonempty_pillars": nonempty,
            "available_nonempty_pillars": available_per_sample,
            "retained_points": retained,
            "average_points_per_pillar": average,
            "maximum_points_per_pillar": maximum_points,
            "empty_pillar_fraction": 1.0 - nonempty.to(reference.dtype) / cells,
        }
        return (
            decorated,
            point_to_pillar,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            statistics,
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
            all_pillar_features = self.feature_net(
                features,
                point_to_pillar,
                total_pillars,
            )
        else:
            all_pillar_features = point_clouds[0].new_empty(
                (0, self.feature_net.output_channels)
            )
        dense_features = self.scatter.forward_batch(
            all_pillar_features,
            pillar_batches,
            pillar_rows,
            pillar_cols,
            len(point_clouds),
        )
        if not return_sparse:
            return dense_features, statistics

        sparse_coordinates = torch.stack(
            (pillar_batches, pillar_rows, pillar_cols), dim=1
        )
        return PointPillarsOutput(
            dense_features=dense_features,
            sparse_features=all_pillar_features,
            sparse_coordinates=sparse_coordinates,
            statistics=statistics,
        )
