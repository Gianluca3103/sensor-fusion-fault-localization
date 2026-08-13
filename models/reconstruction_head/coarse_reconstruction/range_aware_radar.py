"""Learned range-aware local aggregation for handcrafted Radar BEVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RangeAwareRadarConfig:
    """Configuration for adaptive pre-fusion Radar aggregation."""

    enabled: bool = False
    output_channels: int = 16
    hidden_channels: int = 32
    min_radius_m: float = 0.30
    max_radius_m: float = 1.00
    range_min_m: float = 10.0
    range_max_m: float = 60.0
    x_min_m: float = 0.0
    x_max_m: float = 64.0
    y_min_m: float = -32.0
    y_max_m: float = 32.0
    occupancy_threshold: float = 0.5
    spatial_chunk_size: int = 4096

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("range_aware_radar.enabled must be boolean")
        if self.output_channels < 1 or self.hidden_channels < 1:
            raise ValueError(
                "range-aware Radar channel counts must be positive"
            )
        values = (
            self.min_radius_m,
            self.max_radius_m,
            self.range_min_m,
            self.range_max_m,
            self.x_min_m,
            self.x_max_m,
            self.y_min_m,
            self.y_max_m,
            self.occupancy_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("range-aware Radar settings must be finite")
        if self.min_radius_m <= 0 or self.max_radius_m < self.min_radius_m:
            raise ValueError(
                "Radar radii must satisfy 0 < min_radius_m <= max_radius_m"
            )
        if self.range_max_m <= self.range_min_m:
            raise ValueError("range_max_m must exceed range_min_m")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("range-aware Radar BEV bounds must be increasing")
        if not 0.0 <= self.occupancy_threshold <= 1.0:
            raise ValueError("occupancy_threshold must be in [0,1]")
        if self.spatial_chunk_size < 1:
            raise ValueError("spatial_chunk_size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class RangeAwareRadarAggregation(nn.Module):
    """Attend to occupied Radar neighbors within an adaptive metric radius."""

    def __init__(self, input_channels: int, config: RangeAwareRadarConfig):
        super().__init__()
        config.validate()
        if input_channels != 4:
            raise ValueError(
                "Range-aware aggregation expects the four handcrafted Radar "
                f"channels, got {input_channels}"
            )
        self.input_channels = int(input_channels)
        self.config = config
        self.center_encoder = nn.Sequential(
            nn.Conv2d(input_channels, config.hidden_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                config.hidden_channels,
                config.output_channels,
                1,
                bias=False,
            ),
        )
        neighbor_input_channels = input_channels + 4
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(
                neighbor_input_channels,
                config.hidden_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Linear(
                config.hidden_channels,
                config.output_channels,
                bias=False,
            ),
        )
        score_input_channels = config.output_channels + 5
        self.weight_network = nn.Sequential(
            nn.Linear(
                score_input_channels,
                config.hidden_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels, 1, bias=False),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                2 * config.output_channels,
                config.output_channels,
                1,
                bias=False,
            ),
            nn.SiLU(inplace=True),
        )

    def _geometry(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        x_step = (self.config.x_max_m - self.config.x_min_m) / height
        y_step = (self.config.y_max_m - self.config.y_min_m) / width
        x = self.config.x_max_m - (
            torch.arange(height, device=device, dtype=dtype) + 0.5
        ) * x_step
        y = self.config.y_min_m + (
            torch.arange(width, device=device, dtype=dtype) + 0.5
        ) * y_step
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        ranges = torch.sqrt(xx.square() + yy.square())
        return xx, yy, ranges, x_step, y_step

    def radius_map(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the continuous radius at every output cell."""

        _, _, ranges, _, _ = self._geometry(
            height, width, device, dtype
        )
        alpha = (
            (ranges - self.config.range_min_m)
            / (self.config.range_max_m - self.config.range_min_m)
        ).clamp(0.0, 1.0)
        radius = self.config.min_radius_m + alpha * (
            self.config.max_radius_m - self.config.min_radius_m
        )
        return radius[None, None]

    @staticmethod
    def _extract_row_patches(
        padded: torch.Tensor,
        row_start: int,
        row_count: int,
        kernel_size: int,
    ) -> torch.Tensor:
        """Return [B,L,K,C] patches for a contiguous group of output rows."""

        region = padded[
            :, :, row_start : row_start + row_count + kernel_size - 1, :
        ]
        patches = region.unfold(2, kernel_size, 1).unfold(
            3, kernel_size, 1
        )
        batch, channels, rows, columns, _, _ = patches.shape
        return (
            patches.permute(0, 2, 3, 4, 5, 1)
            .reshape(batch, rows * columns, kernel_size**2, channels)
        )

    def forward(
        self,
        radar_bev: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if radar_bev.ndim != 4 or radar_bev.shape[1] != self.input_channels:
            raise ValueError(
                "radar_bev must have shape [B,4,H,W], got "
                f"{tuple(radar_bev.shape)}"
            )
        if active_mask.shape != (
            radar_bev.shape[0], 1, radar_bev.shape[2], radar_bev.shape[3]
        ):
            raise ValueError(
                "active_mask must have shape [B,1,H,W] matching radar_bev"
            )
        if radar_bev.device != active_mask.device:
            raise ValueError("radar_bev and active_mask must share a device")
        if radar_bev.dtype != active_mask.dtype:
            raise ValueError("radar_bev and active_mask must share a dtype")

        batch, _, height, width = radar_bev.shape
        center_x_grid, center_y_grid, _, x_step, y_step = self._geometry(
            height, width, radar_bev.device, radar_bev.dtype
        )
        max_row_offset = math.ceil(self.config.max_radius_m / x_step)
        max_col_offset = math.ceil(self.config.max_radius_m / y_step)
        if max_row_offset != max_col_offset:
            raise ValueError(
                "Range-aware Radar currently requires square metric BEV cells"
            )
        padding = max_row_offset
        kernel_size = 2 * padding + 1
        row_offsets = torch.arange(
            -padding,
            padding + 1,
            device=radar_bev.device,
            dtype=radar_bev.dtype,
        )
        col_offsets = torch.arange(
            -padding,
            padding + 1,
            device=radar_bev.device,
            dtype=radar_bev.dtype,
        )
        row_grid, col_grid = torch.meshgrid(
            row_offsets, col_offsets, indexing="ij"
        )
        dx = (-row_grid * x_step).reshape(1, 1, -1, 1)
        dy = (col_grid * y_step).reshape(1, 1, -1, 1)
        neighbor_distance = torch.sqrt(dx.square() + dy.square())

        masked_radar = radar_bev * active_mask
        padded = F.pad(masked_radar, (padding,) * 4)
        center_encoded = self.center_encoder(masked_radar)
        radius = self.radius_map(
            height,
            width,
            device=radar_bev.device,
            dtype=radar_bev.dtype,
        )
        rows_per_chunk = max(
            1, min(height, self.config.spatial_chunk_size // width)
        )
        aggregated_rows = []
        count_rows = []
        attention_rows = [] if return_attention else None

        for row_start in range(0, height, rows_per_chunk):
            row_count = min(rows_per_chunk, height - row_start)
            neighbors = self._extract_row_patches(
                padded, row_start, row_count, kernel_size
            )
            locations = row_count * width
            center_x = center_x_grid[
                row_start : row_start + row_count
            ].reshape(1, locations, 1, 1)
            center_y = center_y_grid[
                row_start : row_start + row_count
            ].reshape(1, locations, 1, 1)
            neighbor_range = torch.sqrt(
                (center_x + dx).square() + (center_y + dy).square()
            )
            expanded_dx = dx.expand(batch, locations, -1, -1)
            expanded_dy = dy.expand(batch, locations, -1, -1)
            expanded_distance = neighbor_distance.expand(
                batch, locations, -1, -1
            )
            expanded_range = neighbor_range.expand(
                batch, locations, -1, -1
            )
            neighbor_input = torch.cat(
                (
                    neighbors,
                    expanded_dx,
                    expanded_dy,
                    expanded_distance,
                    expanded_range,
                ),
                dim=-1,
            )
            encoded = self.neighbor_encoder(neighbor_input)
            score_input = torch.cat(
                (
                    encoded,
                    expanded_dx,
                    expanded_dy,
                    expanded_distance,
                    neighbors[..., 1:2],
                    expanded_range,
                ),
                dim=-1,
            )
            scores = self.weight_network(score_input).squeeze(-1)
            chunk_radius = radius[
                :, :, row_start : row_start + row_count, :
            ].reshape(1, locations, 1)
            valid = (
                (expanded_distance[..., 0] <= chunk_radius + 1.0e-6)
                & (
                    neighbors[..., 0]
                    >= self.config.occupancy_threshold
                )
            )
            scores = scores.masked_fill(valid.logical_not(), -torch.inf)
            has_valid = valid.any(dim=-1, keepdim=True)
            safe_scores = torch.where(has_valid, scores, torch.zeros_like(scores))
            weights = torch.softmax(safe_scores, dim=-1) * valid.to(scores.dtype)
            aggregated = torch.sum(weights[..., None] * encoded, dim=2)
            aggregated_rows.append(
                aggregated.transpose(1, 2).reshape(
                    batch,
                    self.config.output_channels,
                    row_count,
                    width,
                )
            )
            count_rows.append(
                valid.sum(dim=-1)
                .reshape(batch, 1, row_count, width)
            )
            if attention_rows is not None:
                attention_rows.append(
                    weights.reshape(
                        batch,
                        row_count,
                        width,
                        kernel_size,
                        kernel_size,
                    )
                )

        aggregated_map = torch.cat(aggregated_rows, dim=2)
        fused = self.fusion(torch.cat((center_encoded, aggregated_map), dim=1))
        output = fused * active_mask
        debug = {
            "radius_m": radius,
            "valid_neighbor_count": torch.cat(count_rows, dim=2),
            "center_feature": center_encoded * active_mask,
            "aggregated_feature": aggregated_map * active_mask,
        }
        if attention_rows is not None:
            debug["attention_weights"] = torch.cat(attention_rows, dim=1)
        return output, debug
