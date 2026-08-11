"""Direct deterministic BEV replacement with full-scene cross-attention."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import BEVEncoder, _group_count
from ..pointpillars import BEVGridGeometry, PointPillarsEncoder
from .coarse_config import CoarseReconstructionConfig


class ResidualCoarseBlock(nn.Module):
    """Two-convolution GroupNorm/SiLU residual block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(tensor) + self.shortcut(tensor))


class LocalUNetEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, depth: int, dropout: float):
        super().__init__()
        self.channels = tuple(base_channels * (2**index) for index in range(depth))
        self.blocks = nn.ModuleList(
            ResidualCoarseBlock(
                in_channels if index == 0 else channels,
                channels,
                dropout,
            )
            for index, channels in enumerate(self.channels)
        )
        self.downsamplers = nn.ModuleList(
            nn.Conv2d(
                self.channels[index],
                self.channels[index + 1],
                kernel_size=3,
                stride=2,
                padding=1,
            )
            for index in range(depth - 1)
        )

    def forward(self, tensor: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        skips = []
        for index, block in enumerate(self.blocks):
            tensor = block(tensor)
            if index < len(self.blocks) - 1:
                skips.append(tensor)
                tensor = self.downsamplers[index](tensor)
        return tuple(skips), tensor


class GlobalLidarEncoder(nn.Module):
    def __init__(self, config: CoarseReconstructionConfig):
        super().__init__()
        self.encoder = BEVEncoder(
            config.lidar_channels + 1,
            base_channels=config.global_base_channels,
            channel_multipliers=config.global_channel_multipliers,
        )

    @property
    def out_channels(self) -> int:
        return self.encoder.out_channels[-1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(tensor)


class GlobalRadarEncoder(nn.Module):
    def __init__(self, config: CoarseReconstructionConfig):
        super().__init__()
        self.encoder = BEVEncoder(
            config.radar_channels,
            base_channels=config.global_base_channels,
            channel_multipliers=config.global_channel_multipliers,
        )

    @property
    def out_channels(self) -> int:
        return self.encoder.out_channels[-1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(tensor)


class GlobalFusionBlock(nn.Module):
    def __init__(self, in_channels: int, attention_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, attention_dim, 1, bias=False),
            nn.GroupNorm(_group_count(attention_dim), attention_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(attention_dim, attention_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(attention_dim), attention_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


class AbsolutePositionEncoder(nn.Module):
    def __init__(self, attention_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, attention_dim),
            nn.SiLU(inplace=True),
            nn.Linear(attention_dim, attention_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape [B,N,2]")
        return self.net(coordinates)


def _normalized_grid(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2).expand(
        batch_size, -1, -1
    )


class LocalToGlobalCrossAttention(nn.Module):
    def __init__(self, local_channels: int, config: CoarseReconstructionConfig):
        super().__init__()
        self.local_projection = (
            nn.Identity()
            if local_channels == config.attention_dim
            else nn.Conv2d(local_channels, config.attention_dim, 1)
        )
        self.position_encoder = AbsolutePositionEncoder(config.attention_dim)
        self.query_norm = nn.LayerNorm(config.attention_dim)
        self.context_norm = nn.LayerNorm(config.attention_dim)
        self.attention = nn.MultiheadAttention(
            config.attention_dim,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

    def forward(
        self,
        local_bottleneck: torch.Tensor,
        global_context_map: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        projected = self.local_projection(local_bottleneck)
        batch, channels, local_h, local_w = projected.shape
        global_batch, global_channels, global_h, global_w = global_context_map.shape
        if batch != global_batch or channels != global_channels:
            raise ValueError("Local projection and global context batch/channels must match")
        local_tokens = projected.flatten(2).transpose(1, 2)
        global_tokens = global_context_map.flatten(2).transpose(1, 2)
        local_xy = _normalized_grid(
            batch, local_h, local_w, device=projected.device, dtype=projected.dtype
        )
        global_xy = _normalized_grid(
            batch,
            global_h,
            global_w,
            device=global_context_map.device,
            dtype=global_context_map.dtype,
        )
        query_tokens = self.query_norm(
            local_tokens + self.position_encoder(local_xy)
        )
        context_tokens = self.context_norm(
            global_tokens + self.position_encoder(global_xy)
        )
        attention_tokens, weights = self.attention(
            query_tokens,
            context_tokens,
            context_tokens,
            need_weights=return_attention_weights,
            average_attn_weights=False,
        )
        attention_context = attention_tokens.transpose(1, 2).reshape(
            batch, channels, local_h, local_w
        )
        return attention_context, weights, query_tokens, context_tokens


class BottleneckFusionBlock(nn.Module):
    def __init__(self, local_channels: int, attention_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(local_channels + attention_dim, local_channels, 1, bias=False),
            nn.GroupNorm(_group_count(local_channels), local_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_channels, local_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(local_channels), local_channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, local_bottleneck: torch.Tensor, attention_context: torch.Tensor) -> torch.Tensor:
        if local_bottleneck.shape[-2:] != attention_context.shape[-2:]:
            raise ValueError("Attention context must match the local bottleneck grid")
        update = self.block(torch.cat((local_bottleneck, attention_context), dim=1))
        return local_bottleneck + update


class LocalUNetDecoder(nn.Module):
    def __init__(self, channels: Sequence[int], dropout: float):
        super().__init__()
        current = channels[-1]
        blocks = []
        for skip_channels in reversed(channels[:-1]):
            blocks.append(
                ResidualCoarseBlock(current + skip_channels, skip_channels, dropout)
            )
            current = skip_channels
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = current

    def forward(
        self, bottleneck: torch.Tensor, skips: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if len(skips) != len(self.blocks):
            raise ValueError("Decoder received the wrong number of skip features")
        tensor = bottleneck
        for block, skip in zip(self.blocks, reversed(skips)):
            tensor = F.interpolate(
                tensor,
                size=skip.shape[-2:],
                mode="nearest",
            )
            if tensor.shape[0] != skip.shape[0] or tensor.shape[-2:] != skip.shape[-2:]:
                raise ValueError("Decoder and skip features are not aligned")
            tensor = block(torch.cat((tensor, skip), dim=1))
        return tensor


class CoarseReplacementHead(nn.Module):
    def __init__(self, in_channels: int, lidar_channels: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, lidar_channels, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.head(tensor)


class CoarseReconstructionModel(nn.Module):
    """Reconstruct every LiDAR cell inside the selected editable region."""

    def __init__(
        self,
        config: CoarseReconstructionConfig | None = None,
        *,
        grid_geometry: BEVGridGeometry | None = None,
    ):
        super().__init__()
        self.config = config or CoarseReconstructionConfig()
        self.config.validate()
        self.grid_geometry = grid_geometry
        self.lidar_pillar_encoder = None
        self.radar_pillar_encoder = None
        if self.config.pointpillars.enabled:
            if grid_geometry is None:
                raise ValueError(
                    "grid_geometry is required when PointPillars is enabled"
                )
            grid_geometry.validate()
            if (grid_geometry.height, grid_geometry.width) != (320, 320):
                raise ValueError(
                    "Experiment 1 requires a PointPillars pseudo-image aligned "
                    "to the existing 320x320 reconstruction grid"
                )
            pointpillars = self.config.pointpillars
            self.lidar_pillar_encoder = PointPillarsEncoder(
                grid_geometry,
                raw_channels=pointpillars.lidar_raw_channels,
                output_channels=pointpillars.output_channels,
                max_points_per_pillar=pointpillars.max_points_per_pillar,
                max_pillars=pointpillars.max_pillars,
            )
            self.radar_pillar_encoder = PointPillarsEncoder(
                grid_geometry,
                raw_channels=pointpillars.radar_raw_channels,
                output_channels=pointpillars.output_channels,
                max_points_per_pillar=pointpillars.max_points_per_pillar,
                max_pillars=pointpillars.max_pillars,
            )
        self.local_unet_encoder = LocalUNetEncoder(
            self.config.local_input_channels,
            self.config.unet_base_channels,
            self.config.unet_depth,
            self.config.dropout,
        )
        self.global_lidar_encoder = GlobalLidarEncoder(self.config)
        self.global_radar_encoder = GlobalRadarEncoder(self.config)
        self.global_fusion = GlobalFusionBlock(
            self.global_lidar_encoder.out_channels
            + self.global_radar_encoder.out_channels,
            self.config.attention_dim,
        )
        local_channels = self.local_unet_encoder.channels[-1]
        self.cross_attention = LocalToGlobalCrossAttention(local_channels, self.config)
        self.bottleneck_fusion = BottleneckFusionBlock(
            local_channels, self.config.attention_dim
        )
        self.local_unet_decoder = LocalUNetDecoder(
            self.local_unet_encoder.channels, self.config.dropout
        )
        self.replacement_head = CoarseReplacementHead(
            self.local_unet_decoder.out_channels,
            self.config.target_lidar_channels,
        )

    def _select_lidar_point_fields(
        self, point_clouds: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        expected = 4
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != expected:
                raise ValueError(
                    "faulty_lidar_points must contain aligned "
                    "[x,y,z,reflectivity] rows"
                )
        if self.config.pointpillars.lidar_use_reflectivity:
            return tuple(point_clouds)
        return tuple(points[:, :3] for points in point_clouds)

    def _select_radar_point_fields(
        self, point_clouds: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        selected = []
        for points in point_clouds:
            if points.ndim != 2 or points.shape[1] != 5:
                raise ValueError(
                    "radar_points must contain aligned "
                    "[x,y,z,power,doppler] rows"
                )
            columns = [points[:, :3]]
            if self.config.pointpillars.radar_use_power:
                columns.append(points[:, 3:4])
            if self.config.pointpillars.radar_use_radial_velocity:
                columns.append(points[:, 4:5])
            selected.append(torch.cat(columns, dim=1))
        return tuple(selected)

    def _sensor_features(
        self,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        faulty_lidar_points: Sequence[torch.Tensor] | None,
        radar_points: Sequence[torch.Tensor] | None,
        *,
        radar_enabled: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if not self.config.pointpillars.enabled:
            return faulty_lidar_bev, radar_bev, {}, {}
        if faulty_lidar_points is None:
            raise ValueError(
                "faulty_lidar_points are required when PointPillars is enabled"
            )
        assert self.lidar_pillar_encoder is not None
        assert self.radar_pillar_encoder is not None
        lidar_features, lidar_statistics = self.lidar_pillar_encoder(
            self._select_lidar_point_fields(faulty_lidar_points)
        )
        if radar_enabled:
            if radar_points is None:
                raise ValueError(
                    "radar_points are required when PointPillars is enabled"
                )
            radar_features, radar_statistics = self.radar_pillar_encoder(
                self._select_radar_point_fields(radar_points)
            )
        else:
            radar_features = lidar_features.new_zeros(
                (
                    lidar_features.shape[0],
                    self.config.radar_channels,
                    lidar_features.shape[2],
                    lidar_features.shape[3],
                )
            )
            radar_statistics = {}
        return lidar_features, radar_features, lidar_statistics, radar_statistics

    def forward(
        self,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        healthy_context_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        *,
        local_radar_bev: torch.Tensor | None = None,
        faulty_lidar_points: Sequence[torch.Tensor] | None = None,
        radar_points: Sequence[torch.Tensor] | None = None,
        radar_enabled: bool = True,
        local_radar_enabled: bool = True,
        use_global_map: bool = True,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        (
            lidar_sensor_bev,
            radar_sensor_bev,
            lidar_pillar_statistics,
            radar_pillar_statistics,
        ) = self._sensor_features(
            faulty_lidar_bev,
            radar_bev,
            faulty_lidar_points,
            radar_points,
            radar_enabled=radar_enabled,
        )
        halo_mask = halo_mask * (1.0 - reconstruction_mask)
        active_mask = torch.maximum(reconstruction_mask, halo_mask)

        erased_lidar_bev = (1.0 - reconstruction_mask) * lidar_sensor_bev
        if self.config.use_healthy_context_mask:
            local_context_mask = healthy_context_mask
            local_mask_channels = (reconstruction_mask, healthy_context_mask)
        else:
            local_context_mask = 1.0 - reconstruction_mask
            local_mask_channels = (reconstruction_mask,)
        local_lidar_context = local_context_mask * lidar_sensor_bev
        if local_radar_bev is None:
            local_radar_bev = radar_sensor_bev
        if not local_radar_enabled:
            local_radar_bev = torch.zeros_like(radar_sensor_bev)
        local_radar_active = active_mask * local_radar_bev
        local_input = torch.cat(
            (
                local_lidar_context,
                local_radar_active,
                *local_mask_channels,
            ),
            dim=1,
        )
        skip_features, local_bottleneck = self.local_unet_encoder(local_input)

        global_outputs = {}
        attention_weights = None
        if use_global_map:
            global_lidar_input = torch.cat(
                (erased_lidar_bev, reconstruction_mask), dim=1
            )
            h_lidar_global = self.global_lidar_encoder(global_lidar_input)
            h_radar_global = self.global_radar_encoder(radar_sensor_bev)
            global_context_map = self.global_fusion(
                torch.cat((h_lidar_global, h_radar_global), dim=1)
            )
            attention_context, attention_weights, query_tokens, context_tokens = (
                self.cross_attention(
                    local_bottleneck,
                    global_context_map,
                    return_attention_weights=return_attention_weights,
                )
            )
            fused_bottleneck = self.bottleneck_fusion(
                local_bottleneck, attention_context
            )
            global_outputs = {
                "global_lidar_input": global_lidar_input,
                "attention_context": attention_context,
                "global_context_map": global_context_map,
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
            }
        else:
            fused_bottleneck = local_bottleneck
        decoder_feature = self.local_unet_decoder(fused_bottleneck, skip_features)
        if decoder_feature.shape[-2:] != faulty_lidar_bev.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=faulty_lidar_bev.shape[-2:],
                mode="nearest",
            )
        replacement_raw = self.replacement_head(decoder_feature)
        occupancy_logits = replacement_raw[:, 0:1]
        predicted_density = replacement_raw[:, 1:2]
        predicted_height = replacement_raw[:, 2:3]
        replacement_bev = torch.cat(
            (
                torch.sigmoid(occupancy_logits),
                predicted_density,
                predicted_height,
            ),
            dim=1,
        )
        coarse_lidar_bev = (
            (1.0 - reconstruction_mask) * faulty_lidar_bev
            + reconstruction_mask * replacement_bev
        )
        outputs = {
            "erased_lidar_bev": erased_lidar_bev,
            "erased_lidar_features": erased_lidar_bev,
            "lidar_sensor_bev": lidar_sensor_bev,
            "radar_sensor_bev": radar_sensor_bev,
            "replacement_raw": replacement_raw,
            "replacement_bev": replacement_bev,
            "occupancy_logits": occupancy_logits,
            "predicted_density": predicted_density,
            "predicted_height": predicted_height,
            "coarse_lidar_bev": coarse_lidar_bev,
            "reconstruction_mask": reconstruction_mask,
            "healthy_context_mask": healthy_context_mask,
            "halo_mask": halo_mask,
            "active_mask": active_mask,
            "local_context_mask": local_context_mask,
            "local_lidar_context": local_lidar_context,
            "local_input": local_input,
            "local_bottleneck": local_bottleneck,
            "fused_bottleneck": fused_bottleneck,
        }
        if self.config.pointpillars.enabled:
            outputs.update(
                {
                    "lidar_pillar_bev": lidar_sensor_bev,
                    "radar_pillar_bev": radar_sensor_bev,
                    "lidar_pillar_statistics": lidar_pillar_statistics,
                    "radar_pillar_statistics": radar_pillar_statistics,
                }
            )
        outputs.update(global_outputs)
        if return_attention_weights and use_global_map:
            assert attention_weights is not None
            outputs["attention_weights"] = attention_weights
        return outputs
