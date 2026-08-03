from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from Fault_Localization_Model.model_blocks import ConvBlock
from PFS.pfs_model import match_spatial


def _norm_layer(channels: int, normalization: str) -> nn.Module:
    if normalization == "batch":
        return nn.BatchNorm2d(channels)
    if normalization == "group":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalization in {"none", "", None}:
        return nn.Identity()
    raise ValueError(f"Unsupported normalization {normalization!r}")


class FlexibleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batch", dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm_layer(out_channels, normalization),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm_layer(out_channels, normalization),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, levels: int, normalization: str, dropout: float):
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2")
        channels = [base_channels * (2**level) for level in range(levels)]
        blocks = []
        current = in_channels
        for out_channels in channels:
            blocks.append(FlexibleConvBlock(current, out_channels, normalization, dropout))
            current = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if index != len(self.blocks) - 1:
                x = self.pool(x)
        return skips[:-1], skips[-1]


class CoarseLiDARRadarReconstructor(nn.Module):
    """Dense dual-encoder U-Net for oracle-mask coarse LiDAR feature repair."""

    def __init__(
        self,
        lidar_channels: int = 3,
        radar_channels: int = 4,
        output_channels: int | None = None,
        base_channels: int = 16,
        levels: int = 4,
        normalization: Literal["batch", "group", "none"] = "batch",
        dropout: float = 0.0,
        use_occupancy: bool = True,
        use_occupancy_head: bool = False,
        use_offset_head: bool = False,
        conditioning_channels: int | None = None,
    ):
        super().__init__()
        self.lidar_channels = int(lidar_channels)
        self.radar_channels = int(radar_channels)
        self.output_channels = int(output_channels or lidar_channels)
        self.use_occupancy = bool(use_occupancy)
        lidar_input_channels = self.lidar_channels + 1 + (1 if self.use_occupancy else 0)
        self.lidar_encoder = _Encoder(lidar_input_channels, base_channels, levels, normalization, dropout)
        self.radar_encoder = _Encoder(self.radar_channels, base_channels, levels, normalization, dropout)
        encoder_channels = self.lidar_encoder.out_channels
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck_fusion = FlexibleConvBlock(
            bottleneck_channels * 2,
            bottleneck_channels,
            normalization,
            dropout,
        )
        self.skip_fusions = nn.ModuleList(
            FlexibleConvBlock(ch * 2, ch, normalization, dropout) for ch in encoder_channels[:-1]
        )
        decoder_blocks = []
        upsamplers = []
        current = bottleneck_channels
        for skip_channels in reversed(encoder_channels[:-1]):
            upsamplers.append(nn.ConvTranspose2d(current, skip_channels, 2, 2))
            decoder_blocks.append(
                FlexibleConvBlock(skip_channels * 2, skip_channels, normalization, dropout)
            )
            current = skip_channels
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.delta_head = nn.Conv2d(current, self.output_channels, 1)
        cond_channels = int(conditioning_channels or current)
        self.conditioning_head = nn.Conv2d(current, cond_channels, 1)
        self.occupancy_head = nn.Conv2d(current, 1, 1) if use_occupancy_head else None
        self.offset_head = nn.Conv2d(current, 3, 1) if use_offset_head else None

    @staticmethod
    def infer_occupancy(lidar_features: torch.Tensor) -> torch.Tensor:
        return (lidar_features.abs().sum(dim=1, keepdim=True) > 1e-6).to(lidar_features.dtype)

    def forward(
        self,
        lidar_corrupt: torch.Tensor,
        radar: torch.Tensor,
        mask_gt: torch.Tensor,
        occupancy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lidar_corrupt.ndim != 4 or radar.ndim != 4 or mask_gt.ndim != 4:
            raise ValueError("lidar_corrupt, radar, and mask_gt must be [B,C,H,W]")
        if mask_gt.shape[1] != 1:
            raise ValueError(f"mask_gt must have one channel, got {mask_gt.shape[1]}")
        if radar.shape[-2:] != lidar_corrupt.shape[-2:]:
            radar = F.interpolate(radar, size=lidar_corrupt.shape[-2:], mode="bilinear", align_corners=False)
        if mask_gt.shape[-2:] != lidar_corrupt.shape[-2:]:
            mask_gt = F.interpolate(mask_gt, size=lidar_corrupt.shape[-2:], mode="nearest")
        mask_gt = (mask_gt > 0.5).to(lidar_corrupt.dtype)
        if int(mask_gt.sum().item()) == 0:
            raise ValueError("Every Stage I sample must contain at least one faulty cell")
        if occupancy is None and self.use_occupancy:
            occupancy = self.infer_occupancy(lidar_corrupt)
        lidar_inputs = [lidar_corrupt, mask_gt]
        if self.use_occupancy:
            lidar_inputs.append(occupancy.to(lidar_corrupt.dtype))
        lidar_skips, lidar_bottleneck = self.lidar_encoder(torch.cat(lidar_inputs, dim=1))
        radar_skips, radar_bottleneck = self.radar_encoder(radar)
        fused = self.bottleneck_fusion(torch.cat([lidar_bottleneck, match_spatial(radar_bottleneck, lidar_bottleneck)], dim=1))
        fused_skips = [
            fusion(torch.cat([lidar_skip, match_spatial(radar_skip, lidar_skip)], dim=1))
            for fusion, lidar_skip, radar_skip in zip(self.skip_fusions, lidar_skips, radar_skips)
        ]
        x = fused
        for up, dec, skip in zip(self.upsamplers, self.decoder_blocks, reversed(fused_skips)):
            x = match_spatial(up(x), skip)
            x = dec(torch.cat([x, skip], dim=1))
        delta = self.delta_head(x)
        if delta.shape[1] != lidar_corrupt.shape[1]:
            raise ValueError(
                f"delta output channels {delta.shape[1]} must match LiDAR channels {lidar_corrupt.shape[1]}"
            )
        coarse = lidar_corrupt + mask_gt * delta
        healthy_error = ((1.0 - mask_gt) * (coarse - lidar_corrupt)).abs().max()
        if not torch.isfinite(healthy_error):
            raise FloatingPointError("Non-finite values in coarse output")
        output = {
            "delta_coarse": delta,
            "coarse_features": coarse,
            "conditioning_features": self.conditioning_head(x),
            "healthy_preservation_max": healthy_error,
        }
        if self.occupancy_head is not None:
            output["occupancy_logits"] = self.occupancy_head(x)
        if self.offset_head is not None:
            output["offset"] = self.offset_head(x)
        return output


def coarse_parameter_breakdown(model: CoarseLiDARRadarReconstructor) -> dict[str, int]:
    components = {
        "lidar_encoder": model.lidar_encoder,
        "radar_encoder": model.radar_encoder,
        "bottleneck_fusion": model.bottleneck_fusion,
        "skip_fusions": model.skip_fusions,
        "decoder": nn.ModuleList([model.upsamplers, model.decoder_blocks]),
        "heads": nn.ModuleList(
            [
                model.delta_head,
                model.conditioning_head,
                *(module for module in (model.occupancy_head, model.offset_head) if module is not None),
            ]
        ),
    }
    breakdown = {name: sum(p.numel() for p in module.parameters()) for name, module in components.items()}
    breakdown["total"] = sum(p.numel() for p in model.parameters())
    return breakdown

