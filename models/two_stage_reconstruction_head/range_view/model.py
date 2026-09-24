"""Lightweight deterministic range-view ADD/range/DELETE network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RangeModelConfig:
    hidden_channels: int = 24
    max_range_m: float = 100.0
    min_range_m: float = 0.5
    use_radar: bool = True
    use_fault_map_conditioning: bool = False
    circular_azimuth: bool = True

    def __post_init__(self) -> None:
        if self.hidden_channels < 4 or self.hidden_channels % 4 or not 0 < self.min_range_m < self.max_range_m:
            raise ValueError("invalid range model channels or physical range bounds")


class CircularHorizontalConv(nn.Module):
    def __init__(self, incoming: int, outgoing: int, dilation: int = 1,
                 circular_azimuth: bool = True) -> None:
        super().__init__()
        self.pad = dilation
        self.circular_azimuth = circular_azimuth
        self.conv = nn.Conv2d(incoming, outgoing, 3, dilation=dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.pad(values, (self.pad, self.pad, 0, 0),
                       mode="circular" if self.circular_azimuth else "constant")
        values = F.pad(values, (0, 0, self.pad, self.pad), mode="constant")
        return self.conv(values)


def _block(incoming: int, outgoing: int, dilation: int = 1,
           circular_azimuth: bool = True) -> nn.Sequential:
    return nn.Sequential(
        CircularHorizontalConv(incoming, outgoing, dilation, circular_azimuth), nn.GroupNorm(4, outgoing), nn.SiLU(),
        CircularHorizontalConv(outgoing, outgoing, circular_azimuth=circular_azimuth), nn.GroupNorm(4, outgoing), nn.SiLU(),
    )


class RangeViewReconstructor(nn.Module):
    """No vertical pooling; two horizontal scales with circular azimuth seams."""

    input_channels = 10  # LiDAR range/valid/reflectivity, six radar, one predicted fault map.

    def __init__(self, config: RangeModelConfig = RangeModelConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_channels
        self.stem = _block(self.input_channels, width, circular_azimuth=config.circular_azimuth)
        self.down = _block(width, width * 2, circular_azimuth=config.circular_azimuth)
        self.bottleneck = _block(width * 2, width * 2, dilation=4,
                                 circular_azimuth=config.circular_azimuth)
        self.up = _block(width * 3, width, circular_azimuth=config.circular_azimuth)
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError("expected [batch,10,beam,azimuth] range-view features")
        features = features.clone()
        if not self.config.use_radar:
            features[:, 3:9] = 0
        if not self.config.use_fault_map_conditioning:
            features[:, 9] = 0
        fine = self.stem(features)
        coarse = self.down(F.avg_pool2d(fine, kernel_size=(1, 2), stride=(1, 2)))
        coarse = self.bottleneck(coarse)
        resized = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.head(self.up(torch.cat((fine, resized), dim=1)))
        add_logit, range_logit, delete_logit = logits[:, 0], logits[:, 1], logits[:, 2]
        add_range = self.config.min_range_m + (
            self.config.max_range_m - self.config.min_range_m
        ) * torch.sigmoid(range_logit)
        return {
            "add_logit": add_logit,
            "delete_logit": delete_logit,
            "add_probability": torch.sigmoid(add_logit),
            "delete_probability": torch.sigmoid(delete_logit),
            "add_range_m": add_range,
        }
