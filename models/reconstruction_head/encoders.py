"""Generic multiscale BEV encoder used by coarse global-context branches."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class ResidualEncoderBlock(nn.Module):
    """A stride-aware residual block that remains stable for small batches."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(tensor) + self.shortcut(tensor))


class BEVEncoder(nn.Module):
    """Encode a dense BEV into its lowest-resolution feature map."""

    def __init__(
        self,
        in_channels: int,
        *,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        multipliers = tuple(int(value) for value in channel_multipliers)
        if not multipliers or any(value < 1 for value in multipliers):
            raise ValueError("channel_multipliers must contain positive integers")

        self.in_channels = int(in_channels)
        self.out_channels = tuple(base_channels * value for value in multipliers)
        blocks = []
        current_channels = self.in_channels
        for index, output_channels in enumerate(self.out_channels):
            blocks.append(
                ResidualEncoderBlock(
                    current_channels,
                    output_channels,
                    stride=1 if index == 0 else 2,
                )
            )
            current_channels = output_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tensor = block(tensor)
        return tensor
