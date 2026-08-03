from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from PFS.pfs_model import match_spatial
from .coarse_reconstructor import FlexibleConvBlock


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        device = timestep.device
        scale = math.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        args = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.channels % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimestepBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, normalization: str, dropout: float):
        super().__init__()
        self.conv = FlexibleConvBlock(in_channels, out_channels, normalization, dropout)
        self.time = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y + self.time(time_embedding)[:, :, None, None]


class ResidualDiffusionUNet(nn.Module):
    """Timestep-conditioned dense U-Net that denoises masked residuals."""

    def __init__(
        self,
        residual_channels: int,
        coarse_channels: int,
        conditioning_channels: int,
        base_channels: int = 16,
        levels: int = 4,
        normalization: str = "batch",
        dropout: float = 0.0,
        time_embedding_channels: int | None = None,
    ):
        super().__init__()
        self.residual_channels = int(residual_channels)
        input_channels = residual_channels + coarse_channels + conditioning_channels + 1
        time_channels = int(time_embedding_channels or base_channels * 4)
        self.time_embedding = nn.Sequential(
            SinusoidalTimestepEmbedding(time_channels),
            nn.Linear(time_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        channels = [base_channels * (2**level) for level in range(levels)]
        self.down_blocks = nn.ModuleList()
        current = input_channels
        for out_channels in channels:
            self.down_blocks.append(TimestepBlock(current, out_channels, time_channels, normalization, dropout))
            current = out_channels
        self.pool = nn.MaxPool2d(2)
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.upsamplers.append(nn.ConvTranspose2d(current, skip_channels, 2, 2))
            self.up_blocks.append(TimestepBlock(skip_channels * 2, skip_channels, time_channels, normalization, dropout))
            current = skip_channels
        self.head = nn.Conv2d(current, residual_channels, 1)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        coarse_features: torch.Tensor,
        conditioning_features: torch.Tensor,
        mask_gt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if mask_gt.shape[1] != 1:
            raise ValueError("mask_gt must have one channel")
        spatial = noisy_residual.shape[-2:]
        tensors = [noisy_residual, coarse_features, conditioning_features, mask_gt]
        tensors = [
            tensor if tensor.shape[-2:] == spatial else F.interpolate(tensor, spatial, mode="bilinear", align_corners=False)
            for tensor in tensors
        ]
        x = torch.cat(tensors, dim=1)
        time_embedding = self.time_embedding(timestep)
        skips = []
        for index, block in enumerate(self.down_blocks):
            x = block(x, time_embedding)
            skips.append(x)
            if index != len(self.down_blocks) - 1:
                x = self.pool(x)
        for up, block, skip in zip(self.upsamplers, self.up_blocks, reversed(skips[:-1])):
            x = match_spatial(up(x), skip)
            x = block(torch.cat([x, skip], dim=1), time_embedding)
        prediction = self.head(x)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("ResidualDiffusionUNet produced non-finite values")
        return prediction * mask_gt

