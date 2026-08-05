"""Timestep-conditioned U-Net for direct-BEV masked residual diffusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import _group_count
from .diffusion_process import (
    BEVChannelNormalization,
    DiffusionProcessConfig,
    GaussianNoiseSchedule,
    MaskedEpsilonMSELoss,
    residual_target,
)


@dataclass(frozen=True)
class ResidualDiffusionUNetConfig:
    lidar_channels: int = 3
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    residual_blocks_per_level: int = 2
    time_embedding_dim: int = 256
    dropout: float = 0.0

    @property
    def input_channels(self) -> int:
        return 2 * self.lidar_channels + 1

    def validate(self) -> None:
        if self.lidar_channels < 1 or self.base_channels < 1:
            raise ValueError("channel counts must be positive")
        if not self.channel_multipliers or any(x < 1 for x in self.channel_multipliers):
            raise ValueError("channel_multipliers must contain positive values")
        if self.residual_blocks_per_level < 1 or self.time_embedding_dim < 2:
            raise ValueError("residual block count and time embedding must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class TimeConditionedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, tensor, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(tensor)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.shortcut(tensor)


class DiffusionDownBlock(nn.Module):
    def __init__(self, channels, time_dim, block_count, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            TimeConditionedResidualBlock(channels, channels, time_dim, dropout)
            for _ in range(block_count)
        )

    def forward(self, tensor, time_embedding):
        for block in self.blocks:
            tensor = block(tensor, time_embedding)
        return tensor


class DiffusionUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, block_count, dropout):
        super().__init__()
        self.upsample_projection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        blocks = [
            TimeConditionedResidualBlock(
                out_channels + skip_channels, out_channels, time_dim, dropout
            )
        ]
        blocks.extend(
            TimeConditionedResidualBlock(out_channels, out_channels, time_dim, dropout)
            for _ in range(block_count - 1)
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, tensor, skip, time_embedding):
        tensor = F.interpolate(tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        tensor = self.upsample_projection(tensor)
        if tensor.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Diffusion decoder skip alignment failed")
        tensor = torch.cat((tensor, skip), dim=1)
        for block in self.blocks:
            tensor = block(tensor, time_embedding)
        return tensor


class ResidualDiffusionUNet(nn.Module):
    """Simple convolutional epsilon predictor with no attention or multimodal inputs."""

    def __init__(self, config: ResidualDiffusionUNetConfig | None = None):
        super().__init__()
        self.config = config or ResidualDiffusionUNetConfig()
        self.config.validate()
        channels = tuple(
            self.config.base_channels * multiplier
            for multiplier in self.config.channel_multipliers
        )
        self.channels = channels
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.config.time_embedding_dim),
            nn.Linear(self.config.time_embedding_dim, self.config.time_embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.config.time_embedding_dim, self.config.time_embedding_dim),
        )
        self.input_projection = nn.Conv2d(self.config.input_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            DiffusionDownBlock(
                channel,
                self.config.time_embedding_dim,
                self.config.residual_blocks_per_level,
                self.config.dropout,
            )
            for channel in channels
        )
        self.downsamplers = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
                for i in range(len(channels) - 1)
            ]
            + [nn.Conv2d(channels[-1], channels[-1], 3, stride=2, padding=1)]
        )
        self.middle = nn.ModuleList(
            TimeConditionedResidualBlock(
                channels[-1], channels[-1], self.config.time_embedding_dim, self.config.dropout
            )
            for _ in range(2)
        )
        up_blocks = []
        current = channels[-1]
        for skip_channels in reversed(channels):
            up_blocks.append(
                DiffusionUpBlock(
                    current,
                    skip_channels,
                    skip_channels,
                    self.config.time_embedding_dim,
                    self.config.residual_blocks_per_level,
                    self.config.dropout,
                )
            )
            current = skip_channels
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_norm = nn.GroupNorm(_group_count(channels[0]), channels[0])
        self.output_projection = nn.Conv2d(channels[0], self.config.lidar_channels, 3, padding=1)

    def forward(self, diffusion_input: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if diffusion_input.ndim != 4 or diffusion_input.shape[1] != self.config.input_channels:
            raise ValueError(
                f"diffusion_input must have {self.config.input_channels} channels"
            )
        if timestep.shape != (diffusion_input.shape[0],):
            raise ValueError("timestep must have shape [B]")
        time_embedding = self.time_embedding(timestep)
        tensor = self.input_projection(diffusion_input)
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamplers):
            tensor = block(tensor, time_embedding)
            skips.append(tensor)
            tensor = downsample(tensor)
        for block in self.middle:
            tensor = block(tensor, time_embedding)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            tensor = block(tensor, skip, time_embedding)
        return self.output_projection(F.silu(self.output_norm(tensor)))


class MaskedResidualDiffusion(nn.Module):
    """Training-time masked residual construction and epsilon prediction."""

    def __init__(
        self,
        unet_config: ResidualDiffusionUNetConfig | None = None,
        process_config: DiffusionProcessConfig | None = None,
        normalization: BEVChannelNormalization | None = None,
    ):
        super().__init__()
        self.unet = ResidualDiffusionUNet(unet_config)
        self.schedule = GaussianNoiseSchedule(process_config)
        self.normalization = normalization or BEVChannelNormalization()
        self.loss_function = MaskedEpsilonMSELoss(
            self.schedule.config.denominator_epsilon
        )

    def predict_epsilon(self, residual_t, coarse_lidar_bev, reconstruction_mask, timestep):
        coarse_normalized = self.normalization.normalize(coarse_lidar_bev)
        diffusion_input = torch.cat(
            (residual_t, coarse_normalized, reconstruction_mask), dim=1
        )
        epsilon_pred = self.unet(diffusion_input, timestep)
        return reconstruction_mask * epsilon_pred, diffusion_input

    def forward(
        self,
        clean_lidar_bev,
        coarse_lidar_bev,
        reconstruction_mask,
        *,
        timestep=None,
        epsilon=None,
        generator=None,
    ):
        residual_gt = reconstruction_mask * self.normalization.normalize_residual(
            clean_lidar_bev - coarse_lidar_bev
        )
        batch = clean_lidar_bev.shape[0]
        if timestep is None:
            timestep = torch.randint(
                self.schedule.config.num_train_timesteps,
                (batch,),
                device=clean_lidar_bev.device,
                generator=generator,
            )
        if epsilon is None:
            epsilon = torch.randn(
                residual_gt.shape,
                device=residual_gt.device,
                dtype=residual_gt.dtype,
                generator=generator,
            )
        residual_t, epsilon_masked = self.schedule.add_masked_noise(
            residual_gt, epsilon, timestep, reconstruction_mask
        )
        epsilon_pred, diffusion_input = self.predict_epsilon(
            residual_t, coarse_lidar_bev, reconstruction_mask, timestep
        )
        loss = self.loss_function(epsilon_pred, epsilon, reconstruction_mask)
        return {
            "clean_lidar_bev": clean_lidar_bev,
            "coarse_lidar_bev": coarse_lidar_bev,
            "residual_gt": residual_gt,
            "residual_t": residual_t,
            "epsilon": epsilon,
            "epsilon_masked": epsilon_masked,
            "epsilon_pred": epsilon_pred,
            "diffusion_input": diffusion_input,
            "reconstruction_mask": reconstruction_mask,
            "timestep": timestep,
            "diffusion_loss": loss,
        }

