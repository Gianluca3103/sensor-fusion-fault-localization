"""Plain convolutional U-Net velocity predictor for Fine reconstruction."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..encoders import _group_count


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal refinement-step embedding."""

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


class TimestepResidualBlock(nn.Module):
    """GroupNorm/SiLU residual block conditioned by refinement step."""

    def __init__(self, input_channels: int, output_channels: int, time_dim: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.norm1 = nn.GroupNorm(
            _group_count(input_channels), input_channels
        )
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, output_channels)
        self.norm2 = nn.GroupNorm(
            _group_count(output_channels), output_channels
        )
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.residual_projection = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(
        self,
        tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(tensor)))
        time_bias = self.time_projection(F.silu(timestep_embedding))
        hidden = hidden + time_bias[:, :, None, None].to(dtype=hidden.dtype)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        output = self.residual_projection(tensor) + hidden
        return output * valid_mask


class Downsample2d(nn.Module):
    """Learned stride-two 3x3 downsampling."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, 3, stride=2, padding=1
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)


class Upsample2d(nn.Module):
    """Nearest-neighbor resize followed by a 3x3 convolution."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=1)

    def forward(
        self, tensor: torch.Tensor, target_shape: tuple[int, int]
    ) -> torch.Tensor:
        tensor = F.interpolate(tensor, size=target_shape, mode="nearest")
        return self.conv(tensor)


class BasicDiffusionUNet(nn.Module):
    """Three-downsample convolutional U-Net predicting residual-flow velocity."""

    def __init__(
        self,
        *,
        lidar_channels: int = 3,
        radar_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        num_downsamples: int = 3,
        resblocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        if num_downsamples != 3:
            raise ValueError("Basic Diffusion U-Net requires exactly 3 downsamples")
        if len(channel_multipliers) != num_downsamples + 1:
            raise ValueError(
                "channel_multipliers must contain one entry per U-Net level"
            )
        if resblocks_per_level < 1:
            raise ValueError("resblocks_per_level must be positive")
        self.lidar_channels = int(lidar_channels)
        self.radar_channels = int(radar_channels)
        self.input_channels = 2 * lidar_channels + radar_channels + 3
        self.output_channels = int(lidar_channels)
        self.channel_hierarchy = tuple(
            int(base_channels * multiplier) for multiplier in channel_multipliers
        )
        self.num_downsamples = int(num_downsamples)
        self.resblocks_per_level = int(resblocks_per_level)
        time_dim = int(base_channels * 4)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )
        self.input_projection = nn.Conv2d(
            self.input_channels, self.channel_hierarchy[0], 3, padding=1
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level, channels in enumerate(self.channel_hierarchy):
            blocks = nn.ModuleList(
                TimestepResidualBlock(channels, channels, time_dim)
                for _ in range(resblocks_per_level)
            )
            self.encoder_blocks.append(blocks)
            if level < num_downsamples:
                self.downsamples.append(
                    Downsample2d(channels, self.channel_hierarchy[level + 1])
                )

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in range(num_downsamples - 1, -1, -1):
            skip_channels = self.channel_hierarchy[level]
            input_channels = self.channel_hierarchy[level + 1]
            self.upsamples.append(Upsample2d(input_channels, skip_channels))
            blocks = nn.ModuleList()
            blocks.append(
                TimestepResidualBlock(2 * skip_channels, skip_channels, time_dim)
            )
            blocks.extend(
                TimestepResidualBlock(skip_channels, skip_channels, time_dim)
                for _ in range(resblocks_per_level - 1)
            )
            self.decoder_blocks.append(blocks)

        final_channels = self.channel_hierarchy[0]
        self.output_norm = nn.GroupNorm(
            _group_count(final_channels), final_channels
        )
        self.output_conv = nn.Conv2d(
            final_channels, self.output_channels, 3, padding=1
        )
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        self._shape_log_emitted = False

    @staticmethod
    def _resized_valid(valid: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(valid, size=shape, mode="nearest")

    def forward(
        self,
        current_residual: torch.Tensor,
        coarse_lidar: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        spatial_input = torch.cat(
            (
                current_residual * valid_mask,
                coarse_lidar * valid_mask,
                radar_bev * valid_mask,
                reconstruction_mask * valid_mask,
                halo_mask * valid_mask,
                valid_mask,
            ),
            dim=1,
        )
        if spatial_input.shape[1] != self.input_channels:
            raise ValueError(
                f"Basic Diffusion U-Net expected {self.input_channels} input "
                f"channels, received {spatial_input.shape[1]}"
            )
        time_embedding = self.time_embedding(timestep)
        tensor = self.input_projection(spatial_input) * valid_mask
        encoder_features: list[torch.Tensor] = []
        encoder_shapes: list[tuple[int, ...]] = []
        level_valid_masks: list[torch.Tensor] = []
        for level, blocks in enumerate(self.encoder_blocks):
            level_valid = self._resized_valid(valid_mask, tensor.shape[-2:])
            for block in blocks:
                tensor = block(tensor, time_embedding, level_valid)
            encoder_features.append(tensor)
            encoder_shapes.append(tuple(tensor.shape))
            level_valid_masks.append(level_valid)
            if level < self.num_downsamples:
                tensor = self.downsamples[level](tensor)
                tensor = tensor * self._resized_valid(valid_mask, tensor.shape[-2:])

        bottleneck_shape = tuple(tensor.shape)
        decoder_shapes: list[tuple[int, ...]] = []
        for decoder_index, level in enumerate(
            range(self.num_downsamples - 1, -1, -1)
        ):
            skip = encoder_features[level]
            tensor = self.upsamples[decoder_index](tensor, skip.shape[-2:])
            level_valid = level_valid_masks[level]
            tensor = torch.cat((tensor * level_valid, skip), dim=1)
            for block in self.decoder_blocks[decoder_index]:
                tensor = block(tensor, time_embedding, level_valid)
            decoder_shapes.append(tuple(tensor.shape))

        velocity = torch.tanh(
            self.output_conv(F.silu(self.output_norm(tensor)))
        )
        active_repair = (reconstruction_mask > 0.5) & (valid_mask > 0.5)
        velocity = torch.where(
            active_repair.expand_as(velocity),
            velocity,
            torch.zeros_like(velocity),
        )
        debug: dict[str, object] = {
            "unet_contextual_input": spatial_input,
            "unet_encoder_shapes": encoder_shapes,
            "unet_bottleneck_shape": bottleneck_shape,
            "unet_decoder_shapes": decoder_shapes,
            "unet_output_shape": tuple(velocity.shape),
        }
        if self.training and not self._shape_log_emitted:
            print(
                "Basic Diffusion U-Net debug batch:\n"
                f"  contextual input: {tuple(spatial_input.shape)}\n"
                f"  encoder: {encoder_shapes}\n"
                f"  bottleneck: {bottleneck_shape}\n"
                f"  decoder: {decoder_shapes}\n"
                f"  output: {tuple(velocity.shape)}",
                flush=True,
            )
            self._shape_log_emitted = True
        return velocity, debug
