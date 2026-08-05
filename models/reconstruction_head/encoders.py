"""Independent multiscale encoders for the stage-two BEV inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


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
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
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


@dataclass(frozen=True)
class MultiScaleBEVEncoding:
    """Feature pyramid ordered from highest to lowest spatial resolution."""

    features: tuple[torch.Tensor, ...]

    @property
    def bottleneck(self) -> torch.Tensor:
        return self.features[-1]

    @property
    def skips(self) -> tuple[torch.Tensor, ...]:
        return self.features[:-1]


class BEVEncoder(nn.Module):
    """Encode a dense BEV into a feature pyramid with 2x downsampling per stage."""

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

    def _validate_input(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected a [B,C,H,W] BEV tensor, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {tensor.shape[1]}"
            )
        minimum_size = 2 ** (len(self.blocks) - 1)
        if min(tensor.shape[-2:]) < minimum_size:
            raise ValueError(
                f"BEV height and width must be at least {minimum_size}, got "
                f"{tuple(tensor.shape[-2:])}"
            )
        if not tensor.is_floating_point():
            raise TypeError("BEV encoder inputs must be floating-point tensors")

    def forward(self, tensor: torch.Tensor) -> MultiScaleBEVEncoding:
        self._validate_input(tensor)
        features = []
        for block in self.blocks:
            tensor = block(tensor)
            features.append(tensor)
        return MultiScaleBEVEncoding(tuple(features))


class CleanBEVEncoder(BEVEncoder):
    """Encoder for the clean three-channel LiDAR BEV training reference."""

    def __init__(self, **kwargs):
        super().__init__(in_channels=3, **kwargs)


class FaultyBEVEncoder(BEVEncoder):
    """Encoder for the faulty three-channel LiDAR BEV."""

    def __init__(self, **kwargs):
        super().__init__(in_channels=3, **kwargs)


class RadarBEVEncoder(BEVEncoder):
    """Encoder for occupancy, log-density, velocity, and RCS radar channels."""

    def __init__(self, **kwargs):
        super().__init__(in_channels=4, **kwargs)


class LiDARTrustedBEVEncoder(BEVEncoder):
    """Encoder for the reliable occupied portion of the faulty LiDAR BEV."""

    def __init__(self, **kwargs):
        super().__init__(in_channels=3, **kwargs)


class GoodDataFusion(nn.Module):
    """Concatenate trusted-LiDAR and radar only at the latent bottleneck."""

    def forward(
        self,
        lidar_trusted: MultiScaleBEVEncoding,
        radar: MultiScaleBEVEncoding,
    ) -> torch.Tensor:
        lidar_latent = lidar_trusted.bottleneck
        radar_latent = radar.bottleneck
        if lidar_latent.shape[0] != radar_latent.shape[0] or (
            lidar_latent.shape[-2:] != radar_latent.shape[-2:]
        ):
            raise ValueError(
                "Trusted-LiDAR and radar bottlenecks must share batch and spatial "
                f"dimensions; got {tuple(lidar_latent.shape)} and "
                f"{tuple(radar_latent.shape)}"
            )
        return torch.cat((lidar_latent, radar_latent), dim=1)


class RequiredCorrection(nn.Module):
    """Clean-minus-faulty latent difference restricted to the repair region."""

    @staticmethod
    def _latent_repair_mask(
        repair_mask: torch.Tensor,
        latent_shape: tuple[int, int],
        batch_size: int,
    ) -> torch.Tensor:
        if repair_mask.ndim != 4 or repair_mask.shape[1] != 1:
            raise ValueError("repair_mask must have shape [B,1,H,W]")
        if repair_mask.shape[0] != batch_size:
            raise ValueError("repair_mask must share the latent batch size")
        if not torch.isfinite(repair_mask).all() or torch.any(
            (repair_mask < 0) | (repair_mask > 1)
        ):
            raise ValueError("repair_mask must contain finite values in [0,1]")
        binary_mask = (repair_mask > 0.5).to(dtype=torch.float32)
        source_height, source_width = binary_mask.shape[-2:]
        target_height, target_width = latent_shape
        if (source_height, source_width) == latent_shape:
            return binary_mask
        if target_height <= source_height and target_width <= source_width:
            # Preserve any repair-box overlap when projecting onto coarse latents.
            return F.adaptive_max_pool2d(binary_mask, latent_shape)
        return F.interpolate(binary_mask, size=latent_shape, mode="nearest")

    def forward(
        self,
        clean: MultiScaleBEVEncoding,
        faulty: MultiScaleBEVEncoding,
        repair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean_latent = clean.bottleneck
        faulty_latent = faulty.bottleneck
        if clean_latent.shape != faulty_latent.shape:
            raise ValueError(
                "Clean and faulty bottlenecks must have identical shapes; got "
                f"{tuple(clean_latent.shape)} and {tuple(faulty_latent.shape)}"
            )
        latent_mask = self._latent_repair_mask(
            repair_mask,
            tuple(clean_latent.shape[-2:]),
            clean_latent.shape[0],
        ).to(device=clean_latent.device, dtype=clean_latent.dtype)
        difference = clean_latent - faulty_latent
        return difference * latent_mask, latent_mask


def mask_unreliable_lidar(
    faulty_bev: torch.Tensor,
    reliability_map: torch.Tensor,
    *,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Zero every LiDAR cell whose reliability is below ``threshold``."""
    if faulty_bev.ndim != 4 or reliability_map.ndim != 4:
        raise ValueError("faulty_bev and reliability_map must have shape [B,C,H,W]")
    if faulty_bev.shape[1] != 3:
        raise ValueError(f"faulty_bev must have 3 channels, got {faulty_bev.shape[1]}")
    if reliability_map.shape[1] != 1:
        raise ValueError(
            f"reliability_map must have 1 channel, got {reliability_map.shape[1]}"
        )
    if faulty_bev.shape[0] != reliability_map.shape[0] or (
        faulty_bev.shape[-2:] != reliability_map.shape[-2:]
    ):
        raise ValueError(
            "faulty_bev and reliability_map must share batch and spatial dimensions"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if not reliability_map.is_floating_point():
        raise TypeError("reliability_map must be floating point")
    if not torch.isfinite(reliability_map).all() or torch.any(
        (reliability_map < 0.0) | (reliability_map > 1.0)
    ):
        raise ValueError("reliability_map must contain finite values in [0,1]")
    trusted_mask = reliability_map >= threshold
    return faulty_bev * trusted_mask.to(dtype=faulty_bev.dtype)


class ReconstructionBEVEncoders(nn.Module):
    """Run four independent encoders with matching feature dimensions."""

    def __init__(
        self,
        *,
        base_channels: int = 16,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 16),
        trusted_reliability_threshold: float = 1.0,
    ):
        super().__init__()
        options = {
            "base_channels": base_channels,
            "channel_multipliers": channel_multipliers,
        }
        self.clean = CleanBEVEncoder(**options)
        self.radar = RadarBEVEncoder(**options)
        self.faulty = FaultyBEVEncoder(**options)
        self.lidar_trusted = LiDARTrustedBEVEncoder(**options)
        self.good_data_fusion = GoodDataFusion()
        self.required_correction = RequiredCorrection()
        if not 0.0 <= trusted_reliability_threshold <= 1.0:
            raise ValueError("trusted_reliability_threshold must be in [0,1]")
        self.trusted_reliability_threshold = float(trusted_reliability_threshold)

    def forward(
        self,
        clean_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        faulty_bev: torch.Tensor,
        reliability_map: torch.Tensor,
        repair_mask: torch.Tensor,
    ) -> dict[str, MultiScaleBEVEncoding | torch.Tensor]:
        spatial_shapes = {
            tuple(clean_bev.shape[-2:]),
            tuple(radar_bev.shape[-2:]),
            tuple(faulty_bev.shape[-2:]),
            tuple(reliability_map.shape[-2:]),
        }
        if len(spatial_shapes) != 1:
            raise ValueError(
                "Clean, radar, faulty, and reliability inputs must share a spatial shape; got "
                f"clean={tuple(clean_bev.shape[-2:])}, "
                f"radar={tuple(radar_bev.shape[-2:])}, and "
                f"faulty={tuple(faulty_bev.shape[-2:])}, and "
                f"reliability={tuple(reliability_map.shape[-2:])}"
            )
        batch_sizes = {
            clean_bev.shape[0],
            radar_bev.shape[0],
            faulty_bev.shape[0],
            reliability_map.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError("All BEV and reliability inputs must share a batch size")
        lidar_trusted_bev = mask_unreliable_lidar(
            faulty_bev,
            reliability_map,
            threshold=self.trusted_reliability_threshold,
        )
        clean_encoding = self.clean(clean_bev)
        radar_encoding = self.radar(radar_bev)
        faulty_encoding = self.faulty(faulty_bev)
        lidar_trusted_encoding = self.lidar_trusted(lidar_trusted_bev)
        good_data_encoding = self.good_data_fusion(
            lidar_trusted_encoding,
            radar_encoding,
        )
        required_correction, latent_repair_mask = self.required_correction(
            clean_encoding,
            faulty_encoding,
            repair_mask,
        )
        return {
            "clean": clean_encoding,
            "radar": radar_encoding,
            "faulty": faulty_encoding,
            "lidar_trusted": lidar_trusted_encoding,
            "good_data": good_data_encoding,
            "required_correction": required_correction,
            "latent_repair_mask": latent_repair_mask,
        }


# Compatibility alias for code written before the trusted-LiDAR stream was added.
TripletBEVEncoders = ReconstructionBEVEncoders
