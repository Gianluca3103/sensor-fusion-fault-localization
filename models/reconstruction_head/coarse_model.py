"""Direct deterministic BEV replacement with full-scene cross-attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import BEVEncoder, _group_count


@dataclass(frozen=True)
class CoarseReconstructionConfig:
    lidar_channels: int = 3
    radar_channels: int = 4
    unet_base_channels: int = 16
    unet_depth: int = 5
    dropout: float = 0.0
    global_base_channels: int = 16
    global_channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 16)
    attention_dim: int = 128
    num_heads: int = 4
    attention_dropout: float = 0.0

    @property
    def local_input_channels(self) -> int:
        return self.lidar_channels + self.radar_channels + 2

    def validate(self) -> None:
        integer_values = {
            "lidar_channels": self.lidar_channels,
            "radar_channels": self.radar_channels,
            "unet_base_channels": self.unet_base_channels,
            "unet_depth": self.unet_depth,
            "global_base_channels": self.global_base_channels,
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
        }
        for name, value in integer_values.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not self.global_channel_multipliers or any(
            value < 1 for value in self.global_channel_multipliers
        ):
            raise ValueError("global_channel_multipliers must contain positive values")
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_binary_mask(
    mask: torch.Tensor,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    expected = (reference.shape[0], 1, *reference.shape[-2:])
    if tuple(mask.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(mask.shape)}")
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must be a finite binary mask containing only 0 and 1")
    return mask.to(device=reference.device, dtype=reference.dtype)


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
            config.lidar_channels,
            base_channels=config.global_base_channels,
            channel_multipliers=config.global_channel_multipliers,
        )

    @property
    def out_channels(self) -> int:
        return self.encoder.out_channels[-1]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.encoder(tensor).bottleneck


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
        return self.encoder(tensor).bottleneck


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
        self.local_projection = nn.Conv2d(local_channels, config.attention_dim, 1)
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
        self, local_bottleneck: torch.Tensor, attention_context: torch.Tensor
    ) -> torch.Tensor:
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
                tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False
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

    def __init__(self, config: CoarseReconstructionConfig | None = None):
        super().__init__()
        self.config = config or CoarseReconstructionConfig()
        self.config.validate()
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
            self.local_unet_decoder.out_channels, self.config.lidar_channels
        )

    def forward(
        self,
        faulty_lidar_bev: torch.Tensor,
        radar_bev: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        healthy_context_mask: torch.Tensor,
        halo_mask: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        if faulty_lidar_bev.ndim != 4 or radar_bev.ndim != 4:
            raise ValueError("LiDAR and radar inputs must have shape [B,C,H,W]")
        if faulty_lidar_bev.shape[1] != self.config.lidar_channels:
            raise ValueError("faulty_lidar_bev has the wrong channel count")
        if radar_bev.shape[1] != self.config.radar_channels:
            raise ValueError("radar_bev has the wrong channel count")
        if faulty_lidar_bev.shape[0] != radar_bev.shape[0] or (
            faulty_lidar_bev.shape[-2:] != radar_bev.shape[-2:]
        ):
            raise ValueError("LiDAR and radar must share batch and spatial dimensions")
        reconstruction_mask = _validate_binary_mask(
            reconstruction_mask, "reconstruction_mask", faulty_lidar_bev
        )
        healthy_context_mask = _validate_binary_mask(
            healthy_context_mask, "healthy_context_mask", faulty_lidar_bev
        )
        halo_mask = _validate_binary_mask(halo_mask, "halo_mask", faulty_lidar_bev)
        if torch.any(reconstruction_mask * healthy_context_mask):
            raise ValueError("healthy_context_mask must not overlap reconstruction_mask")
        if torch.any(healthy_context_mask * (1.0 - halo_mask)):
            raise ValueError("healthy_context_mask must be contained inside halo_mask")
        halo_mask = halo_mask * (1.0 - reconstruction_mask)
        active_mask = torch.maximum(reconstruction_mask, halo_mask)

        erased_lidar_bev = (1.0 - reconstruction_mask) * faulty_lidar_bev
        local_lidar_context = healthy_context_mask * faulty_lidar_bev
        local_radar_active = active_mask * radar_bev
        local_input = torch.cat(
            (
                local_lidar_context,
                local_radar_active,
                reconstruction_mask,
                healthy_context_mask,
            ),
            dim=1,
        )
        if local_input.shape[1] != self.config.local_input_channels:
            raise AssertionError("Direct local input channel contract was violated")
        skip_features, local_bottleneck = self.local_unet_encoder(local_input)

        h_lidar_global = self.global_lidar_encoder(erased_lidar_bev)
        h_radar_global = self.global_radar_encoder(radar_bev)
        if h_lidar_global.shape[-2:] != h_radar_global.shape[-2:]:
            raise ValueError("Global LiDAR and radar feature maps must align")
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
        decoder_feature = self.local_unet_decoder(fused_bottleneck, skip_features)
        if decoder_feature.shape[-2:] != faulty_lidar_bev.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=faulty_lidar_bev.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        replacement_raw = self.replacement_head(decoder_feature)
        if replacement_raw.shape != faulty_lidar_bev.shape:
            raise AssertionError("Replacement output must match the LiDAR BEV shape")
        coarse_lidar_bev = (
            (1.0 - reconstruction_mask) * faulty_lidar_bev
            + reconstruction_mask * replacement_raw
        )
        outputs = {
            "erased_lidar_bev": erased_lidar_bev,
            "replacement_raw": replacement_raw,
            "coarse_lidar_bev": coarse_lidar_bev,
            "reconstruction_mask": reconstruction_mask,
            "healthy_context_mask": healthy_context_mask,
            "halo_mask": halo_mask,
            "active_mask": active_mask,
            "local_input": local_input,
            "local_bottleneck": local_bottleneck,
            "attention_context": attention_context,
            "fused_bottleneck": fused_bottleneck,
            "global_context_map": global_context_map,
            "query_tokens": query_tokens,
            "context_tokens": context_tokens,
        }
        if return_attention_weights:
            assert attention_weights is not None
            outputs["attention_weights"] = attention_weights
        return outputs

