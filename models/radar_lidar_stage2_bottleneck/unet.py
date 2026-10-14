"""Fixed-support MinkowskiEngine U-Net with timestep FiLM and one fusion hook."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage2.sparse_unet import ME, SparseLayerNorm, _require_me


def _replace(sites, features: torch.Tensor):
    return ME.SparseTensor(features=features.contiguous(),
                           coordinate_map_key=sites.coordinate_map_key,
                           coordinate_manager=sites.coordinate_manager)


class TimedResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, time_dim: int):
        super().__init__()
        _require_me()
        self.first = ME.MinkowskiConvolution(incoming, outgoing, kernel_size=3,
                                             stride=1, dimension=3, bias=False)
        self.second = ME.MinkowskiConvolution(outgoing, outgoing, kernel_size=3,
                                              stride=1, dimension=3, bias=False)
        self.norm1 = SparseLayerNorm(outgoing)
        self.norm2 = SparseLayerNorm(outgoing)
        self.film = nn.Linear(time_dim, outgoing * 2)
        self.shortcut = (nn.Identity() if incoming == outgoing else
                         ME.MinkowskiConvolution(incoming, outgoing, kernel_size=1,
                                                 stride=1, dimension=3, bias=False))

    def forward(self, sites, time_embedding: torch.Tensor):
        residual = self.shortcut(sites)
        hidden = self.norm1(self.first(sites))
        batch = hidden.C[:, 0].long().to(time_embedding.device)
        scale, shift = self.film(time_embedding)[batch].chunk(2, dim=1)
        hidden = _replace(hidden, F.silu(hidden.F * (1 + scale) + shift))
        hidden = self.norm2(self.second(hidden))
        if hidden.coordinate_map_key != residual.coordinate_map_key:
            raise AssertionError("Sparse residual block changed its coordinate map")
        return _replace(hidden, F.silu(hidden.F + residual.F))


class BottleneckDiffusionUNet(nn.Module):
    def __init__(self, channels: tuple[int, ...], strides: tuple[int, ...],
                 blocks_per_level: int, time_dim: int):
        super().__init__()
        _require_me()
        if len(strides) != len(channels) - 1 or blocks_per_level < 1:
            raise ValueError("Invalid sparse U-Net levels")
        self.channels = channels
        self.strides = strides
        self.encoders = nn.ModuleList()
        self.down = nn.ModuleList()
        for level, width in enumerate(channels):
            if level:
                self.down.append(ME.MinkowskiConvolution(channels[level - 1], width,
                                  kernel_size=strides[level - 1], stride=strides[level - 1], dimension=3))
            self.encoders.append(nn.ModuleList(
                TimedResidualBlock(width, width, time_dim) for _ in range(blocks_per_level)))
        self.up = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(len(channels) - 2, -1, -1):
            self.up.append(ME.MinkowskiConvolutionTranspose(
                channels[level + 1], channels[level], kernel_size=strides[level],
                stride=strides[level], dimension=3))
            self.decoders.append(nn.ModuleList([
                TimedResidualBlock(2 * channels[level], channels[level], time_dim),
                *(TimedResidualBlock(channels[level], channels[level], time_dim)
                  for _ in range(blocks_per_level - 1))]))

    @staticmethod
    def _blocks(blocks, sites, time_embedding):
        for block in blocks:
            sites = block(sites, time_embedding)
        return sites

    def forward(self, sites, time_embedding: torch.Tensor, bottleneck_fusion):
        if time_embedding.ndim != 2:
            raise ValueError("Expected one timestep embedding per frame")
        skips = []
        current = sites
        counts = {"input": len(sites.F)}
        for level, blocks in enumerate(self.encoders):
            if level:
                current = self.down[level - 1](current)
            current = self._blocks(blocks, current, time_embedding)
            skips.append(current)
            counts[f"enc{level}"] = len(current.F)
        # This hook is called exactly once at the lowest spatial resolution.
        current = bottleneck_fusion(current)
        if current.coordinate_map_key != skips[-1].coordinate_map_key:
            raise AssertionError("Bottleneck fusion changed the sparse coordinates")
        counts["bottleneck"] = len(current.F)
        for index, blocks in enumerate(self.decoders):
            level = len(self.channels) - index - 2
            upward = self.up[index](current)
            skip = skips[level]
            if upward.coordinate_map_key != skip.coordinate_map_key:
                raise AssertionError("Upsampled map differs from the encoder skip")
            current = self._blocks(blocks, ME.cat(upward, skip), time_embedding)
            counts[f"dec{level}"] = len(current.F)
        counts["output"] = len(current.F)
        if counts["output"] != counts["input"] or any(v > counts["input"] for v in counts.values()):
            raise AssertionError("Sparse U-Net escaped the candidate support")
        return current, counts
