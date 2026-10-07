"""Four-resolution fixed-domain MinkowskiEngine sparse 3D U-Net."""

from __future__ import annotations

import torch
from torch import nn

_ME_IMPORT_ERROR = None
try:
    import MinkowskiEngine as ME
except ImportError as error:  # Allows CPU-only geometry tests before the professor-machine smoke test.
    ME = None
    _ME_IMPORT_ERROR = error


def _require_me() -> None:
    if ME is None:
        raise RuntimeError(f"MinkowskiEngine import failed: {_ME_IMPORT_ERROR}") from _ME_IMPORT_ERROR


class SparseLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, sites):
        return ME.SparseTensor(features=self.norm(sites.F),
                               coordinate_map_key=sites.coordinate_map_key,
                               coordinate_manager=sites.coordinate_manager)


class SparseSiLU(nn.Module):
    def forward(self, sites):
        return ME.SparseTensor(features=torch.nn.functional.silu(sites.F),
                               coordinate_map_key=sites.coordinate_map_key,
                               coordinate_manager=sites.coordinate_manager)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        _require_me()
        self.first = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                             stride=1, dimension=3, bias=False)
        self.second = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3,
                                              stride=1, dimension=3, bias=False)
        self.norm1 = SparseLayerNorm(out_channels)
        self.norm2 = SparseLayerNorm(out_channels)
        self.activation = SparseSiLU()
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1,
                                                 stride=1, dimension=3, bias=False))

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.activation(self.norm1(self.first(x)))
        y = self.norm2(self.second(y))
        if y.coordinate_map_key != residual.coordinate_map_key:
            raise AssertionError("Residual sparse coordinate maps differ")
        return self.activation(y + residual)


class SparseUNet(nn.Module):
    def __init__(self, channels: tuple[int, int, int, int]):
        super().__init__()
        _require_me()
        a, b, c, d = channels
        self.enc1 = ResidualBlock(a, a)
        self.down1 = ME.MinkowskiConvolution(a, b, kernel_size=2, stride=2, dimension=3)
        self.enc2 = ResidualBlock(b, b)
        self.down2 = ME.MinkowskiConvolution(b, c, kernel_size=2, stride=2, dimension=3)
        self.enc3 = ResidualBlock(c, c)
        self.down3 = ME.MinkowskiConvolution(c, d, kernel_size=2, stride=2, dimension=3)
        self.bottleneck = ResidualBlock(d, d)
        self.up3 = ME.MinkowskiConvolutionTranspose(d, c, kernel_size=2, stride=2, dimension=3)
        self.dec3 = ResidualBlock(c + c, c)
        self.up2 = ME.MinkowskiConvolutionTranspose(c, b, kernel_size=2, stride=2, dimension=3)
        self.dec2 = ResidualBlock(b + b, b)
        self.up1 = ME.MinkowskiConvolutionTranspose(b, a, kernel_size=2, stride=2, dimension=3)
        self.dec1 = ResidualBlock(a + a, a)

    @staticmethod
    def _join(up, skip):
        if up.coordinate_map_key != skip.coordinate_map_key:
            raise AssertionError("U-Net skip coordinates differ after non-generative upsampling")
        return ME.cat(up, skip)

    def forward(self, sparse_state, conditioning=None, timestep=None):
        if timestep is not None:
            raise NotImplementedError("Diffusion is disabled in deterministic Stage II")
        if conditioning is not None:
            raise ValueError("Conditioning must already be projected onto the candidate sparse tensor")
        e1 = self.enc1(sparse_state)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.bottleneck(self.down3(e3))
        d3 = self.dec3(self._join(self.up3(e4), e3))
        d2 = self.dec2(self._join(self.up2(d3), e2))
        d1 = self.dec1(self._join(self.up1(d2), e1))
        counts = {"input": len(sparse_state.F), "enc1": len(e1.F), "enc2": len(e2.F),
                  "enc3": len(e3.F), "bottleneck": len(e4.F), "dec3": len(d3.F),
                  "dec2": len(d2.F), "output": len(d1.F)}
        if any(count > counts["input"] for count in counts.values()):
            raise AssertionError("Sparse U-Net grew beyond its fixed candidate support")
        return d1, counts
