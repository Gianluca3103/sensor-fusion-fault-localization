from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PFS_DIR = REPO_ROOT / "PFS"
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
for path in (REPO_ROOT, PFS_DIR, FAULT_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_v5_like import ConvBlock
from pfs_model import ExpertCorrection, ShiftNormalization, SpatialReliabilityEstimator


class BEVEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, bottleneck


class RadarPostFusionStabilizer(nn.Module):
    """PFS with radar replacing the paper's clean camera modality."""

    def __init__(self, channels: int):
        super().__init__()
        self.shift_norm = ShiftNormalization(channels)
        self.reliability = SpatialReliabilityEstimator(channels)
        # The original semantic/camera expert now learns radar-supported cues.
        self.correction = ExpertCorrection(channels)

    def forward(self, fused_features, lidar_features):
        shifted = self.shift_norm(fused_features)
        reliability = self.reliability(shifted, lidar_features=lidar_features)
        corrected = self.correction(shifted, reliability)
        return corrected, reliability


class ReliabilityDecoder(nn.Module):
    def __init__(self, base_channels: int = 16, dropout: float = 0.0):
        super().__init__()
        bottleneck_channels = base_channels * 16
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up4 = nn.ConvTranspose2d(bottleneck_channels, base_channels * 8, 2, 2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _match(x, target):
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips
        d4 = self._match(self.up4(self.dropout(bottleneck)), e4)
        d4 = self.dropout(self.dec4(torch.cat([d4, e4], dim=1)))
        d3 = self._match(self.up3(d4), e3)
        d3 = self.dropout(self.dec3(torch.cat([d3, e3], dim=1)))
        d2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class PFSRadarReliabilityModel(nn.Module):
    """Predict LiDAR fault heatmaps from degraded LiDAR and uncorrupted radar BEVs."""

    def __init__(
        self,
        lidar_channels: int = 3,
        radar_channels: int = 4,
        base_channels: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lidar_encoder = BEVEncoder(lidar_channels, base_channels)
        self.radar_encoder = BEVEncoder(radar_channels, base_channels)
        channels = base_channels * 16
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pfs = RadarPostFusionStabilizer(channels)
        self.decoder = ReliabilityDecoder(base_channels, dropout)

    def forward(self, faulty_lidar_bev, radar_bev, clean_lidar_bev=None, return_features=False):
        e1, e2, e3, e4, lidar_bottleneck = self.lidar_encoder(faulty_lidar_bev)
        radar_bottleneck = self.radar_encoder(radar_bev)[-1]
        fused = self.fusion(torch.cat([lidar_bottleneck, radar_bottleneck], dim=1))
        stabilized, pfs_reliability = self.pfs(fused, lidar_bottleneck)
        logits = self.decoder(stabilized, (e1, e2, e3, e4))

        if not return_features:
            return logits

        clean_features = None
        if clean_lidar_bev is not None:
            with torch.no_grad():
                clean_features = self.lidar_encoder(clean_lidar_bev)[-1]
        return {
            "logits": logits,
            "stabilized_features": stabilized,
            "clean_features": clean_features,
            "pfs_reliability": pfs_reliability,
            "fused_features": fused,
            "lidar_features": lidar_bottleneck,
            "radar_features": radar_bottleneck,
        }


def parameter_breakdown(model: PFSRadarReliabilityModel) -> dict[str, int]:
    components = {
        "lidar_encoder": model.lidar_encoder,
        "radar_encoder": model.radar_encoder,
        "fusion": model.fusion,
        "pfs": model.pfs,
        "decoder": model.decoder,
    }
    breakdown = {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in components.items()}
    breakdown["total"] = sum(parameter.numel() for parameter in model.parameters())
    return breakdown

