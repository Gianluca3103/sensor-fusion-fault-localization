from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.model_blocks import ConvBlock, frozen_reference_forward
from PFS.pfs_model import BEVEncoder, PostFusionStabilizer, match_spatial


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

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips
        d4 = match_spatial(self.up4(self.dropout(bottleneck)), e4)
        d4 = self.dropout(self.dec4(torch.cat([d4, e4], dim=1)))
        d3 = match_spatial(self.up3(d4), e3)
        d3 = self.dropout(self.dec3(torch.cat([d3, e3], dim=1)))
        d2 = match_spatial(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = match_spatial(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class SkipFusion(nn.Module):
    """Fuse same-resolution LiDAR and radar skip features without widening the decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, lidar_skip, radar_skip):
        radar_skip = match_spatial(radar_skip, lidar_skip)
        return self.fusion(torch.cat([lidar_skip, radar_skip], dim=1))


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
        self.pfs = PostFusionStabilizer(channels)
        self.radar_skip_fusion = nn.ModuleList(
            [
                SkipFusion(base_channels),
                SkipFusion(base_channels * 2),
                SkipFusion(base_channels * 4),
                SkipFusion(base_channels * 8),
            ]
        )
        self.decoder = ReliabilityDecoder(base_channels, dropout)

    def forward(self, faulty_lidar_bev, radar_bev, clean_lidar_bev=None, return_features=False):
        e1, e2, e3, e4, lidar_bottleneck = self.lidar_encoder(faulty_lidar_bev)
        radar_e1, radar_e2, radar_e3, radar_e4, radar_bottleneck = self.radar_encoder(radar_bev)
        fused = self.fusion(torch.cat([lidar_bottleneck, radar_bottleneck], dim=1))
        stabilized, pfs_reliability = self.pfs(fused, lidar_bottleneck)
        skips = tuple(
            fusion(lidar_skip, radar_skip)
            for fusion, lidar_skip, radar_skip in zip(
                self.radar_skip_fusion,
                (e1, e2, e3, e4),
                (radar_e1, radar_e2, radar_e3, radar_e4),
            )
        )
        logits = self.decoder(stabilized, skips)

        if not return_features:
            return logits

        clean_features = None
        if clean_lidar_bev is not None:
            clean_lidar_features = frozen_reference_forward(
                self.lidar_encoder,
                clean_lidar_bev,
            )[-1]
            clean_features = frozen_reference_forward(
                self.fusion,
                torch.cat(
                    [clean_lidar_features, radar_bottleneck.detach()],
                    dim=1,
                ),
            )
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
        "radar_skip_fusion": model.radar_skip_fusion,
        "decoder": model.decoder,
    }
    breakdown = {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in components.items()}
    breakdown["total"] = sum(parameter.numel() for parameter in model.parameters())
    return breakdown


def load_model_checkpoint(path, device):
    """Load a trusted trainer checkpoint and reconstruct its model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = PFSRadarReliabilityModel(
        base_channels=int(saved_args.get("base_channels", 16)),
        dropout=float(saved_args.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
