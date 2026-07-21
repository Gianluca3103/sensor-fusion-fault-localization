from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
FAULT_MODEL_DIR = CURRENT_DIR.parent / "Fault_Localization_Model"
if str(FAULT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FAULT_MODEL_DIR))

from model_v5_like import ConvBlock


class BEVEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


class ShiftNormalization(nn.Module):
    """PFS block 1: gated channel-wise BEV feature normalization."""

    def __init__(self, channels):
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        hidden = max(16, channels // 2)
        self.affine = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        self.alpha = nn.Parameter(torch.tensor(-5.0))
        self._init_identity(channels)

    def _init_identity(self, channels):
        final = self.affine[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[:channels].fill_(1.0)

    def forward(self, x):
        context = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        gamma, beta = self.affine(context).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        normalized = gamma * self.norm(x) + beta
        gate = torch.sigmoid(self.alpha)
        return gate * normalized + (1.0 - gate) * x


class SpatialReliabilityEstimator(nn.Module):
    """PFS block 2: predicts a reliability gate from stabilized BEV features."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(16, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.constant_(self.net[-1].bias, 4.0)

    def forward(self, shifted, lidar_features=None):
        if lidar_features is None:
            lidar_features = shifted
        x = torch.cat([shifted, lidar_features], dim=1)
        reliability = torch.sigmoid(self.net(x))
        return F.interpolate(reliability, size=shifted.shape[-2:], mode="bilinear", align_corners=False)


class ExpertCorrection(nn.Module):
    """PFS block 3: gated residual correction guided by the reliability map."""

    def __init__(self, channels):
        super().__init__()
        in_channels = channels + 1
        self.semantic_expert = ConvBlock(in_channels, channels)
        self.geometric_expert = ConvBlock(in_channels, channels)
        self.expert_logits = nn.Parameter(torch.zeros(2))
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, shifted, reliability):
        clean_features = reliability * shifted
        expert_input = torch.cat([clean_features, reliability], dim=1)
        weights = torch.softmax(self.expert_logits, dim=0)
        residual = weights[0] * self.semantic_expert(expert_input) + weights[1] * self.geometric_expert(expert_input)
        gate = torch.sigmoid(self.gate(expert_input))
        return shifted + gate * residual


class PostFusionStabilizer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.shift_norm = ShiftNormalization(channels)
        self.reliability = SpatialReliabilityEstimator(channels)
        self.correction = ExpertCorrection(channels)

    def forward(self, features, lidar_features=None):
        shifted = self.shift_norm(features)
        reliability = self.reliability(shifted, lidar_features=lidar_features)
        corrected = self.correction(shifted, reliability)
        return corrected, reliability


class LidarSpatialReliabilityEstimator(nn.Module):
    """Predict spatial reliability directly from a single LiDAR BEV feature map."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(16, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.constant_(self.net[-1].bias, 4.0)

    def forward(self, lidar_features):
        reliability = torch.sigmoid(self.net(lidar_features))
        return F.interpolate(
            reliability,
            size=lidar_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class LidarGeometricCorrection(nn.Module):
    """Apply one gated geometric residual instead of camera/LiDAR experts."""

    def __init__(self, channels):
        super().__init__()
        in_channels = channels + 1
        self.geometric_expert = ConvBlock(in_channels, channels)
        self.gate = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, lidar_features, reliability):
        reliable_features = reliability * lidar_features
        expert_input = torch.cat([reliable_features, reliability], dim=1)
        residual = self.geometric_expert(expert_input)
        gate = torch.sigmoid(self.gate(expert_input))
        return lidar_features + gate * residual


class LidarFaultStabilizer(nn.Module):
    """LiDAR-only adaptation of PFS blocks 2 and 3."""

    def __init__(self, channels):
        super().__init__()
        self.reliability = LidarSpatialReliabilityEstimator(channels)
        self.correction = LidarGeometricCorrection(channels)

    def forward(self, lidar_features):
        reliability = self.reliability(lidar_features)
        corrected = self.correction(lidar_features, reliability)
        return corrected, reliability


class PFSReliabilityModel(nn.Module):
    """PFS-style BEV stabilizer adapted to dense fault-localization heatmaps."""

    def __init__(self, in_channels=3, base_channels=16, dropout=0.0):
        super().__init__()
        self.encoder = BEVEncoder(in_channels=in_channels, base_channels=base_channels)
        bottleneck_channels = base_channels * 16
        self.pfs = PostFusionStabilizer(bottleneck_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.up4 = nn.ConvTranspose2d(bottleneck_channels, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def _match(self, x, target):
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def decode(self, stabilized, skips):
        e1, e2, e3, e4 = skips
        stabilized = self.dropout(stabilized)
        d4 = self._match(self.up4(stabilized), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d4 = self.dropout(d4)
        d3 = self._match(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.dropout(d3)
        d2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

    def forward(self, faulty_bev, clean_bev=None, return_features=False):
        e1, e2, e3, e4, bottleneck = self.encoder(faulty_bev)
        stabilized, pfs_reliability = self.pfs(bottleneck)
        logits = self.decode(stabilized, (e1, e2, e3, e4))

        if not return_features:
            return logits

        clean_bottleneck = None
        if clean_bev is not None:
            with torch.no_grad():
                clean_bottleneck = self.encoder(clean_bev)[-1]

        return {
            "logits": logits,
            "stabilized_features": stabilized,
            "clean_features": clean_bottleneck,
            "pfs_reliability": pfs_reliability,
        }


class LidarOnlyReliabilityModel(PFSReliabilityModel):
    """LiDAR-only fault localizer using spatial reliability and geometric correction."""

    def __init__(self, in_channels=3, base_channels=16, dropout=0.0):
        super().__init__(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
        self.pfs = LidarFaultStabilizer(base_channels * 16)

    def forward(self, faulty_bev, clean_bev=None, return_features=False):
        e1, e2, e3, e4, bottleneck = self.encoder(faulty_bev)
        stabilized, pfs_reliability = self.pfs(bottleneck)
        logits = self.decode(stabilized, (e1, e2, e3, e4))

        if not return_features:
            return logits

        clean_bottleneck = None
        if clean_bev is not None:
            with torch.no_grad():
                clean_bottleneck = self.encoder(clean_bev)[-1]

        return {
            "logits": logits,
            "stabilized_features": stabilized,
            "clean_features": clean_bottleneck,
            "pfs_reliability": pfs_reliability,
        }


class NoPFSReliabilityModel(PFSReliabilityModel):
    """Encoder-decoder baseline with no PFS blocks or PFS auxiliary outputs."""

    def __init__(self, in_channels=3, base_channels=16, dropout=0.0):
        super().__init__(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
        self.pfs = nn.Identity()

    def forward(self, faulty_bev, clean_bev=None, return_features=False):
        e1, e2, e3, e4, bottleneck = self.encoder(faulty_bev)
        logits = self.decode(bottleneck, (e1, e2, e3, e4))

        if not return_features:
            return logits

        return {
            "logits": logits,
            "stabilized_features": bottleneck,
            "clean_features": None,
            "pfs_reliability": None,
        }


MODEL_VARIANTS = {
    "pfs": PFSReliabilityModel,
    "lidar-only": LidarOnlyReliabilityModel,
    "no-pfs": NoPFSReliabilityModel,
}


def build_reliability_model(model_variant="pfs", **kwargs):
    try:
        model_class = MODEL_VARIANTS[model_variant]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown model variant {model_variant!r}; choose one of: {choices}") from exc
    return model_class(**kwargs)
