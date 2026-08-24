"""Stage-I reconstruction at the post-PillarScatter feature interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .hrnet_backbone import HRNetBackbone, HRNetConfig


@dataclass(frozen=True)
class PointPillarFeatureReconstructionConfig:
    lidar_feature_channels: int = 64
    radar_feature_channels: int = 64
    lidar_feature_height: int = 320
    lidar_feature_width: int = 320
    lambda_cosine: float = 0.05
    lambda_changed: float = 5.0
    faulty_cell_epsilon: float = 1.0e-6
    hrnet: HRNetConfig = field(default_factory=HRNetConfig)

    def validate(self) -> None:
        for name in (
            "lidar_feature_channels",
            "radar_feature_channels",
            "lidar_feature_height",
            "lidar_feature_width",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.lambda_cosine < 0:
            raise ValueError("lambda_cosine must be non-negative")
        if self.lambda_changed < 0:
            raise ValueError("lambda_changed must be non-negative")
        if self.faulty_cell_epsilon < 0:
            raise ValueError("faulty_cell_epsilon must be non-negative")
        self.hrnet.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "PointPillarFeatureReconstructionConfig":
        values = dict(payload)
        hrnet = values.get("hrnet", {})
        if isinstance(hrnet, dict):
            valid = {item.name for item in fields(HRNetConfig)}
            values["hrnet"] = HRNetConfig(
                **{key: value for key, value in hrnet.items() if key in valid}
            )
        config = cls(**values)
        config.validate()
        return config


class PointPillarFeatureCacheDataset(Dataset):
    """Load cached clean/faulty/radar post-scatter tensors."""

    def __init__(self, sample_paths, cache_root: str | Path, data_root: str | Path):
        self.sample_paths = tuple(Path(path) for path in sample_paths)
        self.cache_root = Path(cache_root)
        self.data_root = Path(data_root)
        if not self.sample_paths:
            raise FileNotFoundError("No feature reconstruction samples were provided")

    def __len__(self) -> int:
        return len(self.sample_paths)

    def cache_path(self, sample_path: Path) -> Path:
        relative = sample_path.relative_to(self.data_root)
        return self.cache_root / relative

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_path = self.sample_paths[index]
        cache_path = self.cache_path(sample_path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing PointPillars feature cache: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            item = {
                key: torch.from_numpy(
                    np.asarray(cached[key], dtype=np.float32)
                )
                for key in (
                    "clean_features",
                    "faulty_features",
                    "radar_features",
                )
            }
        item["sample_path"] = str(sample_path)
        return item


class CoarsePointPillarFeatureReconstructor(nn.Module):
    """Predict a full-grid residual on the detector's post-scatter tensor."""

    def __init__(self, config: PointPillarFeatureReconstructionConfig):
        super().__init__()
        config.validate()
        self.config = config
        input_channels = (
            config.lidar_feature_channels
            + config.radar_feature_channels
        )
        self.backbone = HRNetBackbone(input_channels, config.hrnet)
        self.residual_head = nn.Conv2d(
            self.backbone.out_channels,
            config.lidar_feature_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        faulty_features: torch.Tensor,
        radar_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = (
            self.config.lidar_feature_channels,
            self.config.lidar_feature_height,
            self.config.lidar_feature_width,
        )
        if tuple(faulty_features.shape[1:]) != expected:
            raise ValueError(
                f"faulty_features must be [B,{expected}], got "
                f"{tuple(faulty_features.shape)}"
            )
        if radar_features.shape[0] != faulty_features.shape[0] or tuple(
            radar_features.shape[1:]
        ) != (
            self.config.radar_feature_channels,
            self.config.lidar_feature_height,
            self.config.lidar_feature_width,
        ):
            raise ValueError("radar_features do not match configured feature grid")
        network_input = torch.cat((faulty_features, radar_features), dim=1)
        features, debug = self.backbone(network_input)
        predicted_delta = self.residual_head(features)
        coarse_features = faulty_features + predicted_delta
        output = {
            "predicted_delta": predicted_delta,
            "coarse_features": coarse_features,
            "network_input": network_input,
        }
        output.update(debug)
        return output


def _supported_mean(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    expanded = support.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def pointpillar_feature_reconstruction_loss(
    output: dict[str, torch.Tensor],
    clean_features: torch.Tensor,
    faulty_features: torch.Tensor,
    config: PointPillarFeatureReconstructionConfig,
) -> dict[str, torch.Tensor]:
    """Full-grid target loss with extra weight on target-derived changes."""

    coarse = output["coarse_features"]
    smooth_cells = F.smooth_l1_loss(coarse, clean_features, reduction="none")
    smooth = smooth_cells.mean()

    cosine = 1.0 - F.cosine_similarity(
        coarse.float(), clean_features.float(), dim=1, eps=1.0e-8
    )
    cosine_support = (
        clean_features.float().square().sum(dim=1, keepdim=True) > 1.0e-12
    ).to(coarse.dtype)
    cosine_loss = _supported_mean(cosine[:, None], cosine_support)

    changed = (
        (clean_features - faulty_features).abs().amax(dim=1, keepdim=True)
        > config.faulty_cell_epsilon
    ).to(coarse.dtype)
    unchanged = 1.0 - changed
    changed_smooth = _supported_mean(smooth_cells, changed)
    unchanged_smooth = _supported_mean(smooth_cells, unchanged)
    total = (
        smooth
        + config.lambda_changed * changed_smooth
        + config.lambda_cosine * cosine_loss
    )
    return {
        "loss": total,
        "smooth_l1_full_grid": smooth,
        "smooth_l1_changed_cells": changed_smooth,
        "smooth_l1_unchanged_cells": unchanged_smooth,
        "cosine_feature": cosine_loss,
        "changed_cell_fraction": changed.mean(),
    }
