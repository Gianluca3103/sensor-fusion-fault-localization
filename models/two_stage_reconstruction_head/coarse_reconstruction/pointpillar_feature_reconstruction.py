"""Stage-I reconstruction at the post-PillarScatter feature interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from ..pointpillars import BEVGridGeometry
from .hrnet_backbone import HRNetBackbone, HRNetConfig


@dataclass(frozen=True)
class PointPillarFeatureReconstructionConfig:
    lidar_feature_channels: int = 64
    radar_feature_channels: int = 64
    lidar_feature_height: int = 320
    lidar_feature_width: int = 320
    lambda_cosine: float = 0.05
    faulty_cell_epsilon: float = 1.0e-6
    use_halo_context: bool = True
    hrnet: HRNetConfig = HRNetConfig()

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


def project_mask_between_bev_grids(
    mask: np.ndarray,
    source: BEVGridGeometry,
    destination: BEVGridGeometry,
) -> np.ndarray:
    """Project a binary mask by metric cell centres, never image resizing."""

    source.validate()
    destination.validate()
    array = np.asarray(mask)
    if array.shape == (1, source.height, source.width):
        array = array[0]
    if array.shape != (source.height, source.width):
        raise ValueError(
            f"mask must match source geometry, got {array.shape}"
        )
    destination_rows, destination_cols = np.indices(
        (destination.height, destination.width)
    )
    x = destination.x_min + (
        destination.height - destination_rows - 0.5
    ) * destination.pillar_size_x
    y = destination.y_min + (
        destination_cols + 0.5
    ) * destination.pillar_size_y
    source_rows = source.height - 1 - np.floor(
        (x - source.x_min) / source.pillar_size_x
    ).astype(np.int64)
    source_cols = np.floor(
        (y - source.y_min) / source.pillar_size_y
    ).astype(np.int64)
    valid = (
        (x >= source.x_min)
        & (x < source.x_max)
        & (y >= source.y_min)
        & (y < source.y_max)
        & (source_rows >= 0)
        & (source_rows < source.height)
        & (source_cols >= 0)
        & (source_cols < source.width)
    )
    projected = np.zeros(
        (destination.height, destination.width), dtype=np.float32
    )
    projected[valid] = array[source_rows[valid], source_cols[valid]] > 0.5
    return projected


class PointPillarFeatureCacheDataset(Dataset):
    """Load cached clean/faulty/radar post-scatter tensors and masks."""

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
                    "feature_repair_mask",
                    "feature_halo_mask",
                    "feature_healthy_context_mask",
                )
            }
        item["sample_path"] = str(sample_path)
        return item


class CoarsePointPillarFeatureReconstructor(nn.Module):
    """Predict a masked residual on the detector's post-scatter tensor."""

    def __init__(self, config: PointPillarFeatureReconstructionConfig):
        super().__init__()
        config.validate()
        self.config = config
        input_channels = (
            config.lidar_feature_channels
            + config.radar_feature_channels
            + 2
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
        feature_repair_mask: torch.Tensor,
        feature_halo_mask: torch.Tensor,
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
        for name, mask in (
            ("feature_repair_mask", feature_repair_mask),
            ("feature_halo_mask", feature_halo_mask),
        ):
            if mask.shape != faulty_features.shape[:1] + (1,) + faulty_features.shape[2:]:
                raise ValueError(f"{name} has incompatible shape {tuple(mask.shape)}")

        repair = (feature_repair_mask > 0.5).to(faulty_features.dtype)
        halo = (feature_halo_mask > 0.5).to(faulty_features.dtype)
        context = halo if self.config.use_halo_context else torch.zeros_like(halo)
        active = torch.maximum(repair, context)
        network_input = torch.cat(
            (faulty_features, active * radar_features, repair, context), dim=1
        )
        features, debug = self.backbone(network_input)
        predicted_delta = self.residual_head(features)
        coarse_features = faulty_features + repair * predicted_delta
        output = {
            "predicted_delta": predicted_delta,
            "coarse_features": coarse_features,
            "feature_repair_mask": repair,
            "feature_halo_mask": halo,
            "network_input": network_input,
        }
        output.update(debug)
        return output


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(values).to(values.dtype)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def pointpillar_feature_reconstruction_loss(
    output: dict[str, torch.Tensor],
    clean_features: torch.Tensor,
    faulty_features: torch.Tensor,
    config: PointPillarFeatureReconstructionConfig,
) -> dict[str, torch.Tensor]:
    """Smooth-L1 plus optional cosine loss, supervised only inside repair."""

    coarse = output["coarse_features"]
    repair = output["feature_repair_mask"]
    smooth_cells = F.smooth_l1_loss(coarse, clean_features, reduction="none")
    smooth = _masked_mean(smooth_cells, repair)

    cosine = 1.0 - F.cosine_similarity(
        coarse.float(), clean_features.float(), dim=1, eps=1.0e-8
    )
    cosine_support = (
        clean_features.float().square().sum(dim=1, keepdim=True) > 1.0e-12
    ).to(repair.dtype)
    cosine_loss = _masked_mean(cosine[:, None], repair * cosine_support)
    total = smooth + config.lambda_cosine * cosine_loss

    actual_fault = (
        (clean_features - faulty_features).abs().amax(dim=1, keepdim=True)
        > config.faulty_cell_epsilon
    ).to(repair.dtype) * repair
    sacrificed = repair * (1.0 - actual_fault)
    outside_error = (
        (coarse - faulty_features).abs() * (1.0 - repair)
    ).amax()
    return {
        "loss": total,
        "smooth_l1_feature": smooth,
        "cosine_feature": cosine_loss,
        "smooth_l1_actual_fault": _masked_mean(smooth_cells, actual_fault),
        "smooth_l1_sacrificed_healthy": _masked_mean(smooth_cells, sacrificed),
        "actual_fault_fraction": actual_fault.sum() / repair.sum().clamp_min(1.0),
        "outside_repair_max_change": outside_error,
    }
