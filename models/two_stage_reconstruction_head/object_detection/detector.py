"""A compact CenterNet-style rotated detector for three-channel LiDAR BEVs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry
from .annotations import RotatedBEVBox
from .geometry import rotated_nms


@dataclass(frozen=True)
class BEVDetectorConfig:
    input_channels: int = 3
    base_channels: int = 32
    output_stride: int = 2
    score_threshold: float = 0.2
    nms_iou_threshold: float = 0.1
    match_iou_threshold: float = 0.5
    top_k: int = 200

    def validate(self) -> None:
        if self.input_channels < 1 or self.base_channels < 4:
            raise ValueError("detector channel counts must be positive")
        if self.output_stride not in {1, 2, 4}:
            raise ValueError("output_stride must be 1, 2, or 4")
        for name in ("score_threshold", "nms_iou_threshold", "match_iou_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(tensor + self.layers(tensor))


class LightweightBEVDetector(nn.Module):
    """Dense centre heatmaps plus metric box regression at each peak."""

    def __init__(self, class_names: tuple[str, ...], config: BEVDetectorConfig | None = None):
        super().__init__()
        self.class_names = tuple(class_names)
        self.config = config or BEVDetectorConfig()
        self.config.validate()
        if not self.class_names:
            raise ValueError("At least one detector class is required")
        channels = self.config.base_channels
        stem: list[nn.Module] = [
            nn.Conv2d(self.config.input_channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        ]
        stride = 1
        while stride < self.config.output_stride:
            stem.extend(
                [
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ]
            )
            stride *= 2
        self.backbone = nn.Sequential(*stem, _ResidualBlock(channels), _ResidualBlock(channels))
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, len(self.class_names), 1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 6, 1),
        )
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(bev)
        return {"heatmap_logits": self.heatmap_head(features), "box_regression": self.box_head(features)}


def _continuous_grid(box: RotatedBEVBox, geometry: BEVGridGeometry, stride: int) -> tuple[float, float]:
    # Integer raster coordinates refer to cell centres. The half-cell terms
    # make metric -> grid -> metric decoding exactly reversible.
    full_row = geometry.height - 0.5 - (box.x - geometry.x_min) / geometry.pillar_size_x
    full_col = (box.y - geometry.y_min) / geometry.pillar_size_y - 0.5
    return full_row / stride, full_col / stride


def _gaussian_radius(length_cells: float, width_cells: float) -> int:
    return max(1, int(round(min(length_cells, width_cells) * 0.25)))


def _draw_gaussian(heatmap: torch.Tensor, row: int, col: int, radius: int) -> None:
    y0, y1 = max(0, row - radius), min(heatmap.shape[0], row + radius + 1)
    x0, x1 = max(0, col - radius), min(heatmap.shape[1], col + radius + 1)
    ys = torch.arange(y0, y1, device=heatmap.device, dtype=heatmap.dtype) - row
    xs = torch.arange(x0, x1, device=heatmap.device, dtype=heatmap.dtype) - col
    gaussian = torch.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / max(2.0 * radius**2, 1.0))
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], gaussian)


def make_detection_targets(
    boxes_batch: list[list[RotatedBEVBox]],
    class_names: tuple[str, ...],
    geometry: BEVGridGeometry,
    output_shape: tuple[int, int],
    output_stride: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = len(boxes_batch)
    height, width = output_shape
    heatmap = torch.zeros((batch_size, len(class_names), height, width), device=device)
    regression = torch.zeros((batch_size, 6, height, width), device=device)
    regression_mask = torch.zeros((batch_size, 1, height, width), device=device)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    for batch_index, boxes in enumerate(boxes_batch):
        for box in boxes:
            if box.class_name not in class_to_index:
                continue
            row_float, col_float = _continuous_grid(box, geometry, output_stride)
            row, col = int(math.floor(row_float)), int(math.floor(col_float))
            if not (0 <= row < height and 0 <= col < width):
                continue
            class_index = class_to_index[box.class_name]
            radius = _gaussian_radius(
                box.length / geometry.pillar_size_x / output_stride,
                box.width / geometry.pillar_size_y / output_stride,
            )
            _draw_gaussian(heatmap[batch_index, class_index], row, col, radius)
            heatmap[batch_index, class_index, row, col] = 1.0
            regression[batch_index, :, row, col] = torch.tensor(
                [
                    row_float - row,
                    col_float - col,
                    math.log(box.length),
                    math.log(box.width),
                    math.sin(box.yaw),
                    math.cos(box.yaw),
                ],
                device=device,
            )
            regression_mask[batch_index, 0, row, col] = 1.0
    return {"heatmap": heatmap, "regression": regression, "regression_mask": regression_mask}


def detector_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    logits, target = outputs["heatmap_logits"], targets["heatmap"]
    probability = torch.sigmoid(logits).clamp(1.0e-5, 1.0 - 1.0e-5)
    positive = target.eq(1.0)
    negative = target.lt(1.0)
    negative_weight = (1.0 - target).pow(4)
    focal = -(
        positive * (1.0 - probability).pow(2) * torch.log(probability)
        + negative * negative_weight * probability.pow(2) * torch.log(1.0 - probability)
    )
    positive_count = positive.sum().clamp_min(1)
    heatmap_loss = focal.sum() / positive_count
    mask = targets["regression_mask"]
    regression_loss = (
        F.smooth_l1_loss(outputs["box_regression"], targets["regression"], reduction="none")
        * mask
    ).sum() / (mask.sum().clamp_min(1) * outputs["box_regression"].shape[1])
    total = heatmap_loss + regression_loss
    return {"loss": total, "heatmap_loss": heatmap_loss, "regression_loss": regression_loss}


@torch.no_grad()
def decode_detections(
    outputs: dict[str, torch.Tensor],
    class_names: tuple[str, ...],
    geometry: BEVGridGeometry,
    config: BEVDetectorConfig,
) -> list[list[RotatedBEVBox]]:
    heatmap = torch.sigmoid(outputs["heatmap_logits"])
    peaks = heatmap.eq(F.max_pool2d(heatmap, 3, stride=1, padding=1))
    heatmap = heatmap * peaks
    regression = outputs["box_regression"]
    decoded: list[list[RotatedBEVBox]] = []
    for batch_index in range(heatmap.shape[0]):
        flattened = heatmap[batch_index].flatten()
        count = min(config.top_k, flattened.numel())
        scores, indices = torch.topk(flattened, count)
        boxes: list[RotatedBEVBox] = []
        output_height, output_width = heatmap.shape[-2:]
        for score, flat_index in zip(scores.tolist(), indices.tolist()):
            if score < config.score_threshold:
                break
            class_index = flat_index // (output_height * output_width)
            location = flat_index % (output_height * output_width)
            row, col = location // output_width, location % output_width
            values = regression[batch_index, :, row, col].float()
            row_float = (row + float(values[0])) * config.output_stride
            col_float = (col + float(values[1])) * config.output_stride
            x = geometry.x_min + (geometry.height - row_float - 0.5) * geometry.pillar_size_x
            y = geometry.y_min + (col_float + 0.5) * geometry.pillar_size_y
            length = float(values[2].clamp(-3.0, 5.0).exp())
            width = float(values[3].clamp(-3.0, 5.0).exp())
            yaw = math.atan2(float(values[4]), float(values[5]))
            boxes.append(
                RotatedBEVBox(
                    class_name=class_names[class_index],
                    x=x,
                    y=y,
                    length=length,
                    width=width,
                    yaw=yaw,
                    confidence=score,
                )
            )
        decoded.append(rotated_nms(boxes, config.nms_iou_threshold))
    return decoded
