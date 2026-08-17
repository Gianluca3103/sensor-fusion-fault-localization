"""Baseline channel-aware objectives and metrics for coarse reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn
import torch.nn.functional as F

from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M


OBSERVABILITY_TOLERANCE = 1.0e-6
BEV_RESOLUTION_M = 0.2
OCCUPANCY_TOLERANCE_M = 0.5


@dataclass(frozen=True)
class ObservabilityWeightingConfig:
    enabled: bool = False
    min_empty_weight: float = 0.1

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("observability_weighting.enabled must be a bool")
        if (
            isinstance(self.min_empty_weight, bool)
            or not isinstance(self.min_empty_weight, (int, float))
            or not math.isfinite(float(self.min_empty_weight))
            or not 0.0 <= self.min_empty_weight <= 1.0
        ):
            raise ValueError(
                "observability_weighting.min_empty_weight must be in [0,1]"
            )


@dataclass(frozen=True)
class OccupancyLossConfig:
    """Optional Stage-I occupancy objective; ``existing`` is the baseline."""

    type: str = "existing"
    exact_weight: float = 0.25
    tolerant_recall_weight: float = 1.0
    far_fp_weight: float = 0.5
    tolerance_radius_m: float = 0.5

    def validate(self) -> None:
        if self.type not in {"existing", "tolerance_aware"}:
            raise ValueError(
                "occupancy.type must be 'existing' or 'tolerance_aware'"
            )
        for name in (
            "exact_weight",
            "tolerant_recall_weight",
            "far_fp_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"occupancy.{name} must be finite and non-negative")
        if (
            isinstance(self.tolerance_radius_m, bool)
            or not isinstance(self.tolerance_radius_m, (int, float))
            or not math.isfinite(float(self.tolerance_radius_m))
            or self.tolerance_radius_m < 0
        ):
            raise ValueError(
                "occupancy.tolerance_radius_m must be finite and non-negative"
            )


@dataclass(frozen=True)
class CoarseLossConfig:
    lambda_occupancy: float = 1.0
    lambda_density: float = 1.0
    lambda_height: float = 1.0
    epsilon: float = 1.0e-8
    observability_weighting: ObservabilityWeightingConfig = field(
        default_factory=ObservabilityWeightingConfig
    )
    occupancy: OccupancyLossConfig = field(default_factory=OccupancyLossConfig)

    def validate(self) -> None:
        for name in ("lambda_occupancy", "lambda_density", "lambda_height"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.observability_weighting.validate()
        self.occupancy.validate()


def tolerance_radius_cells(tolerance_radius_m: float, resolution_m: float) -> int:
    """Convert metric tolerance to a conservative symmetric cell radius."""

    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("BEV resolution must be finite and positive")
    if not math.isfinite(tolerance_radius_m) or tolerance_radius_m < 0:
        raise ValueError("Tolerance radius must be finite and non-negative")
    return int(math.floor(tolerance_radius_m / resolution_m + 1.0e-9))


def _validate_observability(
    observability_confidence: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if observability_confidence is None:
        raise ValueError(
            "Observability-aware occupancy weighting is enabled, but the batch "
            "does not contain observability_confidence. Regenerate samples with "
            "the clean-LiDAR observability branch or disable weighting."
        )
    expected_shape = (reference.shape[0], 1, *reference.shape[-2:])
    if observability_confidence.shape != expected_shape:
        raise ValueError(
            "observability_confidence must have shape [B,1,H,W] aligned with "
            f"the clean BEV; expected {expected_shape}, got "
            f"{tuple(observability_confidence.shape)}"
        )
    if observability_confidence.device != reference.device:
        raise ValueError(
            "observability_confidence must be on the same device as clean LiDAR"
        )
    if not observability_confidence.is_floating_point():
        raise TypeError("observability_confidence must use a floating dtype")
    if not torch.isfinite(observability_confidence).all():
        raise ValueError("observability_confidence contains NaN or Inf")
    minimum = float(observability_confidence.min())
    maximum = float(observability_confidence.max())
    if minimum < -OBSERVABILITY_TOLERANCE or maximum > 1.0 + OBSERVABILITY_TOLERANCE:
        raise ValueError(
            "observability_confidence must contain values in [0,1]; "
            f"got range [{minimum}, {maximum}]"
        )
    return observability_confidence.clamp(0.0, 1.0)


def occupancy_bce_weights(
    clean_occupancy: torch.Tensor,
    observability_confidence: torch.Tensor,
    min_empty_weight: float,
) -> torch.Tensor:
    """Return unit occupied weights and confidence-scaled empty weights."""

    empty_weight = min_empty_weight + (
        1.0 - min_empty_weight
    ) * observability_confidence
    return clean_occupancy + (1.0 - clean_occupancy) * empty_weight


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Average values over valid cells and return differentiable zero if empty."""

    return (values * mask).sum() / mask.sum().clamp_min(epsilon)


def _soft_dice_loss_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, logits.ndim))
    intersection = (probabilities * target * mask).sum(dim=dimensions)
    denominator = (probabilities * mask).sum(dim=dimensions) + (
        target * mask
    ).sum(dim=dimensions)
    losses = 1.0 - (2.0 * intersection + epsilon) / (
        denominator + epsilon
    )
    valid_samples = mask.sum(dim=dimensions) > 0
    return (losses * valid_samples).sum() / valid_samples.sum().clamp_min(1)


class MaskedBEVReconstructionLoss(nn.Module):
    """Masked occupancy, density, and P90-height reconstruction objective."""

    def __init__(
        self,
        config: CoarseLossConfig | None = None,
        *,
        bev_resolution_m: float = BEV_RESOLUTION_M,
    ):
        super().__init__()
        self.config = config or CoarseLossConfig()
        self.config.validate()
        self.bev_resolution_m = float(bev_resolution_m)
        self.tolerance_radius_cells = tolerance_radius_cells(
            self.config.occupancy.tolerance_radius_m,
            self.bev_resolution_m,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        clean_lidar_bev: torch.Tensor,
        observability_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        replacement = outputs["replacement_raw"]
        mask = outputs["reconstruction_mask"]
        if replacement.ndim != 4 or replacement.shape[1] != 3:
            raise ValueError("replacement_raw must have shape [B,3,H,W]")
        if clean_lidar_bev.shape != replacement.shape:
            raise ValueError("clean_lidar_bev must match replacement_raw")
        if mask.shape != (replacement.shape[0], 1, *replacement.shape[-2:]):
            raise ValueError("reconstruction_mask must have shape [B,1,H,W]")

        occupancy_logits = replacement[:, 0:1]
        predicted_density = replacement[:, 1:2]
        predicted_height = replacement[:, 2:3]
        clean_occupancy = clean_lidar_bev[:, 0:1]
        clean_density = clean_lidar_bev[:, 1:2]
        clean_height = clean_lidar_bev[:, 2:3]

        occupancy_bce_per_cell = F.binary_cross_entropy_with_logits(
            occupancy_logits,
            clean_occupancy,
            reduction="none",
        )
        weighting = self.config.observability_weighting
        if weighting.enabled:
            observability_confidence = _validate_observability(
                observability_confidence,
                clean_occupancy,
            )
            occupancy_weight = occupancy_bce_weights(
                clean_occupancy,
                observability_confidence,
                weighting.min_empty_weight,
            )
            valid_weight = mask * occupancy_weight
            occupancy_bce = _masked_mean(
                occupancy_bce_per_cell,
                valid_weight,
                self.config.epsilon,
            )
        else:
            occupancy_weight = None
            occupancy_bce = _masked_mean(
                occupancy_bce_per_cell,
                mask,
                self.config.epsilon,
            )
        occupancy_dice = _soft_dice_loss_per_sample(
            occupancy_logits,
            clean_occupancy,
            mask,
            self.config.epsilon,
        )
        occupancy_exact = occupancy_bce + occupancy_dice
        zero = occupancy_logits.sum() * 0.0
        occupancy_tolerant_recall = zero
        occupancy_far_fp = zero
        occupancy_config = self.config.occupancy
        if occupancy_config.type == "tolerance_aware":
            (
                occupancy_tolerant_recall,
                occupancy_far_fp,
            ) = _tolerance_aware_occupancy_terms(
                occupancy_logits,
                clean_occupancy,
                mask,
                radius_cells=self.tolerance_radius_cells,
                epsilon=self.config.epsilon,
            )
            occupancy_loss = (
                occupancy_config.exact_weight * occupancy_exact
                + occupancy_config.tolerant_recall_weight
                * occupancy_tolerant_recall
                + occupancy_config.far_fp_weight * occupancy_far_fp
            )
        else:
            occupancy_loss = occupancy_exact

        continuous_mask = mask * clean_occupancy
        density_loss = _masked_mean(
            F.smooth_l1_loss(
                predicted_density, clean_density, reduction="none"
            ),
            continuous_mask,
            self.config.epsilon,
        )
        height_loss = _masked_mean(
            F.smooth_l1_loss(
                predicted_height, clean_height, reduction="none"
            ),
            continuous_mask,
            self.config.epsilon,
        )

        total = (
            self.config.lambda_occupancy * occupancy_loss
            + self.config.lambda_density * density_loss
            + self.config.lambda_height * height_loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite coarse BEV reconstruction loss")
        result = {
            "loss": total,
            "loss_total": total,
            "loss_occupancy": occupancy_loss,
            "loss_occupancy_exact": occupancy_exact,
            "loss_occupancy_tolerant_recall": occupancy_tolerant_recall,
            "loss_occupancy_far_fp": occupancy_far_fp,
            "loss_occupancy_bce": occupancy_bce,
            "loss_occupancy_dice": occupancy_dice,
            "loss_density": density_loss,
            "loss_height": height_loss,
        }
        if weighting.enabled:
            empty_mask = mask * (1.0 - clean_occupancy)
            result.update(
                _observability_loss_diagnostics(
                    observability_confidence,
                    occupancy_weight,
                    mask,
                    empty_mask,
                    self.config.epsilon,
                )
            )
        return result


def _tolerance_aware_occupancy_terms(
    occupancy_logits: torch.Tensor,
    clean_occupancy: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    *,
    radius_cells: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable tolerant-recall and distant-FP penalties."""

    valid = reconstruction_mask > 0
    target_positive = (clean_occupancy >= 0.5) & valid
    kernel_size = 2 * radius_cells + 1

    # max(sigmoid(x)) == sigmoid(max(x)); pooling logits keeps BCE stable.
    minimum = torch.finfo(occupancy_logits.dtype).min
    valid_logits = occupancy_logits.masked_fill(~valid, minimum)
    nearby_logits = F.max_pool2d(
        valid_logits,
        kernel_size=kernel_size,
        stride=1,
        padding=radius_cells,
    )
    tolerant_recall_per_cell = torch.where(
        target_positive,
        F.softplus(-nearby_logits),
        torch.zeros_like(nearby_logits),
    )
    tolerant_recall = _masked_mean(
        tolerant_recall_per_cell,
        target_positive.to(dtype=occupancy_logits.dtype),
        epsilon,
    )

    target_tolerance_region = F.max_pool2d(
        target_positive.to(dtype=occupancy_logits.dtype),
        kernel_size=kernel_size,
        stride=1,
        padding=radius_cells,
    ) > 0
    far_region = valid & ~target_tolerance_region
    far_fp = _masked_mean(
        F.softplus(occupancy_logits),
        far_region.to(dtype=occupancy_logits.dtype),
        epsilon,
    )
    return tolerant_recall, far_fp


def _observability_loss_diagnostics(
    confidence: torch.Tensor,
    occupancy_weight: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    empty_mask: torch.Tensor,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        mean_repair = _masked_mean(confidence, reconstruction_mask, epsilon)
        mean_empty = _masked_mean(confidence, empty_mask, epsilon)
        mean_empty_weight = _masked_mean(
            occupancy_weight, empty_mask, epsilon
        )
        valid_empty = empty_mask > 0
        if valid_empty.any():
            minimum = occupancy_weight[valid_empty].min()
            maximum = occupancy_weight[valid_empty].max()
        else:
            minimum = maximum = confidence.new_zeros(())
        return {
            "mean_observability_repair": mean_repair,
            "mean_empty_observability_repair": mean_empty,
            "mean_empty_occupancy_weight": mean_empty_weight,
            "min_empty_occupancy_weight": minimum,
            "max_empty_occupancy_weight": maximum,
        }


def _safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return numerator / denominator.clamp_min(epsilon)


def _occupancy_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    *,
    include_tolerant: bool,
    resolution_m: float,
    tolerance_m: float,
) -> dict[str, torch.Tensor]:
    predicted = probability >= 0.5
    occupied = target >= 0.5
    valid = mask > 0
    tp = (predicted & occupied & valid).sum(dtype=torch.float32)
    fp = (predicted & ~occupied & valid).sum(dtype=torch.float32)
    fn = (~predicted & occupied & valid).sum(dtype=torch.float32)
    tn = (~predicted & ~occupied & valid).sum(dtype=torch.float32)
    precision = _safe_ratio(tp, tp + fp, epsilon)
    recall = _safe_ratio(tp, tp + fn, epsilon)
    exact_f1 = _safe_ratio(
        2.0 * precision * recall, precision + recall, epsilon
    )
    exact_iou = _safe_ratio(tp, tp + fp + fn, epsilon)
    result = {
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": exact_f1,
        "occupancy_iou": exact_iou,
        "occupancy_exact_precision": precision,
        "occupancy_exact_recall": recall,
        "occupancy_exact_f1": exact_f1,
        "occupancy_exact_iou": exact_iou,
        "occupancy_hallucination_rate": _safe_ratio(fp, fp + tn, epsilon),
    }
    if include_tolerant:
        result.update(
            _tolerant_occupancy_metrics(
                predicted,
                occupied,
                valid,
                tolerance_m=tolerance_m,
                resolution_m=resolution_m,
                epsilon=epsilon,
            )
        )
    return result


def _dilate_with_metric_disk(
    values: torch.Tensor,
    tolerance_m: float,
    resolution_m: float,
) -> torch.Tensor:
    radius_cells = int(tolerance_m // resolution_m)
    offsets = torch.arange(
        -radius_cells,
        radius_cells + 1,
        device=values.device,
        dtype=torch.float32,
    )
    rows, columns = torch.meshgrid(offsets, offsets, indexing="ij")
    kernel = (
        torch.sqrt(rows.square() + columns.square()) * resolution_m
        <= tolerance_m + 1.0e-6
    ).to(dtype=torch.float32)
    dilated = F.conv2d(
        values.to(dtype=torch.float32),
        kernel[None, None],
        padding=radius_cells,
    )
    return dilated > 0


def _tolerant_occupancy_metrics(
    predicted: torch.Tensor,
    occupied: torch.Tensor,
    valid: torch.Tensor,
    *,
    tolerance_m: float,
    resolution_m: float,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    predicted = predicted & valid
    occupied = occupied & valid
    target_neighborhood = _dilate_with_metric_disk(
        occupied, tolerance_m, resolution_m
    )
    prediction_neighborhood = _dilate_with_metric_disk(
        predicted, tolerance_m, resolution_m
    )
    matched_predictions = (predicted & target_neighborhood).sum(
        dtype=torch.float32
    )
    matched_targets = (occupied & prediction_neighborhood).sum(
        dtype=torch.float32
    )
    prediction_count = predicted.sum(dtype=torch.float32)
    target_count = occupied.sum(dtype=torch.float32)
    precision = _safe_ratio(matched_predictions, prediction_count, epsilon)
    recall = _safe_ratio(matched_targets, target_count, epsilon)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall, epsilon)
    # Bidirectional tolerant matching has no single TP count. This monotonic
    # F1-equivalent reports IoU on the familiar scale: IoU = F1 / (2 - F1).
    iou = _safe_ratio(f1, 2.0 - f1, epsilon)
    return {
        "occupancy_tolerant_precision": precision,
        "occupancy_tolerant_recall": recall,
        "occupancy_tolerant_f1": f1,
        "occupancy_tolerant_iou": iou,
        "occupancy_tolerant_0_5m_precision": precision,
        "occupancy_tolerant_0_5m_recall": recall,
        "occupancy_tolerant_0_5m_f1": f1,
        "occupancy_tolerant_0_5m_iou": iou,
    }


def _continuous_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = prediction - target
    mae = _masked_mean(difference.abs(), mask, epsilon)
    rmse = torch.sqrt(_masked_mean(difference.square(), mask, epsilon))
    return mae, rmse


@torch.no_grad()
def coarse_reconstruction_metrics(
    outputs: dict[str, torch.Tensor],
    faulty_lidar_bev: torch.Tensor,
    clean_lidar_bev: torch.Tensor,
    epsilon: float = 1.0e-8,
    observability_confidence: torch.Tensor | None = None,
    *,
    include_tolerant: bool = True,
    resolution_m: float = BEV_RESOLUTION_M,
    tolerance_m: float = OCCUPANCY_TOLERANCE_M,
) -> dict[str, torch.Tensor]:
    """Compare reconstructed and faulty baselines inside the repair region."""

    mask = outputs["reconstruction_mask"]
    occupancy_target = clean_lidar_bev[:, 0:1]
    continuous_mask = mask * occupancy_target
    coarse = outputs["coarse_lidar_bev"]

    coarse_metrics = _occupancy_metrics(
        torch.sigmoid(outputs["occupancy_logits"]),
        occupancy_target,
        mask,
        epsilon,
        include_tolerant=include_tolerant,
        resolution_m=resolution_m,
        tolerance_m=tolerance_m,
    )
    faulty_metrics = _occupancy_metrics(
        faulty_lidar_bev[:, 0:1],
        occupancy_target,
        mask,
        epsilon,
        include_tolerant=include_tolerant,
        resolution_m=resolution_m,
        tolerance_m=tolerance_m,
    )
    density_mae, density_rmse = _continuous_metrics(
        coarse[:, 1:2], clean_lidar_bev[:, 1:2], continuous_mask, epsilon
    )
    faulty_density_mae, faulty_density_rmse = _continuous_metrics(
        faulty_lidar_bev[:, 1:2],
        clean_lidar_bev[:, 1:2],
        continuous_mask,
        epsilon,
    )
    height_mae, height_rmse = _continuous_metrics(
        coarse[:, 2:3], clean_lidar_bev[:, 2:3], continuous_mask, epsilon
    )
    faulty_height_mae, faulty_height_rmse = _continuous_metrics(
        faulty_lidar_bev[:, 2:3],
        clean_lidar_bev[:, 2:3],
        continuous_mask,
        epsilon,
    )
    height_scale_m = float(HEIGHT_RANGE_M[1] - HEIGHT_RANGE_M[0])
    outside_change = (
        (1.0 - mask) * (coarse - faulty_lidar_bev)
    ).abs().max()

    result = {f"coarse_{key}": value for key, value in coarse_metrics.items()}
    result.update(
        {f"faulty_{key}": value for key, value in faulty_metrics.items()}
    )
    result.update(
        {
            "coarse_density_mae": density_mae,
            "coarse_density_rmse": density_rmse,
            "faulty_density_mae": faulty_density_mae,
            "faulty_density_rmse": faulty_density_rmse,
            "coarse_height_mae": height_mae,
            "coarse_height_rmse": height_rmse,
            "faulty_height_mae": faulty_height_mae,
            "faulty_height_rmse": faulty_height_rmse,
            "coarse_height_mae_m": height_mae * height_scale_m,
            "coarse_height_rmse_m": height_rmse * height_scale_m,
            "faulty_height_mae_m": faulty_height_mae * height_scale_m,
            "faulty_height_rmse_m": faulty_height_rmse * height_scale_m,
            "outside_mask_max_change": outside_change,
        }
    )
    if observability_confidence is not None:
        confidence = _validate_observability(
            observability_confidence,
            occupancy_target,
        )
        predicted_occupied = torch.sigmoid(
            outputs["occupancy_logits"]
        ) >= 0.5
        clean_empty = (mask > 0) & (occupancy_target < 0.5)
        bins = {
            "low": confidence < 0.25,
            "medium": (confidence >= 0.25) & (confidence < 0.75),
            "high": confidence >= 0.75,
        }
        for name, bin_mask in bins.items():
            valid = clean_empty & bin_mask
            count = valid.sum(dtype=torch.float32)
            false_positives = (predicted_occupied & valid).sum(
                dtype=torch.float32
            )
            result[f"hallucination_rate_{name}_observability"] = _safe_ratio(
                false_positives,
                count,
                epsilon,
            )
            result[f"empty_cells_{name}_observability"] = count
    return result


@torch.no_grad()
def coarse_reconstruction_range_metrics(
    outputs: dict[str, torch.Tensor],
    clean_lidar_bev: torch.Tensor,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    epsilon: float = 1.0e-8,
    include_tolerant: bool = True,
) -> dict[str, torch.Tensor]:
    """Report occupancy reconstruction metrics in fixed radial-distance bins."""

    mask = outputs["reconstruction_mask"]
    batch, _, height, width = mask.shape
    device = mask.device
    dtype = mask.dtype
    x_step = (x_range[1] - x_range[0]) / height
    y_step = (y_range[1] - y_range[0]) / width
    x = x_range[1] - (
        torch.arange(height, device=device, dtype=dtype) + 0.5
    ) * x_step
    y = y_range[0] + (
        torch.arange(width, device=device, dtype=dtype) + 0.5
    ) * y_step
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    distance = torch.sqrt(xx.square() + yy.square())[None, None]
    distance = distance.expand(batch, -1, -1, -1)
    probability = torch.sigmoid(outputs["occupancy_logits"])
    target = clean_lidar_bev[:, 0:1]
    bins = (
        ("0_15m", 0.0, 15.0),
        ("15_30m", 15.0, 30.0),
        ("30_45m", 30.0, 45.0),
        ("45_60m", 45.0, 60.0),
        ("over_60m", 60.0, float("inf")),
    )
    result = {}
    for name, minimum, maximum in bins:
        range_mask = (distance >= minimum) & (distance < maximum)
        metrics = _occupancy_metrics(
            probability,
            target,
            mask * range_mask.to(dtype=mask.dtype),
            epsilon,
            include_tolerant=include_tolerant,
            resolution_m=(x_step + y_step) / 2.0,
            tolerance_m=OCCUPANCY_TOLERANCE_M,
        )
        result.update(
            {f"range_{name}/{key}": value for key, value in metrics.items()}
        )
        result[f"range_{name}/repair_cells"] = (
            mask * range_mask.to(dtype=mask.dtype)
        ).sum(dtype=torch.float32)
    return result
