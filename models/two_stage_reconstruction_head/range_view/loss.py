"""Asymmetric range-view edit losses with separately logged free-space cost."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RangeLossConfig:
    lambda_add: float = 1.0
    lambda_range: float = 1.0
    lambda_delete: float = 1.0
    lambda_geometry: float = 0.0
    lambda_free_space: float = 0.1
    lambda_intensity: float = 0.1
    add_positive_weight: float = 10.0
    false_delete_penalty: float = 30.0
    missed_delete_penalty: float = 1.0
    free_space_margin_m: float = 0.1
    radar_focused_objective: bool = False
    object_class_weights: tuple[float, float, float] = (2.0, 4.0, 4.0)
    lambda_scanline: float = 0.0
    scanline_edge_threshold_m: float = 0.75

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values() if isinstance(value, (int, float))):
            raise ValueError("loss weights and tolerances must be nonnegative")
        if len(self.object_class_weights) != 3 or any(value < 1 for value in self.object_class_weights):
            raise ValueError("car, pedestrian and cyclist weights must each be at least one")
        if self.false_delete_penalty <= self.missed_delete_penalty:
            raise ValueError("false-delete penalty must exceed missed-delete penalty")


def range_edit_loss(
    prediction: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
    config: RangeLossConfig = RangeLossConfig(),
) -> dict[str, torch.Tensor]:
    add = target["add"].float()
    delete = target["delete"].float()
    observed = target["delete_valid"].float()
    clean_valid = target["clean_valid"].float()
    clean_range = target["clean_range_m"].float()
    region = (target.get("radar_region", torch.ones_like(add)).float().clamp(0, 1)
              if config.radar_focused_objective else torch.ones_like(add))
    classes = target.get("object_class", torch.zeros_like(add)).float()
    class_weight = torch.ones_like(add)
    if config.radar_focused_objective:
        for class_id, weight in enumerate(config.object_class_weights, start=1):
            class_weight = torch.where(classes == class_id, weight, class_weight)
    positive = add * region * class_weight
    add_bce = F.binary_cross_entropy_with_logits(prediction["add_logit"], add, reduction="none")
    add_weight = torch.where(add > 0.5, config.add_positive_weight * class_weight, 1.0)
    add_loss = (add_bce * add_weight * region).sum() / region.sum().clamp_min(1)
    range_error = F.smooth_l1_loss(prediction["add_range_m"], target["add_range_m"], reduction="none")
    range_loss = (range_error * positive).sum() / positive.sum().clamp_min(1)
    delete_bce = F.binary_cross_entropy_with_logits(prediction["delete_logit"], delete, reduction="none")
    delete_weight = torch.where(delete > 0.5, config.missed_delete_penalty, config.false_delete_penalty)
    delete_region = observed * region
    delete_loss = (delete_bce * delete_weight * delete_region).sum() / delete_region.sum().clamp_min(1)
    # Same-ray XYZ distance equals absolute radial distance. Keep a separate
    # knob for later geometric targets using noncentral real returns.
    geometry_loss = ((prediction["add_range_m"] - target["add_range_m"]).abs() * positive).sum() / positive.sum().clamp_min(1)
    premature = F.relu(clean_range - prediction["add_range_m"] - config.free_space_margin_m)
    free_space_term = clean_valid * premature / clean_range.detach().amax().clamp_min(1.0)
    if not config.radar_focused_objective:
        # Legacy objective retained for reproducibility of prior runs.
        free_space_term = free_space_term + (1.0 - clean_valid)
    free_space_loss = (prediction["add_probability"] * free_space_term * region).sum() / region.sum().clamp_min(1)
    scanline_loss = add_loss.new_zeros(())
    if config.radar_focused_objective and config.lambda_scanline:
        # Compare horizontal depth changes only on clean same-surface pairs.
        # Healthy neighboring returns are fixed anchors; gradients act on ADD
        # predictions. The clean target defines edges only during training.
        predicted_or_clean = torch.where(add > 0.5, prediction["add_range_m"], clean_range)
        clean_delta = clean_range[..., 1:] - clean_range[..., :-1]
        predicted_delta = predicted_or_clean[..., 1:] - predicted_or_clean[..., :-1]
        pair = (clean_valid[..., 1:] * clean_valid[..., :-1]
                * region[..., 1:] * region[..., :-1]
                * ((add[..., 1:] + add[..., :-1]) > 0).float()
                * (clean_delta.abs() <= config.scanline_edge_threshold_m).float())
        scanline_loss = (F.smooth_l1_loss(predicted_delta, clean_delta, reduction="none")
                         * pair).sum() / pair.sum().clamp_min(1)
    intensity_loss = add_loss.new_zeros(())
    if "add_log_intensity" in prediction:
        if "add_intensity" not in target:
            raise KeyError("Intensity prediction requires add_intensity targets")
        intensity_target = torch.log1p(target["add_intensity"].float().clamp_min(0))
        intensity_error = F.smooth_l1_loss(
            prediction["add_log_intensity"], intensity_target, reduction="none"
        )
        intensity_loss = (intensity_error * positive).sum() / positive.sum().clamp_min(1)
    total = (
        config.lambda_add * add_loss + config.lambda_range * range_loss
        + config.lambda_delete * delete_loss + config.lambda_geometry * geometry_loss
        + config.lambda_free_space * free_space_loss
        + config.lambda_intensity * intensity_loss
        + config.lambda_scanline * scanline_loss
    )
    return {"loss": total, "add_loss": add_loss, "range_loss": range_loss,
            "delete_loss": delete_loss, "geometry_loss": geometry_loss,
            "free_space_loss": free_space_loss, "intensity_loss": intensity_loss,
            "scanline_loss": scanline_loss}
