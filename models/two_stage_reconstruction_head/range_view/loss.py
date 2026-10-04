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
    radar_to_clean_objective: bool = False
    object_class_weights: tuple[float, float, float] = (2.0, 4.0, 4.0)
    background_positive_weight: float = 0.1
    lambda_scanline: float = 0.0
    scanline_edge_threshold_m: float = 0.75

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values() if isinstance(value, (int, float))):
            raise ValueError("loss weights and tolerances must be nonnegative")
        if len(self.object_class_weights) != 3 or any(value < 1 for value in self.object_class_weights):
            raise ValueError("car, pedestrian and cyclist weights must each be at least one")
        if self.false_delete_penalty <= self.missed_delete_penalty:
            raise ValueError("false-delete penalty must exceed missed-delete penalty")
        if self.radar_to_clean_objective and not self.radar_focused_objective:
            raise ValueError("radar-to-clean training requires radar-focused supervision")


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
    # In radar-to-clean mode, predict the clean scan from radar alone. An ADD
    # label depends on which faulty returns survived, information intentionally
    # hidden from this model, so it cannot be the occupancy training target.
    desired_add = clean_valid if config.radar_to_clean_objective else add
    class_weight = torch.ones_like(add)
    if config.radar_focused_objective:
        ground = target.get("ground_mask", torch.zeros_like(add)).float() > 0.5
        # In this task a missing road return is intentionally a no-add target.
        # Object labels override the ground estimate at target construction.
        desired_add = desired_add * (~ground).float()
        class_weight = torch.full_like(add, config.background_positive_weight)
        for class_id, weight in enumerate(config.object_class_weights, start=1):
            class_weight = torch.where(classes == class_id, weight, class_weight)
    positive = desired_add * region * class_weight
    add_bce = F.binary_cross_entropy_with_logits(prediction["add_logit"], desired_add, reduction="none")
    if config.radar_focused_objective:
        negative = (1 - desired_add) * region
        # Normalize positives and negatives separately: a large radar/road
        # region must not drown out the comparatively few missing object rays.
        add_loss = (config.add_positive_weight * (add_bce * positive).sum()
                    / positive.sum().clamp_min(1)
                    + (add_bce * negative).sum() / negative.sum().clamp_min(1))
    else:
        add_weight = torch.where(add > 0.5, config.add_positive_weight, 1.0)
        add_loss = (add_bce * add_weight).mean()
    range_target = clean_range if config.radar_to_clean_objective else target["add_range_m"]
    range_error = F.smooth_l1_loss(prediction["add_range_m"], range_target, reduction="none")
    range_loss = (range_error * positive).sum() / positive.sum().clamp_min(1)
    delete_bce = F.binary_cross_entropy_with_logits(prediction["delete_logit"], delete, reduction="none")
    delete_weight = torch.where(delete > 0.5, config.missed_delete_penalty, config.false_delete_penalty)
    delete_region = observed * region
    delete_loss = (delete_bce * delete_weight * delete_region).sum() / delete_region.sum().clamp_min(1)
    # Same-ray XYZ distance equals absolute radial distance. Keep a separate
    # knob for later geometric targets using noncentral real returns.
    geometry_loss = ((prediction["add_range_m"] - range_target).abs() * positive).sum() / positive.sum().clamp_min(1)
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
        predicted_or_clean = torch.where(desired_add > 0.5, prediction["add_range_m"], clean_range)
        clean_delta = clean_range[..., 1:] - clean_range[..., :-1]
        predicted_delta = predicted_or_clean[..., 1:] - predicted_or_clean[..., :-1]
        pair_interest = torch.maximum(positive[..., 1:], positive[..., :-1])
        pair = (clean_valid[..., 1:] * clean_valid[..., :-1]
                * region[..., 1:] * region[..., :-1]
                * pair_interest
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
        # The cached intensity label covers missing returns only. In this
        # mode the network predicts all clean rays, but only generated rays
        # need an intensity during conservative merging.
        intensity_positive = (add * desired_add * region * class_weight
                              if config.radar_to_clean_objective else positive)
        intensity_loss = ((intensity_error * intensity_positive).sum()
                          / intensity_positive.sum().clamp_min(1))
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
