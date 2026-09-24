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
    add_positive_weight: float = 10.0
    false_delete_penalty: float = 30.0
    missed_delete_penalty: float = 1.0
    free_space_margin_m: float = 0.1

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("loss weights and tolerances must be nonnegative")
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
    add_bce = F.binary_cross_entropy_with_logits(prediction["add_logit"], add, reduction="none")
    add_weight = torch.where(add > 0.5, config.add_positive_weight, 1.0)
    add_loss = (add_bce * add_weight).mean()
    range_error = F.smooth_l1_loss(prediction["add_range_m"], target["add_range_m"], reduction="none")
    range_loss = (range_error * add).sum() / add.sum().clamp_min(1)
    delete_bce = F.binary_cross_entropy_with_logits(prediction["delete_logit"], delete, reduction="none")
    delete_weight = torch.where(delete > 0.5, config.missed_delete_penalty, config.false_delete_penalty)
    delete_loss = (delete_bce * delete_weight * observed).sum() / observed.sum().clamp_min(1)
    # Same-ray XYZ distance equals absolute radial distance. Keep a separate
    # knob for later geometric targets using noncentral real returns.
    geometry_loss = ((prediction["add_range_m"] - target["add_range_m"]).abs() * add).sum() / add.sum().clamp_min(1)
    premature = F.relu(clean_range - prediction["add_range_m"] - config.free_space_margin_m)
    free_space_loss = (
        prediction["add_probability"] * (
            clean_valid * premature / clean_range.detach().amax().clamp_min(1.0)
            + (1.0 - clean_valid)
        )
    ).mean()
    total = (
        config.lambda_add * add_loss + config.lambda_range * range_loss
        + config.lambda_delete * delete_loss + config.lambda_geometry * geometry_loss
        + config.lambda_free_space * free_space_loss
    )
    return {"loss": total, "add_loss": add_loss, "range_loss": range_loss,
            "delete_loss": delete_loss, "geometry_loss": geometry_loss,
            "free_space_loss": free_space_loss}
