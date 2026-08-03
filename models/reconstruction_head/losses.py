from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_denominator(mask: torch.Tensor, channels: int = 1, eps: float = 1e-6) -> torch.Tensor:
    return mask.sum().clamp_min(eps) * int(channels)


def masked_feature_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "smooth_l1",
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if mask.shape[1] != 1:
        raise ValueError("mask must have one channel")
    error = mask * (prediction - target)
    denom = _mask_denominator(mask, prediction.shape[1])
    if mode == "l1":
        return error.abs().sum() / denom
    if mode == "smooth_l1":
        return F.smooth_l1_loss(error, torch.zeros_like(error), reduction="sum") / denom
    raise ValueError(f"Unsupported feature loss mode: {mode!r}")


def masked_occupancy_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / _mask_denominator(mask)


def masked_offset_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
    valid = mask * occupancy
    loss = F.smooth_l1_loss(prediction, target, reduction="none") * valid
    return loss.sum() / _mask_denominator(valid, prediction.shape[1])


def healthy_region_change(candidate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    healthy = 1.0 - mask
    return (healthy * (candidate - reference).abs()).sum() / _mask_denominator(healthy, candidate.shape[1])


def coarse_reconstruction_loss(
    outputs: dict[str, torch.Tensor],
    lidar_corrupt: torch.Tensor,
    lidar_clean: torch.Tensor,
    mask: torch.Tensor,
    *,
    feature_weight: float = 1.0,
    occupancy_weight: float = 0.0,
    offset_weight: float = 0.0,
    feature_loss_mode: str = "smooth_l1",
    clean_occupancy: torch.Tensor | None = None,
    clean_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature = masked_feature_loss(outputs["coarse_features"], lidar_clean, mask, mode=feature_loss_mode)
    total = feature_weight * feature
    diagnostics = {
        "feature": feature.detach(),
        "healthy_change": healthy_region_change(outputs["coarse_features"], lidar_corrupt, mask).detach(),
    }
    if "occupancy_logits" in outputs and clean_occupancy is not None and occupancy_weight > 0:
        occ = masked_occupancy_bce(outputs["occupancy_logits"], clean_occupancy, mask)
        total = total + occupancy_weight * occ
        diagnostics["occupancy_bce"] = occ.detach()
    if "offset" in outputs and clean_offsets is not None and clean_occupancy is not None and offset_weight > 0:
        offset = masked_offset_loss(outputs["offset"], clean_offsets, mask, clean_occupancy)
        total = total + offset_weight * offset
        diagnostics["offset"] = offset.detach()
    diagnostics["loss"] = total.detach()
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite coarse reconstruction loss")
    return total, diagnostics


def diffusion_noise_loss(
    epsilon: torch.Tensor,
    epsilon_prediction: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if epsilon.shape != epsilon_prediction.shape:
        raise ValueError(f"epsilon shapes differ: {epsilon.shape} vs {epsilon_prediction.shape}")
    error = mask * (epsilon - epsilon_prediction)
    loss = (error.square().sum() / _mask_denominator(mask, epsilon.shape[1]))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite diffusion noise loss")
    return loss


def residual_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (mask * (prediction - target).abs()).sum() / _mask_denominator(mask, prediction.shape[1])

