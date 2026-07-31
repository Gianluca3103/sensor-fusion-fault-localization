from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BoundaryWeightedBCELoss(nn.Module):
    """Boundary-focused BCE for sparse BEV fault heatmaps.

    Boundary strength is computed from the local 3x3 mixture of soft target
    values. Homogeneous all-healthy and all-faulty neighborhoods receive zero
    extra weight; mixed healthy/faulty neighborhoods receive higher weight.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        eps: float = 1e-6,
        use_evidence_confidence: bool = True,
        evidence_n_ref: float = 10.0,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if evidence_n_ref <= 0.0:
            raise ValueError("evidence_n_ref must be positive")
        self.kernel_size = int(kernel_size)
        self.eps = float(eps)
        self.use_evidence_confidence = bool(use_evidence_confidence)
        self.evidence_n_ref = float(evidence_n_ref)

    def _local_average(self, values: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        padded = F.pad(
            values.float(),
            (padding, padding, padding, padding),
            mode="replicate",
        )
        return F.avg_pool2d(
            padded,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        evidence_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.ndim != 4 or target.ndim != 4:
            raise ValueError("logits and target must have shape [B,C,H,W]")
        if logits.shape != target.shape:
            raise ValueError(
                f"logits and target shapes must match, got {logits.shape} and {target.shape}"
            )
        if torch.any((target < 0.0) | (target > 1.0)):
            raise ValueError("target values must lie in [0,1]")

        target = target.float()
        local_fault_proportion = self._local_average(target)
        boundary_strength = (
            4.0 * local_fault_proportion * (1.0 - local_fault_proportion)
        ).clamp(0.0, 1.0)

        if self.use_evidence_confidence and evidence_count is not None:
            if evidence_count.shape != target.shape:
                raise ValueError(
                    "evidence_count must match target shape, got "
                    f"{evidence_count.shape} and {target.shape}"
                )
            cell_confidence = (evidence_count.float() / self.evidence_n_ref).clamp(
                0.0,
                1.0,
            )
            neighborhood_confidence = self._local_average(cell_confidence)
        else:
            neighborhood_confidence = torch.ones_like(boundary_strength)

        boundary_weight = boundary_strength * neighborhood_confidence
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        weighted_bce_sum = (boundary_weight * bce_map).flatten(1).sum(dim=1)
        boundary_weight_sum = boundary_weight.flatten(1).sum(dim=1)
        valid_boundary = boundary_weight_sum > self.eps
        sample_boundary_bce = torch.zeros_like(weighted_bce_sum)
        sample_boundary_bce = torch.where(
            valid_boundary,
            weighted_bce_sum / boundary_weight_sum.clamp_min(self.eps),
            sample_boundary_bce,
        )
        loss = sample_boundary_bce.mean()

        diagnostics = {
            "boundary_strength_mean": boundary_strength.detach().mean(),
            "boundary_weight_mean": boundary_weight.detach().mean(),
            "boundary_weight_max": boundary_weight.detach().amax(),
            "boundary_cell_fraction": (boundary_weight.detach() > self.eps).float().mean(),
            "valid_boundary_fraction": valid_boundary.detach().float().mean(),
        }
        return loss, diagnostics
