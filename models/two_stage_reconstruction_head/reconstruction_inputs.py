"""Shared aligned inputs for the coarse and fine reconstruction stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ReconstructionInputs:
    """One batch of sensor evidence shared by both reconstruction stages."""

    faulty_lidar_bev: torch.Tensor
    radar_bev: torch.Tensor
    reconstruction_mask: torch.Tensor
    healthy_context_mask: torch.Tensor
    halo_mask: torch.Tensor
    observability_confidence: torch.Tensor | None = None
    faulty_lidar_points: Sequence[torch.Tensor] | None = None
    radar_points: Sequence[torch.Tensor] | None = None
    lidar_pillar_bev: torch.Tensor | None = None
    radar_pillar_bev: torch.Tensor | None = None
    trusted_faulty: torch.Tensor = field(init=False, repr=False)
    effective_halo: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mask_shape = tuple(self.reconstruction_mask.shape)
        if self.reconstruction_mask.ndim != 4 or mask_shape[1] != 1:
            raise ValueError("reconstruction_mask must have shape [B,1,H,W]")
        for name, tensor in (
            ("faulty_lidar_bev", self.faulty_lidar_bev),
            ("radar_bev", self.radar_bev),
        ):
            if tensor.ndim != 4:
                raise ValueError(f"{name} must have shape [B,C,H,W]")
            if tensor.shape[0] != mask_shape[0] or tensor.shape[-2:] != mask_shape[-2:]:
                raise ValueError(f"{name} must align with reconstruction_mask")
        for name, tensor in (
            ("healthy_context_mask", self.healthy_context_mask),
            ("halo_mask", self.halo_mask),
        ):
            if tuple(tensor.shape) != mask_shape:
                raise ValueError(f"{name} must match reconstruction_mask")
        if self.observability_confidence is not None:
            if tuple(self.observability_confidence.shape) != mask_shape:
                raise ValueError(
                    "observability_confidence must match reconstruction_mask"
                )
        shared = (
            self.faulty_lidar_bev,
            self.radar_bev,
            self.reconstruction_mask,
            self.healthy_context_mask,
            self.halo_mask,
        )
        if self.observability_confidence is not None:
            shared = (*shared, self.observability_confidence)
        if any(tensor.device != self.faulty_lidar_bev.device for tensor in shared):
            raise ValueError("Shared reconstruction tensors must use one device")
        if any(tensor.dtype != self.faulty_lidar_bev.dtype for tensor in shared):
            raise TypeError("Shared reconstruction tensors must use one dtype")
        for name, tensor in (
            ("lidar_pillar_bev", self.lidar_pillar_bev),
            ("radar_pillar_bev", self.radar_pillar_bev),
        ):
            if tensor is None:
                continue
            if tensor.ndim != 4:
                raise ValueError(f"{name} must have shape [B,C,H,W]")
            if tensor.shape[0] != mask_shape[0] or tensor.shape[-2:] != mask_shape[-2:]:
                raise ValueError(f"{name} must align with reconstruction_mask")
            if tensor.device != self.faulty_lidar_bev.device:
                raise ValueError(f"{name} must use the shared input device")
        object.__setattr__(
            self,
            "trusted_faulty",
            self.faulty_lidar_bev * (1.0 - self.reconstruction_mask),
        )
        object.__setattr__(
            self,
            "effective_halo",
            self.halo_mask * (1.0 - self.reconstruction_mask),
        )

    def fine_crop_tensors(
        self,
        coarse_lidar_bev: torch.Tensor,
        *,
        clean_lidar_bev: torch.Tensor | None = None,
        residual_gt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return named references used by local fine-stage cropping."""

        tensors = {
            "coarse": coarse_lidar_bev,
            "faulty": self.faulty_lidar_bev,
            "trusted_faulty": self.trusted_faulty,
            "radar": self.radar_bev,
            "repair": self.reconstruction_mask,
            "halo": self.effective_halo,
        }
        if clean_lidar_bev is not None:
            tensors["clean"] = clean_lidar_bev
        if residual_gt is not None:
            tensors["residual_gt"] = residual_gt
        if self.observability_confidence is not None:
            tensors["observability_confidence"] = self.observability_confidence
        if self.lidar_pillar_bev is not None:
            tensors["lidar_pillars"] = self.lidar_pillar_bev
        if self.radar_pillar_bev is not None:
            tensors["radar_pillars"] = self.radar_pillar_bev
        return tensors
