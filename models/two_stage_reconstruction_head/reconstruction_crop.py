"""Shared spatially aligned repair/halo crop extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


@dataclass
class ReconstructionCropBatch:
    """Aligned local crops, real-cell validity, and full-BEV coordinates."""

    tensors: dict[str, torch.Tensor]
    boxes: torch.Tensor
    source_boxes: torch.Tensor
    valid_mask: torch.Tensor
    active_samples: torch.Tensor
    crop_heights: torch.Tensor
    crop_widths: torch.Tensor
    full_height: int
    full_width: int

    def paste(
        self,
        crop: torch.Tensor,
        *,
        channels: int | None = None,
    ) -> torch.Tensor:
        output_channels = channels or crop.shape[1]
        output = crop.new_zeros(
            (crop.shape[0], output_channels, self.full_height, self.full_width)
        )
        for index, box in enumerate(self.boxes.tolist()):
            top, bottom, left, right = box
            height, width = bottom - top, right - left
            if bool(self.active_samples[index]):
                output[index, :, top:bottom, left:right] = crop[
                    index, :output_channels, :height, :width
                ]
        return output


class ReconstructionCropExtractor:
    """Crop repair/halo, optionally add real context, then technical padding."""

    def __init__(self, pad_multiple: int = 8, minimum_size: int = 1):
        if pad_multiple < 1:
            raise ValueError("pad_multiple must be positive")
        if minimum_size < 1:
            raise ValueError("minimum_size must be positive")
        self.pad_multiple = int(pad_multiple)
        self.minimum_size = int(minimum_size)

    @staticmethod
    def _extent(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
        locations = torch.nonzero(mask, as_tuple=False)
        if locations.numel() == 0:
            return None
        rows, columns = locations[:, -2], locations[:, -1]
        return (
            int(rows.min()),
            int(rows.max()) + 1,
            int(columns.min()),
            int(columns.max()) + 1,
        )

    @staticmethod
    def _expand_axis(start: int, end: int, minimum: int, limit: int) -> tuple[int, int]:
        target = min(limit, max(end - start, minimum))
        extra = target - (end - start)
        expanded_start = start - extra // 2
        expanded_end = end + (extra - extra // 2)
        if expanded_start < 0:
            expanded_end -= expanded_start
            expanded_start = 0
        if expanded_end > limit:
            expanded_start -= expanded_end - limit
            expanded_end = limit
        return max(expanded_start, 0), min(expanded_end, limit)

    def _boxes(
        self,
        repair: torch.Tensor,
        halo: torch.Tensor,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], bool]:
        crop_extent = self._extent(torch.maximum(repair, halo) > 0.5)
        if crop_extent is None:
            return (0, 1, 0, 1), (0, 1, 0, 1), False
        top, bottom, left, right = crop_extent
        height, width = repair.shape[-2:]
        top, bottom = self._expand_axis(top, bottom, self.minimum_size, height)
        left, right = self._expand_axis(left, right, self.minimum_size, width)
        return (top, bottom, left, right), crop_extent, True

    def _box(
        self,
        repair: torch.Tensor,
        halo: torch.Tensor,
    ) -> tuple[tuple[int, int, int, int], bool]:
        box, _source, active = self._boxes(repair, halo)
        return box, active

    def extract(
        self,
        tensors: Mapping[str, torch.Tensor],
        reconstruction_mask: torch.Tensor,
        halo_mask: torch.Tensor,
    ) -> ReconstructionCropBatch:
        if reconstruction_mask.ndim != 4 or reconstruction_mask.shape[1] != 1:
            raise ValueError("reconstruction_mask must have shape [B,1,H,W]")
        if halo_mask.shape != reconstruction_mask.shape:
            raise ValueError("halo_mask must match reconstruction_mask")
        batch, _one, height, width = reconstruction_mask.shape
        for name, tensor in tensors.items():
            if tensor.ndim != 4 or tensor.shape[0] != batch:
                raise ValueError(f"{name} must have shape [B,C,H,W]")
            if tensor.shape[-2:] != (height, width):
                raise ValueError(f"{name} is not spatially aligned")
        box_records = [
            self._boxes(reconstruction_mask[index, 0], halo_mask[index, 0])
            for index in range(batch)
        ]
        boxes = [item[0] for item in box_records]
        source_boxes = [item[1] for item in box_records]
        active = reconstruction_mask.new_tensor(
            [item[2] for item in box_records], dtype=torch.bool
        )
        crop_heights = [bottom - top for top, bottom, _left, _right in boxes]
        crop_widths = [right - left for _top, _bottom, left, right in boxes]
        padded_height = math.ceil(max(crop_heights) / self.pad_multiple) * self.pad_multiple
        padded_width = math.ceil(max(crop_widths) / self.pad_multiple) * self.pad_multiple
        cropped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            output = tensor.new_zeros(
                (batch, tensor.shape[1], padded_height, padded_width)
            )
            for index, box in enumerate(boxes):
                top, bottom, left, right = box
                output[index, :, : bottom - top, : right - left] = tensor[
                    index, :, top:bottom, left:right
                ]
            cropped[name] = output
        valid = reconstruction_mask.new_zeros((batch, 1, padded_height, padded_width))
        for index, (crop_height, crop_width) in enumerate(zip(crop_heights, crop_widths)):
            if bool(active[index]):
                valid[index, :, :crop_height, :crop_width] = 1
        return ReconstructionCropBatch(
            tensors=cropped,
            boxes=torch.tensor(boxes, device=reconstruction_mask.device),
            source_boxes=torch.tensor(source_boxes, device=reconstruction_mask.device),
            valid_mask=valid,
            active_samples=active,
            crop_heights=torch.tensor(crop_heights, device=reconstruction_mask.device),
            crop_widths=torch.tensor(crop_widths, device=reconstruction_mask.device),
            full_height=height,
            full_width=width,
        )
