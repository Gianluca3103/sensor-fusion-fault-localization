from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import warnings

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchMetadata:
    """Coordinates required to place a patch back into its original BEV frame."""

    top: int
    left: int
    height: int
    width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    full_height: int
    full_width: int
    used_full_frame: bool = False


def _dilate_mask(mask: torch.Tensor, halo_radius: int) -> torch.Tensor:
    if halo_radius <= 0:
        return mask
    kernel = 2 * int(halo_radius) + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=halo_radius) > 0


def compute_oracle_patch_metadata(
    mask: torch.Tensor,
    *,
    patch_size: int = 128,
    halo_radius: int = 12,
    full_frame_fallback: bool = True,
) -> PatchMetadata:
    """Return a fixed-size crop centered on the oracle faulty region.

    ``mask`` must have shape ``[1,H,W]`` or ``[H,W]`` and contain at least one
    faulty cell. If the dilated region cannot fit inside ``patch_size`` this
    function warns and falls back to the full frame by default.
    """

    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError(f"mask must have one channel, got {tuple(mask.shape)}")
        mask_2d = mask[0]
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        raise ValueError(f"mask must have shape [1,H,W] or [H,W], got {tuple(mask.shape)}")
    height, width = mask_2d.shape
    if height < 1 or width < 1:
        raise ValueError("mask spatial dimensions must be positive")
    if int((mask_2d > 0).sum().item()) == 0:
        raise ValueError("Stage I reconstruction samples must contain at least one faulty cell")

    dilated = _dilate_mask(mask_2d[None, None], halo_radius)[0, 0]
    coords = torch.nonzero(dilated, as_tuple=False)
    min_row, min_col = coords.min(dim=0).values.tolist()
    max_row, max_col = coords.max(dim=0).values.tolist()
    box_height = int(max_row - min_row + 1)
    box_width = int(max_col - min_col + 1)

    if box_height > patch_size or box_width > patch_size:
        message = (
            f"Dilated fault region {box_height}x{box_width} exceeds patch_size={patch_size}."
        )
        if full_frame_fallback:
            warnings.warn(message + " Falling back to full frame.", RuntimeWarning)
            return PatchMetadata(0, 0, height, width, 0, 0, 0, 0, height, width, True)
        raise ValueError(message)

    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    top = int(center_row - patch_size // 2)
    left = int(center_col - patch_size // 2)
    bottom = top + patch_size
    right = left + patch_size
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)
    top = max(0, top)
    left = max(0, left)
    bottom = min(height, bottom)
    right = min(width, right)
    return PatchMetadata(
        top=top,
        left=left,
        height=bottom - top,
        width=right - left,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        full_height=height,
        full_width=width,
        used_full_frame=False,
    )


def apply_patch_crop(tensor: torch.Tensor, metadata: PatchMetadata) -> torch.Tensor:
    """Apply the same crop/pad metadata to a ``[C,H,W]`` tensor."""

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor [C,H,W], got {tuple(tensor.shape)}")
    crop = tensor[
        :,
        metadata.top : metadata.top + metadata.height,
        metadata.left : metadata.left + metadata.width,
    ]
    if any((metadata.pad_left, metadata.pad_right, metadata.pad_top, metadata.pad_bottom)):
        crop = F.pad(
            crop,
            (metadata.pad_left, metadata.pad_right, metadata.pad_top, metadata.pad_bottom),
        )
    return crop


def crop_sample_tensors(
    tensors: Mapping[str, torch.Tensor],
    mask_key: str = "mask",
    *,
    patch_size: int = 128,
    halo_radius: int = 12,
    full_frame_fallback: bool = True,
) -> tuple[dict[str, torch.Tensor], PatchMetadata]:
    metadata = compute_oracle_patch_metadata(
        tensors[mask_key],
        patch_size=patch_size,
        halo_radius=halo_radius,
        full_frame_fallback=full_frame_fallback,
    )
    return {key: apply_patch_crop(value, metadata) for key, value in tensors.items()}, metadata

