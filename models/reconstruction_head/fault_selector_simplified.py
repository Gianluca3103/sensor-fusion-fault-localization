"""Small morphology-based selector for LiDAR reconstruction regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label


@dataclass(frozen=True)
class SimplifiedFaultSelectorConfig:
    """The few controls that directly change simplified mask geometry."""

    fault_threshold: float = 0.0
    min_blob_cells: int = 5
    merge_radius_cells: int = 4
    repair_dilation_cells: int = 2
    halo_width_cells: int = 4
    max_blobs: int | None = 3

    def validate(self) -> None:
        if not np.isfinite(self.fault_threshold):
            raise ValueError("fault_threshold must be finite")
        if self.min_blob_cells < 1:
            raise ValueError("min_blob_cells must be positive")
        for name in (
            "merge_radius_cells",
            "repair_dilation_cells",
            "halo_width_cells",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_blobs is not None and self.max_blobs < 1:
            raise ValueError("max_blobs must be positive or None")


@dataclass(frozen=True)
class SimplifiedFaultComponent:
    """Diagnostics for an original (not dilated) grouped fault component."""

    component_id: int
    fault_cell_count: int
    fault_mass: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class SimplifiedFaultSelection:
    """Masks produced by the simplified selector plus lightweight diagnostics."""

    reconstruction_mask: np.ndarray
    halo_mask: np.ndarray
    healthy_context_mask: np.ndarray
    selected_components: tuple[SimplifiedFaultComponent, ...]
    rejected_small_components: tuple[SimplifiedFaultComponent, ...]
    thresholded_cell_count: int
    selected_fault_cell_count: int

    @property
    def selected_cell_count(self) -> int:
        return int(self.reconstruction_mask.sum())

    @property
    def halo_cell_count(self) -> int:
        return int(self.halo_mask.sum())

    @property
    def healthy_context_cell_count(self) -> int:
        return int(self.healthy_context_mask.sum())


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W] or [1,H,W]")
    return values


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0 or not mask.any():
        return mask.copy()
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask, structure=footprint)


def _component_description(
    component_id: int,
    component_mask: np.ndarray,
    heatmap: np.ndarray,
) -> SimplifiedFaultComponent:
    rows, cols = np.nonzero(component_mask)
    return SimplifiedFaultComponent(
        component_id=component_id,
        fault_cell_count=int(rows.size),
        fault_mass=float(heatmap[component_mask].sum()),
        bbox=(
            int(rows.min()),
            int(cols.min()),
            int(rows.max()) + 1,
            int(cols.max()) + 1,
        ),
    )


class FaultSelectorSimplified:
    """Select irregular fault regions using connected components and dilation."""

    def __init__(self, config: SimplifiedFaultSelectorConfig | None = None):
        self.config = config or SimplifiedFaultSelectorConfig()
        self.config.validate()

    def select(
        self,
        fault_heatmap: np.ndarray,
        *,
        reliability_map: np.ndarray,
        faulty_counts: np.ndarray,
        valid_support_mask: np.ndarray | None = None,
    ) -> SimplifiedFaultSelection:
        heatmap = _as_2d(fault_heatmap, "fault_heatmap").astype(
            np.float32, copy=False
        )
        reliability = _as_2d(reliability_map, "reliability_map")
        occupancy_counts = _as_2d(faulty_counts, "faulty_counts")
        if reliability.shape != heatmap.shape or occupancy_counts.shape != heatmap.shape:
            raise ValueError(
                "fault_heatmap, reliability_map, and faulty_counts must share a shape"
            )

        if valid_support_mask is None:
            valid_support = np.ones(heatmap.shape, dtype=bool)
        else:
            valid_support = _as_2d(
                valid_support_mask, "valid_support_mask"
            ).astype(bool, copy=False)
            if valid_support.shape != heatmap.shape:
                raise ValueError(
                    "valid_support_mask and fault_heatmap must share a shape"
                )

        fault_mask = (heatmap > self.config.fault_threshold) & valid_support

        # Dilation is used only to decide which original fault cells belong to the
        # same group. It does not become part of the repair mask by itself.
        grouping_mask = _dilate(fault_mask, self.config.merge_radius_cells)
        labels, component_count = label(
            grouping_mask,
            structure=generate_binary_structure(2, 2),
        )

        qualifying: list[tuple[SimplifiedFaultComponent, np.ndarray]] = []
        rejected: list[SimplifiedFaultComponent] = []
        for component_id in range(1, component_count + 1):
            original_cells = fault_mask & (labels == component_id)
            if not original_cells.any():
                continue
            description = _component_description(
                component_id, original_cells, heatmap
            )
            if description.fault_cell_count < self.config.min_blob_cells:
                rejected.append(description)
            else:
                qualifying.append((description, original_cells))

        priority = lambda item: (
            -item[0].fault_cell_count,
            -item[0].fault_mass,
            item[0].component_id,
        )
        qualifying.sort(key=priority)
        rejected.sort(
            key=lambda item: (-item.fault_cell_count, -item.fault_mass, item.component_id)
        )
        if self.config.max_blobs is not None:
            qualifying = qualifying[: self.config.max_blobs]

        selected_faults = np.zeros(heatmap.shape, dtype=bool)
        for _description, component_mask in qualifying:
            selected_faults |= component_mask

        reconstruction_mask = _dilate(
            selected_faults, self.config.repair_dilation_cells
        ) & valid_support
        halo_mask = (
            _dilate(reconstruction_mask, self.config.halo_width_cells)
            & ~reconstruction_mask
            & valid_support
        )
        # The current oracle reliability maps use exactly 1.0 for reliable cells.
        healthy_occupied = (reliability >= 1.0) & (occupancy_counts > 0)
        healthy_context_mask = halo_mask & healthy_occupied

        return SimplifiedFaultSelection(
            reconstruction_mask=reconstruction_mask,
            halo_mask=halo_mask,
            healthy_context_mask=healthy_context_mask,
            selected_components=tuple(item[0] for item in qualifying),
            rejected_small_components=tuple(rejected),
            thresholded_cell_count=int(fault_mask.sum()),
            selected_fault_cell_count=int(selected_faults.sum()),
        )

