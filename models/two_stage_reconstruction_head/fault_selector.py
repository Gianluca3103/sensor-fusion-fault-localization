"""Select only LiDAR cells whose original returns were almost completely lost."""

from __future__ import annotations

from dataclasses import dataclass
import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import binary_dilation, label


@dataclass(frozen=True)
class FaultSelectorConfig:
    """Per-cell repair policy and its surrounding context halo."""

    min_lidar_loss_fraction: float = 0.95
    min_repair_box_cells: int = 5
    min_repair_fault_fraction: float = 0.95
    primary_expansion_gap_cells: int = 0
    max_secondary_repair_boxes: int = 0
    min_secondary_lidar_loss_fraction: float = 0.70
    min_secondary_repair_fault_fraction: float = 0.70
    min_secondary_repair_cells: int = 1
    secondary_merge_gap_cells: int = 4
    min_secondary_box_spatial_density: float = 0.01
    min_secondary_box_side_cells: int = 1
    min_halo_healthy_fraction: float = 0.90
    min_halo_healthy_cells: int = 64
    min_halo_context_ratio: float = 0.25
    min_halo_width_cells: int = 4
    max_halo_dilation_cells: int = 64
    distance_bin_m: float = 10.0
    x_min_m: float = 0.0
    x_cell_size_m: float = 0.2

    def validate(self) -> None:
        for name in (
            "min_lidar_loss_fraction",
            "min_secondary_lidar_loss_fraction",
            "min_repair_fault_fraction",
            "min_secondary_repair_fault_fraction",
            "min_halo_healthy_fraction",
            "min_secondary_box_spatial_density",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.min_halo_healthy_cells < 0:
            raise ValueError("min_halo_healthy_cells must be non-negative")
        if self.min_repair_box_cells < 1:
            raise ValueError("min_repair_box_cells must be positive")
        if self.primary_expansion_gap_cells < 0:
            raise ValueError("primary_expansion_gap_cells must be non-negative")
        if self.max_secondary_repair_boxes < 0:
            raise ValueError("max_secondary_repair_boxes must be non-negative")
        if self.min_secondary_repair_cells < 1:
            raise ValueError("min_secondary_repair_cells must be positive")
        if self.secondary_merge_gap_cells < 0:
            raise ValueError("secondary_merge_gap_cells must be non-negative")
        if self.min_secondary_box_side_cells < 1:
            raise ValueError("min_secondary_box_side_cells must be positive")
        if (
            not np.isfinite(self.min_halo_context_ratio)
            or self.min_halo_context_ratio < 0.0
        ):
            raise ValueError(
                "min_halo_context_ratio must be finite and non-negative"
            )
        if self.min_halo_width_cells < 1:
            raise ValueError("min_halo_width_cells must be positive")
        if self.max_halo_dilation_cells < self.min_halo_width_cells:
            raise ValueError(
                "max_halo_dilation_cells must be at least min_halo_width_cells"
            )
        if not np.isfinite(self.distance_bin_m) or self.distance_bin_m <= 0.0:
            raise ValueError("distance_bin_m must be finite and positive")
        if not np.isfinite(self.x_min_m) or self.x_min_m < 0.0:
            raise ValueError("x_min_m must be finite and non-negative")
        if not np.isfinite(self.x_cell_size_m) or self.x_cell_size_m <= 0.0:
            raise ValueError("x_cell_size_m must be finite and positive")


@dataclass(frozen=True)
class FaultBlob:
    """Summary of the complete severe-loss repair mask for diagnostics."""

    component_id: int
    cell_count: int
    fault_mass: float
    nearest_distance_m: float
    centroid_distance_m: float
    distance_bin: int
    bbox: tuple[int, int, int, int]
    halo_bbox: tuple[int, int, int, int] | None = None
    repair_fault_fraction: float = 0.0
    repair_target_met: bool = False
    healthy_occupied_cell_count: int = 0
    required_healthy_context_cell_count: int = 0
    halo_healthy_fraction: float = 0.0
    halo_target_met: bool = False
    halo_dilation_cells: int = 0


@dataclass(frozen=True)
class FaultSelection:
    reconstruction_mask: np.ndarray
    halo_mask: np.ndarray
    healthy_context_mask: np.ndarray
    selected_blobs: tuple[FaultBlob, ...]
    qualifying_blobs: tuple[FaultBlob, ...]
    rejected_small_blobs: tuple[FaultBlob, ...]
    original_fault_cell_count: int
    thresholded_cell_count: int
    excluded_added_only_cell_count: int
    selected_fault_cell_count: int

    @property
    def selected_cell_count(self) -> int:
        return int(self.reconstruction_mask.sum())

    @property
    def healthy_context_cell_count(self) -> int:
        return int(self.healthy_context_mask.sum())

    @property
    def halo_cell_count(self) -> int:
        return int(self.halo_mask.sum())


def _array(values, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {result.shape}")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return (0, 0, 0, 0)
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()) + 1,
        int(cols.max()) + 1,
    )


def _box_spatial_density(
    rectangle: tuple[int, int, int, int],
    fault_mask: np.ndarray,
) -> float:
    """Return fault-evidence cells divided by complete rectangle area."""

    top, left, bottom, right = rectangle
    area = (bottom - top) * (right - left)
    if area <= 0:
        return 0.0
    return float(fault_mask[top:bottom, left:right].sum()) / area


def _rectangle_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _rectangle_gap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    row_gap = max(first[0] - second[2], second[0] - first[2], 0)
    col_gap = max(first[1] - second[3], second[1] - first[3], 0)
    return max(row_gap, col_gap)


def _merge_touching_rectangles(
    rectangles: list[tuple[int, int, int, int]],
    candidate: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Insert a rectangle while unioning every overlap or shared boundary."""

    remaining = list(rectangles)
    while True:
        separated = []
        merged = False
        for rectangle in remaining:
            if _rectangle_gap(rectangle, candidate) == 0:
                candidate = _rectangle_union(rectangle, candidate)
                merged = True
            else:
                separated.append(rectangle)
        if not merged:
            return separated + [candidate]
        remaining = separated


def _expand_primary_box(
    rectangle: tuple[int, int, int, int],
    secondary_loss: np.ndarray,
    healthy_occupied: np.ndarray,
    *,
    maximum_gap_cells: int,
    minimum_component_cells: int,
    minimum_fault_fraction: float,
) -> tuple[int, int, int, int]:
    """Absorb nearby unreliable components without sacrificing box purity."""

    labels, count = label(
        secondary_loss,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    components = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        cell_count = int(component.sum())
        if cell_count >= minimum_component_cells:
            components.append((cell_count, _bbox(component)))

    expanded = rectangle
    remaining = components
    while remaining:
        eligible = [
            item
            for item in remaining
            if _rectangle_gap(expanded, item[1]) <= maximum_gap_cells
        ]
        if not eligible:
            break
        eligible.sort(key=lambda item: (_rectangle_gap(expanded, item[1]), -item[0]))
        accepted = False
        for _cell_count, component_box in eligible:
            proposal = _rectangle_union(expanded, component_box)
            top, left, bottom, right = proposal
            fault_count = int(secondary_loss[top:bottom, left:right].sum())
            healthy_count = int(healthy_occupied[top:bottom, left:right].sum())
            if fault_count / (fault_count + healthy_count) < minimum_fault_fraction:
                continue
            expanded = proposal
            remaining.remove((_cell_count, component_box))
            accepted = True
            break
        if not accepted:
            break
    return expanded


@lru_cache(maxsize=None)
def _row_intervals(height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return every non-empty ``[top, bottom)`` row interval."""
    tops, inclusive_bottoms = np.triu_indices(height)
    return tops, inclusive_bottoms + 1


def _best_repair_box(
    severe_loss: np.ndarray,
    healthy_occupied: np.ndarray,
    min_fault_cells: int,
    min_fault_fraction: float,
) -> tuple[int, int, int, int] | None:
    """Choose the best depth band, then remove empty lateral margins."""
    height = severe_loss.shape[0]
    fault_prefix = np.concatenate(
        ([0], np.cumsum(severe_loss.sum(axis=1), dtype=np.int64))
    )
    healthy_prefix = np.concatenate(
        ([0], np.cumsum(healthy_occupied.sum(axis=1), dtype=np.int64))
    )
    tops, bottoms = _row_intervals(height)
    fault_counts = fault_prefix[bottoms] - fault_prefix[tops]
    healthy_counts = healthy_prefix[bottoms] - healthy_prefix[tops]
    informative_counts = fault_counts + healthy_counts
    valid = fault_counts >= min_fault_cells
    fault_fractions = np.divide(
        fault_counts,
        informative_counts,
        out=np.zeros_like(fault_counts, dtype=np.float64),
        where=informative_counts > 0,
    )
    valid &= fault_fractions >= min_fault_fraction
    candidate_indices = np.flatnonzero(valid)
    if not len(candidate_indices):
        return None

    candidate_tops = tops[candidate_indices]
    candidate_bottoms = bottoms[candidate_indices]
    # This ascending lexicographic order is exactly equivalent to maximizing
    # (fault_count, -healthy_count, -height, -top) in the former Python loop.
    order = np.lexsort(
        (
            candidate_tops,
            candidate_bottoms - candidate_tops,
            healthy_counts[candidate_indices],
            -fault_counts[candidate_indices],
        )
    )
    selected = candidate_indices[int(order[0])]
    top = int(tops[selected])
    bottom = int(bottoms[selected])
    band_columns = np.flatnonzero(severe_loss[top:bottom].any(axis=0))
    return (
        top,
        int(band_columns[0]),
        bottom,
        int(band_columns[-1]) + 1,
    )


def _expand_rectangle_to_minimum_size(
    rectangle: tuple[int, int, int, int],
    shape: tuple[int, int],
    minimum_side_cells: int,
) -> tuple[int, int, int, int]:
    """Expand a rectangle around its center to the requested minimum size."""

    top, left, bottom, right = rectangle
    height, width = shape

    def expand_axis(start: int, stop: int, limit: int) -> tuple[int, int]:
        target = min(minimum_side_cells, limit)
        missing = max(target - (stop - start), 0)
        start -= missing // 2
        stop += missing - missing // 2
        if start < 0:
            stop = min(stop - start, limit)
            start = 0
        if stop > limit:
            start = max(start - (stop - limit), 0)
            stop = limit
        return start, stop

    top, bottom = expand_axis(top, bottom, height)
    left, right = expand_axis(left, right, width)
    return top, left, bottom, right


def _next_secondary_repair_box(
    severe_loss: np.ndarray,
    healthy_occupied: np.ndarray,
    primary_box: tuple[int, int, int, int] | None,
    *,
    minimum_fault_cells: int,
    minimum_fault_fraction: float,
    minimum_spatial_density: float,
    minimum_box_side_cells: int,
    merge_gap_cells: int,
) -> tuple[int, int, int, int] | None:
    """Select the strongest coherent remaining secondary repair region.

    Nearby residual cells are grouped before choosing the strongest eligible
    group. A secondary region is independent from the primary region: it is
    never stretched across the healthy/empty gap merely to touch the primary
    box. Its bounds cover the complete merged unreliable cluster.
    """

    remaining = severe_loss.copy()
    if primary_box is not None:
        top, left, bottom, right = primary_box
        remaining[top:bottom, left:right] = False
    grouping_mask = (
        binary_dilation(
            remaining,
            structure=np.ones((3, 3), dtype=bool),
            iterations=merge_gap_cells,
        )
        if merge_gap_cells > 0
        else remaining
    )
    connectivity = np.ones((3, 3), dtype=np.uint8)
    group_labels, group_count = label(grouping_mask, structure=connectivity)
    candidates = []

    def add_candidate(grouped_fault: np.ndarray) -> bool:
        fault_count = int(grouped_fault.sum())
        if fault_count < minimum_fault_cells:
            return False
        rectangle = _bbox(grouped_fault)
        top, left, bottom, right = rectangle
        healthy_count = int(healthy_occupied[top:bottom, left:right].sum())
        fault_fraction = fault_count / (fault_count + healthy_count)
        if fault_fraction < minimum_fault_fraction:
            return False
        area = (bottom - top) * (right - left)
        if fault_count / area < minimum_spatial_density:
            return False
        rectangle = _expand_rectangle_to_minimum_size(
            rectangle,
            severe_loss.shape,
            minimum_box_side_cells,
        )
        candidates.append((-fault_count, healthy_count, area, rectangle))
        return True

    for group_id in range(1, group_count + 1):
        grouped_fault = remaining & (group_labels == group_id)
        if add_candidate(grouped_fault):
            continue
        # A gap-merged group can still span too much empty area. In that case,
        # fall back to its tight original components instead of accepting one
        # hallucination-prone rectangle or discarding every local fault.
        component_labels, component_count = label(
            grouped_fault,
            structure=connectivity,
        )
        for component_id in range(1, component_count + 1):
            add_candidate(component_labels == component_id)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def _adaptive_halo(
    reconstruction_mask: np.ndarray,
    repair_fault_cell_count: int,
    trusted_occupied: np.ndarray,
    occupied: np.ndarray,
    valid_support: np.ndarray,
    config: FaultSelectorConfig,
) -> tuple[np.ndarray, int, int, float, bool]:
    required_healthy = max(
        config.min_halo_healthy_cells,
        math.ceil(
            repair_fault_cell_count * config.min_halo_context_ratio
        ),
    )
    best = None
    previous = None
    structure = np.ones((3, 3), dtype=bool)
    expanded = binary_dilation(
        reconstruction_mask,
        structure=structure,
        iterations=config.min_halo_width_cells,
    )
    for amount in range(
        config.min_halo_width_cells,
        config.max_halo_dilation_cells + 1,
    ):
        if amount > config.min_halo_width_cells:
            expanded = binary_dilation(expanded, structure=structure)
        halo = expanded & valid_support & ~reconstruction_mask
        if previous is not None and np.array_equal(halo, previous):
            break
        previous = halo
        healthy_count = int(np.logical_and(halo, trusted_occupied).sum())
        occupied_count = int(np.logical_and(halo, occupied).sum())
        healthy_fraction = (
            healthy_count / occupied_count if occupied_count else 0.0
        )
        target_met = (
            healthy_count >= required_healthy
            and healthy_fraction >= config.min_halo_healthy_fraction
        )
        candidate = (
            halo,
            amount,
            healthy_count,
            float(healthy_fraction),
            target_met,
        )
        if best is None or (
            target_met,
            min(healthy_count, required_healthy),
            healthy_fraction,
            -amount,
        ) > (
            best[4],
            min(best[2], required_healthy),
            best[3],
            -best[1],
        ):
            best = candidate
        if target_met:
            break
    assert best is not None
    return best


class FaultSelector:
    """Repair cells only when more than the configured LiDAR fraction was lost."""

    def __init__(self, config: FaultSelectorConfig | None = None):
        self.config = config or FaultSelectorConfig()
        self.config.validate()

    def select(
        self,
        fault_heatmap: np.ndarray,
        *,
        reliability_map: np.ndarray | None = None,
        faulty_counts: np.ndarray | None = None,
        added_faulty_counts: np.ndarray | None = None,
        missing_faulty_counts: np.ndarray | None = None,
        moved_faulty_counts: np.ndarray | None = None,
        valid_support_mask: np.ndarray | None = None,
    ) -> FaultSelection:
        heatmap = _array(fault_heatmap, "fault_heatmap")
        shape = heatmap.shape
        if faulty_counts is None or missing_faulty_counts is None:
            raise ValueError(
                "faulty_counts and missing_faulty_counts are required to "
                "measure per-cell LiDAR loss"
            )
        faulty = _array(faulty_counts, "faulty_counts", shape)
        missing = _array(
            missing_faulty_counts, "missing_faulty_counts", shape
        )
        added = (
            np.zeros(shape, dtype=np.float32)
            if added_faulty_counts is None
            else _array(added_faulty_counts, "added_faulty_counts", shape)
        )
        if reliability_map is not None:
            _array(reliability_map, "reliability_map", shape)
        if moved_faulty_counts is not None:
            _array(moved_faulty_counts, "moved_faulty_counts", shape)
        valid_support = (
            np.ones(shape, dtype=bool)
            if valid_support_mask is None
            else _array(valid_support_mask, "valid_support_mask", shape).astype(
                bool
            )
        )

        # Added synthetic returns do not count as retained LiDAR evidence.
        surviving = np.maximum(faulty - added, 0.0)
        original_count = surviving + missing
        loss_fraction = np.zeros(shape, dtype=np.float32)
        observed_original = original_count > 0
        loss_fraction[observed_original] = (
            missing[observed_original] / original_count[observed_original]
        )

        severe_loss = (
            (loss_fraction >= self.config.min_lidar_loss_fraction)
            & observed_original
            & valid_support
        )
        secondary_severe_loss = (
            (
                loss_fraction
                >= self.config.min_secondary_lidar_loss_fraction
            )
            & observed_original
            & valid_support
        )
        occupied = (surviving > 0) & valid_support
        healthy_for_box = occupied & ~severe_loss
        rectangle = _best_repair_box(
            severe_loss,
            healthy_for_box,
            self.config.min_repair_box_cells,
            self.config.min_repair_fault_fraction,
        )
        secondary_healthy_for_box = occupied & ~secondary_severe_loss
        if rectangle is not None and self.config.primary_expansion_gap_cells > 0:
            rectangle = _expand_primary_box(
                rectangle,
                secondary_severe_loss,
                secondary_healthy_for_box,
                maximum_gap_cells=self.config.primary_expansion_gap_cells,
                minimum_component_cells=self.config.min_secondary_repair_cells,
                minimum_fault_fraction=self.config.min_repair_fault_fraction,
            )
        remaining_secondary_loss = secondary_severe_loss.copy()
        if rectangle is not None:
            top, left, bottom, right = rectangle
            remaining_secondary_loss[top:bottom, left:right] = False
        secondary_rectangles = []
        while len(secondary_rectangles) < self.config.max_secondary_repair_boxes:
            secondary_rectangle = _next_secondary_repair_box(
                remaining_secondary_loss,
                secondary_healthy_for_box,
                rectangle,
                minimum_fault_cells=self.config.min_secondary_repair_cells,
                minimum_fault_fraction=(
                    self.config.min_secondary_repair_fault_fraction
                ),
                minimum_spatial_density=(
                    self.config.min_secondary_box_spatial_density
                ),
                minimum_box_side_cells=(
                    self.config.min_secondary_box_side_cells
                ),
                merge_gap_cells=self.config.secondary_merge_gap_cells,
            )
            if secondary_rectangle is None:
                break
            secondary_rectangles = _merge_touching_rectangles(
                secondary_rectangles,
                secondary_rectangle,
            )
            for top, left, bottom, right in secondary_rectangles:
                remaining_secondary_loss[top:bottom, left:right] = False
        reconstruction_mask = np.zeros(shape, dtype=bool)
        # Smaller boxes are listed first for diagnostics. The model
        # reconstructs the union of all selected boxes in one forward pass.
        rectangle_entries = [
            (secondary_rectangle, secondary_severe_loss)
            for secondary_rectangle in secondary_rectangles
        ]
        if rectangle is not None:
            rectangle_entries.append((rectangle, severe_loss))
        rectangles = [entry[0] for entry in rectangle_entries]
        for top, left, bottom, right in rectangles:
            reconstruction_mask[top:bottom, left:right] = True
        reconstruction_mask &= valid_support
        selector_threshold_mask = (
            secondary_severe_loss
            if self.config.max_secondary_repair_boxes > 0
            else severe_loss
        )
        selected_fault_evidence = (
            reconstruction_mask & selector_threshold_mask
        )
        trusted_occupied = occupied & ~reconstruction_mask
        original_fault = (missing > 0) & valid_support
        excluded_added_only = (
            (heatmap > 0) & (missing <= 0) & (added > 0) & valid_support
        )

        if not reconstruction_mask.any():
            empty = np.zeros(shape, dtype=bool)
            return FaultSelection(
                reconstruction_mask=empty.copy(),
                halo_mask=empty.copy(),
                healthy_context_mask=empty.copy(),
                selected_blobs=(),
                qualifying_blobs=(),
                rejected_small_blobs=(),
                original_fault_cell_count=int(original_fault.sum()),
                thresholded_cell_count=int(selector_threshold_mask.sum()),
                excluded_added_only_cell_count=int(excluded_added_only.sum()),
                selected_fault_cell_count=0,
            )

        halo_mask, dilation, healthy_count, healthy_fraction, target_met = (
            _adaptive_halo(
                reconstruction_mask,
                int(selected_fault_evidence.sum()),
                trusted_occupied,
                occupied,
                valid_support,
                self.config,
            )
        )
        healthy_context_mask = halo_mask & trusted_occupied
        height = shape[0]
        required_healthy = max(
            self.config.min_halo_healthy_cells,
            math.ceil(
                int(selected_fault_evidence.sum())
                * self.config.min_halo_context_ratio
            ),
        )
        halo_bbox = _bbox(reconstruction_mask | halo_mask)
        blobs = []
        for component_id, (rectangle, evidence_mask) in enumerate(
            rectangle_entries,
            start=1,
        ):
            top, left, bottom, right = rectangle
            rows = np.arange(top, bottom)
            nearest_distance = self.config.x_min_m + (
                height - bottom + 0.5
            ) * self.config.x_cell_size_m
            centroid_distance = self.config.x_min_m + (
                height - 1 - float(rows.mean()) + 0.5
            ) * self.config.x_cell_size_m
            rectangle_loss = loss_fraction[top:bottom, left:right]
            rectangle_fault = evidence_mask[top:bottom, left:right]
            rectangle_healthy = (
                occupied & ~evidence_mask
            )[top:bottom, left:right]
            fault_count = int(rectangle_fault.sum())
            healthy_in_box = int(rectangle_healthy.sum())
            blobs.append(
                FaultBlob(
                    component_id=component_id,
                    cell_count=fault_count,
                    fault_mass=float(rectangle_loss[rectangle_fault].sum()),
                    nearest_distance_m=float(nearest_distance),
                    centroid_distance_m=float(centroid_distance),
                    distance_bin=int(
                        nearest_distance // self.config.distance_bin_m
                    ),
                    bbox=rectangle,
                    halo_bbox=halo_bbox,
                    repair_fault_fraction=(
                        fault_count / (fault_count + healthy_in_box)
                    ),
                    repair_target_met=True,
                    healthy_occupied_cell_count=healthy_count,
                    required_healthy_context_cell_count=required_healthy,
                    halo_healthy_fraction=healthy_fraction,
                    halo_target_met=target_met,
                    halo_dilation_cells=dilation,
                )
            )
        return FaultSelection(
            reconstruction_mask=reconstruction_mask,
            halo_mask=halo_mask,
            healthy_context_mask=healthy_context_mask,
            selected_blobs=tuple(blobs),
            qualifying_blobs=tuple(blobs),
            rejected_small_blobs=(),
            original_fault_cell_count=int(original_fault.sum()),
            thresholded_cell_count=int(selector_threshold_mask.sum()),
            excluded_added_only_cell_count=int(excluded_added_only.sum()),
            selected_fault_cell_count=int(
                selected_fault_evidence.sum()
            ),
        )
