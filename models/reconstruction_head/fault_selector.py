"""Select coherent perfect-heatmap regions for ordered reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label


@dataclass(frozen=True)
class FaultSelectorConfig:
    """Tunable first-pass policy for selecting reconstruction regions."""

    threshold: float = 0.0
    min_blob_cells: int = 5
    max_blobs: int | None = 3
    connectivity: int = 8
    merge_radius_cells: int = 4
    box_padding_cells: int = 2
    combine_gap_cells: int = 40
    min_relative_blob_size: float = 0.05
    bbox_quantile: float = 0.01
    min_repair_fault_fraction: float = 0.75
    min_halo_healthy_fraction: float = 0.90
    min_halo_healthy_cells: int = 64
    min_halo_context_ratio: float = 0.25
    min_halo_width_cells: int = 4
    healthy_reliability_threshold: float = 1.0
    max_halo_dilation_cells: int | None = None
    distance_bin_m: float = 10.0
    x_min_m: float = 0.0
    x_cell_size_m: float = 0.2
    ignore_added_only: bool = True

    def validate(self) -> None:
        for name in (
            "threshold",
            "min_relative_blob_size",
            "min_repair_fault_fraction",
            "min_halo_healthy_fraction",
            "healthy_reliability_threshold",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        for name in (
            "merge_radius_cells",
            "box_padding_cells",
            "combine_gap_cells",
            "min_halo_healthy_cells",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("min_blob_cells", "min_halo_width_cells"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_blobs is not None and self.max_blobs < 1:
            raise ValueError("max_blobs must be positive or None")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        if not 0.0 <= self.bbox_quantile < 0.5:
            raise ValueError("bbox_quantile must be in [0,0.5)")
        if (
            not np.isfinite(self.min_halo_context_ratio)
            or self.min_halo_context_ratio < 0.0
        ):
            raise ValueError("min_halo_context_ratio must be finite and non-negative")
        if (
            self.max_halo_dilation_cells is not None
            and self.max_halo_dilation_cells < self.min_halo_width_cells
        ):
            raise ValueError(
                "max_halo_dilation_cells must be at least min_halo_width_cells or None"
            )
        if not np.isfinite(self.distance_bin_m) or self.distance_bin_m <= 0.0:
            raise ValueError("distance_bin_m must be finite and positive")
        if not np.isfinite(self.x_min_m) or self.x_min_m < 0.0:
            raise ValueError("x_min_m must be finite and non-negative")
        if not np.isfinite(self.x_cell_size_m) or self.x_cell_size_m <= 0.0:
            raise ValueError("x_cell_size_m must be finite and positive")


@dataclass(frozen=True)
class FaultBlob:
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
    """Selected reconstruction mask and ordered qualifying blob descriptions."""

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


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    amount: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    top, left, bottom, right = bbox
    height, width = shape
    return (
        max(0, top - amount),
        max(0, left - amount),
        min(height, bottom + amount),
        min(width, right + amount),
    )


def _repair_counts(
    bbox: tuple[int, int, int, int],
    fault: np.ndarray,
    healthy_occupied: np.ndarray,
) -> tuple[int, int, float]:
    top, left, bottom, right = bbox
    fault_count = int(fault[top:bottom, left:right].sum())
    healthy_count = int(healthy_occupied[top:bottom, left:right].sum())
    informative_count = fault_count + healthy_count
    fraction = fault_count / informative_count if informative_count else 0.0
    return fault_count, healthy_count, float(fraction)


def _fit_repair_bbox(
    bbox: tuple[int, int, int, int],
    fault: np.ndarray,
    healthy_occupied: np.ndarray,
    target_fraction: float,
) -> tuple[tuple[int, int, int, int], int, int, float, bool]:
    """Trim low-value edges until the repair region reaches its fault target."""
    current = bbox
    fault_count, healthy_count, fraction = _repair_counts(
        current, fault, healthy_occupied
    )
    best = (current, fault_count, healthy_count, fraction, False)
    if fraction >= target_fraction:
        return current, fault_count, healthy_count, fraction, True

    while True:
        top, left, bottom, right = current
        candidates = []
        if bottom - top > 1:
            candidates.extend(
                ((top + 1, left, bottom, right), (top, left, bottom - 1, right))
            )
        if right - left > 1:
            candidates.extend(
                ((top, left + 1, bottom, right), (top, left, bottom, right - 1))
            )
        scored = []
        for candidate in candidates:
            candidate_faults, candidate_healthy, candidate_fraction = _repair_counts(
                candidate, fault, healthy_occupied
            )
            if candidate_faults == 0:
                continue
            scored.append(
                (
                    candidate_fraction,
                    candidate_faults,
                    -candidate_healthy,
                    candidate,
                )
            )
        if not scored:
            break
        candidate_fraction, candidate_faults, neg_healthy, current = max(scored)
        candidate_healthy = -neg_healthy
        if candidate_fraction > best[3]:
            best = (
                current,
                candidate_faults,
                candidate_healthy,
                candidate_fraction,
                False,
            )
        if candidate_fraction >= target_fraction:
            return (
                current,
                candidate_faults,
                candidate_healthy,
                candidate_fraction,
                True,
            )
    return best


def _halo_counts(
    halo_bbox: tuple[int, int, int, int],
    repair_bbox: tuple[int, int, int, int],
    healthy_occupied: np.ndarray,
    occupied: np.ndarray,
) -> tuple[int, int, float]:
    top, left, bottom, right = halo_bbox
    healthy_count = int(healthy_occupied[top:bottom, left:right].sum())
    occupied_count = int(occupied[top:bottom, left:right].sum())
    top, left, bottom, right = repair_bbox
    healthy_count -= int(healthy_occupied[top:bottom, left:right].sum())
    occupied_count -= int(occupied[top:bottom, left:right].sum())
    fraction = healthy_count / occupied_count if occupied_count else 0.0
    return healthy_count, occupied_count, float(fraction)


def _dilate_halo_bbox(
    repair_bbox: tuple[int, int, int, int],
    healthy_occupied: np.ndarray,
    occupied: np.ndarray,
    target_fraction: float,
    target_healthy_cells: int,
    min_width: int,
    max_dilation: int | None,
) -> tuple[tuple[int, int, int, int], int, int, float, bool, int]:
    """Find the smallest halo meeting both healthy fraction and cell-count targets."""
    shape = occupied.shape
    limit = max(shape) if max_dilation is None else max_dilation
    best = None
    previous_bbox = None
    for amount in range(min_width, limit + 1):
        candidate = _expand_bbox(repair_bbox, amount, shape)
        if candidate == previous_bbox:
            break
        previous_bbox = candidate
        healthy_count, occupied_count, fraction = _halo_counts(
            candidate, repair_bbox, healthy_occupied, occupied
        )
        candidate_score = (
            fraction >= target_fraction,
            min(healthy_count, target_healthy_cells),
            fraction,
        )
        best_score = None if best is None else (
            best[3] >= target_fraction,
            min(best[1], target_healthy_cells),
            best[3],
        )
        if best is None or candidate_score > best_score:
            best = (
                candidate,
                healthy_count,
                occupied_count,
                fraction,
                False,
                amount,
            )
        if fraction >= target_fraction and healthy_count >= target_healthy_cells:
            return (
                candidate,
                healthy_count,
                occupied_count,
                fraction,
                True,
                amount,
            )
    assert best is not None
    return best

def _connected_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    structure = generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, count = label(mask, structure=structure)
    return [np.argwhere(labels == index) for index in range(1, count + 1)]


def _dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Join evidence separated by at most a small spatial gap."""
    if radius == 0:
        return mask.copy()
    return binary_dilation(
        mask,
        structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool),
    )


def _robust_bbox(
    cells: np.ndarray,
    shape: tuple[int, int],
    padding: int,
    quantile: float,
) -> tuple[int, int, int, int]:
    """Return a clipped rectangle without letting sparse tails inflate it."""
    rows, cols = cells[:, 0], cells[:, 1]
    height, width = shape
    top = max(int(np.floor(np.quantile(rows, quantile))) - padding, 0)
    left = max(int(np.floor(np.quantile(cols, quantile))) - padding, 0)
    bottom = min(
        int(np.ceil(np.quantile(rows, 1.0 - quantile))) + 1 + padding,
        height,
    )
    right = min(
        int(np.ceil(np.quantile(cols, 1.0 - quantile))) + 1 + padding,
        width,
    )
    return top, left, bottom, right


def _raw_bbox(cells: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = cells[:, 0], cells[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()) + 1,
        int(cols.max()) + 1,
    )


def _boxes_are_close(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int,
) -> bool:
    first_top, first_left, first_bottom, first_right = first
    second_top, second_left, second_bottom, second_right = second
    row_gap = max(first_top - second_bottom, second_top - first_bottom, 0)
    col_gap = max(first_left - second_right, second_left - first_right, 0)
    return row_gap <= gap and col_gap <= gap


def _cluster_cells_by_bbox_gap(
    components: list[np.ndarray], gap: int
) -> list[np.ndarray]:
    """Transitively cluster substantial components whose boxes are nearby."""
    remaining = set(range(len(components)))
    boxes = [_raw_bbox(cells) for cells in components]
    clusters = []
    while remaining:
        seed = remaining.pop()
        members = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbors = {
                candidate
                for candidate in remaining
                if _boxes_are_close(boxes[current], boxes[candidate], gap)
            }
            remaining.difference_update(neighbors)
            members.update(neighbors)
            frontier.extend(neighbors)
        clusters.append(np.concatenate([components[index] for index in members]))
    return clusters


class FaultSelector:
    """Build fault-dense repair regions with enclosing healthy context halos."""

    def __init__(self, config: FaultSelectorConfig | None = None):
        self.config = config or FaultSelectorConfig()

    def _describe_blob(
        self,
        component_id: int,
        cells: np.ndarray,
        heatmap: np.ndarray,
    ) -> FaultBlob:
        rows, cols = cells[:, 0], cells[:, 1]
        values = heatmap[rows, cols]
        closest_row = int(rows.max())
        centroid_row = float(rows.mean())
        height = heatmap.shape[0]
        nearest_distance = self.config.x_min_m + (
            height - 1 - closest_row + 0.5
        ) * self.config.x_cell_size_m
        centroid_distance = self.config.x_min_m + (
            height - 1 - centroid_row + 0.5
        ) * self.config.x_cell_size_m
        return FaultBlob(
            component_id=component_id,
            cell_count=len(cells),
            fault_mass=float(values.sum()),
            nearest_distance_m=float(nearest_distance),
            centroid_distance_m=float(centroid_distance),
            distance_bin=int(math.floor(nearest_distance / self.config.distance_bin_m)),
            bbox=_robust_bbox(
                cells,
                heatmap.shape,
                self.config.box_padding_cells,
                self.config.bbox_quantile,
            ),
        )

    @staticmethod
    def _priority(blob: FaultBlob) -> tuple[float, ...]:
        return (
            -float(blob.cell_count),
            -blob.fault_mass,
            float(blob.distance_bin),
            blob.nearest_distance_m,
            float(blob.component_id),
        )

    @staticmethod
    def _evidence_array(array) -> np.ndarray:
        values = np.asarray(array)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        return values

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
        heatmap = np.asarray(fault_heatmap, dtype=np.float32)
        if heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]

        if valid_support_mask is None:
            valid_support = np.ones_like(heatmap, dtype=bool)
        else:
            valid_support = self._evidence_array(valid_support_mask).astype(bool)
            if valid_support.shape != heatmap.shape:
                raise ValueError(
                    "valid_support_mask and fault_heatmap must have the same "
                    f"shape, got {valid_support.shape} and {heatmap.shape}"
                )

        healthy_occupied = np.zeros_like(heatmap, dtype=bool)
        occupied = np.zeros_like(heatmap, dtype=bool)
        if reliability_map is not None or faulty_counts is not None:
            if reliability_map is None or faulty_counts is None:
                raise ValueError(
                    "reliability_map and faulty_counts must be provided together"
                )
            reliability = self._evidence_array(reliability_map)
            faulty_occupancy_counts = self._evidence_array(faulty_counts)
            occupied = (faulty_occupancy_counts > 0) & valid_support
            healthy_occupied = occupied & (
                reliability >= self.config.healthy_reliability_threshold
            )
        elif (
            self.config.min_repair_fault_fraction > 0.0
            or self.config.min_halo_healthy_fraction > 0.0
            or self.config.min_halo_healthy_cells > 0
            or self.config.min_halo_context_ratio > 0.0
        ):
            raise ValueError(
                "repair/halo fraction targets require reliability_map and faulty_counts"
            )

        original_thresholded = heatmap > self.config.threshold
        excluded_added_only = np.zeros_like(original_thresholded, dtype=bool)
        if self.config.ignore_added_only:
            evidence = {
                "added_faulty_counts": added_faulty_counts,
                "missing_faulty_counts": missing_faulty_counts,
                "moved_faulty_counts": moved_faulty_counts,
            }
            missing_names = [name for name, value in evidence.items() if value is None]
            if missing_names:
                raise ValueError(
                    "ignore_added_only=True requires evidence arrays: "
                    + ", ".join(missing_names)
                )
            added = self._evidence_array(added_faulty_counts)
            missing = self._evidence_array(missing_faulty_counts)
            moved = self._evidence_array(moved_faulty_counts)
            excluded_added_only = (added > 0) & (missing <= 0) & (moved <= 0)
        thresholded = original_thresholded & ~excluded_added_only & valid_support
        grouping_mask = _dilate_square(
            thresholded, self.config.merge_radius_cells
        ) & valid_support
        grouped_regions = _connected_components(grouping_mask, self.config.connectivity)
        component_cells = []
        for region in grouped_regions:
            rows, cols = region[:, 0], region[:, 1]
            evidence_cells = region[thresholded[rows, cols]]
            if len(evidence_cells):
                component_cells.append(evidence_cells)
        component_descriptions = [
            self._describe_blob(index + 1, cells, heatmap)
            for index, cells in enumerate(component_cells)
        ]
        qualifying_components = sorted(
            (
                blob
                for blob in component_descriptions
                if blob.cell_count >= self.config.min_blob_cells
            ),
            key=self._priority,
        )
        rejected = sorted(
            (
                blob
                for blob in component_descriptions
                if blob.cell_count < self.config.min_blob_cells
            ),
            key=self._priority,
        )
        significant_cells = []
        if qualifying_components:
            largest_component = qualifying_components[0].cell_count
            relative_cutoff = largest_component * self.config.min_relative_blob_size
            significant_ids = {
                blob.component_id
                for blob in qualifying_components
                if blob.cell_count >= relative_cutoff
            }
            significant_cells = [
                cells
                for index, cells in enumerate(component_cells, start=1)
                if index in significant_ids
            ]
        clustered_cells = _cluster_cells_by_bbox_gap(
            significant_cells, self.config.combine_gap_cells
        ) if significant_cells else []
        clusters = sorted(
            (
                (self._describe_blob(index + 1, cells, heatmap), cells)
                for index, cells in enumerate(clustered_cells)
            ),
            key=lambda pair: self._priority(pair[0]),
        )
        qualifying = clusters
        if clusters:
            largest_cluster = clusters[0][0].cell_count
            cluster_cutoff = largest_cluster * self.config.min_relative_blob_size
            qualifying = [
                pair
                for pair in clusters
                if pair[0].cell_count >= cluster_cutoff
            ]
        qualifying_blobs = [blob for blob, _cells in qualifying]
        reconstruction_mask = np.zeros_like(thresholded, dtype=bool)
        halo_mask = np.zeros_like(thresholded, dtype=bool)
        selected = []
        for blob, cells in qualifying:
            if reconstruction_mask[cells[:, 0], cells[:, 1]].all():
                continue
            (
                repair_bbox,
                repair_fault_count,
                _repair_healthy_count,
                repair_fault_fraction,
                repair_target_met,
            ) = _fit_repair_bbox(
                blob.bbox,
                thresholded,
                healthy_occupied,
                self.config.min_repair_fault_fraction,
            )
            required_healthy_cells = max(
                self.config.min_halo_healthy_cells,
                math.ceil(repair_fault_count * self.config.min_halo_context_ratio),
            )
            (
                halo_bbox,
                healthy_count,
                _halo_occupied_count,
                halo_healthy_fraction,
                halo_target_met,
                halo_dilation_cells,
            ) = _dilate_halo_bbox(
                repair_bbox,
                healthy_occupied,
                occupied,
                self.config.min_halo_healthy_fraction,
                required_healthy_cells,
                self.config.min_halo_width_cells,
                self.config.max_halo_dilation_cells,
            )
            selected.append(
                replace(
                    blob,
                    bbox=repair_bbox,
                    halo_bbox=halo_bbox,
                    repair_fault_fraction=repair_fault_fraction,
                    repair_target_met=repair_target_met,
                    healthy_occupied_cell_count=healthy_count,
                    required_healthy_context_cell_count=required_healthy_cells,
                    halo_healthy_fraction=halo_healthy_fraction,
                    halo_target_met=halo_target_met,
                    halo_dilation_cells=halo_dilation_cells,
                )
            )
            top, left, bottom, right = repair_bbox
            reconstruction_mask[top:bottom, left:right] = True
            top, left, bottom, right = halo_bbox
            halo_mask[top:bottom, left:right] = True
            if self.config.max_blobs is not None and len(selected) >= self.config.max_blobs:
                break
        halo_mask &= ~reconstruction_mask
        reconstruction_mask &= valid_support
        halo_mask &= valid_support
        healthy_context_mask = halo_mask & healthy_occupied
        return FaultSelection(
            reconstruction_mask=reconstruction_mask,
            halo_mask=halo_mask,
            healthy_context_mask=healthy_context_mask,
            selected_blobs=tuple(selected),
            qualifying_blobs=tuple(qualifying_blobs),
            rejected_small_blobs=tuple(rejected),
            original_fault_cell_count=int(original_thresholded.sum()),
            thresholded_cell_count=int(thresholded.sum()),
            excluded_added_only_cell_count=int(
                np.logical_and(original_thresholded, excluded_added_only).sum()
            ),
            selected_fault_cell_count=int(
                np.logical_and(reconstruction_mask, thresholded).sum()
            ),
        )
