"""Select coherent perfect-heatmap regions for ordered reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


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
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in [0,1]")
        if self.min_blob_cells < 1:
            raise ValueError("min_blob_cells must be positive")
        if self.max_blobs is not None and self.max_blobs < 1:
            raise ValueError("max_blobs must be positive or None")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        if self.merge_radius_cells < 0:
            raise ValueError("merge_radius_cells must be non-negative")
        if self.box_padding_cells < 0:
            raise ValueError("box_padding_cells must be non-negative")
        if self.combine_gap_cells < 0:
            raise ValueError("combine_gap_cells must be non-negative")
        if not 0.0 <= self.min_relative_blob_size <= 1.0:
            raise ValueError("min_relative_blob_size must be in [0,1]")
        if not 0.0 <= self.bbox_quantile < 0.5:
            raise ValueError("bbox_quantile must be in [0,0.5)")
        if not 0.0 <= self.min_repair_fault_fraction <= 1.0:
            raise ValueError("min_repair_fault_fraction must be in [0,1]")
        if not 0.0 <= self.min_halo_healthy_fraction <= 1.0:
            raise ValueError("min_halo_healthy_fraction must be in [0,1]")
        if self.min_halo_healthy_cells < 0:
            raise ValueError("min_halo_healthy_cells must be non-negative")
        if (
            not np.isfinite(self.min_halo_context_ratio)
            or self.min_halo_context_ratio < 0.0
        ):
            raise ValueError("min_halo_context_ratio must be finite and non-negative")
        if self.min_halo_width_cells < 1:
            raise ValueError("min_halo_width_cells must be positive")
        if not 0.0 <= self.healthy_reliability_threshold <= 1.0:
            raise ValueError("healthy_reliability_threshold must be in [0,1]")
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
    mean_unreliability: float
    peak_unreliability: float
    nearest_distance_m: float
    centroid_distance_m: float
    distance_bin: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    halo_bbox: tuple[int, int, int, int] | None = None
    repair_fault_cell_count: int = 0
    repair_healthy_cell_count: int = 0
    repair_fault_fraction: float = 0.0
    repair_target_met: bool = False
    healthy_occupied_cell_count: int = 0
    required_healthy_context_cell_count: int = 0
    halo_occupied_cell_count: int = 0
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
    halo_region = np.zeros_like(occupied, dtype=bool)
    top, left, bottom, right = halo_bbox
    halo_region[top:bottom, left:right] = True
    top, left, bottom, right = repair_bbox
    halo_region[top:bottom, left:right] = False
    healthy_count = int(np.logical_and(halo_region, healthy_occupied).sum())
    occupied_count = int(np.logical_and(halo_region, occupied).sum())
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

def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    return tuple(
        (row_delta, col_delta)
        for row_delta in (-1, 0, 1)
        for col_delta in (-1, 0, 1)
        if row_delta != 0 or col_delta != 0
    )


def _connected_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    offsets = _neighbor_offsets(connectivity)
    components = []
    for start_row, start_col in np.argwhere(mask):
        start_row, start_col = int(start_row), int(start_col)
        if visited[start_row, start_col]:
            continue
        visited[start_row, start_col] = True
        stack = [(start_row, start_col)]
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for row_delta, col_delta in offsets:
                neighbor_row = row + row_delta
                neighbor_col = col + col_delta
                if not (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_col < width
                    and mask[neighbor_row, neighbor_col]
                    and not visited[neighbor_row, neighbor_col]
                ):
                    continue
                visited[neighbor_row, neighbor_col] = True
                stack.append((neighbor_row, neighbor_col))
        components.append(np.asarray(cells, dtype=np.int32))
    return components


def _dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Join evidence separated by at most a small spatial gap."""
    if radius == 0:
        return mask.copy()
    window = 2 * radius + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    integral = np.pad(
        padded.astype(np.int32), ((1, 0), (1, 0)), mode="constant"
    ).cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    counts = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return counts > 0


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
        self.config.validate()

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
            mean_unreliability=float(values.mean()),
            peak_unreliability=float(values.max()),
            nearest_distance_m=float(nearest_distance),
            centroid_distance_m=float(centroid_distance),
            distance_bin=int(math.floor(nearest_distance / self.config.distance_bin_m)),
            bbox=_robust_bbox(
                cells,
                heatmap.shape,
                self.config.box_padding_cells,
                self.config.bbox_quantile,
            ),
            centroid=(centroid_row, float(cols.mean())),
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
    def _evidence_array(array, name: str, shape: tuple[int, int]) -> np.ndarray:
        values = np.asarray(array)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} must contain finite non-negative values")
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
    ) -> FaultSelection:
        heatmap = np.asarray(fault_heatmap, dtype=np.float32)
        if heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        if heatmap.ndim != 2 or min(heatmap.shape) < 1:
            raise ValueError(
                f"fault_heatmap must have shape [H,W] or [1,H,W], got {heatmap.shape}"
            )
        if not np.isfinite(heatmap).all() or np.any((heatmap < 0.0) | (heatmap > 1.0)):
            raise ValueError("fault_heatmap must contain finite values in [0,1]")

        healthy_occupied = np.zeros_like(heatmap, dtype=bool)
        occupied = np.zeros_like(heatmap, dtype=bool)
        if reliability_map is not None or faulty_counts is not None:
            if reliability_map is None or faulty_counts is None:
                raise ValueError(
                    "reliability_map and faulty_counts must be provided together"
                )
            reliability = self._evidence_array(
                reliability_map, "reliability_map", heatmap.shape
            )
            if np.any(reliability > 1.0):
                raise ValueError("reliability_map must contain values in [0,1]")
            faulty_occupancy_counts = self._evidence_array(
                faulty_counts, "faulty_counts", heatmap.shape
            )
            occupied = faulty_occupancy_counts > 0
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
            added = self._evidence_array(
                added_faulty_counts, "added_faulty_counts", heatmap.shape
            )
            missing = self._evidence_array(
                missing_faulty_counts, "missing_faulty_counts", heatmap.shape
            )
            moved = self._evidence_array(
                moved_faulty_counts, "moved_faulty_counts", heatmap.shape
            )
            excluded_added_only = (added > 0) & (missing <= 0) & (moved <= 0)
        thresholded = original_thresholded & ~excluded_added_only
        grouping_mask = _dilate_square(
            thresholded, self.config.merge_radius_cells
        )
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
        cluster_descriptions = sorted(
            (
                self._describe_blob(index + 1, cells, heatmap)
                for index, cells in enumerate(clustered_cells)
            ),
            key=self._priority,
        )
        qualifying = cluster_descriptions
        if cluster_descriptions:
            largest_cluster = cluster_descriptions[0].cell_count
            cluster_cutoff = largest_cluster * self.config.min_relative_blob_size
            qualifying = [
                blob
                for blob in cluster_descriptions
                if blob.cell_count >= cluster_cutoff
            ]
        cells_by_id = {
            index + 1: cells for index, cells in enumerate(clustered_cells)
        }
        reconstruction_mask = np.zeros_like(thresholded, dtype=bool)
        halo_mask = np.zeros_like(thresholded, dtype=bool)
        healthy_context_mask = np.zeros_like(thresholded, dtype=bool)
        selected = []
        for blob in qualifying:
            cells = cells_by_id[blob.component_id]
            if reconstruction_mask[cells[:, 0], cells[:, 1]].all():
                continue
            (
                repair_bbox,
                repair_fault_count,
                repair_healthy_count,
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
                math.ceil(
                    repair_fault_count * self.config.min_halo_context_ratio
                ),
            )
            (
                halo_bbox,
                healthy_count,
                halo_occupied_count,
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
            selected_blob = replace(
                blob,
                bbox=repair_bbox,
                halo_bbox=halo_bbox,
                repair_fault_cell_count=repair_fault_count,
                repair_healthy_cell_count=repair_healthy_count,
                repair_fault_fraction=repair_fault_fraction,
                repair_target_met=repair_target_met,
                healthy_occupied_cell_count=healthy_count,
                required_healthy_context_cell_count=required_healthy_cells,
                halo_occupied_cell_count=halo_occupied_count,
                halo_healthy_fraction=halo_healthy_fraction,
                halo_target_met=halo_target_met,
                halo_dilation_cells=halo_dilation_cells,
            )
            selected.append(selected_blob)
            top, left, bottom, right = repair_bbox
            reconstruction_mask[top:bottom, left:right] = True
            top, left, bottom, right = halo_bbox
            halo_mask[top:bottom, left:right] = True
            top, left, bottom, right = repair_bbox
            halo_mask[top:bottom, left:right] = False
            healthy_context_mask |= np.logical_and(halo_mask, healthy_occupied)
            healthy_context_mask[top:bottom, left:right] = False
            if (
                self.config.max_blobs is not None
                and len(selected) >= self.config.max_blobs
            ):
                break
        halo_mask &= ~reconstruction_mask
        healthy_context_mask &= halo_mask
        return FaultSelection(
            reconstruction_mask=reconstruction_mask,
            halo_mask=halo_mask,
            healthy_context_mask=healthy_context_mask,
            selected_blobs=tuple(selected),
            qualifying_blobs=tuple(qualifying),
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
