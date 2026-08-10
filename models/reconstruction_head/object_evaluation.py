"""Object-level geometry for evaluating masked BEV reconstruction regions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from Fault_Localization_Model.kradar_dataset import KRadarObjectAnnotation


@dataclass(frozen=True)
class BEVGeometry:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    height: int
    width: int

    @classmethod
    def from_metadata(
        cls,
        metadata: dict,
        mask_shape: tuple[int, int],
    ) -> "BEVGeometry":
        height, width = (int(value) for value in mask_shape)
        x_range = tuple(float(value) for value in metadata["x_range"])
        y_range = tuple(float(value) for value in metadata["y_range"])
        if (
            len(x_range) != 2
            or len(y_range) != 2
            or x_range[0] >= x_range[1]
            or y_range[0] >= y_range[1]
            or height < 1
            or width < 1
        ):
            raise ValueError("Invalid BEV geometry")
        x_resolution = (x_range[1] - x_range[0]) / height
        y_resolution = (y_range[1] - y_range[0]) / width
        metadata_resolution = float(metadata["resolution"])
        if not (
            math.isclose(x_resolution, metadata_resolution, abs_tol=1e-6)
            and math.isclose(y_resolution, metadata_resolution, abs_tol=1e-6)
        ):
            raise ValueError(
                "Mask shape does not agree with the sample BEV ranges and resolution"
            )
        return cls(x_range, y_range, height, width)

    @property
    def x_resolution(self) -> float:
        return (self.x_range[1] - self.x_range[0]) / self.height

    @property
    def y_resolution(self) -> float:
        return (self.y_range[1] - self.y_range[0]) / self.width

    def cell_bounds(self, row: int, column: int) -> tuple[float, ...]:
        x_high = self.x_range[1] - row * self.x_resolution
        x_low = x_high - self.x_resolution
        y_low = self.y_range[0] + column * self.y_resolution
        y_high = y_low + self.y_resolution
        return x_low, x_high, y_low, y_high


@dataclass(frozen=True)
class ObjectMaskOverlap:
    annotation: KRadarObjectAnnotation
    any_overlap: bool
    affected_fraction: float
    overlap_area_m2: float


def oriented_box_corners_xy(
    annotation: KRadarObjectAnnotation,
) -> np.ndarray:
    """Return the four corners of a K-Radar box footprint in LiDAR XY."""

    local = np.asarray(
        (
            (-annotation.length / 2.0, -annotation.width / 2.0),
            (annotation.length / 2.0, -annotation.width / 2.0),
            (annotation.length / 2.0, annotation.width / 2.0),
            (-annotation.length / 2.0, annotation.width / 2.0),
        ),
        dtype=np.float64,
    )
    cosine = math.cos(annotation.yaw)
    sine = math.sin(annotation.yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    return local @ rotation.T + np.asarray((annotation.x, annotation.y))


def _clip_axis(
    polygon: np.ndarray,
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        if keep_greater:
            return bool(point[axis] >= boundary)
        return bool(point[axis] <= boundary)

    output = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            if delta != 0.0:
                amount = (boundary - previous[axis]) / delta
                output.append(previous + amount * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _clip_to_cell(
    polygon: np.ndarray,
    bounds: tuple[float, ...],
) -> np.ndarray:
    x_low, x_high, y_low, y_high = bounds
    clipped = _clip_axis(
        polygon, axis=0, boundary=x_low, keep_greater=True
    )
    clipped = _clip_axis(
        clipped, axis=0, boundary=x_high, keep_greater=False
    )
    clipped = _clip_axis(
        clipped, axis=1, boundary=y_low, keep_greater=True
    )
    return _clip_axis(
        clipped, axis=1, boundary=y_high, keep_greater=False
    )


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


def object_mask_overlap(
    annotation: KRadarObjectAnnotation,
    reconstruction_mask: np.ndarray,
    geometry: BEVGeometry,
) -> ObjectMaskOverlap:
    """Measure exact area overlap between one footprint and masked cells."""

    mask = np.asarray(reconstruction_mask, dtype=bool)
    if mask.shape != (geometry.height, geometry.width):
        raise ValueError(
            f"Mask shape {mask.shape} does not match geometry "
            f"{(geometry.height, geometry.width)}"
        )
    polygon = oriented_box_corners_xy(annotation)
    polygon_min = polygon.min(axis=0)
    polygon_max = polygon.max(axis=0)
    row_start = max(
        0,
        int(math.floor((geometry.x_range[1] - polygon_max[0]) / geometry.x_resolution)),
    )
    row_stop = min(
        geometry.height,
        int(math.ceil((geometry.x_range[1] - polygon_min[0]) / geometry.x_resolution)),
    )
    column_start = max(
        0,
        int(math.floor((polygon_min[1] - geometry.y_range[0]) / geometry.y_resolution)),
    )
    column_stop = min(
        geometry.width,
        int(math.ceil((polygon_max[1] - geometry.y_range[0]) / geometry.y_resolution)),
    )
    overlap_area = 0.0
    if row_start >= row_stop or column_start >= column_stop:
        candidate_cells = np.empty((0, 2), dtype=np.int64)
    else:
        candidate_cells = np.argwhere(
            mask[row_start:row_stop, column_start:column_stop]
        )
        candidate_cells += np.asarray((row_start, column_start))
    for row, column in candidate_cells:
        bounds = geometry.cell_bounds(int(row), int(column))
        x_low, x_high, y_low, y_high = bounds
        if (
            x_high <= polygon_min[0]
            or x_low >= polygon_max[0]
            or y_high <= polygon_min[1]
            or y_low >= polygon_max[1]
        ):
            continue
        overlap_area += _polygon_area(_clip_to_cell(polygon, bounds))
    footprint_area = annotation.length * annotation.width
    affected_fraction = min(1.0, max(0.0, overlap_area / footprint_area))
    return ObjectMaskOverlap(
        annotation=annotation,
        any_overlap=overlap_area > 1e-12,
        affected_fraction=affected_fraction,
        overlap_area_m2=overlap_area,
    )


def summarize_object_overlaps(
    records: Iterable[dict],
    *,
    fraction_bins: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    records = list(records)
    if len(fraction_bins) < 2 or fraction_bins[0] != 0.0 or fraction_bins[-1] != 1.0:
        raise ValueError("affected-fraction bins must begin at 0 and end at 1")
    if any(left >= right for left, right in zip(fraction_bins, fraction_bins[1:])):
        raise ValueError("affected-fraction bins must be strictly increasing")

    def counts(items: list[dict]) -> dict:
        affected = sum(bool(item["any_overlap"]) for item in items)
        return {
            "objects": len(items),
            "affected_objects": affected,
            "unaffected_objects": len(items) - affected,
            "mean_affected_fraction": (
                float(np.mean([item["affected_fraction"] for item in items]))
                if items
                else 0.0
            ),
        }

    classes = sorted({str(record["class"]) for record in records})
    by_class = {
        class_name: counts(
            [record for record in records if record["class"] == class_name]
        )
        for class_name in classes
    }
    by_fraction = {"unaffected": counts([r for r in records if not r["any_overlap"]])}
    for lower, upper in zip(fraction_bins, fraction_bins[1:]):
        label = f"({lower:g},{upper:g}]"
        selected = [
            r for r in records if lower < r["affected_fraction"] <= upper
        ]
        by_fraction[label] = counts(selected)
    return {
        "overall": counts(records),
        "by_class": by_class,
        "by_affected_fraction": by_fraction,
    }
