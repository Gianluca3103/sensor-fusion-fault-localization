"""Rotated BEV geometry shared by detector targets, matching, and plots."""

from __future__ import annotations

import numpy as np

from .annotations import RotatedBEVBox


def box_corners(box: RotatedBEVBox) -> np.ndarray:
    local = np.asarray(
        [
            [box.length / 2.0, box.width / 2.0],
            [-box.length / 2.0, box.width / 2.0],
            [-box.length / 2.0, -box.width / 2.0],
            [box.length / 2.0, -box.width / 2.0],
        ],
        dtype=np.float64,
    )
    cosine, sine = np.cos(box.yaw), np.sin(box.yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray([box.x, box.y])


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _inside(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    return _cross(end - start, point - start) >= -1.0e-9


def _intersection(
    first: np.ndarray,
    second: np.ndarray,
    clip_start: np.ndarray,
    clip_end: np.ndarray,
) -> np.ndarray:
    direction = second - first
    clip_direction = clip_end - clip_start
    denominator = _cross(direction, clip_direction)
    if abs(denominator) < 1.0e-12:
        return second
    fraction = _cross(clip_start - first, clip_direction) / denominator
    return first + fraction * direction


def polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = [point for point in np.asarray(subject, dtype=np.float64)]
    for index, clip_end in enumerate(clipper):
        clip_start = clipper[index - 1]
        input_points, output = output, []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, clip_start, clip_end)
            previous_inside = _inside(previous, clip_start, clip_end)
            if current_inside:
                if not previous_inside:
                    output.append(_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_intersection(previous, current, clip_start, clip_end))
            previous = current
    return np.asarray(output, dtype=np.float64)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(
        abs(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
        / 2.0
    )


def rotated_box_iou(first: RotatedBEVBox, second: RotatedBEVBox) -> float:
    intersection = polygon_area(polygon_clip(box_corners(first), box_corners(second)))
    union = first.length * first.width + second.length * second.width - intersection
    return intersection / union if union > 0.0 else 0.0


def rotated_nms(boxes: list[RotatedBEVBox], threshold: float) -> list[RotatedBEVBox]:
    remaining = sorted(boxes, key=lambda box: box.confidence, reverse=True)
    kept: list[RotatedBEVBox] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            box
            for box in remaining
            if box.class_name != current.class_name
            or rotated_box_iou(current, box) < threshold
        ]
    return kept
