"""K-Radar 3D object annotations following the official label convention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_LABEL_VERSIONS = {"auto", "v1_0", "v1_1", "v2_0", "v2_1"}


@dataclass(frozen=True)
class KRadarObjectAnnotation:
    class_name: str
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float

    def box(self) -> tuple[float, ...]:
        return (
            self.x,
            self.y,
            self.z,
            self.yaw,
            self.length,
            self.width,
            self.height,
        )


def _repair_v2_header_line_break(lines: list[str]) -> list[str]:
    """Mirror the official workaround for malformed v2.0 first lines."""

    if not lines:
        raise ValueError("K-Radar label file is empty")
    header = lines[0].rstrip("\n")
    try:
        header.split(", ", maxsplit=1)[1]
        return lines
    except (IndexError, ValueError):
        pass
    try:
        _empty, header_prime, first_object = header.split("*")
        repaired = list(lines)
        repaired[0] = "*" + header_prime + "\n"
        repaired.insert(1, "*" + first_object + "\n")
        return repaired
    except ValueError as exc:
        raise ValueError(f"Malformed K-Radar label header: {header!r}") from exc


def _detect_version(values: list[str]) -> str:
    if len(values) == 10:
        return "v2_0"
    if len(values) == 12:
        return "v1_1"
    if len(values) == 11:
        try:
            int(values[1].strip())
            return "v1_0"
        except ValueError:
            return "v2_1"
    raise ValueError(
        f"Cannot infer K-Radar label version from {len(values)} fields"
    )


def _object_columns(values: list[str], version: str) -> tuple[int, int]:
    expected = {"v1_0": 11, "v1_1": 12, "v2_0": 10, "v2_1": 11}
    if len(values) != expected[version]:
        raise ValueError(
            f"K-Radar {version} object row requires {expected[version]} "
            f"fields, got {len(values)}"
        )
    if version == "v1_0":
        return 3, 4
    if version == "v1_1":
        return 4, 5
    if version == "v2_0":
        return 2, 3
    return 3, 4


def load_kradar_annotations(
    path: str | Path,
    *,
    version: str = "auto",
    translation_xyz: tuple[float, float, float] | None = None,
) -> tuple[KRadarObjectAnnotation, ...]:
    """Load one K-Radar label file using the official v1/v2 conventions.

    K-Radar stores yaw in degrees and half box dimensions. Returned yaw is in
    radians and length/width/height are full dimensions. ``translation_xyz``
    is optional because it must only be used when the sensor data are shifted
    into the same calibrated coordinate frame.
    """

    if version not in _LABEL_VERSIONS:
        raise ValueError(f"Unsupported K-Radar label version: {version}")
    path = Path(path)
    lines = _repair_v2_header_line_break(
        path.read_text(encoding="utf-8").splitlines(keepends=True)
    )
    translation = np.zeros(3, dtype=np.float64)
    if translation_xyz is not None:
        translation = np.asarray(translation_xyz, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation_xyz must contain three finite values")

    objects = []
    for line_number, line in enumerate(lines[1:], start=2):
        stripped = line.strip()
        if not stripped:
            continue
        values = [value.strip() for value in stripped.split(",")]
        if values[0] != "*":
            continue
        row_version = _detect_version(values) if version == "auto" else version
        try:
            class_column, geometry_column = _object_columns(values, row_version)
            class_name = values[class_column]
            x, y, z, yaw_degrees, half_length, half_width, half_height = (
                float(value)
                for value in values[geometry_column : geometry_column + 7]
            )
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Malformed K-Radar {row_version} object at {path}:{line_number}: "
                f"{stripped!r}"
            ) from exc
        geometry = np.asarray(
            (x, y, z, yaw_degrees, half_length, half_width, half_height),
            dtype=np.float64,
        )
        if not class_name or not np.isfinite(geometry).all():
            raise ValueError(
                f"Invalid K-Radar object at {path}:{line_number}: {stripped!r}"
            )
        if min(half_length, half_width, half_height) <= 0.0:
            raise ValueError(
                f"K-Radar box dimensions must be positive at {path}:{line_number}"
            )
        objects.append(
            KRadarObjectAnnotation(
                class_name=class_name,
                x=x + float(translation[0]),
                y=y + float(translation[1]),
                z=z + float(translation[2]),
                yaw=float(np.deg2rad(yaw_degrees)),
                length=2.0 * half_length,
                width=2.0 * half_width,
                height=2.0 * half_height,
            )
        )
    return tuple(objects)


def resolve_kradar_label_path(
    kradar_root: str | Path,
    metadata: dict,
    *,
    revised_label_root: str | Path | None = None,
) -> Path:
    """Prefer a revised label with the same frame name, then use metadata."""

    kradar_root = Path(kradar_root)
    sequence = str(metadata["sequence"])
    original_relative = Path(str(metadata["label_relative_path"]))
    filename = original_relative.name
    candidates = []
    if revised_label_root is not None:
        revised_root = Path(revised_label_root)
        candidates.extend(
            (
                revised_root / sequence / filename,
                revised_root / sequence / "info_label" / filename,
                revised_root / sequence / "info_label_v2_0" / filename,
                revised_root / "KRadar_refined_label_by_UWIPL" / sequence / filename,
                revised_root / "KRadar_revised_visibility" / sequence / filename,
                revised_root / f"{sequence}_info_label_revised" / filename,
                revised_root / filename,
            )
        )
    sequence_root = kradar_root / "lidar" / sequence
    candidates.extend(
        (
            sequence_root / "info_label_v2_0" / filename,
            sequence_root / "info_label_v2" / filename,
            kradar_root / original_relative,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No K-Radar label found; checked: "
        + ", ".join(str(path) for path in candidates)
    )
