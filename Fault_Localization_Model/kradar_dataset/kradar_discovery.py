"""Discover paired K-Radar os2-64 LiDAR and pc10p radar frames."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
import re


_LABEL_HEADER = re.compile(
    r"idx\([^)]*\)=([0-9_]+),\s*timestamp=([0-9]+(?:\.[0-9]+)?)"
)


def kradar_sequence_root(data_root: str | Path, sequence: str | int) -> Path:
    """Return the canonical LiDAR/support directory for one K-Radar sequence."""

    return Path(data_root) / "lidar" / str(int(sequence))


def _timestamp_ns(text: str) -> int:
    return int(Decimal(text) * Decimal(1_000_000_000))


def parse_label_frame(path: str | Path) -> tuple[str, str, int]:
    """Return paired radar index, os2-64 index, and LiDAR timestamp."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    match = _LABEL_HEADER.search(header)
    if match is None:
        raise ValueError(f"Malformed K-Radar label header in {path}: {header!r}")
    indices = match.group(1).split("_")
    if len(indices) != 5:
        raise ValueError(
            f"K-Radar label {path} must identify five sensor indices, got {indices}"
        )
    return indices[0], indices[1], _timestamp_ns(match.group(2))


@lru_cache(maxsize=64)
def _sequence_pairings(sequence_root_text: str) -> dict[str, tuple[str, int, Path]]:
    sequence_root = Path(sequence_root_text)
    pairings = {}
    for label_path in sorted((sequence_root / "info_label").glob("*.txt")):
        radar_index, lidar_index, timestamp_ns = parse_label_frame(label_path)
        if lidar_index in pairings:
            raise ValueError(
                f"Duplicate os2-64 index {lidar_index} under {sequence_root}"
            )
        pairings[lidar_index] = (radar_index, timestamp_ns, label_path)
    return pairings


def list_kradar_lidar_frames(
    sequence_root: str | Path,
    radar_point_root: str | Path,
) -> list[Path]:
    """Return os2-64 frames that have both label pairing and pc10p radar."""

    sequence_root = Path(sequence_root)
    radar_point_root = Path(radar_point_root)
    pairings = _sequence_pairings(str(sequence_root.resolve()))
    frames = []
    for lidar_index, (radar_index, _timestamp, _label) in pairings.items():
        lidar_path = sequence_root / "os2-64" / f"os2-64_{lidar_index}.pcd"
        radar_path = radar_point_root / sequence_root.name / f"rpc_{radar_index}.npy"
        if lidar_path.is_file() and radar_path.is_file():
            frames.append(lidar_path)
    frames.sort(key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    return frames


def list_all_kradar_lidar_frames(
    data_root: str | Path,
) -> tuple[list[Path], list[Path]]:
    """Return all locally available paired K-Radar os2-64 frames."""

    data_root = Path(data_root)
    lidar_root = data_root / "lidar"
    if not lidar_root.is_dir():
        raise FileNotFoundError(f"K-Radar LiDAR root does not exist: {lidar_root}")
    radar_point_root = data_root / "radar" / "pc10p"
    if not radar_point_root.is_dir():
        raise FileNotFoundError(
            f"K-Radar pc10p root does not exist: {radar_point_root}"
        )
    sequence_dirs = sorted(
        (
            path
            for path in lidar_root.iterdir()
            if path.is_dir()
            and path.name.isdigit()
            and (path / "info_label").is_dir()
            and (path / "info_calib" / "calib_radar_lidar.txt").is_file()
            and (path / "os2-64").is_dir()
        ),
        key=lambda path: int(path.name),
    )
    frames = [
        frame
        for sequence_dir in sequence_dirs
        for frame in list_kradar_lidar_frames(sequence_dir, radar_point_root)
    ]
    populated_sequences = sorted(
        {frame.parents[1] for frame in frames}, key=lambda path: int(path.name)
    )
    if not frames:
        raise FileNotFoundError(
            "No paired K-Radar os2-64/pc10p frames were found under "
            f"{data_root}"
        )
    return frames, populated_sequences


def kradar_source_metadata(
    lidar_path: str | Path,
    data_root: str | Path,
) -> dict[str, object]:
    """Describe one paired K-Radar LiDAR source frame."""

    lidar_path = Path(lidar_path)
    data_root = Path(data_root)
    relative = lidar_path.relative_to(data_root)
    if (
        len(relative.parts) < 4
        or relative.parts[0] != "lidar"
        or not relative.parts[1].isdigit()
    ):
        raise ValueError(f"Unexpected K-Radar LiDAR path: {lidar_path}")
    sequence = str(int(relative.parts[1]))
    lidar_index = f"{int(lidar_path.stem.rsplit('_', 1)[-1]):05d}"
    sequence_root = kradar_sequence_root(data_root, sequence)
    try:
        radar_index, timestamp_ns, label_path = _sequence_pairings(
            str(sequence_root.resolve())
        )[lidar_index]
    except KeyError as exc:
        raise FileNotFoundError(
            f"No K-Radar label pairing for sequence {sequence}, LiDAR {lidar_index}"
        ) from exc
    radar_path = data_root / "radar" / "pc10p" / sequence / f"rpc_{radar_index}.npy"
    if not radar_path.is_file():
        raise FileNotFoundError(f"Missing paired K-Radar pc10p frame: {radar_path}")
    return {
        "sequence": sequence,
        "scene": sequence,
        "day": sequence,
        "session": "",
        "lidar_index": lidar_index,
        "radar_index": radar_index,
        "timestamp": str(timestamp_ns),
        "timestamp_ns": timestamp_ns,
        "source_relative_path": str(relative),
        "source_lidar_dir": str(lidar_path.parent),
        "label_relative_path": str(label_path.relative_to(data_root)),
        "radar_relative_path": str(radar_path.relative_to(data_root)),
        "calibration_path": str(
            sequence_root / "info_calib" / "calib_radar_lidar.txt"
        ),
    }
