"""Discover Aeva frames and describe their Hercules source location."""

from pathlib import Path


def list_aeva_bins(aeva_dir: str | Path) -> list[Path]:
    aeva_dir = Path(aeva_dir)
    bins = sorted(aeva_dir.glob("*.bin"), key=lambda path: path.stem)
    if not bins:
        raise FileNotFoundError(f"No Hercules Aeva .bin files found in {aeva_dir}")
    return bins


def list_all_aeva_bins(data_root: str | Path) -> tuple[list[Path], list[Path]]:
    """Return every raw Aeva frame beneath the Hercules data root."""
    data_root = Path(data_root)
    aeva_dirs = [
        candidate
        for candidate in sorted(data_root.rglob("Aeva"))
        if candidate.is_dir() and any(candidate.glob("*.bin"))
    ]
    if not aeva_dirs:
        raise FileNotFoundError(
            f"No Hercules Aeva folders with .bin files found under {data_root}"
        )

    bins = [
        bin_path
        for aeva_dir in aeva_dirs
        for bin_path in list_aeva_bins(aeva_dir)
    ]
    bins.sort(key=lambda path: str(path.relative_to(data_root)).casefold())
    return bins, aeva_dirs


def hercules_source_metadata(
    bin_path: str | Path,
    data_root: str | Path,
) -> dict[str, str]:
    bin_path = Path(bin_path)
    data_root = Path(data_root)
    relative = bin_path.relative_to(data_root)
    parts = relative.parts
    scene = parts[0] if parts else ""
    session = ""
    if "LiDAR" in parts:
        lidar_index = parts.index("LiDAR")
        if lidar_index > 1:
            session = parts[lidar_index - 1]
    return {
        "scene": scene,
        "day": scene,
        "session": session,
        "source_relative_path": str(relative),
        "source_aeva_dir": str(bin_path.parent),
    }
