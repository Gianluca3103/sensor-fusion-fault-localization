"""Chronological train/validation/test selection for Hercules Aeva frames."""

from pathlib import Path

from .hercules_discovery import hercules_source_metadata


def select_temporal_split_bins(
    bins,
    data_root: str | Path,
    split_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """Select a chronological split independently inside each Aeva folder."""
    data_root = Path(data_root)
    grouped = {}
    for bin_path in bins:
        bin_path = Path(bin_path)
        source_dir = hercules_source_metadata(bin_path, data_root)["source_aeva_dir"]
        grouped.setdefault(source_dir, []).append(bin_path)

    selected = []
    split_counts = []
    for source_dir, folder_bins in sorted(grouped.items()):
        folder_bins = sorted(folder_bins, key=lambda path: path.stem)
        count = len(folder_bins)
        train_end = int(count * train_ratio)
        val_end = int(count * (train_ratio + val_ratio))
        if split_name == "train":
            split_bins = folder_bins[:train_end]
        elif split_name == "val":
            split_bins = folder_bins[train_end:val_end]
        elif split_name == "test":
            split_bins = folder_bins[val_end:]
        else:
            raise ValueError(f"Unknown temporal split: {split_name}")
        selected.extend(split_bins)
        split_counts.append((source_dir, count, len(split_bins)))

    selected.sort(key=lambda path: str(path.relative_to(data_root)).casefold())
    if not selected:
        raise FileNotFoundError(
            f"No frames selected for temporal split {split_name!r}."
        )
    return selected, split_counts
