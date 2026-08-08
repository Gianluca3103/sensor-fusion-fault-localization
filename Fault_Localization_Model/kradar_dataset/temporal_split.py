"""Chronological train/validation/test selection inside each K-Radar sequence."""

from pathlib import Path

from .kradar_discovery import kradar_source_metadata


def select_temporal_split_frames(
    frames,
    data_root: str | Path,
    split_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    data_root = Path(data_root)
    grouped = {}
    for frame in frames:
        frame = Path(frame)
        sequence = kradar_source_metadata(frame, data_root)["sequence"]
        grouped.setdefault(sequence, []).append(frame)

    selected = []
    split_counts = []
    for sequence, sequence_frames in sorted(grouped.items(), key=lambda row: int(row[0])):
        sequence_frames.sort(
            key=lambda path: int(path.stem.rsplit("_", 1)[-1])
        )
        count = len(sequence_frames)
        train_end = int(count * train_ratio)
        val_end = int(count * (train_ratio + val_ratio))
        if split_name == "train":
            split_frames = sequence_frames[:train_end]
        elif split_name == "val":
            split_frames = sequence_frames[train_end:val_end]
        elif split_name == "test":
            split_frames = sequence_frames[val_end:]
        else:
            raise ValueError(f"Unknown temporal split: {split_name}")
        selected.extend(split_frames)
        split_counts.append((sequence, count, len(split_frames)))

    selected.sort(
        key=lambda path: (
            int(Path(path).parents[1].name),
            int(Path(path).stem.rsplit("_", 1)[-1]),
        )
    )
    if not selected:
        raise FileNotFoundError(
            f"No frames selected for temporal split {split_name!r}."
        )
    return selected, split_counts
