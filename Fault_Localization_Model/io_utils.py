import csv
import json
import os
from pathlib import Path
import uuid


def _temporary_path(path: Path) -> Path:
    """Return a unique sibling path suitable for an atomic replacement."""
    path = Path(path)
    return path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp{path.suffix}"
    )


def _atomic_replace(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path, text, encoding="utf-8"):
    """Write text without exposing a partially written destination file."""
    path = Path(path)
    _atomic_replace(path, lambda temporary: temporary.write_text(text, encoding=encoding))


def atomic_write_json(path, payload, *, indent=2):
    """Serialize JSON through an atomic sibling-file replacement."""
    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False),
    )


def atomic_savez_compressed(path, **arrays):
    """Atomically save a compressed NumPy archive."""
    import numpy as np

    path = Path(path)
    _atomic_replace(path, lambda temporary: np.savez_compressed(temporary, **arrays))


def atomic_torch_save(payload, path):
    """Atomically save a trusted PyTorch checkpoint."""
    import torch

    path = Path(path)
    _atomic_replace(path, lambda temporary: torch.save(payload, temporary))


def write_csv_rows(path, rows, fieldnames=None):
    """Write dictionaries to CSV and return whether any rows were written."""
    if not rows:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})

    def write(temporary):
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _atomic_replace(path, write)
    return True
