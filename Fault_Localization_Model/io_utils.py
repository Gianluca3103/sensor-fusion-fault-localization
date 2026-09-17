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


def atomic_savez(path, *, compression_level=1, **arrays):
    """NumPy-compatible, lossless NPZ with explicit ZIP compression effort.

    Level 0 stores arrays verbatim; 1 is fast DEFLATE; 6 matches normal ZIP
    effort. Only container encoding changes, never array values/dtypes.
    """
    import numpy as np
    import zipfile
    if not isinstance(compression_level, int) or not 0 <= compression_level <= 9:
        raise ValueError('compression_level must be an integer from 0 to 9')
    def write(temporary):
        compression = zipfile.ZIP_STORED if compression_level == 0 else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(temporary, 'w', compression=compression,
                             compresslevel=compression_level if compression_level else None,
                             allowZip64=True) as archive:
            for name, value in arrays.items():
                with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.asanyarray(value), allow_pickle=False)
    _atomic_replace(Path(path), write)


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
