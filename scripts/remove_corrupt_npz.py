from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("faulty_rgb", "clean_rgb", "fault_heatmap", "metadata_json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find unreadable reliability-map NPZ files and optionally delete them."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def validate_npz(path: Path) -> tuple[Path, str | None]:
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in REQUIRED_KEYS:
                _ = data[key]
        return path, None
    except Exception as error:
        return path, f"{type(error).__name__}: {error}"


def main() -> None:
    args = parse_args()
    paths = sorted(args.dataset_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {args.dataset_root}")

    corrupt: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (path, error) in enumerate(executor.map(validate_npz, paths), 1):
            if error is not None:
                corrupt.append((path, error))
                print(f"CORRUPT: {path} ({error})", flush=True)
                if args.delete:
                    path.unlink(missing_ok=True)
            if index % 5000 == 0 or index == len(paths):
                print(
                    f"Checked {index}/{len(paths)} | corrupt: {len(corrupt)}",
                    flush=True,
                )

    action = "deleted" if args.delete else "found"
    print(f"Done: {len(corrupt)} corrupt files {action}.")
    if corrupt and not args.delete:
        print("Run again with --delete so the generator can recreate them.")


if __name__ == "__main__":
    main()
