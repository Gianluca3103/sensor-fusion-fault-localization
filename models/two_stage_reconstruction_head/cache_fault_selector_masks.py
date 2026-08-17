"""Precompute versioned Fault Selector masks for reconstruction training."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.two_stage_reconstruction_head import build_selector_config, load_config
from models.two_stage_reconstruction_head.fault_selector_cache import (
    build_selector_cache_entry,
    selector_cache_root,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "coarse_reconstruction.json"),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
    )
    return parser.parse_args()


def _discover_sample_paths(data_root: Path) -> list[Path]:
    """Find sample archives in either a split dataset or a flat preview set."""

    sample_paths = sorted(data_root.rglob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No NPZ samples were found under {data_root}")
    return sample_paths


def main():
    args = _parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {data_root}")
    cache_root = selector_cache_root(data_root)
    config = build_selector_config(load_config(args.config))

    sample_paths = _discover_sample_paths(data_root)
    printed_split = False
    for split in ("train", "val"):
        split_root = data_root / split
        if split_root.is_dir():
            print(f"{split}: {len(list(split_root.rglob('*.npz')))} samples")
            printed_split = True
    if not printed_split:
        print(f"samples: {len(sample_paths)}")

    print(f"Cache root: {cache_root}")
    print(f"Workers: {args.num_workers}")
    started = time.perf_counter()
    counts = {"created": 0, "cached": 0}
    worker = partial(
        build_selector_cache_entry,
        data_root=data_root,
        config=config,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for index, status in enumerate(
            executor.map(worker, sample_paths, chunksize=16),
            1,
        ):
            counts[status] += 1
            if index % 1_000 == 0 or index == len(sample_paths):
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                print(
                    f"Processed {index}/{len(sample_paths)}; "
                    f"created={counts['created']}; cached={counts['cached']}; "
                    f"rate={rate:.1f} samples/s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    print(
        f"Fault Selector cache complete in {elapsed / 60.0:.1f} minutes: "
        f"created={counts['created']}, cached={counts['cached']}"
    )


if __name__ == "__main__":
    main()
