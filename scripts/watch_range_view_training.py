"""Show a clean live progress bar for a background range-view training run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time


def _render(progress: dict, *, width: int = 32) -> str:
    completed = int(progress.get("completed_batches", 0))
    total = max(int(progress.get("total_batches", 0)), 1)
    fraction = min(max(completed / total, 0.0), 1.0)
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    loss = progress.get("running_loss")
    loss_text = "" if loss is None else f" | loss {float(loss):.4f}"
    return (f"Epoch {progress.get('epoch', '?')} {progress.get('phase', '?'):10s} "
            f"[{bar}] {completed}/{total} ({fraction:.0%}){loss_text}")


def _read_summaries(path: Path, last_epoch: int | None) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row.get("epoch") and row.get("val_reconstructed_f1_at_0_2m")]
    if last_epoch is None:
        return rows[-1:]
    return [row for row in rows if int(row["epoch"]) > last_epoch]


def _format_summary(row: dict) -> str:
    return (
        f"Epoch {row['epoch']} result | train loss {float(row['train_loss']):.4f} | "
        f"val F1@0.2m {float(row['val_reconstructed_f1_at_0_2m']):.4f} "
        f"(faulty {float(row['val_faulty_f1_at_0_2m']):.4f}, "
        f"change {float(row['val_net_f1_improvement']):+.4f}) | "
        f"add precision {float(row['val_addition_precision']):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    progress_path = args.run_root / "progress.json"
    summary_path = args.run_root / "summary.csv"
    config_path = args.run_root / "resolved_config.json"
    total_epochs = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        total_epochs = int(config["arguments"]["epochs"])
    previous_width = 0
    last_summary_epoch = None
    try:
        while True:
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {"epoch": "?", "phase": "waiting", "completed_batches": 0,
                            "total_batches": 1}
            completed = _read_summaries(summary_path, last_summary_epoch)
            if completed:
                sys.stdout.write("\r" + " " * previous_width + "\r")
                for row in completed:
                    sys.stdout.write(_format_summary(row) + "\n")
                last_summary_epoch = int(completed[-1]["epoch"])
                previous_width = 0
            line = _render(progress)
            sys.stdout.write("\r" + line.ljust(previous_width))
            sys.stdout.flush()
            previous_width = len(line)
            if (total_epochs is not None and progress.get("phase") == "complete"
                    and progress.get("epoch") == total_epochs):
                sys.stdout.write("\n")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
