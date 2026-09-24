"""Show a clean live progress bar for a background range-view training run."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    progress_path = args.run_root / "progress.json"
    config_path = args.run_root / "resolved_config.json"
    total_epochs = None
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        total_epochs = int(config["arguments"]["epochs"])
    previous_width = 0
    try:
        while True:
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {"epoch": "?", "phase": "waiting", "completed_batches": 0,
                            "total_batches": 1}
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
