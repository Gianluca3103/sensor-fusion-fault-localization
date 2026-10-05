"""Readable epoch summaries with complete machine-readable metrics on disk."""

from __future__ import annotations

import json
from pathlib import Path


def record_epoch(output_root: Path, message: dict, summary: str) -> None:
    with (output_root / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(message) + "\n")
    print(summary, flush=True)


def blueprint_summary(message: dict, total_epochs: int) -> str:
    epoch = message["epoch"]
    train = message["train"]
    line = (f"Epoch {epoch:02d}/{total_epochs} | train loss {train['loss']:.4f}"
            f" | train F1@3m {train['depth_f1_3m']:.1%}")
    val = message.get("val")
    if val is None:
        return line
    line += (f"\n  val loss {val['loss']:.4f} | coverage {val['candidate_coverage']:.1%}"
             f" | P/R/F1@3m "
             f"{val['depth_precision_3m']:.1%}/{val['depth_recall_3m']:.1%}/"
             f"{val['depth_f1_3m']:.1%} | MAE {val['matched_depth_mae_m']:.2f} m"
             f" | clean hits {int(val['clean_hits']):,}")
    if message.get("warning"):
        line += f"\n  WARNING: {message['warning']}"
    return line


def diffusion_summary(message: dict, total_epochs: int) -> str:
    epoch = message["epoch"]
    train = message["train"]
    line = (f"Epoch {epoch:02d}/{total_epochs} | train loss {train['loss']:.4f}"
            f" (blueprint {train['blueprint']:.4f}, diffusion {train['diffusion']:.4f})")
    val = message.get("val")
    if val is None:
        return line
    return (line + f"\n  val loss {val['loss']:.4f}"
            f" | correctable/tile {val['correctable']:.1f}/{val['supported']:.1f}"
            f" | gate precision {val['calibration_precision']:.1%}"
            f" | accepted {int(val['calibration_accepted']):,}"
            f" | threshold {val['calibration_threshold']:.3f}")
