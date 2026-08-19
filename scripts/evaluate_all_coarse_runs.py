"""Evaluate every saved coarse-reconstruction run on one held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


EVALUATOR_MODULE = (
    "models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Fallback test dataset when a run's saved data root is unavailable.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--radar3-root", type=Path)
    parser.add_argument("--radar5-root", type=Path)
    parser.add_argument("--radar10-root", type=Path)
    parser.add_argument("--radar20-root", type=Path)
    parser.add_argument(
        "--selector-config",
        type=Path,
        help=(
            "Optional shared Fault Selector configuration used to validate "
            "the evaluation mask cache. Model architecture is still loaded "
            "from each checkpoint."
        ),
    )
    parser.add_argument("--pattern", default="coarse_*")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize-samples-per-fault", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate runs that already contain summary.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and print commands without evaluating models.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _run_stack(run: Path, resolved: dict[str, Any]) -> int | None:
    saved_root = str(resolved.get("args", {}).get("radar_root", ""))
    searchable = f"{run.name} {saved_root}".lower()
    match = re.search(r"radar[_-]?(20|10|5|3)(?:\D|$)", searchable)
    return int(match.group(1)) if match else None


def _has_test_files(root: Path | None) -> bool:
    if root is None:
        return False
    split = root / "test"
    return split.is_dir() and next(split.rglob("*.npz"), None) is not None


def _choose_radar_root(
    run: Path,
    resolved: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path | None, str]:
    stack = _run_stack(run, resolved)

    saved = resolved.get("args", {}).get("radar_root")
    saved_root = Path(saved) if saved else None
    candidates: list[Path | None] = []
    if stack == 3:
        candidates = [args.radar3_root, saved_root]
    elif stack == 5:
        candidates = [args.radar5_root, saved_root]
    elif stack == 10:
        candidates = [args.radar10_root, saved_root]
    elif stack == 20:
        candidates = [args.radar20_root, saved_root]
    else:
        candidates = [saved_root]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_test_files(candidate):
            return candidate, f"radar{stack or '?'}"
    requirement = f"radar stack={stack or 'unknown'}"
    return None, requirement


def _choose_data_root(
    resolved: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    saved = resolved.get("args", {}).get("data_root")
    candidates = [
        Path(saved) if saved else None,
        args.data_root,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_test_files(candidate):
            return candidate
    return None


def _checkpoint_path(run: Path) -> Path:
    tolerant = run / "best_tolerant_iou.pt"
    return tolerant if tolerant.is_file() else run / "best_model.pt"


def _discover_runs(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_dir()
        and (path / "best_model.pt").is_file()
        and (path / "resolved_config.json").is_file()
    )


def _run_metadata(
    run: Path,
    resolved: dict[str, Any],
    radar_root: Path | None,
    data_root: Path | None,
    checkpoint: Path,
) -> dict[str, Any]:
    model = resolved.get("model", {})
    loss = resolved.get("loss", {})
    training = resolved.get("training", {})
    pointpillars = model.get("pointpillars", {})
    observability = loss.get("observability_weighting", {})
    return {
        "run": run.name,
        "radar_stack": _run_stack(run, resolved) or "",
        "radar_root": str(radar_root or ""),
        "data_root": str(data_root or ""),
        "checkpoint": checkpoint.name,
        "pointpillars": bool(pointpillars.get("enabled", False)),
        "lidar_channels": model.get("lidar_channels", ""),
        "radar_channels": model.get("radar_channels", ""),
        "dropout": model.get(
            "dropout", model.get("hrnet", {}).get("dropout", "")
        ),
        "weight_decay": training.get("weight_decay", ""),
        "observability": bool(observability.get("enabled", False)),
        "healthy_context": model.get("use_healthy_context_mask", ""),
        "halo_context": model.get("use_halo_context", ""),
    }


def _flatten_scalars(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = value
    return flattened


def _write_comparison(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "all_models_test_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_root / "all_models_test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete = [row for row in rows if row.get("status") == "complete"]
    complete.sort(
        key=lambda row: float(row.get("tolerant_iou_0_5m", 0.0)),
        reverse=True,
    )
    leaderboard_fields = [
        "run",
        "checkpoint",
        "exact_iou",
        "exact_f1",
        "exact_precision",
        "exact_recall",
        "tolerant_iou_0_5m",
        "faulty_tolerant_iou_0_5m",
        "tolerant_iou_0_5m_improvement",
        "tolerant_f1_0_5m",
        "tolerant_precision_0_5m",
        "tolerant_recall_0_5m",
    ]
    with (output_root / "coarse_model_leaderboard.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard_fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in leaderboard_fields}
            for row in complete
        )


def _print_leaderboard(rows: list[dict[str, Any]]) -> None:
    complete = [row for row in rows if row.get("status") == "complete"]
    complete.sort(
        key=lambda row: float(row.get("tolerant_iou_0_5m", 0.0)),
        reverse=True,
    )
    print("\nFINAL TEST LEADERBOARD (ranked by tolerant IoU@0.5m)")
    for rank, row in enumerate(complete, 1):
        print(f"\n{rank:2d}. {row['run']}  [{row['checkpoint']}]")
        print(
            "    Exact:  "
            f"IoU={row['exact_iou']:.3%}  "
            f"F1={row['exact_f1']:.3%}  "
            f"P={row['exact_precision']:.3%}  "
            f"R={row['exact_recall']:.3%}"
        )
        print(
            "    @0.5m:  "
            f"IoU={row['tolerant_iou_0_5m']:.3%}  "
            f"Faulty IoU={row['faulty_tolerant_iou_0_5m']:.3%}  "
            f"Improvement={row['tolerant_iou_0_5m_improvement']:+.3%}  "
            f"F1={row['tolerant_f1_0_5m']:.3%}  "
            f"P={row['tolerant_precision_0_5m']:.3%}  "
            f"R={row['tolerant_recall_0_5m']:.3%}"
        )


def main() -> None:
    args = _parse_args()
    if args.selector_config is not None and not args.selector_config.is_file():
        raise FileNotFoundError(
            f"Selector configuration is missing: {args.selector_config}"
        )
    if not _has_test_files(args.data_root):
        raise FileNotFoundError(
            f"Test split is missing or empty: {args.data_root / 'test'}"
        )
    runs = _discover_runs(args.runs_root, args.pattern)
    if not runs:
        raise FileNotFoundError(
            f"No completed runs matching {args.pattern!r} under {args.runs_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    print(f"Discovered {len(runs)} model runs", flush=True)
    for index, run in enumerate(runs, 1):
        resolved_path = run / "resolved_config.json"
        resolved = _read_json(resolved_path)
        radar_root, radar_description = _choose_radar_root(run, resolved, args)
        data_root = _choose_data_root(resolved, args)
        checkpoint = _checkpoint_path(run)
        row = _run_metadata(run, resolved, radar_root, data_root, checkpoint)
        selector_config = args.selector_config or resolved_path
        row["selector_config"] = str(selector_config)
        destination = args.output_root / run.name
        summary_path = destination / "summary.json"
        print(f"\n[{index}/{len(runs)}] {run.name}", flush=True)

        if radar_root is None:
            row.update(
                status="skipped",
                error=f"No test radar cache satisfies {radar_description}",
            )
            print(f"  SKIP: {row['error']}", flush=True)
            rows.append(row)
            _write_comparison(args.output_root, rows)
            continue
        if data_root is None:
            row.update(status="skipped", error="No compatible test dataset found")
            print(f"  SKIP: {row['error']}", flush=True)
            rows.append(row)
            _write_comparison(args.output_root, rows)
            continue

        command = [
            sys.executable,
            "-u",
            "-m",
            EVALUATOR_MODULE,
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--radar-root",
            str(radar_root),
            "--output-root",
            str(destination),
            "--config",
            str(selector_config),
            "--split",
            "test",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            "--visualize-samples-per-fault",
            str(args.visualize_samples_per_fault),
        ]
        if args.limit_samples is not None:
            command.extend(("--limit-samples", str(args.limit_samples)))
        if args.no_amp:
            command.append("--no-amp")

        print("  " + " ".join(command), flush=True)
        if args.dry_run:
            row["status"] = "dry-run"
            rows.append(row)
            _write_comparison(args.output_root, rows)
            continue

        if args.force or not summary_path.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / "evaluation.log").open(
                "w", encoding="utf-8"
            ) as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print("  | " + line, end="", flush=True)
                returncode = process.wait()
            if returncode != 0:
                row.update(
                    status="failed",
                    error=(
                        f"Evaluator exited with {returncode}; see "
                        f"{destination / 'evaluation.log'}"
                    ),
                )
                print(f"  FAIL: {row['error']}", flush=True)
                rows.append(row)
                _write_comparison(args.output_root, rows)
                continue

        summary = _read_json(summary_path)
        row.update(
            status="complete",
            checkpoint_epoch=summary.get("checkpoint_epoch", ""),
        )
        overall = summary.get("overall", {})
        row.update(_flatten_scalars("", overall))
        row.update(
            exact_iou=overall.get("micro/coarse_iou", 0.0),
            exact_f1=overall.get("micro/coarse_f1", 0.0),
            exact_precision=overall.get("micro/coarse_precision", 0.0),
            exact_recall=overall.get("micro/coarse_recall", 0.0),
            tolerant_iou_0_5m=overall.get(
                "macro/coarse_occupancy_tolerant_0_5m_iou", 0.0
            ),
            faulty_tolerant_iou_0_5m=overall.get(
                "macro/faulty_occupancy_tolerant_0_5m_iou", 0.0
            ),
            tolerant_iou_0_5m_improvement=overall.get(
                "macro/tolerant_0_5m_iou_improvement", 0.0
            ),
            tolerant_f1_0_5m=overall.get(
                "macro/coarse_occupancy_tolerant_0_5m_f1", 0.0
            ),
            tolerant_precision_0_5m=overall.get(
                "macro/coarse_occupancy_tolerant_0_5m_precision", 0.0
            ),
            tolerant_recall_0_5m=overall.get(
                "macro/coarse_occupancy_tolerant_0_5m_recall", 0.0
            ),
        )
        rows.append(row)
        _write_comparison(args.output_root, rows)
        print(
            "  DONE: exact IoU="
            f"{row.get('micro/coarse_iou', 0.0):.2%}, "
            "IoU@0.5m="
            f"{row['tolerant_iou_0_5m']:.2%} "
            f"({row['tolerant_iou_0_5m_improvement']:+.2%}), "
            f"F1@0.5m={row['tolerant_f1_0_5m']:.2%}",
            flush=True,
        )

    complete = sum(row["status"] == "complete" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    skipped = sum(row["status"] == "skipped" for row in rows)
    print(
        f"\nFinished: complete={complete}, failed={failed}, skipped={skipped}",
        flush=True,
    )
    print(f"Comparison: {args.output_root / 'all_models_test_metrics.csv'}")
    print(f"Leaderboard: {args.output_root / 'coarse_model_leaderboard.csv'}")
    _print_leaderboard(rows)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
