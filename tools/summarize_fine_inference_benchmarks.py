"""Summarize Fine inference timings and reject unfair model comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing one inference_timing.json per model/step run.",
    )
    return parser.parse_args()


def _fairness_mismatches(results: list[dict]) -> list[str]:
    if not results:
        return []
    ignored = {"sampling_steps"}
    reference = results[0]["comparison_signature"]
    mismatches: set[str] = set()
    for result in results[1:]:
        candidate = result["comparison_signature"]
        for key in set(reference) | set(candidate):
            if key not in ignored and reference.get(key) != candidate.get(key):
                mismatches.add(key)
    return sorted(mismatches)


def main() -> None:
    args = _parse_args()
    paths = sorted(args.root.glob("**/inference_timing.json"))
    if not paths:
        raise FileNotFoundError(
            f"No inference_timing.json files found below {args.root}"
        )
    results = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_label"] = str(path.parent.relative_to(args.root))
        results.append(payload)

    mismatches = _fairness_mismatches(results)
    if mismatches:
        raise ValueError(
            "Benchmarks are not directly comparable; mismatched settings: "
            + ", ".join(mismatches)
        )

    print(
        f"{'Model':<34} {'Steps':>5} {'Params':>11} {'Mean':>10} "
        f"{'P50':>10} {'P95':>10} {'P99':>10} {'Hz':>8}"
    )
    print("-" * 105)
    for result in sorted(
        results,
        key=lambda item: (item["sampling_steps"], item["_label"]),
    ):
        latency = result["latency_per_sample"]["end_to_end"]
        print(
            f"{result['_label']:<34} {result['sampling_steps']:>5d} "
            f"{result['parameters']['fine']:>11,d} "
            f"{latency['mean_ms']:>8.2f}ms {latency['p50_ms']:>8.2f}ms "
            f"{latency['p95_ms']:>8.2f}ms {latency['p99_ms']:>8.2f}ms "
            f"{result['end_to_end_throughput_samples_per_second']:>7.2f}"
        )

    print()
    print("Fairness check: PASS")
    print(
        "Identical samples, coarse checkpoint, radar/data roots, batch size, "
        "precision, buckets, compilation mode, and Fine PointPillars setting."
    )


if __name__ == "__main__":
    main()
