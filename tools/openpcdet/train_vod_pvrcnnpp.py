"""Train upstream PV-RCNN++ on clean VoD and select best validation 3D mAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcdet_integration.openpcdet_eval import (
    add_openpcdet_to_path,
    evaluate_checkpoint_on_condition,
    load_openpcdet_config,
    validation_3d_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Official OpenPCDet ckpt directory; auto-discovered when omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable OpenPCDet's native CUDA automatic mixed precision training.",
    )
    parser.add_argument("--extra-tag", default="vod_pvrcnnpp_clean")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    openpcdet_root = add_openpcdet_to_path(args.openpcdet_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    data_link = openpcdet_root / "data" / "vod"
    data_link.parent.mkdir(parents=True, exist_ok=True)
    if data_link.exists() or data_link.is_symlink():
        if data_link.resolve() != args.data_root.resolve():
            raise RuntimeError(
                f"{data_link} already points to {data_link.resolve()}, not "
                f"{args.data_root.resolve()}"
            )
    else:
        data_link.symlink_to(args.data_root.resolve(), target_is_directory=True)

    if not args.skip_training:
        command = [
            sys.executable,
            "train.py",
            "--cfg_file",
            str(args.config.resolve()),
            "--batch_size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--workers",
            str(args.workers),
            "--extra_tag",
            args.extra_tag,
        ]
        if args.amp:
            command.append("--use_amp")
        print("Running official OpenPCDet training:\n  " + " ".join(command))
        subprocess.run(command, cwd=openpcdet_root / "tools", check=True)

    add_openpcdet_to_path(openpcdet_root)
    cfg = load_openpcdet_config(openpcdet_root, args.config)
    checkpoint_roots = [args.output_root]
    if args.checkpoint_dir is not None:
        checkpoint_roots.insert(0, args.checkpoint_dir)
    checkpoint_roots.extend(
        (openpcdet_root / "output").glob(f"**/{args.extra_tag}/ckpt")
    )
    checkpoints = sorted(
        {path.resolve() for root in checkpoint_roots for path in root.glob("checkpoint_epoch_*.pth")}
    )
    if not checkpoints:
        raise FileNotFoundError(
            "No OpenPCDet epoch checkpoints found. Pass --output-root pointing "
            "to the checkpoint directory or inspect the official output tree."
        )

    best = None
    model = None
    records = []
    for checkpoint in checkpoints:
        evaluation_root = args.output_root / "validation" / checkpoint.stem
        metrics, _predictions, model = evaluate_checkpoint_on_condition(
            openpcdet_root,
            cfg,
            checkpoint,
            args.data_root,
            evaluation_root,
            split="val",
            batch_size=args.batch_size,
            workers=args.workers,
            model=None,
        )
        score = validation_3d_map(metrics)
        record = {"checkpoint": str(checkpoint), "validation_3d_map": score, "metrics": metrics}
        records.append(record)
        if best is None or score > best[0]:
            best = (score, checkpoint)
    assert best is not None
    best_path = args.output_root / "best_validation_map.pth"
    shutil.copy2(best[1], best_path)
    summary = {
        "training_data": "clean VoD train only",
        "selection_data": "clean VoD val only",
        "best_checkpoint": str(best_path.resolve()),
        "source_checkpoint": str(best[1]),
        "best_validation_3d_map": best[0],
        "frozen_for_conditions": ["clean", "faulty", "coarse", "fine"],
        "checkpoints": records,
    }
    (args.output_root / "checkpoint_selection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
