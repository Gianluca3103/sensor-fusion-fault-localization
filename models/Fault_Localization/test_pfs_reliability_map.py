from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.Fault_Localization.datasets import PFSReliabilityDataset, collate_reliability_batch
from models.Fault_Localization.pfs_model import MODEL_VARIANTS, load_model_checkpoint
from models.Fault_Localization.training_utils import resolve_device, save_predictions, split_paths


DEFAULT_DATASET_ROOT = FAULT_MODEL_DIR / "grid_reliability_7500_fog_s3_x64_y32"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "runs" / "pfs_7500_fog_s3_x64_y32" / "checkpoints" / "last_checkpoint.pt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "pfs_7500_fog_s3_x64_y32" / "test_10_predictions"


def main():
    parser = argparse.ArgumentParser(description="Save PFS model reliability-map prediction samples.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--model-variant", choices=sorted(MODEL_VARIANTS), default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--visual-grid-size", type=int, default=100)
    parser.add_argument("--localization-threshold", type=float, default=0.5)
    parser.add_argument("--localization-tolerance-m", type=float, default=0.20)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument(
        "--use-all-samples",
        action="store_true",
        help="Use the supplied folder directly instead of taking another random validation split.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_images < 1:
        parser.error("--batch-size and --max-images must be at least 1")
    if (
        args.resize_height < 1
        or args.resize_width < 1
        or args.visual_grid_size < 1
    ):
        parser.error("Resize dimensions and --visual-grid-size must be positive")
    if args.visual_grid_size > min(args.resize_height, args.resize_width):
        parser.error(
            "--visual-grid-size cannot exceed the smaller resized input dimension"
        )
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must lie strictly between 0 and 1")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 0.0 < args.localization_threshold < 1.0:
        parser.error("--localization-threshold must lie strictly between 0 and 1")
    if args.localization_tolerance_m < 0.0:
        parser.error("--localization-tolerance-m must be non-negative")
    if not 0.0 <= args.target_fault_threshold < 1.0:
        parser.error("--target-fault-threshold must lie in [0,1)")
    if args.base_channels is not None and args.base_channels < 1:
        parser.error("--base-channels must be positive")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0,1)")

    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    paths = sorted(dataset_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_root}")
    device = resolve_device(args.device)
    model, _, model_info = load_model_checkpoint(
        checkpoint_path,
        device,
        base_channels=args.base_channels,
        dropout=args.dropout,
        model_variant=args.model_variant,
    )

    if args.use_all_samples:
        test_paths = paths
    else:
        _, test_paths = split_paths(paths, args.val_ratio, args.seed)
    resize_hw = (args.resize_height, args.resize_width)
    loader = DataLoader(
        PFSReliabilityDataset(test_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_reliability_batch,
        pin_memory=device.type == "cuda",
    )

    rows = save_predictions(
        model,
        loader,
        output_root,
        device,
        args.max_images,
        visual_grid_size=args.visual_grid_size,
        localization_threshold=args.localization_threshold,
        localization_tolerance_m=args.localization_tolerance_m,
        target_fault_threshold=args.target_fault_threshold,
    )
    print(
        f"Saved {len(rows)} {model_info['model_variant']} "
        f"prediction comparisons: {output_root}"
    )


if __name__ == "__main__":
    main()
