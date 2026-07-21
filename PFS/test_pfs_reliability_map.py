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
if str(FAULT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FAULT_MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pfs_model import MODEL_VARIANTS, build_reliability_model
from train_pfs_reliability_map import PFSReliabilityDataset, collate
from train_reliability_map import save_predictions, split_paths


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

    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    paths = sorted(dataset_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    base_channels = args.base_channels or int(checkpoint_args.get("base_channels", 16))
    dropout = args.dropout if args.dropout is not None else float(checkpoint_args.get("dropout", 0.0))
    model_variant = args.model_variant or checkpoint_args.get("model_variant", "pfs")

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
        collate_fn=collate,
    )

    model = build_reliability_model(
        model_variant,
        in_channels=3,
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    print(f"Saved {len(rows)} {model_variant} prediction comparisons: {output_root}")


if __name__ == "__main__":
    main()
