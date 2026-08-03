from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PFS_Radar.train_pfs_radar import make_scheduler, sample_paths_from_root
from PFS_Radar.radar_data import filter_samples_with_radar_cache
from Fault_Localization_Model.sample_utils import filter_paths_by_fault, require_disjoint_splits


def prepare_stage1_paths(args):
    train_paths = sample_paths_from_root(Path(args.train_root))
    val_paths = sample_paths_from_root(Path(args.val_root))
    if not train_paths or not val_paths:
        raise FileNotFoundError("Both --train-root and --val-root must contain .npz files")
    train_paths, train_counts = filter_paths_by_fault(
        train_paths,
        include_faults=args.include_faults,
        exclude_faults=args.exclude_faults,
        strict_fault_names=True,
    )
    val_paths, val_counts = filter_paths_by_fault(
        val_paths,
        include_faults=args.include_faults,
        exclude_faults=args.exclude_faults,
        strict_fault_names=True,
    )
    if not train_paths or not val_paths:
        raise FileNotFoundError("Fault filtering removed every train or validation sample")
    require_disjoint_splits({"train": train_paths, "validation": val_paths})
    train_paths, missing_train = filter_samples_with_radar_cache(
        train_paths,
        Path(args.radar_root),
        max_delta_ms=args.max_radar_delta_ms,
    )
    val_paths, missing_val = filter_samples_with_radar_cache(
        val_paths,
        Path(args.radar_root),
        max_delta_ms=args.max_radar_delta_ms,
    )
    if missing_train or missing_val:
        print(f"Skipping samples without aligned radar cache: train={len(missing_train)} validation={len(missing_val)}")
    if args.debug_overfit:
        train_paths = train_paths[: max(1, min(2, len(train_paths)))]
        val_paths = val_paths[: max(1, min(2, len(val_paths)))]
    return train_paths, val_paths, train_counts, val_counts


def build_optimizer_scheduler(model, args):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        args.warmup_epochs,
        args.learning_rate,
        args.min_learning_rate,
        args.plateau_factor,
        args.plateau_patience,
        args.plateau_threshold,
    )
    return optimizer, scheduler


def assert_finite_tensor(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")


def gpu_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

