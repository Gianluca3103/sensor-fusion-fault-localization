from pathlib import Path
import math

def require_directory(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def require_positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive number, got {value}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive number, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be positive, got {value}")
    return value


def require_range(min_value, max_value, label):
    if isinstance(min_value, bool) or isinstance(max_value, bool):
        raise ValueError(f"{label} bounds must be numeric")
    try:
        minimum = float(min_value)
        maximum = float(max_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} bounds must be numeric, got {min_value!r}, {max_value!r}"
        ) from exc
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError(f"{label} min must be smaller than max, got {min_value} >= {max_value}")


def require_non_negative(value, label):
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


def validate_generation_args(args):
    require_directory(args.data_root, "K-Radar data root")
    require_positive(args.num_samples, "num_samples")
    require_positive(args.grid_size, "grid_size")
    require_positive(args.resolution, "resolution")
    require_positive(args.num_workers, "num_workers")
    require_positive(args.source_batch_size, "source_batch_size")
    require_positive(args.observability_num_z_bins, "observability_num_z_bins")
    require_positive(
        args.observability_ray_support_tau,
        "observability_ray_support_tau",
    )
    if int(args.observability_num_z_bins) != args.observability_num_z_bins:
        raise ValueError("observability_num_z_bins must be an integer")
    require_positive(args.movement_tolerance_m, "movement_tolerance_m")
    require_non_negative(args.seed, "--seed")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be less than 1.0 so a test split remains.")
    require_range(args.x_min, args.x_max, "x range")
    require_range(args.y_min, args.y_max, "y range")
    require_range(args.min_range, args.max_range, "point range")
    if args.frames and args.temporal_split:
        raise ValueError(
            "--frames cannot be combined with --temporal-split because frame "
            "indexes are global while temporal splits are computed per folder."
        )
    if args.fault_plan and (args.faults or args.severities):
        raise ValueError(
            "--fault-plan cannot be combined with --faults or --severities."
        )
