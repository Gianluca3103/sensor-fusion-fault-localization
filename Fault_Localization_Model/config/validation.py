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
