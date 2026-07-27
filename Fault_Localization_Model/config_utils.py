from pathlib import Path
import json
import logging
import math
import sys


def load_json_config(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config root in {path} must be a JSON object")
    return config


def config_get(config, dotted_key, default=None):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def setup_logging(level_name="INFO"):
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


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
