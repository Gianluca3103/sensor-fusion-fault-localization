from pathlib import Path
import json
import logging
import sys


def load_json_config(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def require_positive(value, label):
    if value <= 0:
        raise ValueError(f"{label} must be positive, got {value}")
    return value


def require_range(min_value, max_value, label):
    if min_value >= max_value:
        raise ValueError(f"{label} min must be smaller than max, got {min_value} >= {max_value}")
