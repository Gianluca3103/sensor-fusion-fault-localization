import json
from itertools import combinations
from pathlib import Path

import numpy as np


class InvalidSampleError(ValueError):
    """Raised when a generated sample violates the training-data contract."""


def load_sample_metadata(path):
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data:
                raise KeyError("metadata_json is missing")
            metadata = json.loads(str(data["metadata_json"]))
    except Exception as exc:
        raise InvalidSampleError(f"Cannot read metadata from {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise InvalidSampleError(f"metadata_json in {path} must decode to an object")
    return metadata


def load_fault_name(path):
    return str(load_sample_metadata(path).get("fault", "unknown"))


def filter_paths_by_fault(
    paths,
    include_faults=None,
    exclude_faults=None,
    *,
    strict_fault_names=False,
):
    include = {fault.strip() for fault in include_faults or [] if fault.strip()}
    exclude = {fault.strip() for fault in exclude_faults or [] if fault.strip()}
    overlap = include & exclude
    if overlap:
        raise ValueError(
            "Fault names cannot be both included and excluded: "
            + ", ".join(sorted(overlap))
        )

    selected = []
    counts = {}
    for path in paths:
        fault = load_fault_name(path)
        counts[fault] = counts.get(fault, 0) + 1
        if include and fault not in include:
            continue
        if fault in exclude:
            continue
        selected.append(path)

    if strict_fault_names:
        requested = include | exclude
        unknown = requested - set(counts)
        if unknown:
            raise ValueError(
                "Requested fault names are absent from the dataset: "
                + ", ".join(sorted(unknown))
            )
    return selected, counts


def sample_frame_identity(metadata):
    """Return the physical source-frame identity used for split leakage checks."""
    relative = str(metadata.get("source_relative_path", "")).strip()
    if relative:
        return ("source_relative_path", relative.replace("\\", "/").casefold())

    timestamp = str(metadata.get("timestamp", "")).strip()
    scene = str(metadata.get("scene") or metadata.get("day") or "").strip()
    session = str(metadata.get("session", "")).strip()
    if timestamp and scene:
        return ("scene_timestamp", scene.casefold(), session.casefold(), timestamp)

    raise InvalidSampleError(
        "Sample metadata must contain source_relative_path or scene/day plus timestamp"
    )


def sample_data_contract(metadata):
    """Return fields that must remain identical throughout one experiment."""
    x_range = metadata.get("x_range")
    y_range = metadata.get("y_range")
    if x_range is not None:
        x_range = tuple(float(value) for value in x_range)
        if (
            len(x_range) != 2
            or not np.isfinite(x_range).all()
            or x_range[0] >= x_range[1]
        ):
            raise InvalidSampleError(f"Invalid x_range in sample metadata: {x_range}")
    if y_range is not None:
        y_range = tuple(float(value) for value in y_range)
        if (
            len(y_range) != 2
            or not np.isfinite(y_range).all()
            or y_range[0] >= y_range[1]
        ):
            raise InvalidSampleError(f"Invalid y_range in sample metadata: {y_range}")

    numeric_fields = {}
    for key in (
        "resolution",
        "movement_tolerance_m",
        "grid_size",
        "image_height",
        "image_width",
        "max_range",
    ):
        value = metadata.get(key)
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise InvalidSampleError(
                    f"Invalid positive metadata field {key}={value}"
                )
        numeric_fields[key] = value

    min_range = metadata.get("min_range")
    if min_range is not None:
        min_range = float(min_range)
        if not np.isfinite(min_range) or min_range < 0.0:
            raise InvalidSampleError(
                f"Invalid non-negative metadata field min_range={min_range}"
            )
    if (
        min_range is not None
        and numeric_fields["max_range"] is not None
        and min_range >= numeric_fields["max_range"]
    ):
        raise InvalidSampleError(
            "Sample metadata max_range must exceed min_range"
        )
    fog_noise = metadata.get("fog_noise")
    if fog_noise is not None:
        fog_noise = float(fog_noise)
        if not np.isfinite(fog_noise) or fog_noise < 0.0:
            raise InvalidSampleError(
                f"Invalid non-negative metadata field fog_noise={fog_noise}"
            )

    return (
        str(metadata.get("dataset", "")).casefold(),
        int(metadata.get("generator_version", 1)),
        str(metadata.get("ground_truth_method", "")).casefold(),
        x_range,
        y_range,
        numeric_fields["resolution"],
        numeric_fields["movement_tolerance_m"],
        numeric_fields["grid_size"],
        numeric_fields["image_height"],
        numeric_fields["image_width"],
        min_range,
        numeric_fields["max_range"],
        fog_noise,
    )


def require_disjoint_splits(splits):
    """Fail on source-frame leakage or incompatible split data contracts."""
    identities = {}
    contracts = {}
    for split_name, paths in splits.items():
        split_identities = {}
        split_contracts = {}
        for path in paths:
            metadata = load_sample_metadata(path)
            identity = sample_frame_identity(metadata)
            split_identities.setdefault(identity, Path(path))
            split_contracts.setdefault(sample_data_contract(metadata), Path(path))
        identities[split_name] = split_identities
        contracts[split_name] = split_contracts
        if len(split_contracts) > 1:
            examples = list(split_contracts.values())[:2]
            raise ValueError(
                f"Dataset contract mismatch inside {split_name}: "
                f"{examples[0].name} and {examples[1].name} use different "
                "generator versions, target definitions, or BEV geometry."
            )

    failures = []
    for (left_name, left), (right_name, right) in combinations(identities.items(), 2):
        overlap = set(left) & set(right)
        if overlap:
            example = next(iter(overlap))
            failures.append(
                f"{left_name}/{right_name}: {len(overlap)} overlapping source frames "
                f"(example: {left[example].name} and {right[example].name})"
            )
    if failures:
        raise ValueError(
            "Dataset split leakage detected. A physical frame must belong to only one "
            "split:\n  " + "\n  ".join(failures)
        )

    nonempty_contracts = {
        split_name: next(iter(split_contracts))
        for split_name, split_contracts in contracts.items()
        if split_contracts
    }
    if len(set(nonempty_contracts.values())) > 1:
        details = ", ".join(
            f"{name}={contract}" for name, contract in nonempty_contracts.items()
        )
        raise ValueError(
            "Dataset contract mismatch across splits. Generator version, target "
            f"definition, and BEV geometry must agree: {details}"
        )


def validate_rgb_array(array, *, name, path):
    array = np.asarray(array)
    if (
        array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] < 1
        or array.shape[1] < 1
    ):
        raise InvalidSampleError(
            f"{name} in {path} must have shape [H,W,3], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidSampleError(f"{name} in {path} contains non-finite values")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 255.0):
        raise InvalidSampleError(f"{name} in {path} must be in the range [0,255]")
    return array


def validate_heatmap_array(array, *, path):
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise InvalidSampleError(
            f"fault_heatmap in {path} must have shape [H,W], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidSampleError(f"fault_heatmap in {path} contains non-finite values")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise InvalidSampleError(
            f"fault_heatmap in {path} must be in the range [0,1]"
        )
    return array


def validate_radar_array(array, *, path):
    array = np.asarray(array)
    if (
        array.ndim != 3
        or array.shape[0] != 4
        or array.shape[1] < 1
        or array.shape[2] < 1
    ):
        raise InvalidSampleError(
            f"radar_bev in {path} must have shape [4,H,W], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise InvalidSampleError(f"radar_bev in {path} contains non-finite values")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise InvalidSampleError(f"radar_bev in {path} must be in the range [0,1]")
    return array
