"""Small disk cache for deterministic Fault Selector masks."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from .fault_selector import FaultSelector, FaultSelectorConfig


MASK_SIZE = (320, 320)
CACHE_VERSION = 8
MASK_NAMES = (
    "reconstruction_mask",
    "halo_mask",
    "healthy_context_mask",
)


class InvalidSelectorCacheError(RuntimeError):
    pass


def selector_cache_root(data_root: str | Path) -> Path:
    data_root = Path(data_root)
    return data_root.parent / f"{data_root.name}_fault_selector_cache"


def selector_cache_path(sample_path: str | Path, data_root: str | Path) -> Path:
    sample_path = Path(sample_path)
    try:
        relative = sample_path.relative_to(Path(data_root))
    except ValueError as exc:
        raise ValueError(f"{sample_path} is not under {data_root}") from exc
    return selector_cache_root(data_root) / relative


def _config_json(config: FaultSelectorConfig) -> str:
    payload = asdict(config)
    # Preserve compatibility with version-8 caches created before optional
    # secondary boxes existed. Explicitly enabled multi-box selectors retain
    # both fields in their cache identity.
    if config.max_secondary_repair_boxes == 0:
        payload.pop("max_secondary_repair_boxes")
        payload.pop("min_secondary_repair_cells")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_selector_inputs(sample_path: str | Path) -> dict[str, np.ndarray]:
    """Load native-resolution NumPy evidence consumed by ``FaultSelector``."""
    names = (
        "fault_heatmap",
        "reliability_map",
        "faulty_counts",
        "added_faulty_counts",
        "missing_faulty_counts",
        "moved_faulty_counts",
    )
    with np.load(sample_path, allow_pickle=False) as sample:
        arrays = {
            name: np.asarray(sample[name], dtype=np.float32)
            for name in names
        }
        stored_support = (
            np.asarray(sample["valid_support_mask"], dtype=np.float32)
            if "valid_support_mask" in sample.files
            else None
        )
    if stored_support is None:
        raise ValueError(
            f"{sample_path} does not contain valid_support_mask; regenerate the "
            "View-of-Delft sample before building selector caches"
        )
    if stored_support.shape != arrays["fault_heatmap"].shape:
        raise ValueError(
            "valid_support_mask must align with fault_heatmap; got "
            f"{stored_support.shape} and {arrays['fault_heatmap'].shape}"
        )
    arrays["valid_support_mask"] = stored_support
    return arrays


def _resize_binary_mask(mask: np.ndarray) -> np.ndarray:
    source_height, source_width = mask.shape
    target_height, target_width = MASK_SIZE
    rows = np.minimum(
        (np.arange(target_height) * source_height // target_height),
        source_height - 1,
    )
    cols = np.minimum(
        (np.arange(target_width) * source_width // target_width),
        source_width - 1,
    )
    return mask[np.ix_(rows, cols)]


def _load_faulty_occupancy(sample_path: str | Path) -> np.ndarray:
    with np.load(sample_path, allow_pickle=False) as sample:
        density = np.asarray(sample["faulty_density"])
    return density > 0


def load_selector_cache(
    path: str | Path,
    config: FaultSelectorConfig,
) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise InvalidSelectorCacheError(
            f"Fault Selector cache is missing: {path}. Run "
            "python -m models.reconstruction_head.cache_fault_selector_masks first."
        )
    try:
        with np.load(path, allow_pickle=False) as cache:
            if (
                "cache_version" not in cache.files
                or int(cache["cache_version"].item()) != CACHE_VERSION
            ):
                raise InvalidSelectorCacheError(
                    f"Fault Selector cache is stale: {path}. Regenerate it."
                )
            if str(cache["selector_config"].item()) != _config_json(config):
                raise InvalidSelectorCacheError(
                    f"Fault Selector cache is stale: {path}. Regenerate it."
                )
            masks = {
                name: np.asarray(cache[name], dtype=np.uint8)
                for name in MASK_NAMES
            }
    except InvalidSelectorCacheError:
        raise
    except Exception as exc:
        raise InvalidSelectorCacheError(
            f"Cannot load Fault Selector cache {path}: {exc}"
        ) from exc

    if any(mask.shape != MASK_SIZE for mask in masks.values()):
        raise InvalidSelectorCacheError(
            f"Fault Selector masks in {path} must have shape {MASK_SIZE}"
        )
    return masks


def build_selector_cache_entry(
    sample_path: str | Path,
    data_root: str | Path,
    config: FaultSelectorConfig,
) -> str:
    path = selector_cache_path(sample_path, data_root)
    if path.is_file():
        try:
            load_selector_cache(path, config)
            return "cached"
        except InvalidSelectorCacheError:
            pass

    inputs = load_selector_inputs(sample_path)
    selection = FaultSelector(config).select(
        inputs.pop("fault_heatmap"),
        **inputs,
    )
    masks = {
        name: _resize_binary_mask(
            getattr(selection, name),
        ).astype(np.uint8)
        for name in MASK_NAMES
    }
    masks["healthy_context_mask"] &= _load_faulty_occupancy(sample_path).astype(
        np.uint8
    )
    masks["healthy_context_mask"] &= masks["halo_mask"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                cache_version=np.asarray(CACHE_VERSION, dtype=np.int64),
                selector_config=np.asarray(_config_json(config)),
                **masks,
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return "created"
