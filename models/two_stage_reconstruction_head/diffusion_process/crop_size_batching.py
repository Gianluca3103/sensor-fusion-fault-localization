"""Shape-aware batching for cropped Fine Diffusion training."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable, Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler

from ..fault_selector_cache import load_selector_cache, selector_cache_path


def _crop_shape(
    reconstruction_mask: np.ndarray,
    halo_mask: np.ndarray,
    *,
    pad_multiple: int,
    minimum_height: int,
    minimum_width: int,
) -> tuple[int, int]:
    """Return the technical padded repair+halo bounding-box shape."""

    active = np.maximum(reconstruction_mask, halo_mask) > 0
    rows, columns = np.nonzero(active)
    if rows.size:
        height = int(rows.max() - rows.min() + 1)
        width = int(columns.max() - columns.min() + 1)
    else:
        height = width = 1
    height = min(active.shape[0], max(height, int(minimum_height)))
    width = min(active.shape[1], max(width, int(minimum_width)))
    return (
        math.ceil(height / pad_multiple) * pad_multiple,
        math.ceil(width / pad_multiple) * pad_multiple,
    )


def dataset_repair_halo_crop_shapes(
    dataset,
    *,
    pad_multiple: int,
    minimum_height: int = 1,
    minimum_width: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Read cached masks once and profile each dataset item's local crop shape."""

    shapes = []
    for sample_path in dataset.sample_paths:
        cached = load_selector_cache(
            selector_cache_path(sample_path, dataset.data_root),
            dataset.selector_config,
        )
        shapes.append(
            _crop_shape(
                cached["reconstruction_mask"],
                cached["halo_mask"],
                pad_multiple=pad_multiple,
                minimum_height=minimum_height,
                minimum_width=minimum_width,
            )
        )
    return tuple(shapes)


class CropSizeBatchSampler(Sampler[list[int]]):
    """Batch nearby 2-D crop shapes while retaining per-epoch shuffling.

    Full batches are formed inside quantized height/width buckets. Sparse
    bucket tails are globally combined in padded-area order so every sample is
    retained without producing many undersized batches.
    """

    def __init__(
        self,
        crop_shapes: Sequence[tuple[int, int]],
        batch_size: int,
        *,
        bucket_multiple: int = 32,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if not crop_shapes:
            raise ValueError("crop_shapes must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if bucket_multiple < 1:
            raise ValueError("bucket_multiple must be positive")
        self.crop_shapes = tuple(
            (int(height), int(width)) for height, width in crop_shapes
        )
        if any(height < 1 or width < 1 for height, width in self.crop_shapes):
            raise ValueError("crop dimensions must be positive")
        self.batch_size = int(batch_size)
        self.bucket_multiple = int(bucket_multiple)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _bucket_key(self, index: int) -> tuple[int, int]:
        height, width = self.crop_shapes[index]
        return (
            math.ceil(height / self.bucket_multiple),
            math.ceil(width / self.bucket_multiple),
        )

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in range(len(self.crop_shapes)):
            buckets[self._bucket_key(index)].append(index)

        batches: list[list[int]] = []
        leftovers: list[int] = []
        for key in sorted(buckets):
            indices = buckets[key]
            if self.shuffle:
                rng.shuffle(indices)
            full_count = len(indices) // self.batch_size
            for offset in range(full_count):
                start = offset * self.batch_size
                batches.append(indices[start : start + self.batch_size])
            leftovers.extend(indices[full_count * self.batch_size :])

        # Preserve shape locality for bucket tails instead of emitting one
        # small batch per sparsely populated shape bucket.
        leftovers.sort(
            key=lambda index: (
                self.crop_shapes[index][0] * self.crop_shapes[index][1],
                self.crop_shapes[index][0],
                self.crop_shapes[index][1],
            )
        )
        for start in range(0, len(leftovers), self.batch_size):
            batch = leftovers[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                if self.shuffle:
                    rng.shuffle(batch)
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        count = len(self.crop_shapes)
        if self.drop_last:
            return count // self.batch_size
        return math.ceil(count / self.batch_size)


def crop_padding_efficiency(
    batches: Iterable[Sequence[int]],
    crop_shapes: Sequence[tuple[int, int]],
) -> float:
    """Ratio of real per-sample crop cells to batch-padded crop cells."""

    real_cells = 0
    padded_cells = 0
    for batch in batches:
        if not batch:
            continue
        heights = [crop_shapes[index][0] for index in batch]
        widths = [crop_shapes[index][1] for index in batch]
        real_cells += sum(height * width for height, width in zip(heights, widths))
        padded_cells += len(batch) * max(heights) * max(widths)
    return real_cells / max(padded_cells, 1)
