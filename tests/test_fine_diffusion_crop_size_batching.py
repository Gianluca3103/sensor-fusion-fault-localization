import unittest

import numpy as np

from models.two_stage_reconstruction_head.diffusion_process.crop_size_batching import (
    CropSizeBatchSampler,
    _crop_shape,
    crop_padding_efficiency,
)


class CropSizeBatchSamplerTests(unittest.TestCase):
    def test_crop_shape_uses_repair_halo_union_and_technical_padding(self):
        repair = np.zeros((320, 320), dtype=np.uint8)
        halo = np.zeros_like(repair)
        repair[20:30, 40:50] = 1
        halo[10:35, 35:60] = 1

        shape = _crop_shape(
            repair,
            halo,
            pad_multiple=8,
            minimum_height=1,
            minimum_width=1,
        )

        self.assertEqual(shape, (32, 32))

    def test_every_sample_appears_exactly_once(self):
        shapes = [(16 + index * 3, 24 + index * 5) for index in range(23)]
        sampler = CropSizeBatchSampler(shapes, 4, bucket_multiple=32, seed=7)

        indices = [index for batch in sampler for index in batch]

        self.assertEqual(sorted(indices), list(range(len(shapes))))
        self.assertEqual(len(sampler), 6)

    def test_epoch_shuffle_is_deterministic_and_changes_order(self):
        shapes = [(64, 64)] * 32
        first = CropSizeBatchSampler(shapes, 4, seed=11)
        second = CropSizeBatchSampler(shapes, 4, seed=11)
        first.set_epoch(3)
        second.set_epoch(3)
        expected = list(first)

        self.assertEqual(expected, list(second))
        first.set_epoch(4)
        self.assertNotEqual(expected, list(first))

    def test_shape_bucketing_reduces_padding_waste(self):
        shapes = (
            [(32, 256), (256, 32), (40, 240), (240, 40)] * 4
            + [(128, 128), (136, 136), (144, 144), (152, 152)] * 4
        )
        sampler = CropSizeBatchSampler(
            shapes, 4, bucket_multiple=32, shuffle=False
        )
        grouped_efficiency = crop_padding_efficiency(list(sampler), shapes)
        mixed_batches = [list(range(start, start + 4)) for start in range(0, 32, 4)]
        mixed_efficiency = crop_padding_efficiency(mixed_batches, shapes)

        self.assertGreater(grouped_efficiency, mixed_efficiency)


if __name__ == "__main__":
    unittest.main()
