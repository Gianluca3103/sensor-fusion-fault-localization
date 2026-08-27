import unittest

import numpy as np

from tools.profile_reconstruction_crop_sizes import (
    _crop_dimensions,
    _distribution,
    _summarize,
)


class ReconstructionCropProfileTests(unittest.TestCase):
    def test_crop_dimensions_use_exact_repair_halo_union(self):
        repair = np.zeros((12, 14), dtype=np.uint8)
        halo = np.zeros_like(repair)
        repair[4:7, 5:9] = 1
        halo[2:9, 3:11] = 1

        self.assertEqual(_crop_dimensions(repair, halo), (7, 8))
        self.assertIsNone(_crop_dimensions(np.zeros_like(repair), np.zeros_like(halo)))

    def test_distribution_and_area_are_calculated_over_active_samples(self):
        summary = _summarize([2, 4, 6], [3, 5, 7], total_samples=4)

        self.assertEqual(summary["active_samples"], 3)
        self.assertEqual(summary["empty_samples"], 1)
        self.assertEqual(summary["crop_height_cells"]["median"], 4.0)
        self.assertEqual(summary["crop_width_cells"]["mean"], 5.0)
        self.assertEqual(summary["crop_area_cells"]["maximum"], 42)

    def test_empty_distribution_is_finite(self):
        values = _distribution([])

        self.assertTrue(all(value == 0 for value in values.values()))


if __name__ == "__main__":
    unittest.main()
