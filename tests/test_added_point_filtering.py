import unittest

import numpy as np

from Fault_Localization_Model.create_grid_reliability_heatmaps import (
    faulty_point_keep_mask,
)


class AddedPointFilteringTests(unittest.TestCase):
    def test_option_removes_only_rows_without_clean_source_identity(self):
        range_mask = np.ones(4, dtype=bool)
        overlap_mask = np.ones(4, dtype=bool)
        labels = np.asarray([1, 2, 1, 0], dtype=np.int8)
        source_ids = np.asarray([10, -1, 11, 12], dtype=np.int64)

        unfiltered = faulty_point_keep_mask(
            range_mask,
            overlap_mask,
            labels,
            source_ids,
            remove_added_points=False,
        )
        filtered = faulty_point_keep_mask(
            range_mask,
            overlap_mask,
            labels,
            source_ids,
            remove_added_points=True,
        )

        self.assertTrue(np.array_equal(unfiltered, [True, True, True, False]))
        self.assertTrue(np.array_equal(filtered, [True, False, True, False]))

    def test_range_and_radar_support_still_apply(self):
        keep = faulty_point_keep_mask(
            np.asarray([True, False, True]),
            np.asarray([False, True, True]),
            np.ones(3, dtype=np.int8),
            np.arange(3, dtype=np.int64),
            remove_added_points=True,
        )

        self.assertTrue(np.array_equal(keep, [False, False, True]))


if __name__ == "__main__":
    unittest.main()
