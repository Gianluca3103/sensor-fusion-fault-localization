from pathlib import Path
import sys
import unittest

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "Fault_Localization_Model"
sys.path.insert(0, str(MODEL_DIR))

from create_grid_reliability_heatmaps import (
    make_reliability_maps,
    mark_bev_point_statuses,
    point_counts_grid,
)


class ReliabilityMapTests(unittest.TestCase):
    def test_missing_and_wrong_added_do_not_cancel(self):
        clean = np.array(
            [
                [1.05, 1.05, 0.0, 1.0],
                [1.10, 1.05, 0.0, 1.0],
                [1.15, 1.05, 0.0, 1.0],
                [1.20, 1.05, 0.0, 1.0],
                [1.25, 1.05, 0.0, 1.0],
                [1.30, 1.05, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        clean_ids = np.arange(len(clean), dtype=np.int64)
        faulty_missing_only = clean[:3].copy()
        missing_only_point_ids = clean_ids[:3].copy()
        missing_only_source_ids = clean_ids[:3].copy()
        faulty_missing_plus_added = np.vstack(
            [
                clean[:3],
                np.array(
                    [
                        [1.05, 1.80, 0.0, 1.0],
                        [1.10, 1.80, 0.0, 1.0],
                        [1.15, 1.80, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        missing_plus_added_point_ids = np.array([0, 1, 2, 6, 7, 8], dtype=np.int64)
        missing_plus_added_source_ids = np.array([0, 1, 2, -1, -1, -1], dtype=np.int64)

        missing_only = make_reliability_maps(
            clean,
            clean_ids,
            faulty_missing_only,
            missing_only_point_ids,
            missing_only_source_ids,
            0.05,
            0,
            2,
            0,
            2,
            1,
            1,
        )
        missing_plus_added = make_reliability_maps(
            clean,
            clean_ids,
            faulty_missing_plus_added,
            missing_plus_added_point_ids,
            missing_plus_added_source_ids,
            0.05,
            0,
            2,
            0,
            2,
            1,
            1,
        )

        self.assertEqual(float(missing_only["missing_faulty_counts"][0, 0]), 3.0)
        self.assertEqual(float(missing_only["added_faulty_counts"][0, 0]), 0.0)
        self.assertEqual(float(missing_plus_added["missing_faulty_counts"][0, 0]), 3.0)
        self.assertEqual(float(missing_plus_added["added_faulty_counts"][0, 0]), 3.0)
        self.assertEqual(float(missing_plus_added["faulty_point_counts"][0, 0]), 6.0)
        self.assertAlmostEqual(
            float(missing_plus_added["reliability_map"][0, 0]),
            3.0 / (3.0 + 6.0),
        )
        self.assertGreater(
            float(missing_plus_added["fault_heatmap"][0, 0]),
            float(missing_only["fault_heatmap"][0, 0]),
        )

    def test_point_movement_uses_five_centimeter_tolerance(self):
        clean = np.array(
            [[1.0, 1.0, 0.0, 1.0], [1.5, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        faulty = clean.copy()
        faulty[0, 0] += 0.04
        faulty[1, 0] += 0.06
        ids = np.arange(2, dtype=np.int64)

        maps = make_reliability_maps(
            clean,
            ids,
            faulty,
            ids.copy(),
            ids.copy(),
            0.05,
            0,
            2,
            0,
            2,
            1,
            1,
        )

        self.assertEqual(float(maps["clean_point_counts"][0, 0]), 1.0)
        self.assertEqual(float(maps["moved_faulty_counts"][0, 0]), 1.0)
        self.assertEqual(float(maps["missing_faulty_counts"][0, 0]), 0.0)
        self.assertAlmostEqual(float(maps["reliability_map"][0, 0]), 0.5)
        np.testing.assert_array_equal(maps["moved_source_ids"], np.array([1]))
        np.testing.assert_array_equal(maps["clean_point_status"], np.array([0, 2], dtype=np.int8))
        np.testing.assert_array_equal(maps["faulty_point_status"], np.array([0, 2], dtype=np.int8))

    def test_synthetic_replacement_is_missing_plus_added(self):
        clean = np.array([[1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        faulty = np.array([[1.2, 1.0, 0.0, 1.0]], dtype=np.float32)

        maps = make_reliability_maps(
            clean,
            np.array([0]),
            faulty,
            np.array([1]),
            np.array([-1]),
            0.05,
            0,
            2,
            0,
            2,
            1,
            1,
        )

        self.assertEqual(float(maps["missing_faulty_counts"][0, 0]), 1.0)
        self.assertEqual(float(maps["added_faulty_counts"][0, 0]), 1.0)
        self.assertEqual(float(maps["moved_faulty_counts"][0, 0]), 0.0)
        self.assertEqual(float(maps["reliability_map"][0, 0]), 0.0)
        np.testing.assert_array_equal(maps["clean_point_status"], np.array([1], dtype=np.int8))
        np.testing.assert_array_equal(maps["faulty_point_status"], np.array([3], dtype=np.int8))

    def test_point_counts_grid_uses_expected_shape(self):
        points = np.array([[1.0, -1.0, 0.0, 1.0], [9.9, 4.9, 0.0, 1.0]], dtype=np.float32)
        counts = point_counts_grid(points, 0, 10, -5, 5, 10, 10)
        self.assertEqual(counts.shape, (10, 10))
        self.assertEqual(float(counts.sum()), 2.0)

    def test_status_overlay_uses_only_id_based_fault_classes(self):
        clean = np.array([[0.25, 0.25, 0.0, 1.0]], dtype=np.float32)
        faulty = np.array(
            [[1.0, 1.0, 0.0, 1.0], [1.75, 1.75, 0.0, 1.0]],
            dtype=np.float32,
        )
        overlay, counts = mark_bev_point_statuses(
            clean,
            faulty,
            np.array([1], dtype=np.int8),
            np.array([2, 3], dtype=np.int8),
            np.zeros((20, 20, 3), dtype=np.uint8),
            0,
            2,
            0,
            2,
        )

        self.assertEqual(counts["missing_points_marked"], 1)
        self.assertEqual(counts["moved_points_marked"], 1)
        self.assertEqual(counts["added_points_marked"], 1)
        self.assertTrue(np.any(np.all(overlay == np.array([255, 80, 0]), axis=2)))
        self.assertTrue(np.any(np.all(overlay == np.array([0, 255, 255]), axis=2)))
        self.assertTrue(np.any(np.all(overlay == np.array([255, 255, 0]), axis=2)))


if __name__ == "__main__":
    unittest.main()
