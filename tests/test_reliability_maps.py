from pathlib import Path
import sys
import unittest

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "Fault_Localization_Model"
sys.path.insert(0, str(MODEL_DIR))

from create_grid_reliability_heatmaps import make_reliability_maps, point_counts_grid


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
        faulty_missing_only = clean[:3].copy()
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

        missing_only = make_reliability_maps(clean, faulty_missing_only, 0, 2, 0, 2, 1, 1)
        missing_plus_added = make_reliability_maps(clean, faulty_missing_plus_added, 0, 2, 0, 2, 1, 1)

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

    def test_point_counts_grid_uses_expected_shape(self):
        points = np.array([[1.0, -1.0, 0.0, 1.0], [9.9, 4.9, 0.0, 1.0]], dtype=np.float32)
        counts = point_counts_grid(points, 0, 10, -5, 5, 10, 10)
        self.assertEqual(counts.shape, (10, 10))
        self.assertEqual(float(counts.sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
