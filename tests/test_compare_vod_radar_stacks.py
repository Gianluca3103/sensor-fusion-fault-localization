import unittest

import numpy as np

from scripts.compare_vod_radar_stacks import _metric_counts, _metrics


class CompareVoDRadarStacksTests(unittest.TestCase):
    def test_exact_metrics_use_standard_occupancy_intersection(self):
        prediction = np.zeros((3, 3), dtype=bool)
        target = np.zeros((3, 3), dtype=bool)
        prediction[1, 1:] = True
        target[1, :2] = True

        metrics = _metrics(
            _metric_counts(
                prediction,
                target,
                tolerance_m=0.0,
                resolution=0.2,
            )
        )

        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)

    def test_tolerance_matches_one_cell_offset_bidirectionally(self):
        prediction = np.zeros((3, 3), dtype=bool)
        target = np.zeros((3, 3), dtype=bool)
        prediction[1, 2] = True
        target[1, 1] = True

        within = _metrics(
            _metric_counts(
                prediction,
                target,
                tolerance_m=0.2,
                resolution=0.2,
            )
        )
        outside = _metrics(
            _metric_counts(
                prediction,
                target,
                tolerance_m=0.19,
                resolution=0.2,
            )
        )

        self.assertEqual(within["iou"], 1.0)
        self.assertEqual(within["f1"], 1.0)
        self.assertEqual(outside["iou"], 0.0)
        self.assertEqual(outside["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
