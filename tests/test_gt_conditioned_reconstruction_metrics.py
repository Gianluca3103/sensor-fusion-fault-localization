import unittest

import numpy as np

from models.two_stage_reconstruction_head.gt_conditioned_reconstruction_metrics import (
    evaluate_bev_condition,
    rasterize_rotated_box,
)
from models.two_stage_reconstruction_head.object_detection.annotations import (
    RotatedBEVBox,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


class GTConditionedReconstructionMetricTests(unittest.TestCase):
    def setUp(self):
        self.geometry = BEVGridGeometry(
            x_min=0.0,
            x_max=2.0,
            y_min=-1.0,
            y_max=1.0,
            height=10,
            width=10,
        )

    def test_rotated_box_is_rasterized_at_cell_centres(self):
        box = RotatedBEVBox(
            class_name="Car",
            x=1.0,
            y=0.0,
            length=0.8,
            width=0.4,
            yaw=0.0,
        )
        mask = rasterize_rotated_box(box, self.geometry)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(int(mask.sum()), 8)

    def test_exact_and_tolerant_metrics_are_detector_independent(self):
        clean = np.zeros((3, 10, 10), dtype=np.float32)
        condition = np.zeros_like(clean)
        clean[0, 5, 5] = 1.0
        clean[1, 5, 5] = 0.8
        clean[2, 5, 5] = 0.75
        condition[0, 5, 6] = 1.0
        condition[1, 5, 5] = 0.6
        condition[2, 5, 5] = 0.5
        scope = np.ones((10, 10), dtype=bool)

        metrics = evaluate_bev_condition(
            clean,
            condition,
            scope,
            resolution_m=0.2,
            tolerances_m=(0.2, 0.5),
        )

        self.assertEqual(metrics["exact_iou"], 0.0)
        self.assertEqual(metrics["tolerant_0_2m_precision"], 1.0)
        self.assertEqual(metrics["tolerant_0_2m_recall"], 1.0)
        self.assertEqual(metrics["tolerant_0_5m_iou"], 1.0)
        self.assertAlmostEqual(metrics["density_mae_clean_support"], 0.2)
        self.assertAlmostEqual(metrics["height_mae_m_clean_support"], 2.0)

    def test_empty_scope_is_reported_without_nan(self):
        bev = np.zeros((3, 10, 10), dtype=np.float32)
        metrics = evaluate_bev_condition(
            bev,
            bev,
            np.zeros((10, 10), dtype=bool),
            resolution_m=0.2,
        )
        self.assertEqual(metrics["scope_cells"], 0)
        self.assertEqual(metrics["exact_iou"], 0.0)
        self.assertIsNone(metrics["symmetric_chamfer_m"])
        self.assertIsNone(metrics["height_mae_m_clean_support"])


if __name__ == "__main__":
    unittest.main()
