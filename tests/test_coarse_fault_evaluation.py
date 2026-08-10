import unittest

from models.reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    summarize_records,
)


class CoarseFaultEvaluationTests(unittest.TestCase):
    def test_summary_reports_macro_and_micro_occupancy(self):
        records = [
            {
                "sample_path": "a.npz",
                "fault": "fog_sim",
                "severity": 3,
                "fault_group": "fog_sim_s3",
                "sequence_id": "1",
                "frame_id": "1",
                "repair_cells": 10,
                "coarse_occupancy_exact_iou": 0.5,
                "faulty_occupancy_exact_iou": 0.25,
                "coarse_tp": 3,
                "coarse_fp": 1,
                "coarse_fn": 1,
                "coarse_tn": 5,
                "faulty_tp": 2,
                "faulty_fp": 2,
                "faulty_fn": 2,
                "faulty_tn": 4,
            },
            {
                "sample_path": "b.npz",
                "fault": "fog_sim",
                "severity": 3,
                "fault_group": "fog_sim_s3",
                "sequence_id": "1",
                "frame_id": "2",
                "repair_cells": 20,
                "coarse_occupancy_exact_iou": 0.75,
                "faulty_occupancy_exact_iou": 0.5,
                "coarse_tp": 6,
                "coarse_fp": 2,
                "coarse_fn": 2,
                "coarse_tn": 10,
                "faulty_tp": 4,
                "faulty_fp": 4,
                "faulty_fn": 4,
                "faulty_tn": 8,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["repair_cells"], 30)
        self.assertAlmostEqual(
            summary["macro/coarse_occupancy_exact_iou"], 0.625
        )
        self.assertAlmostEqual(summary["micro/coarse_iou"], 9 / 15)
        self.assertAlmostEqual(summary["micro/faulty_iou"], 6 / 18)
        self.assertAlmostEqual(summary["micro/iou_improvement"], 9 / 15 - 6 / 18)


if __name__ == "__main__":
    unittest.main()
