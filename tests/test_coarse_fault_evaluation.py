import unittest
import tempfile
from pathlib import Path

import torch

from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _save_comparison,
    summarize_records,
)


class CoarseFaultEvaluationTests(unittest.TestCase):
    def test_comparison_visualization_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "comparison.png"
            clean = torch.zeros(3, 8, 8)
            clean[0, 2:5, 2:5] = 1.0
            faulty = clean.clone()
            radar = torch.zeros(4, 8, 8)
            radar[0, 1:4, 1:4] = 1.0
            coarse = clean.clone()
            coarse[:, 3, 3] = 0.5
            mask = torch.zeros(1, 8, 8)
            mask[:, 1:6, 1:6] = 1.0
            _save_comparison(
                destination,
                clean,
                faulty,
                radar,
                coarse,
                mask,
                {
                    "fault_group": "fog_sim_s3",
                    "sequence_id": "1",
                    "frame_id": "1",
                    "faulty_occupancy_exact_iou": 0.25,
                    "coarse_occupancy_exact_iou": 0.5,
                },
            )
            self.assertTrue(destination.is_file())
            self.assertGreater(destination.stat().st_size, 0)

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

    def test_summary_excludes_selector_rejected_samples_from_metrics(self):
        records = []
        for name, repair_cells, target_cells, tolerant_iou in (
            ("rejected", 0, 4, 1.0),
            ("empty", 10, 0, 1.0),
            ("failure", 10, 4, 0.0),
            ("success", 10, 4, 0.8),
        ):
            records.append(
                {
                    "sample_path": f"{name}.npz",
                    "fault": "fog_sim",
                    "severity": 4,
                    "fault_group": "fog_sim_s4",
                    "sequence_id": "1",
                    "frame_id": name,
                    "repair_cells": repair_cells,
                    "target_occupied_cells": target_cells,
                    "coarse_occupancy_tolerant_0_5m_iou": tolerant_iou,
                    "coarse_tp": 0,
                    "coarse_fp": 0,
                    "coarse_fn": target_cells,
                    "coarse_tn": 10 - target_cells,
                    "faulty_tp": 0,
                    "faulty_fp": 0,
                    "faulty_fn": target_cells,
                    "faulty_tn": 10 - target_cells,
                }
            )

        summary = summarize_records(records)

        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["metric_samples"], 2)
        self.assertEqual(summary["occupancy_metric_samples"], 2)
        self.assertEqual(summary["excluded_empty_target_samples"], 1)
        self.assertEqual(summary["excluded_selector_rejected_samples"], 1)
        self.assertEqual(summary["excluded_metric_samples"], 2)
        self.assertAlmostEqual(
            summary["macro/coarse_occupancy_tolerant_0_5m_iou"],
            0.4,
        )


if __name__ == "__main__":
    unittest.main()
