"""Compact reporting stays readable while detailed validation remains available."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.train_range_view_reconstruction import (
    FAULT_FIELDS, SUMMARY_FIELDS, _append_csv, _format_epoch_summary,
    _summary_rows, _write_progress,
)
from scripts.watch_range_view_training import _render


class RangeViewTrainingReportingTests(unittest.TestCase):
    def test_epoch_summary_and_csv(self) -> None:
        metrics = {
            "faulty_f1_at_0.2m": 0.42,
            "reconstructed_precision_at_0.2m": 0.22,
            "reconstructed_recall_at_0.2m": 0.39,
            "reconstructed_f1_at_0.2m": 0.27,
            "reconstructed_iou_at_0.2m": 0.17,
            "net_f1_improvement": -0.15,
            "addition_precision": 0.02,
            "addition_recall": 0.12,
            "generated_hallucination_rate": 0.98,
            "generated_count": 53746.16,
        }
        fault_metrics = {
            "count": 34, "faulty_f1_at_0.2m": 0.61,
            "reconstructed_f1_at_0.2m": 0.38,
            "net_f1_improvement": -0.23,
            "addition_precision": 0.02, "addition_recall": 0.13,
            "generated_count": 48867.59,
        }
        record = {
            "epoch": 2, "seconds": 79.2,
            "train": {"loss": 12.23, "add_loss": 1.26,
                      "range_loss": 10.69, "delete_loss": 0.24,
                      "free_space_loss": 0.41},
            "val_overall": metrics,
            "val_by_fault": {"fov_filter": fault_metrics},
        }
        summary, faults = _summary_rows(record)
        self.assertEqual(summary["val_net_f1_improvement"], -0.15)
        self.assertEqual(faults[0]["fault"], "fov_filter")
        message = _format_epoch_summary(summary, faults, 10)
        self.assertIn("Epoch 2/10", message)
        self.assertIn("change -0.1500", message)
        self.assertIn("generated 53,746 per sample", message)
        self.assertNotIn("chamfer", message)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _append_csv(root / "summary.csv", SUMMARY_FIELDS, [summary])
            _append_csv(root / "summary.csv", SUMMARY_FIELDS, [summary])
            _append_csv(root / "fault_summary.csv", FAULT_FIELDS, faults)
            with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["val_reconstructed_f1_at_0_2m"]), 0.27)
            self.assertEqual((root / "summary.csv").read_text().count("epoch,seconds"), 1)
            _write_progress(root / "progress.json", epoch=2, phase="train",
                            completed=25, total=1750, loss=12.23)
            progress = json.loads((root / "progress.json").read_text())
            self.assertEqual(progress["completed_batches"], 25)
            self.assertIn("25/1750", _render(progress))


if __name__ == "__main__":
    unittest.main()
