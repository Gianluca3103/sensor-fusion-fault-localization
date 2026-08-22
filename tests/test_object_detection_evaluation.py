from pathlib import Path
import tempfile
import unittest

import torch

from models.two_stage_reconstruction_head.object_detection import (
    BEVDetectorConfig,
    LightweightBEVDetector,
    RotatedBEVBox,
    VODAnnotationLoader,
    decode_detections,
    evaluate_detection_conditions,
    make_detection_targets,
    rotated_box_iou,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


def _geometry():
    return BEVGridGeometry(0.0, 64.0, -32.0, 32.0)


def _box(x=10.0, y=0.0, confidence=1.0):
    return RotatedBEVBox("Car", x, y, 4.0, 2.0, 0.0, confidence=confidence)


class ObjectDetectionEvaluationTests(unittest.TestCase):
    def test_rotated_iou_identity_and_disjoint(self):
        self.assertAlmostEqual(rotated_box_iou(_box(), _box()), 1.0)
        self.assertAlmostEqual(rotated_box_iou(_box(), _box(x=20.0)), 0.0)

    def test_vod_annotation_loader_uses_kitti_dimensions_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            public = tmp_path / "view_of_delft_PUBLIC" / "lidar" / "training"
            (public / "label_2").mkdir(parents=True)
            (public / "calib").mkdir()
            (public / "calib" / "00001.txt").write_text(
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8"
            )
            (public / "label_2" / "00001.txt").write_text(
                "Car 0 0 0 0 0 1 1 1.5 2.0 4.0 10.0 0.0 1.0 0.0\n"
                "bicycle 0 0 0 0 0 1 1 1 1 2 8 0 1 0\n",
                encoding="utf-8",
            )
            boxes = VODAnnotationLoader(tmp_path, _geometry()).load("00001")
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].length, 4.0)
        self.assertEqual(boxes[0].width, 2.0)
        self.assertEqual(boxes[0].height, 1.5)
        self.assertAlmostEqual(boxes[0].x, 10.0)
        self.assertAlmostEqual(boxes[0].z, 1.0)

    def test_detector_targets_and_decode_share_geometry(self):
        config = BEVDetectorConfig(output_stride=2, score_threshold=0.5, top_k=5)
        geometry = _geometry()
        targets = make_detection_targets(
            [[_box()]], ("Car",), geometry, (160, 160), 2, device=torch.device("cpu")
        )
        outputs = {
            "heatmap_logits": torch.logit(targets["heatmap"].clamp(1e-4, 1 - 1e-4)),
            "box_regression": targets["regression"],
        }
        decoded = decode_detections(outputs, ("Car",), geometry, config)[0]
        self.assertTrue(decoded)
        self.assertAlmostEqual(decoded[0].x, 10.0, delta=0.25)
        self.assertAlmostEqual(decoded[0].y, 0.0, delta=0.25)
        self.assertAlmostEqual(decoded[0].length, 4.0)

    def test_recovery_metrics_use_same_gt_identity_across_conditions(self):
        gt = [_box()]
        prediction = [_box(confidence=0.9)]
        records = [
            {
                "frame_id": "00001",
                "ground_truth": gt,
                "predictions": {
                    "clean": prediction,
                    "faulty": [],
                    "coarse": [],
                    "fine": prediction,
                },
            }
        ]
        summary, frame_rows, object_rows = evaluate_detection_conditions(
            records, ("Car",), 0.5
        )
        self.assertEqual(summary["object_recovery"]["lost_after_fault"], 1)
        self.assertEqual(summary["object_recovery"]["coarse_recovered"], 0)
        self.assertEqual(summary["object_recovery"]["fine_recovered"], 1)
        self.assertEqual(summary["object_recovery"]["additional_fine"], 1)
        self.assertEqual(len(frame_rows), 4)
        self.assertIs(object_rows[0]["additional_fine_over_coarse"], True)

    def test_lightweight_detector_output_contract(self):
        model = LightweightBEVDetector(("Car", "Pedestrian", "Cyclist"))
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(output["heatmap_logits"].shape, (2, 3, 32, 32))
        self.assertEqual(output["box_regression"].shape, (2, 6, 32, 32))


if __name__ == "__main__":
    unittest.main()
