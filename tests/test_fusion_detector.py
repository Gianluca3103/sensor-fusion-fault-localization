import unittest

import torch

from models.two_stage_reconstruction_head.coarse_reconstruction.hrnet_backbone import (
    HRNetConfig,
)
from models.two_stage_reconstruction_head.object_detection import (
    BEVDetectorConfig,
    FusionDetectorConfig,
    PointPillarsHRNetFusionDetector,
    RotatedBEVBox,
    make_detection_targets,
)
from models.two_stage_reconstruction_head.object_detection.train_fusion_detector import (
    fusion_detection_collate,
)
from models.two_stage_reconstruction_head.pointpillars import (
    BEVGridGeometry,
    PointPillarsConfig,
)


def _geometry():
    return BEVGridGeometry(0.0, 16.0, -8.0, 8.0, height=16, width=16)


def _config():
    return FusionDetectorConfig(
        pointpillars=PointPillarsConfig(
            enabled=True,
            output_channels=4,
            max_points_per_pillar=8,
            max_pillars=None,
        ),
        hrnet=HRNetConfig(
            base_channels=4,
            num_stages=2,
            blocks_per_stage=1,
            residual_blocks_per_branch=1,
            dropout=0.0,
        ),
        detector=BEVDetectorConfig(
            input_channels=32,
            base_channels=8,
            output_stride=2,
            box_regression_channels=8,
        ),
    )


class FusionDetectorTests(unittest.TestCase):
    def test_separate_encoders_align_concatenate_and_detect(self):
        model = PointPillarsHRNetFusionDetector(
            ("Car", "Pedestrian", "Cyclist"), _geometry(), _config()
        )
        self.assertIsNot(
            model.lidar_pillar_encoder, model.radar_pillar_encoder
        )
        lidar = (
            torch.tensor([[2.0, 0.0, 0.2, 0.8], [2.1, 0.1, 0.4, 0.5]]),
            torch.tensor([[6.0, -2.0, 0.1, 0.7]]),
        )
        radar = (
            torch.tensor([[2.0, 0.0, 0.1, 0.9, 1.5]]),
            torch.tensor([[6.0, -2.0, 0.2, 0.4, -0.5]]),
        )
        outputs = model(lidar, radar, return_diagnostics=True)
        debug = outputs["diagnostics"]
        self.assertEqual(debug["lidar_pillar_features"].shape, (2, 4, 16, 16))
        self.assertEqual(debug["radar_pillar_features"].shape, (2, 4, 16, 16))
        self.assertEqual(debug["concatenated_features"].shape, (2, 8, 16, 16))
        self.assertEqual(debug["fused_features"].shape, (2, 32, 16, 16))
        self.assertEqual(outputs["heatmap_logits"].shape, (2, 3, 8, 8))
        self.assertEqual(outputs["box_regression"].shape, (2, 8, 8, 8))

    def test_lidar_only_ablation_zeros_radar_feature_map(self):
        model = PointPillarsHRNetFusionDetector(("Car",), _geometry(), _config())
        model.eval()
        outputs = model(
            (torch.tensor([[2.0, 0.0, 0.2, 0.8]]),),
            (torch.tensor([[2.0, 0.0, 0.1, 0.9, 1.5]]),),
            radar_enabled=False,
            return_diagnostics=True,
        )
        self.assertTrue(
            torch.equal(
                outputs["diagnostics"]["radar_pillar_features"],
                torch.zeros_like(
                    outputs["diagnostics"]["radar_pillar_features"]
                ),
            )
        )

    def test_3d_center_targets_include_z_and_height(self):
        box = RotatedBEVBox(
            "Car", 8.0, 0.0, 4.0, 2.0, 0.25, z=1.2, height=1.6
        )
        targets = make_detection_targets(
            [[box]],
            ("Car",),
            _geometry(),
            (8, 8),
            2,
            device=torch.device("cpu"),
            box_regression_channels=8,
        )
        location = targets["regression_mask"][0, 0].nonzero()[0]
        row, column = location.tolist()
        self.assertAlmostEqual(
            float(targets["regression"][0, 6, row, column]), 1.2, places=5
        )
        self.assertAlmostEqual(
            float(targets["regression"][0, 7, row, column].exp()), 1.6, places=5
        )

    def test_configuration_round_trip_and_unknown_key_rejection(self):
        restored = FusionDetectorConfig.from_dict(_config().to_dict())
        self.assertEqual(restored, _config())
        with self.assertRaisesRegex(ValueError, "Unknown fusion_detector"):
            FusionDetectorConfig.from_dict({"unexpected": True})

    def test_collate_preserves_optional_clean_lidar_for_diagnostics(self):
        batch = [
            {
                "faulty_lidar_points": torch.zeros(2, 4),
                "clean_lidar_points": torch.ones(3, 4),
                "radar_points": torch.zeros(4, 5),
                "frame_id": "00001",
                "sample_path": "sample.npz",
                "boxes": [],
            }
        ]
        collated = fusion_detection_collate(batch)
        self.assertEqual(collated["clean_lidar_points"][0].shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
