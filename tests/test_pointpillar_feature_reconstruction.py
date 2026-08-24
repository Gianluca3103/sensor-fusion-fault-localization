import unittest

import torch

from models.two_stage_reconstruction_head.coarse_reconstruction.hrnet_backbone import (
    HRNetConfig,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.pointpillar_feature_reconstruction import (
    CoarsePointPillarFeatureReconstructor,
    PointPillarFeatureReconstructionConfig,
    pointpillar_feature_reconstruction_loss,
)


class PointPillarFeatureReconstructionTests(unittest.TestCase):
    def test_zero_initialized_residual_starts_from_faulty_features(self):
        config = PointPillarFeatureReconstructionConfig(
            lidar_feature_channels=4,
            radar_feature_channels=4,
            lidar_feature_height=16,
            lidar_feature_width=16,
            hrnet=HRNetConfig(
                base_channels=4,
                num_stages=2,
                blocks_per_stage=1,
                residual_blocks_per_branch=1,
            ),
        )
        model = CoarsePointPillarFeatureReconstructor(config)
        faulty = torch.randn(2, 4, 16, 16)
        radar = torch.randn(2, 4, 16, 16)
        output = model(faulty, radar)
        self.assertTrue(torch.equal(output["coarse_features"], faulty))
        self.assertEqual(output["predicted_delta"].shape, faulty.shape)
        self.assertEqual(output["network_input"].shape, (2, 8, 16, 16))

        clean = torch.randn_like(faulty)
        losses = pointpillar_feature_reconstruction_loss(
            output, clean, faulty, config
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertIn("smooth_l1_changed_cells", losses)


if __name__ == "__main__":
    unittest.main()
