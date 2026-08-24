import unittest

import numpy as np
import torch

from models.two_stage_reconstruction_head.coarse_reconstruction.hrnet_backbone import (
    HRNetConfig,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.pointpillar_feature_reconstruction import (
    CoarsePointPillarFeatureReconstructor,
    PointPillarFeatureReconstructionConfig,
    pointpillar_feature_reconstruction_loss,
    project_mask_between_bev_grids,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


class PointPillarFeatureReconstructionTests(unittest.TestCase):
    def test_metric_mask_projection_is_exact_on_matching_grid(self):
        geometry = BEVGridGeometry(0.0, 4.0, -2.0, 2.0, 4, 4)
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[1:3, 2:4] = 1.0
        projected = project_mask_between_bev_grids(mask, geometry, geometry)
        np.testing.assert_array_equal(projected, mask)

    def test_residual_composition_preserves_outside_exactly(self):
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
        repair = torch.zeros(2, 1, 16, 16)
        repair[:, :, 4:10, 6:12] = 1.0
        halo = torch.zeros_like(repair)
        halo[:, :, 2:12, 4:14] = 1.0 - repair[:, :, 2:12, 4:14]
        output = model(faulty, radar, repair, halo)
        outside = (1.0 - repair).expand_as(faulty).bool()
        self.assertTrue(torch.equal(output["coarse_features"][outside], faulty[outside]))

        clean = torch.randn_like(faulty)
        losses = pointpillar_feature_reconstruction_loss(
            output, clean, faulty, config
        )
        self.assertEqual(float(losses["outside_repair_max_change"]), 0.0)


if __name__ == "__main__":
    unittest.main()
