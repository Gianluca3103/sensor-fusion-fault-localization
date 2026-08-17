import math
import unittest

import torch

from models.two_stage_reconstruction_head import (
    BEVGridGeometry,
    GeometricAugmentationConfig,
    GeometricTransform,
    ReconstructionGeometricAugmentation,
)


class GeometricAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.geometry = BEVGridGeometry(
            x_min=0.0,
            x_max=2.0,
            y_min=-1.0,
            y_max=1.0,
            height=10,
            width=10,
        )
        self.config = GeometricAugmentationConfig.from_dict(
            {
                "enabled": True,
                "horizontal_flip": {"enabled": True, "probability": 0.5},
                "translation": {
                    "enabled": True,
                    "max_x_m": 0.5,
                    "max_y_m": 0.5,
                },
                "yaw": {"enabled": True, "max_degrees": 5.0},
                "scale": {"enabled": True, "min": 0.95, "max": 1.05},
            }
        )
        self.augmentation = ReconstructionGeometricAugmentation(
            self.config, self.geometry
        )

    def _aligned_item(self):
        lidar = torch.zeros(3, 10, 10)
        radar = torch.zeros(4, 10, 10)
        mask = torch.zeros(1, 10, 10, dtype=torch.uint8)
        lidar[0, 4, 2] = 1.0
        lidar[1:, 4, 2] = 0.75
        radar[0, 4, 2] = 1.0
        radar[1:, 4, 2] = 0.5
        mask[0, 4, 2] = 1
        return {
            "clean_bev": lidar.clone(),
            "faulty_bev": lidar.clone(),
            "radar_bev": radar,
            "reconstruction_mask": mask.clone(),
            "halo_mask": torch.zeros_like(mask),
            "healthy_context_mask": torch.zeros_like(mask),
            "observability_confidence": mask.float(),
            "faulty_lidar_points": torch.tensor(
                [[1.1, -0.5, 0.2, 0.8]], dtype=torch.float32
            ),
            "radar_points": torch.tensor(
                [[1.1, -0.5, 0.2, 7.0, -3.5]], dtype=torch.float32
            ),
        }

    def test_flip_keeps_all_raster_support_aligned_and_masks_binary(self):
        output = self.augmentation.apply(
            self._aligned_item(), transform=GeometricTransform(flip_y=True)
        )
        supports = (
            output["clean_bev"][0] > 0,
            output["faulty_bev"][0] > 0,
            output["radar_bev"][0] > 0,
            output["reconstruction_mask"][0] > 0,
        )
        for support in supports[1:]:
            self.assertTrue(torch.equal(supports[0], support))
        for name in (
            "reconstruction_mask",
            "halo_mask",
            "healthy_context_mask",
        ):
            self.assertTrue(
                set(output[name].unique().tolist()).issubset({0, 1})
            )
            self.assertEqual(output[name].shape, (1, 10, 10))

    def test_raw_lidar_and_radar_share_transform_and_scalar_doppler_is_unchanged(self):
        transform = GeometricTransform(
            flip_y=True,
            scale=1.02,
            yaw_radians=math.radians(4.0),
            translation_x_m=0.1,
            translation_y_m=0.1,
        )
        output = self.augmentation.apply(self._aligned_item(), transform=transform)
        self.assertTrue(
            torch.allclose(
                output["faulty_lidar_points"][0, :3],
                output["radar_points"][0, :3],
            )
        )
        self.assertAlmostEqual(float(output["faulty_lidar_points"][0, 3]), 0.8)
        self.assertAlmostEqual(float(output["radar_points"][0, 3]), 7.0)
        self.assertAlmostEqual(float(output["radar_points"][0, 4]), -3.5)

    def test_translation_drops_out_of_bounds_points_and_does_not_wrap_raster(self):
        item = self._aligned_item()
        item["faulty_lidar_points"] = torch.tensor(
            [[1.9, 0.0, 0.0, 1.0]], dtype=torch.float32
        )
        item["radar_points"] = torch.tensor(
            [[1.9, 0.0, 0.0, 1.0, 2.0]], dtype=torch.float32
        )
        item["clean_bev"].zero_()
        item["clean_bev"][0, 0, 5] = 1.0
        item["faulty_bev"] = item["clean_bev"].clone()
        item["radar_bev"].zero_()
        item["radar_bev"][0, 0, 5] = 1.0
        output = self.augmentation.apply(
            item, transform=GeometricTransform(translation_x_m=0.5)
        )
        self.assertEqual(len(output["faulty_lidar_points"]), 0)
        self.assertEqual(len(output["radar_points"]), 0)
        self.assertEqual(int(output["clean_bev"][0].sum()), 0)
        self.assertEqual(int(output["radar_bev"][0].sum()), 0)

    def test_zero_transform_returns_original_without_resampling(self):
        item = self._aligned_item()
        output = self.augmentation.apply(item, transform=GeometricTransform())
        self.assertIs(output, item)

    def test_fixed_generator_reproduces_parameters_without_reseeding_getitem(self):
        first = torch.Generator().manual_seed(1234)
        second = torch.Generator().manual_seed(1234)
        self.assertEqual(
            self.augmentation.sample_transform(generator=first),
            self.augmentation.sample_transform(generator=second),
        )

    def test_disabled_configuration_is_identity(self):
        disabled = ReconstructionGeometricAugmentation(
            GeometricAugmentationConfig(), self.geometry
        )
        self.assertTrue(disabled.sample_transform().is_identity)


if __name__ == "__main__":
    unittest.main()
