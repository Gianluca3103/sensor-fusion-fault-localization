import math
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

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


class OnlineDatasetAugmentationTests(unittest.TestCase):
    def make_dataset(self, root, dataset_name='VoD', enabled=True):
        from models.two_stage_reconstruction_head.coarse_dataset import CoarseReconstructionDataset
        from models.two_stage_reconstruction_head.fault_selector import FaultSelectorConfig
        from models.two_stage_reconstruction_head.fault_selector_cache import (
            selector_cache_path, CACHE_VERSION, _config_json,
        )
        data_root = root / 'data'
        sample = data_root / 'train' / '00001.npz'
        sample.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((320, 320, 3), dtype=np.uint8)
        rgb[250, 135] = 255
        points = np.array([[10., -5., .2, 42.]], dtype=np.float32)
        np.savez(sample, clean_rgb=rgb, faulty_rgb=rgb, faulty_lidar_points=points,
                 metadata_json=json.dumps({'dataset': dataset_name, 'split': 'train',
                     'frame_id': '1', 'x_range': [0., 64.], 'y_range': [-32., 32.], 'resolution': .2}))
        radar = root / 'radar' / 'train'
        radar.mkdir(parents=True, exist_ok=True)
        np.savez(radar / '00001.npz', radar_bev=np.zeros((4, 320, 320), dtype=np.float32),
                 radar_points=np.array([[10., -5., .2, 30., -2.]], dtype=np.float32))
        selector = FaultSelectorConfig()
        cache = selector_cache_path(sample, data_root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((320, 320), dtype=np.uint8)
        mask[240:260, 125:145] = 1
        np.savez(cache, cache_version=CACHE_VERSION, selector_config=_config_json(selector),
                 reconstruction_mask=mask, halo_mask=np.zeros_like(mask),
                 healthy_context_mask=np.zeros_like(mask))
        return CoarseReconstructionDataset([sample], root / 'radar', data_root=data_root,
            use_pointpillars=True, augmentation_seed=42,
            augmentation_config=GeometricAugmentationConfig(enabled=True) if enabled else None)

    def test_fresh_epochs_repeatable_and_shared_sensor_transform_for_both_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('VoD', 'HeRCULES'):
                dataset = self.make_dataset(root / name, name)
                source = dataset.sample_paths[0].read_bytes()
                dataset.set_epoch(1)
                first = dataset[0]
                dataset.set_epoch(2)
                second = dataset[0]
                self.assertNotEqual(first['augmentation_transform'], second['augmentation_transform'])
                dataset.set_epoch(1)
                repeated = dataset[0]
                self.assertEqual(first['augmentation_transform'], repeated['augmentation_transform'])
                self.assertTrue(torch.equal(first['faulty_lidar_points'], repeated['faulty_lidar_points']))
                self.assertTrue(torch.equal(first['clean_bev'], first['faulty_bev']))
                self.assertTrue(torch.equal(first['faulty_lidar_points'][:, :3], first['radar_points'][:, :3]))
                self.assertEqual(source, dataset.sample_paths[0].read_bytes())

    def test_persistent_workers_observe_epoch_and_match_single_process(self):
        from torch.utils.data import DataLoader
        from models.two_stage_reconstruction_head.coarse_dataset import coarse_reconstruction_collate
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory))
            loader = DataLoader(dataset, batch_size=1, num_workers=1,
                persistent_workers=True, collate_fn=coarse_reconstruction_collate)
            try:
                dataset.set_epoch(3)
                first = next(iter(loader))['faulty_lidar_points'][0]
                self.assertTrue(torch.equal(first, dataset[0]['faulty_lidar_points']))
                dataset.set_epoch(4)
                second = next(iter(loader))['faulty_lidar_points'][0]
                self.assertTrue(torch.equal(second, dataset[0]['faulty_lidar_points']))
                self.assertFalse(torch.equal(first, second))
            finally:
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()

    def test_no_augmentation_dataset_stays_unchanged_across_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_dataset(Path(directory), enabled=False)
            dataset.set_epoch(1)
            first = dataset[0]
            dataset.set_epoch(2)
            second = dataset[0]
            self.assertNotIn('augmentation_transform', second)
            self.assertTrue(torch.equal(first['faulty_lidar_points'], second['faulty_lidar_points']))
            self.assertTrue(torch.equal(first['clean_bev'], second['clean_bev']))


if __name__ == "__main__":
    unittest.main()
