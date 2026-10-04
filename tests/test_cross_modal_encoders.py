import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from models.two_stage_reconstruction_head.cross_modal_data import (
    CrossModalVoDDataset, collate_cross_modal, observed_lidar_height_mask,
)
from models.two_stage_reconstruction_head.cross_modal_encoders import (
    EncoderGrid,
    RadarGeometryEncoder,
    RadarLidarEncoders,
)


class CrossModalEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.grid = EncoderGrid((0, -2, -1), (4, 2, 1), (1, 1, 1))
        self.radar = torch.tensor([[[1.2, 0.2, 0.0, 10, 2, 1, 0],
                                    [1.3, 0.3, 0.0, 12, -2, -1, -1],
                                    [0, 0, 0, 0, 0, 0, 0]]], dtype=torch.float32)
        self.radar_valid = torch.tensor([[True, True, False]])
        self.lidar = torch.tensor([[[1.2, 0.2, 0.0, 4],
                                    [2.2, 0.2, 0.0, 8]]], dtype=torch.float32)
        self.lidar_valid = torch.tensor([[True, True]])

    def test_radar_uses_temporal_and_doppler_fields_with_spatial_support(self):
        encoder = RadarGeometryEncoder(self.grid, width=8, history_scans=20)
        original = encoder(self.radar, self.radar_valid)
        changed = self.radar.clone()
        changed[0, 1, 4:7] = torch.tensor([8., 5., -10.])
        updated = encoder(changed, self.radar_valid)
        self.assertEqual(original.features.shape, (1, 8, 2, 4, 4))
        self.assertEqual(int(original.occupied.sum()), 1)
        tokens, positions = original.active_tokens(occupied_only=True)[0]
        self.assertEqual(tokens.shape, (1, 8))
        self.assertEqual(positions.shape, (1, 3))
        self.assertFalse(torch.allclose(original.features, updated.features))
        self.assertTrue(torch.equal(original.support, updated.support))

    def test_clean_teacher_is_training_only_and_does_not_change_inputs(self):
        encoder = RadarLidarEncoders(self.grid, width=8)
        encoder.train()
        without_teacher = encoder(self.radar, self.radar_valid,
                                  self.lidar, self.lidar_valid)
        with_teacher = encoder(self.radar, self.radar_valid,
                               self.lidar, self.lidar_valid,
                               clean_lidar=self.lidar * 1.1,
                               clean_valid=self.lidar_valid)
        self.assertIn("clean_teacher", with_teacher)
        for key in ("radar", "observed_lidar"):
            torch.testing.assert_close(without_teacher[key].features,
                                       with_teacher[key].features)
        encoder.eval()
        with self.assertRaisesRegex(ValueError, "training supervision"):
            encoder(self.radar, self.radar_valid, self.lidar,
                    self.lidar_valid, clean_lidar=self.lidar,
                    clean_valid=self.lidar_valid)

    def test_empty_radar_and_gradient_flow(self):
        encoder = RadarGeometryEncoder(self.grid, width=8)
        empty = encoder(self.radar, torch.zeros_like(self.radar_valid))
        self.assertEqual(int(empty.occupied.sum()), 0)
        self.assertEqual(float(empty.features.abs().sum().detach()), 0.0)
        populated = encoder(self.radar, self.radar_valid)
        populated.features.sum().backward()
        self.assertIsNotNone(encoder.point_mlp[0].weight.grad)

    def test_observed_lidar_height_gate_is_inclusive_and_handles_total_loss(self):
        radar = np.asarray([[0, 0, -2, 0, 0, 0, 0],
                            [0, 0, -1, 0, 0, 0, -1],
                            [0, 0, 1, 0, 0, 0, -2],
                            [0, 0, 2, 0, 0, 0, -3]], dtype=np.float32)
        observed = np.asarray([[0, 0, -1, 1], [0, 0, 1, 1]], dtype=np.float32)
        np.testing.assert_array_equal(
            observed_lidar_height_mask(radar, observed),
            [False, True, True, False],
        )
        self.assertTrue(observed_lidar_height_mask(
            radar, np.empty((0, 4), np.float32)).all())
        self.assertTrue(observed_lidar_height_mask(radar, observed[:1]).all())

    def test_full_scan_loader_preserves_all_radar_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lidar = root / "lidar" / "training"
            radar = root / "radar_20frames_verified" / "training" / "velodyne"
            raw_calib = root / "radar" / "training" / "calib"
            for folder in (root / "lidar" / "ImageSets", lidar / "velodyne", lidar / "calib",
                           radar, raw_calib, root / "samples"):
                folder.mkdir(parents=True, exist_ok=True)
            (root / "lidar" / "ImageSets" / "train.txt").write_text("00001\n")
            calibration = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            (lidar / "calib" / "00001.txt").write_text(calibration)
            (raw_calib / "00001.txt").write_text(calibration)
            np.asarray([[1, 0, 0, 9], [2, 0, 0, 8]], dtype=np.float32).tofile(
                lidar / "velodyne" / "00001.bin"
            )
            np.asarray([[1, 0, -1, 11, -2, -1, -1],
                        [2, 0, 0, 12, 3, 2, 0],
                        [3, 0, 2, 13, 4, 3, -2]], dtype=np.float32).tofile(
                radar / "00001.bin"
            )
            sample = root / "samples" / "00001_fault.npz"
            np.savez(sample,
                     faulty_lidar_points=np.asarray([[1, 0, -1, 7],
                                                     [2, 0, 0, 7]], dtype=np.float32),
                     metadata_json=np.asarray(json.dumps({
                         "dataset": "View-of-Delft",
                         "split": "train", "frame_id": "00001",
                         "range_view_full_scan": True,
                     })))
            data = CrossModalVoDDataset([sample], root,
                                        radar_variant="radar_20frames_verified",
                                        include_clean=True)
            batch = collate_cross_modal([data[0]])
            self.assertEqual(batch["radar"].shape, (1, 3, 7))
            self.assertEqual(batch["radar"][0, 0, 6].item(), -1)
            self.assertEqual(batch["radar"][0, 0, 4].item(), -2)
            self.assertEqual(batch["clean_lidar"].shape, (1, 2, 4))
            no_teacher = CrossModalVoDDataset(
                [sample], root, radar_variant="radar_20frames_verified")[0]
            self.assertNotIn("clean_lidar", no_teacher)
            gated = CrossModalVoDDataset(
                [sample], root, radar_variant="radar_20frames_verified",
                include_clean=True, radar_height_filter=True)[0]
            self.assertEqual(gated["radar"].shape, (2, 7))
            self.assertEqual(gated["radar"][:, 2].tolist(), [-1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
