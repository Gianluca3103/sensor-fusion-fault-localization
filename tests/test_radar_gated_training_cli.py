import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.two_stage_reconstruction_head.cross_modal_data import CrossModalVoDDataset
from scripts.train_radar_gated_ray_diffusion import main
from scripts.export_radar_gated_ray_diffusion import main as export_main


class RadarGatedTrainingCliTests(unittest.TestCase):
    def test_one_epoch_with_real_dataset_loader_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vod = root / "vod"
            samples = root / "samples"
            geometry = root / "geometry.json"
            output = root / "run"
            geometry.write_text(json.dumps({
                "beam_elevations_rad": [-0.1, 0.0, 0.1],
                "azimuth_bins": 8, "min_range_m": 0.5,
                "max_range_m": 12.0,
                "azimuth_span_rad": 2 * np.pi,
            }))
            direction0 = np.asarray([np.cos(np.pi / 8), np.sin(np.pi / 8), 0],
                                    dtype=np.float32)
            direction1 = np.asarray([np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8), 0],
                                    dtype=np.float32)
            identity = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            for split, frame_id in (("train", "00001"), ("val", "00002")):
                for relative in (
                    "lidar/ImageSets", "lidar/training/velodyne",
                    "lidar/training/calib", "radar/training/calib",
                    "radar_20frames_verified_doppler_radial/training/velodyne",
                ):
                    (vod / relative).mkdir(parents=True, exist_ok=True)
                (vod / "lidar" / "ImageSets" / f"{split}.txt").write_text(frame_id + "\n")
                for sensor in ("lidar", "radar"):
                    (vod / sensor / "training" / "calib" / f"{frame_id}.txt").write_text(identity)
                clean = np.asarray([[*(direction0 * 4.5), 5.0],
                                    [*(direction1 * 4.0), 7.0]], dtype=np.float32)
                clean.tofile(vod / "lidar" / "training" / "velodyne" / f"{frame_id}.bin")
                radar = np.asarray([[*(direction0 * 4), 12.0, 1.0, 1.0, 0.0],
                                    [*(direction0 * 4 + [0, 0, 1]),
                                     8.0, 2.0, 2.0, -1.0]], dtype=np.float32)
                radar.tofile(vod / "radar_20frames_verified_doppler_radial" / "training" /
                             "velodyne" / f"{frame_id}.bin")
                folder = samples / split
                folder.mkdir(parents=True)
                np.savez(folder / f"{frame_id}_fault.npz",
                    faulty_lidar_points=np.asarray([
                        [*(direction1 * 4), 7.0],
                        [*(direction1 * 4 + [0, 0, 0.1]), 7.0],
                    ], dtype=np.float32),
                    metadata_json=np.asarray(json.dumps({
                        "dataset": "View-of-Delft", "split": split,
                        "frame_id": frame_id, "range_view_full_scan": True,
                    })))
            probe = CrossModalVoDDataset(
                [samples / "train" / "00001_fault.npz"], vod,
                radar_variant="radar_20frames_verified_doppler_radial",
                radar_height_filter=True,
            )[0]
            self.assertEqual(len(probe["radar"]), 1)
            arguments = ["train_radar_gated_ray_diffusion", "--samples-root", str(samples),
                         "--vod-root", str(vod), "--geometry", str(geometry),
                         "--output-root", str(output), "--epochs", "1",
                         "--validate-every", "1", "--train-limit", "1",
                         "--val-limit", "1", "--tile-rows", "1",
                         "--tile-cols", "4", "--width", "8", "--hidden", "8",
                         "--timesteps", "8", "--num-workers", "0",
                         "--device", "cpu"]
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                main()
            checkpoint = torch.load(output / "last_checkpoint.pt",
                                    map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertIn("diffusion", checkpoint)
            self.assertIn("blueprint", checkpoint)
            self.assertEqual(checkpoint["radar_variant"],
                             "radar_20frames_verified_doppler_radial")
            self.assertTrue(checkpoint["radar_height_filter"])
            resumed = arguments.copy()
            resumed[resumed.index("--epochs") + 1] = "2"
            resumed += ["--resume", str(output / "last_checkpoint.pt")]
            with patch.object(sys, "argv", resumed), contextlib.redirect_stdout(io.StringIO()):
                main()
            checkpoint = torch.load(output / "last_checkpoint.pt",
                                    map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            export = root / "detector_export"
            export_args = [
                "export_radar_gated_ray_diffusion",
                "--samples-root", str(samples),
                "--vod-root", str(vod),
                "--checkpoint", str(output / "last_checkpoint.pt"),
                "--output-root", str(export),
                "--steps", "3", "--device", "cpu",
            ]
            with patch.object(sys, "argv", export_args), contextlib.redirect_stdout(io.StringIO()):
                export_main()
            scan = export / "lidar" / "reconstructed" / "training" / "velodyne" / "00002.bin"
            self.assertTrue(scan.is_file())
            self.assertEqual(np.fromfile(scan, dtype="<f4").size % 4, 0)
            manifest = json.loads((export / "lidar" / "reconstructed" /
                                   "export_manifest.json").read_text())
            self.assertTrue(manifest["complete_official_validation"])
            self.assertEqual(manifest["checkpoint_epoch"], 2)
            self.assertEqual(manifest["radar_variant"],
                             "radar_20frames_verified_doppler_radial")
            self.assertTrue(manifest["radar_height_filter"])


if __name__ == "__main__":
    unittest.main()
