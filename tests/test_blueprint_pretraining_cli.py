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

from scripts.train_ray_depth_blueprint import main as blueprint_main
from scripts.train_radar_gated_ray_diffusion import main as diffusion_main


class BlueprintPretrainingCliTests(unittest.TestCase):
    def test_pretrain_then_freeze_and_unfreeze_for_diffusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vod, samples = root / "vod", root / "samples"
            geometry = root / "geometry.json"
            geometry.write_text(json.dumps({
                "beam_elevations_rad": [-0.1, 0.0, 0.1],
                "azimuth_bins": 8, "min_range_m": 0.5,
                "max_range_m": 12.0, "azimuth_span_rad": 2 * np.pi,
            }))
            direction_radar = np.asarray([
                np.cos(np.pi / 8), np.sin(np.pi / 8), 0,
            ], dtype=np.float32)
            direction_lidar = np.asarray([
                np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8), 0,
            ], dtype=np.float32)
            identity = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            for split, frame_id in (("train", "00001"),
                                    ("train", "00003"),
                                    ("train", "00004"), ("val", "00002")):
                for relative in (
                    "lidar/ImageSets", "lidar/training/velodyne",
                    "lidar/training/calib", "radar/training/calib",
                    "radar_20frames_verified_doppler_radial/training/velodyne",
                ):
                    (vod / relative).mkdir(parents=True, exist_ok=True)
                with (vod / "lidar/ImageSets" / f"{split}.txt").open("a") as ids:
                    ids.write(frame_id + "\n")
                for sensor in ("lidar", "radar"):
                    (vod / sensor / "training/calib" / f"{frame_id}.txt").write_text(identity)
                np.asarray([
                    [*(direction_radar * 4.5), 5.0],
                    [*(direction_lidar * 4.0), 7.0],
                ], dtype=np.float32).tofile(
                    vod / "lidar/training/velodyne" / f"{frame_id}.bin")
                np.asarray([
                    [*(direction_radar * 4.0), 12.0, 1.0, 1.0, 0.0],
                ], dtype=np.float32).tofile(
                    vod / "radar_20frames_verified_doppler_radial/training/velodyne" /
                    f"{frame_id}.bin")
                (samples / split).mkdir(parents=True, exist_ok=True)
                np.savez(samples / split / f"{frame_id}_fault.npz",
                    faulty_lidar_points=np.asarray([
                        [*(direction_lidar * 4.0), 7.0],
                        [*(direction_lidar * 4.0 + [0, 0, 0.1]), 7.0],
                    ], dtype=np.float32),
                    metadata_json=np.asarray(json.dumps({
                        "dataset": "View-of-Delft", "split": split,
                        "frame_id": frame_id, "range_view_full_scan": True,
                    })))
            blueprint_root = root / "blueprint_run"
            shared = ["--samples-root", str(samples), "--vod-root", str(vod),
                      "--geometry", str(geometry), "--epochs", "1",
                      "--train-limit", "1", "--val-limit", "1",
                      "--tile-rows", "1", "--tile-cols", "4", "--width", "8",
                      "--num-workers", "0", "--device", "cpu"]
            blueprint_args = ["blueprint", *shared, "--output-root", str(blueprint_root),
                              "--validate-every", "1", "--grad-accum-steps", "2",
                              "--teacher-epochs", "1"]
            blueprint_args[blueprint_args.index("--train-limit") + 1] = "3"
            with patch.object(sys, "argv", blueprint_args), \
                    contextlib.redirect_stdout(io.StringIO()):
                blueprint_main()
            stage1 = torch.load(blueprint_root / "best_checkpoint.pt",
                                map_location="cpu", weights_only=False)
            self.assertEqual(stage1["stage"], "blueprint_pretraining")
            self.assertEqual(stage1["relationship_version"],
                             "radar_only_clean_teacher_v1")
            self.assertFalse(stage1["radar_height_filter"])
            self.assertTrue((blueprint_root / "teacher_best_checkpoint.pt").is_file())
            self.assertEqual(stage1["settings"]["grad_accum_steps"], 2)
            self.assertGreater(stage1["validation"]["clean_hits"], 0)
            self.assertIn("depth_f1_3m", stage1["train_eval"])
            diffusion_root = root / "diffusion_run"
            diffusion_args = ["diffusion", *shared, "--output-root", str(diffusion_root),
                              "--hidden", "8", "--timesteps", "8",
                              "--validate-every", "1", "--pretrained-blueprint",
                              str(blueprint_root / "best_checkpoint.pt"),
                              "--freeze-blueprint-epochs", "1"]
            with patch.object(sys, "argv", diffusion_args), \
                    contextlib.redirect_stdout(io.StringIO()):
                diffusion_main()
            frozen = torch.load(diffusion_root / "last_checkpoint.pt",
                                map_location="cpu", weights_only=False)
            self.assertEqual(frozen["freeze_blueprint_epochs"], 1)
            self.assertTrue(all(torch.equal(stage1["blueprint"][key], value)
                                for key, value in frozen["blueprint"].items()))
            resumed = diffusion_args.copy()
            resumed[resumed.index("--epochs") + 1] = "2"
            pretrained_position = resumed.index("--pretrained-blueprint")
            del resumed[pretrained_position:pretrained_position + 2]
            resumed += ["--resume", str(diffusion_root / "last_checkpoint.pt")]
            with patch.object(sys, "argv", resumed), \
                    contextlib.redirect_stdout(io.StringIO()):
                diffusion_main()
            unfrozen = torch.load(diffusion_root / "last_checkpoint.pt",
                                  map_location="cpu", weights_only=False)
            self.assertEqual(unfrozen["epoch"], 2)
            self.assertTrue(any(not torch.equal(stage1["blueprint"][key], value)
                                for key, value in unfrozen["blueprint"].items()))


if __name__ == "__main__":
    unittest.main()
