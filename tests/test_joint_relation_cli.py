"""One-frame integration test for teacher, joint training, and export."""

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

from scripts.export_joint_relation_vod import main as export_main
from scripts.train_joint_relation_diffusion import main as joint_main
from scripts.train_ray_depth_blueprint import main as teacher_main
from scripts.visualize_joint_relation_reconstruction import main as visualize_main


class JointRelationCliTests(unittest.TestCase):
    def test_teacher_only_joint_training_and_inference_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vod, samples = root / "vod", root / "samples"
            geometry = root / "geometry.json"
            geometry.write_text(json.dumps({
                "beam_elevations_rad": [-0.1, 0.0, 0.1],
                "azimuth_bins": 8, "min_range_m": 0.5,
                "max_range_m": 12.0, "azimuth_span_rad": 2 * np.pi,
            }))
            radar_direction = np.asarray([
                np.cos(np.pi / 8), np.sin(np.pi / 8), 0,
            ], dtype=np.float32)
            lidar_direction = np.asarray([
                np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8), 0,
            ], dtype=np.float32)
            identity = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
            for split, frame_id in (("train", "00001"), ("train", "00003"),
                                    ("val", "00002"), ("val", "00004")):
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
                    [*(radar_direction * 8.5), 5.0],
                    [*(lidar_direction * 4.0), 7.0],
                ], dtype=np.float32).tofile(
                    vod / "lidar/training/velodyne" / f"{frame_id}.bin")
                np.asarray([
                    [*(radar_direction * 4.0), 12.0, 1.0, 1.0, 0.0],
                ], dtype=np.float32).tofile(
                    vod / "radar_20frames_verified_doppler_radial/training/velodyne" /
                    f"{frame_id}.bin")
                (samples / split).mkdir(parents=True, exist_ok=True)
                np.savez(samples / split / f"{frame_id}_fault.npz",
                    faulty_lidar_points=np.asarray([
                        [*(lidar_direction * 4.0), 7.0],
                        [*(lidar_direction * 4.0 + [0, 0, 0.1]), 7.0],
                    ], dtype=np.float32),
                    metadata_json=np.asarray(json.dumps({
                        "dataset": "View-of-Delft", "split": split,
                        "frame_id": frame_id, "range_view_full_scan": True,
                    })))
            teacher_root = root / "teacher"
            shared = ["--samples-root", str(samples), "--vod-root", str(vod),
                      "--geometry", str(geometry), "--epochs", "1",
                      "--tile-rows", "1", "--tile-cols", "4",
                      "--width", "8", "--num-workers", "0", "--device", "cpu"]
            args = ["teacher", *shared, "--output-root", str(teacher_root),
                    "--teacher-epochs", "1", "--validate-every", "1",
                    "--teacher-only"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                teacher_main()
            teacher_checkpoint = teacher_root / "teacher_best_checkpoint.pt"
            self.assertTrue(teacher_checkpoint.is_file())
            self.assertFalse((teacher_root / "best_checkpoint.pt").exists())
            run = root / "joint"
            joint_args = ["joint", *shared, "--output-root", str(run),
                          "--teacher-checkpoint", str(teacher_checkpoint),
                          "--hidden", "8", "--timesteps", "8",
                          "--batch-size", "2", "--grad-accum-steps", "1",
                          "--validate-every", "1", "--no-audit-train-at-end"]
            with patch.object(sys, "argv", joint_args), \
                    contextlib.redirect_stdout(io.StringIO()):
                joint_main()
            saved = torch.load(run / "best_checkpoint.pt", map_location="cpu",
                               weights_only=False)
            self.assertEqual(saved["stage"], "joint_radar_relation_diffusion_v1")
            self.assertNotIn("teacher", saved)
            self.assertGreater(saved["validation"]["supported"], 0)
            resumed = joint_args.copy()
            resumed[resumed.index("--epochs") + 1] = "2"
            resumed[resumed.index("--batch-size") + 1] = "1"
            resumed[resumed.index("--grad-accum-steps") + 1] = "2"
            resumed += ["--resume", str(run / "last_checkpoint.pt")]
            with patch.object(sys, "argv", resumed), \
                    contextlib.redirect_stdout(io.StringIO()):
                joint_main()
            continued = torch.load(run / "last_checkpoint.pt", map_location="cpu",
                                   weights_only=False)
            self.assertEqual(continued["epoch"], 2)
            exported = root / "export"
            export_args = ["export", "--vod-root", str(vod),
                           "--samples-root", str(samples),
                           "--checkpoint", str(run / "best_checkpoint.pt"),
                           "--output-root", str(exported),
                           "--limit", "1", "--steps", "2",
                           "--num-workers", "0", "--device", "cpu"]
            with patch.object(sys, "argv", export_args), \
                    contextlib.redirect_stdout(io.StringIO()):
                export_main()
            destination = (exported / "lidar/reconstructed/training/velodyne/00002.bin")
            self.assertTrue(destination.is_file())
            original_bytes = destination.read_bytes()
            with patch.object(sys, "argv", export_args), \
                    contextlib.redirect_stdout(io.StringIO()):
                export_main()
            self.assertEqual(destination.read_bytes(), original_bytes)
            cloud = np.fromfile(destination, dtype="<f4").reshape(-1, 4)
            self.assertTrue(np.isfinite(cloud).all())
            manifest = json.loads((exported / "lidar/reconstructed/export_manifest.json")
                                  .read_text())
            self.assertTrue(manifest["complete"])
            self.assertFalse(manifest["clean_lidar_used_at_inference"])
            viewed = root / "viewed"
            viewer_args = ["viewer", "--vod-root", str(vod),
                           "--samples-root", str(samples),
                           "--checkpoint", str(run / "best_checkpoint.pt"),
                           "--output-root", str(viewed),
                           "--sample-indices", "0", "--steps", "2",
                           "--device", "cpu"]
            with patch.object(sys, "argv", viewer_args), \
                    contextlib.redirect_stdout(io.StringIO()):
                visualize_main()
            self.assertTrue((viewed / "00002_fault/interactive.html").is_file())
            self.assertTrue((viewed / "00002_fault/generated.ply").is_file())
            view_metadata = json.loads((viewed / "00002_fault/metadata.json").read_text())
            self.assertFalse(view_metadata["clean_lidar_used_at_inference"])


if __name__ == "__main__":
    unittest.main()
