"""Cached geometry must preserve Stage-I supervision across shuffled batches."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.radar_lidar_stage1 import pfe_spconv
from models.radar_lidar_stage1.bottleneck_support import BottleneckSupportStage1
from models.radar_lidar_stage1.config import Stage1Config, VoxelGrid
from models.radar_lidar_stage1.static_geometry_cache import Stage1StaticGeometryCache
from scripts.cache_stage1_static_geometry import prebuild_split
from tests.test_bottleneck_support_interface import ReferenceConv, ReferenceTensor


def _frame(root: Path, frame_id: str):
    files = []
    for name in ("radar", "lidar", "radar_calibration", "lidar_calibration"):
        path = root / f"{frame_id}_{name}.bin"
        path.write_bytes(frame_id.encode() + name.encode())
        files.append(path)
    return SimpleNamespace(
        split="train", frame_id=frame_id,
        radar_path=files[0], lidar_path=files[1],
        radar_calibration_path=files[2], lidar_calibration_path=files[3],
        radar_variant="radar_20frames_verified_doppler_radial")


class Stage1StaticCacheTests(unittest.TestCase):
    def test_cold_warm_and_shuffled_batches_match_uncached_loss(self):
        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = replace(Stage1Config(grid=grid, channels=(8, 16, 32, 64)),
                         encoder_backend="pfe_spconv", growth_scales=(),
                         bottleneck_cross_attention_only=True,
                         bottleneck_support_only=True,
                         corr_scale_weights=(0., 0., 0., 1.),
                         confidence_weight=0., attention_dim=16)
        xyz = torch.tensor([[1., 1., 1.], [2., 3., 3.], [3., 5., 6.],
                            [6., 3., 9.], [7., 7., 12.], [9., 11., 4.],
                            [12., 4., 6.], [14., 14., 14.]])
        radar = torch.stack((torch.cat((xyz, torch.ones(8, 4)), dim=1),
                             torch.cat((xyz + .2, torch.ones(8, 4)), dim=1)))
        lidar = torch.stack((torch.cat((xyz + .1, torch.ones(8, 1)), dim=1),
                             torch.cat((xyz + .3, torch.ones(8, 1)), dim=1)))
        valid = torch.ones((2, 8), dtype=torch.bool)
        reference = SimpleNamespace(SubMConv3d=ReferenceConv,
                                    SparseConv3d=ReferenceConv,
                                    SparseConvTensor=ReferenceTensor)
        with TemporaryDirectory(dir=Path.cwd()) as directory, patch.object(pfe_spconv, "spconv", reference):
            root = Path(directory)
            frames = [_frame(root, "00001"), _frame(root, "00002")]
            cache = Stage1StaticGeometryCache(root / "cache", config, frames)
            model = BottleneckSupportStage1(config).eval()
            self.assertFalse(cache.has_entry("train", "00001"))

            def run(order, use_cache):
                args = (radar[order], valid[order], lidar[order], valid[order])
                ids = [frames[index].frame_id for index in order]
                losses, diagnostics = model.forward_train(
                    *args, static_cache=cache if use_cache else None,
                    frame_ids=ids, split="train")
                return losses, diagnostics

            uncached, _ = run([0, 1], False)
            cold, cold_stats = run([0, 1], True)
            warm, warm_stats = run([0, 1], True)
            self.assertEqual(cold_stats["cache_builds"], 2)
            self.assertEqual(warm_stats["cache_hits"], 2)
            self.assertTrue(cache.has_entry("train", "00001"))
            for key in uncached:
                torch.testing.assert_close(uncached[key], cold[key])
                torch.testing.assert_close(uncached[key], warm[key])

            # A resumed process must read the saved entries, while changed
            # learned scores still come from the current model weights.
            cache = Stage1StaticGeometryCache(root / "cache", config, frames)
            with torch.no_grad():
                model.radar_only.support_head[-1].bias.add_(.2)
            changed_reference, _ = run([0, 1], False)
            changed_cached, changed_stats = run([0, 1], True)
            self.assertEqual(changed_stats["cache_hits"], 2)
            for key in changed_reference:
                torch.testing.assert_close(changed_reference[key], changed_cached[key])

            shuffled_reference, _ = run([1, 0], False)
            shuffled_cached, shuffled_stats = run([1, 0], True)
            self.assertEqual(shuffled_stats["cache_hits"], 2)
            for key in shuffled_reference:
                torch.testing.assert_close(shuffled_reference[key], shuffled_cached[key])

            model.zero_grad(set_to_none=True)
            shuffled_reference["loss/total"].backward()
            reference_grad = model.radar_only.support_head[-1].weight.grad.clone()
            model.zero_grad(set_to_none=True)
            shuffled_cached["loss/total"].backward()
            torch.testing.assert_close(reference_grad,
                                       model.radar_only.support_head[-1].weight.grad)

            class SyntheticDataset:
                def __init__(self):
                    self.frames = frames.copy()

                def __len__(self):
                    return len(self.frames)

                def __getitem__(self, index):
                    frame = self.frames[index]
                    position = next(i for i, item in enumerate(frames)
                                    if item.frame_id == frame.frame_id)
                    return {"frame_id": frame.frame_id, "split": "train",
                            "radar": radar[position], "clean_lidar": lidar[position]}

            prebuilt = Stage1StaticGeometryCache(root / "prebuilt", config, frames)
            prebuild_split(model, prebuilt, SyntheticDataset(), "train",
                           batch_size=2, num_workers=0, device="cpu")
            self.assertTrue(all(prebuilt.has_entry("train", frame.frame_id)
                                for frame in frames))
            prebuild_split(model, prebuilt, SyntheticDataset(), "train",
                           batch_size=2, num_workers=0, device="cpu")

    def test_changed_source_invalidates_disk_entry(self):
        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = Stage1Config(grid=grid)
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            frame = _frame(root, "00001")
            cache = Stage1StaticGeometryCache(root / "cache", config, [frame])
            before = cache._signature("train", "00001")
            frame.radar_path.write_bytes(b"changed radar file contents")
            fresh = Stage1StaticGeometryCache(root / "cache", config, [frame])
            self.assertNotEqual(before, fresh._signature("train", "00001"))


if __name__ == "__main__":
    unittest.main()
