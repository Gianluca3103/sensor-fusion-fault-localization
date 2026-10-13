"""The disk cache preserves Stage-II candidate geometry and trainable features."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from unittest.mock import patch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2.candidate_domain import make_candidates
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning
from models.radar_lidar_stage2_diffusion.cache import (
    cached_loader, collate_cached, prepare_frame, restore_cached, save_frame,
)
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_diffusion.model import ResidualSparseDiffusion
from models.radar_lidar_stage2_diffusion.train import _context_target, _to_device
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.coverage import faulty_coverage
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from scripts.smoke_stage2_minkowski import _stage1


def test_cached_candidates_targets_and_radar_projection_match(tmp_path):
    torch.manual_seed(31)
    grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))
    base = ResidualStage2Config(channels=(8, 12, 16, 24), conditioning_dim=8,
                                query_radii_m=(.75, 1.5, 3., 6.), max_neighbors=4)
    config = DiffusionConfig(residual=base, timesteps=4)
    evidence = _stage1(grid, "cpu")
    clean = torch.tensor([[[2.25, 2.25, 2.25, .1], [2.75, 2.75, 2.75, .2]]])
    faulty = clean[:, :1].clone()
    batch = {"frame_id": ["00001"], "clean_lidar": clean,
             "clean_lidar_valid": torch.ones((1, 2), dtype=torch.bool),
             "faulty_lidar": faulty,
             "faulty_lidar_valid": torch.ones((1, 1), dtype=torch.bool),
             "fault_region": [{"full_scan": True}]}
    domain = make_candidates(evidence, grid, confidence_threshold=base.confidence_threshold,
                             expansion_zyx=base.expansion_zyx)
    coverage = faulty_coverage(domain, faulty, batch["faulty_lidar_valid"],
                               batch["fault_region"], ray_tolerance_m=base.ray_tolerance_m)
    target = make_residual_targets(domain, clean, batch["clean_lidar_valid"], coverage,
                                   free_ray_tolerance_m=base.free_ray_tolerance_m)
    arrays = prepare_frame(evidence, grid, batch, config)
    assert all(arrays[f"neighbor_index_{scale}"].dtype == np.uint16 for scale in range(4))
    assert all(f"neighbor_distance_{scale}" not in arrays for scale in range(4))
    assert "target_offsets" not in arrays
    path = tmp_path / "00001.npz"
    save_frame(path, arrays, "fast")
    with np.load(path, allow_pickle=False) as archive:
        loaded = {name: archive[name] for name in archive.files}
    cached = collate_cached([loaded], grid)
    cache = cached["cache"]
    assert torch.equal(cache["coordinates"], domain.coordinates)
    assert torch.equal(cache["measured_ray"], coverage.measured_ray)
    assert torch.equal(cache["target_addition"], target.addition)
    assert torch.equal(cache["target_known_free"], target.known_free)
    assert torch.allclose(cache["clean_centroids"], target.clean.clean_centroid_xyz)

    conditioner = Stage1Conditioning((4, 4, 4, 4), base)
    direct = conditioner(evidence, domain)
    precomputed = conditioner.forward_cached(domain, cache["sources"],
                                             cache["indices"], None,
                                             source_centers=cache["source_centers"])
    assert torch.allclose(precomputed, direct, atol=1e-5, rtol=1e-5)
    doubled_batch = collate_cached([loaded, loaded], grid)
    doubled = doubled_batch["cache"]
    for scale in range(4):
        source_count = len(cache["sources"][scale])
        local = cache["indices"][scale]
        second = doubled["indices"][scale][len(local):]
        assert torch.equal(second, torch.where(local >= 0, local + source_count, -1))

    with (patch("models.radar_lidar_stage2_diffusion.model._require_me"),
          patch("models.radar_lidar_stage2_diffusion.model.SparseUNet",
                side_effect=lambda channels: nn.Identity())):
        model = ResidualSparseDiffusion((4, 4, 4, 4), config)
    direct_context = model.prepare(evidence, grid, faulty, batch["faulty_lidar_valid"],
                                   batch["fault_region"])
    restored, cached_target = restore_cached(cached, model)
    assert torch.equal(restored.domain.coordinates, domain.coordinates)
    assert torch.allclose(restored.condition, direct_context.condition, atol=1e-5, rtol=1e-5)
    assert torch.equal(cached_target.clean.occupied, target.clean.occupied)
    doubled_context, doubled_target = restore_cached(doubled_batch, model)
    count = len(domain.coordinates)
    assert torch.allclose(doubled_context.condition[:count], restored.condition,
                          atol=1e-5, rtol=1e-5)
    assert torch.allclose(doubled_context.condition[count:], restored.condition,
                          atol=1e-5, rtol=1e-5)
    assert torch.equal(doubled_target.addition[:count], cached_target.addition)

    manifest = {"complete": True, "splits": {"train": ["00001"], "val": ["00001"]},
                "grid": asdict(grid)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    train_arrays = prepare_frame(evidence, grid, batch, config,
                                 include_validation_clouds=False)
    assert "clean_lidar" not in train_arrays and "faulty_lidar" not in train_arrays
    save_frame(tmp_path / "train" / "00001.npz", train_arrays, "fast")
    args = SimpleNamespace(preprocess_cache_root=tmp_path, batch_size=1, num_workers=0)
    loaded_batch = _to_device(next(iter(cached_loader(args, "train", 0, False, {}))), "cpu")
    loaded_context, loaded_target = _context_target(model, None, loaded_batch, config)
    assert torch.allclose(loaded_context.condition, direct_context.condition,
                          atol=1e-5, rtol=1e-5)
    assert torch.equal(loaded_target.addition, target.addition)
    assert loaded_batch["clean_lidar"].shape[1] == 0
    assert loaded_batch["faulty_lidar"].shape[1] == 0
