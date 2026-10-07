"""Engine-independent invariants for radar-only Stage-II training."""

from __future__ import annotations

import numpy as np
import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage2.candidate_domain import make_candidates
from models.radar_lidar_stage2.config import Stage2Config
from models.radar_lidar_stage2.losses import reconstruction_loss
from models.radar_lidar_stage2.metrics import Stage2MetricAccumulator
from models.radar_lidar_stage2.reconstruction_model import Stage2Output
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning, physical_neighbor_map
from models.radar_lidar_stage2.voxel_target import make_targets


GRID = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))


def _evidence():
    levels = {}
    for i in range(4):
        stride = 2**i
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]])
        levels[f"s{i+1}"] = SparseSites(coords, torch.ones(2, 4), GRID.scale_shape(stride), stride)
    conf = levels["s1"].replace_features(torch.tensor([[.9], [.9]]))
    return Stage1Output(levels, conf, {k: v.coords for k,v in levels.items()}, {})


def test_physical_queries_keep_batch_and_radius():
    query = np.array([[0., 0., 0.], [0., 0., 0.]], np.float32)
    source = np.array([[.1, 0., 0.], [10., 0., 0.]], np.float32)
    idx, dist = physical_neighbor_map(query, np.array([0, 1]), source,
                                      np.array([0, 1]), 1., 1)
    assert idx[:, 0].tolist() == [0, -1]
    assert abs(float(dist[0, 0]) - .1) < 1e-5


def test_conditioning_uses_all_four_physical_scales_and_ablation():
    evidence = _evidence()
    cfg = Stage2Config(query_radii_m=(1., 2., 4., 8.), expansion_zyx=(0, 0, 0))
    domain = make_candidates(evidence, GRID, confidence_threshold=.5,
                             expansion_zyx=(0, 0, 0))
    module = Stage1Conditioning((4,4,4,4), cfg)
    real = module(evidence, domain)
    zero = module(evidence, domain, ablation="zero")
    assert real.shape == (2, cfg.channels[0])
    assert torch.isfinite(real).all()
    assert not torch.allclose(real, zero)


def test_loss_masks_unknown_and_regresses_only_occupied():
    evidence = _evidence()
    domain = make_candidates(evidence, GRID, confidence_threshold=.5,
                             expansion_zyx=(0, 0, 0))
    clean = torch.tensor([[[.25,.25,.25,.1]]])
    target = make_targets(domain, clean, torch.ones((1,1), dtype=torch.bool))
    assert target.occupied.tolist() == [True, False]
    assert not bool(target.known_free[1])
    logits = torch.tensor([0., 10.], requires_grad=True)
    offsets = torch.zeros((2,3), requires_grad=True)
    output = Stage2Output(domain.coordinates, logits, torch.sigmoid(logits), offsets,
                          torch.empty((0,3)), torch.empty(0), domain.confidence, {}, domain)
    loss = reconstruction_loss(output, target, Stage2Config())
    loss["total"].backward()
    assert logits.grad[0] != 0 and logits.grad[1] == 0
    assert torch.all(offsets.grad[1] == 0)


def test_dataset_metrics_aggregate_counts_not_frame_averages():
    evidence = _evidence()
    domain = make_candidates(evidence, GRID, confidence_threshold=.5,
                             expansion_zyx=(0, 0, 0))
    clean = torch.tensor([[[.25,.25,.25,.1]]])
    valid = torch.ones((1,1), dtype=torch.bool)
    target = make_targets(domain, clean, valid)
    logits = torch.tensor([10., -10.])
    offsets = torch.zeros((2,3))
    output = Stage2Output(domain.coordinates, logits, torch.sigmoid(logits), offsets,
                          domain.centers_xyz[:1], domain.confidence[:1], domain.confidence, {}, domain)
    metrics = Stage2MetricAccumulator()
    metrics.add(output, target, clean, valid)
    metrics.add(output, target, clean, valid)
    summary = metrics.summary()
    assert summary["frames"] == 2
    assert summary["occupancy"]["f1"] == 1.
    assert summary["geometry"]["0.1m"]["f1"] == 1.
    assert summary["clean_points_in_candidates"] == 2
