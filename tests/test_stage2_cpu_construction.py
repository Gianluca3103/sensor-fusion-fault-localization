"""Candidate support and spatial queries remain invariant under CPU optimizations."""

from __future__ import annotations

from itertools import product
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.model import Stage1Output, SurfaceProposals
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage2.candidate_domain import make_candidates
from models.radar_lidar_stage2.cpu_query import query_workers
from models.radar_lidar_stage2.stage1_conditioning import physical_neighbor_map


def test_overlapping_regions_keep_the_strongest_score_and_batch_identity():
    grid = VoxelGrid((0., 0., 0.), (4., 4., 4.), (.5, .5, .5))
    coords = torch.tensor([[0, 2, 2, 2], [1, 2, 2, 2]])
    fine = SparseSites(coords, torch.ones((2, 4)), grid.shape_zyx)
    proposals = SurfaceProposals(
        xyz=torch.tensor([[[1.25, 1.25, 1.25], [1.75, 1.25, 1.25]],
                          [[1.25, 1.25, 1.25], [7., 7., 7.]]]),
        score=torch.tensor([[.6, .9], [.8, .99]]),
        radii_xyz=torch.tensor([[[1., 1., 1.], [1., 1., 1.]],
                                [[1., 1., 1.], [1., 1., 1.]]]),
        anchor_batch=torch.tensor([0, 1]),
        anchor_xyz=torch.tensor([[1., 1., 1.], [1., 1., 1.]]),
    )
    evidence = Stage1Output({"s1": fine}, fine.replace_features(torch.ones((2, 1))),
                            {"s1": coords}, {}, proposals)
    domain = make_candidates(evidence, grid, confidence_threshold=.25)
    expected = {}
    for batch, z, y, x in product(range(2), range(8), range(8), range(8)):
        center = np.array([x, y, z], dtype=np.float64) * .5 + .25
        strongest = 0.
        for anchor in range(2):
            if anchor != batch:
                continue
            for slot in range(2):
                xyz = proposals.xyz[anchor, slot].numpy().astype(np.float64)
                radius = proposals.radii_xyz[anchor, slot].numpy().astype(np.float64)
                squared = (((center - xyz) / radius) ** 2).sum()
                if squared <= 1 + 1e-9:
                    strongest = max(strongest, float(proposals.score[anchor, slot]) * (1 - .5 * squared))
        if strongest:
            expected[(batch, z, y, x)] = strongest
    assert domain.counts["accepted_region_proposals"] == 3
    assert domain.coordinates.tolist() == [list(row) for row in sorted(expected)]
    np.testing.assert_allclose(domain.confidence.numpy(), [expected[row] for row in sorted(expected)],
                               rtol=1e-6, atol=1e-7)


def test_parallel_neighbor_queries_match_serial_queries():
    rng = np.random.default_rng(31)
    query = rng.normal(size=(5000, 3)).astype(np.float32)
    source = rng.normal(size=(800, 3)).astype(np.float32)
    query_batch = np.arange(len(query)) % 2
    source_batch = np.arange(len(source)) % 2
    with patch.dict("os.environ", {"STAGE2_KDTREE_WORKERS": "1"}):
        serial = physical_neighbor_map(query, query_batch, source, source_batch, .8, 8)
    with patch.dict("os.environ", {"STAGE2_KDTREE_WORKERS": "4"}):
        parallel = physical_neighbor_map(query, query_batch, source, source_batch, .8, 8)
    np.testing.assert_array_equal(parallel[0], serial[0])
    np.testing.assert_array_equal(parallel[1], serial[1])
    assert query_workers(10) == 1


def test_frozen_source_pooling_matches_projected_source_and_parameter_gradients():
    torch.manual_seed(47)
    source = torch.randn(29, 128)
    indices = torch.randint(0, 29, (107, 8))
    weights = torch.rand(107, 8)
    weights = weights / weights.sum(1, keepdim=True)
    weights[0] = 0  # A query without neighbors still receives the linear bias.
    linear = torch.nn.Linear(128, 32)
    with torch.no_grad():
        pooled = source.new_zeros((len(indices), source.shape[1]))
        for neighbor in range(indices.shape[1]):
            pooled.add_(source.index_select(0, indices[:, neighbor])
                        * weights[:, neighbor, None])
    original = linear(pooled)
    projected_source = F.linear(source, linear.weight, bias=None)
    optimized = (projected_source[indices] * weights[..., None]).sum(1) + linear.bias
    torch.testing.assert_close(optimized, original, rtol=1e-5, atol=1e-6)
    old_grads = torch.autograd.grad(optimized.square().sum(),
                                   (linear.weight, linear.bias), retain_graph=True)
    new_grads = torch.autograd.grad(original.square().sum(),
                                   (linear.weight, linear.bias))
    for old, new in zip(old_grads, new_grads):
        torch.testing.assert_close(old, new, rtol=1e-5, atol=2e-5)
