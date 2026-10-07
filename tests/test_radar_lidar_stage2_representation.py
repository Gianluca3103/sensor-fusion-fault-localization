"""Stage-II pre-network tests: physical support and clean centroid targets."""

from __future__ import annotations

import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage2 import decode_centroids, make_candidates, make_targets


GRID = VoxelGrid((0., 0., 0.), (2., 2., 2.), (.5, .5, .5))


def stage1(confidences=(.7, .1)):
    coords = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]])
    sites = SparseSites(coords, torch.ones(2, 4), GRID.shape_zyx)
    confidence = sites.replace_features(torch.tensor(confidences)[:, None])
    return Stage1Output({"s1": sites}, confidence, {"s1": coords}, {})


def test_candidate_threshold_expansion_is_bounded_and_does_not_use_clean_scan():
    output = stage1()
    domain = make_candidates(output, GRID, confidence_threshold=.5, expansion_zyx=(1, 1, 1))
    assert len(domain.coordinates) == 27
    assert torch.all(domain.confidence == .7)
    assert domain.counts["initial_stage1_sites"] == 2
    assert domain.counts["candidate_sites_after_confidence"] == 1
    assert domain.counts["candidate_expansion_ratio"] == 27
    assert len(make_candidates(output, GRID, confidence_threshold=.5,
                               expansion_zyx=(1, 1, 1), max_sites=27).coordinates) <= 27
    empty = make_candidates(output, GRID, confidence_threshold=.9)
    assert len(empty.coordinates) == 0
    assert len(empty.confidence) == 0


def test_clean_centroid_target_and_physical_decode():
    domain = make_candidates(stage1(), GRID, confidence_threshold=.5,
                             expansion_zyx=(0, 0, 0))
    clean = torch.tensor([[[.65, .70, .80, .2], [.75, .80, .90, .3],
                           [1.30, 1.30, 1.30, .4]]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    targets = make_targets(domain, clean, valid)
    assert targets.occupied.tolist() == [True]
    assert targets.clean_point_count.tolist() == [2]
    assert targets.clean_points_in_grid == 3
    assert targets.clean_points_in_candidates == 2
    assert abs(targets.clean_point_coverage - 2 / 3) < 1e-6
    expected = clean[0, :2, :3].mean(0)
    assert torch.allclose(targets.clean_centroid_xyz[0], expected)
    assert torch.allclose(decode_centroids(domain, targets.offsets_normalized)[0], expected)
    assert bool((targets.offsets_normalized.abs() <= .5).all())


def test_empty_candidate_is_not_automatically_occupied():
    domain = make_candidates(stage1(), GRID, confidence_threshold=.5,
                             expansion_zyx=(1, 1, 1))
    clean = torch.tensor([[[.70, .75, .80, .1]]])
    targets = make_targets(domain, clean, torch.ones((1, 1), dtype=torch.bool))
    assert int(targets.occupied.sum()) == 1
    assert int((~targets.occupied).sum()) == 26
    assert torch.all(targets.offsets_normalized[~targets.occupied] == 0)


def test_candidate_cap_prefers_stronger_radar_and_stays_in_grid():
    output = stage1((.95, .65))
    domain = make_candidates(output, GRID, confidence_threshold=.5,
                             expansion_zyx=(1, 1, 1), max_sites=27)
    assert len(domain.coordinates) == 27
    assert domain.counts["candidate_sites_after_confidence"] == 2
    assert domain.counts["selected_seeds_after_cap"] == 1
    assert domain.counts["cap_applied"] == 1
    assert torch.all(domain.confidence == .95)
    assert bool((domain.coordinates[:, 1:] >= 0).all())
    assert bool((domain.coordinates[:, 1:] < torch.tensor(GRID.shape_zyx)).all())


def test_empty_and_multibatch_targets_do_not_cross_match():
    coords = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    sites = SparseSites(coords, torch.ones(2, 4), GRID.shape_zyx)
    output = Stage1Output({"s1": sites}, sites.replace_features(torch.ones(2, 1)),
                          {"s1": coords}, {})
    domain = make_candidates(output, GRID, confidence_threshold=.5,
                             expansion_zyx=(0, 0, 0))
    clean = torch.tensor([[[.65, .70, .80, .2]], [[1.55, 1.55, 1.55, .3]]])
    valid = torch.ones((2, 1), dtype=torch.bool)
    targets = make_targets(domain, clean, valid)
    assert targets.occupied.tolist() == [True, False]
    assert targets.clean_point_count.tolist() == [1, 0]
    empty = make_targets(domain, clean, torch.zeros_like(valid))
    assert not bool(empty.occupied.any())
    assert empty.clean_points_in_grid == 0


def test_only_visible_space_before_clean_first_return_is_negative():
    grid = VoxelGrid((0., -1., -1.), (5., 1., 1.), (.5, .5, .5))
    coords = torch.tensor([[0, 2, 2, x] for x in (1, 3, 5, 7)])
    sites = SparseSites(coords, torch.ones(4, 4), grid.shape_zyx)
    evidence = Stage1Output({"s1": sites}, sites.replace_features(torch.ones(4, 1)),
                            {"s1": coords}, {})
    domain = make_candidates(evidence, grid, confidence_threshold=.5, expansion_zyx=(0, 0, 0))
    # Centers are x=.75,1.75,2.75,3.75, y=z=.25. The clean ray ends
    # at x=2.75,y=z=.25; x=3.75 is behind the first return.
    clean = torch.tensor([[[2.75, .25, .25, .1]]])
    target = make_targets(domain, clean, torch.ones((1, 1), dtype=torch.bool),
                          free_ray_tolerance_m=.15)
    assert target.occupied.tolist() == [False, False, True, False]
    assert target.known_free.tolist() == [False, True, False, False]


def test_stage2_configuration_rejects_active_diffusion():
    from models.radar_lidar_stage2.config import Stage2Config
    try:
        Stage2Config(diffusion_enabled=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Diffusion must remain disabled")


def test_nearer_return_prevents_free_label_behind_it():
    grid = VoxelGrid((0., -1., -1.), (6., 1., 1.), (.5, .5, .5))
    coords = torch.tensor([[0, 2, 2, 6]])
    sites = SparseSites(coords, torch.ones(1, 4), grid.shape_zyx)
    evidence = Stage1Output({"s1": sites}, sites.replace_features(torch.ones(1, 1)),
                            {"s1": coords}, {})
    domain = make_candidates(evidence, grid, confidence_threshold=.5, expansion_zyx=(0, 0, 0))
    clean = torch.tensor([[[2.25, .25, .25, .1], [4.25, .25, .25, .2]]])
    target = make_targets(domain, clean, torch.ones((1, 2), dtype=torch.bool),
                          free_ray_tolerance_m=.25)
    assert not bool(target.occupied[0])
    assert not bool(target.known_free[0])
