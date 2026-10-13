"""Behavioral checks for the inference-only region comparison methods."""

import numpy as np

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.interpolation import (
    edge_aware_seed_field, gaussian_seed_field, inverse_distance_seed_field,
    local_signed_distance_field, nearest_seed_field, trilinear_seed_field,
)


GRID = VoxelGrid(minimum_xyz=(-4., -4., -4.), maximum_xyz=(4., 4., 4.),
                 size_xyz=(.2, .2, .25))


def test_single_seed_has_compact_support_without_ellipsoid() -> None:
    field = trilinear_seed_field(np.array([[0., 0., 0.]]), np.array([.8]), GRID,
                                 lattice_stride=2, seed_threshold=.25)
    xyz = field.centers_xyz(GRID)
    assert len(xyz) > 0
    assert np.max(np.abs(xyz), axis=0).max() <= .5
    assert np.isclose(field.scores.max(), .8)
    assert np.all((field.scores >= 0) & (field.scores <= 1))


def test_edge_guidance_blocks_incompatible_proposals() -> None:
    points = np.array([[0., 0., 0.], [1., 0., 0.]])
    scores = np.array([.8, .8])
    opposite = np.array([[1., 0.], [-1., 0.]])
    same = np.array([[1., 0.], [1., 0.]])
    blocked = edge_aware_seed_field(points, scores, opposite, GRID,
                                    groups=np.array([0, 1]), max_link_m=2.)
    connected = edge_aware_seed_field(points, scores, same, GRID,
                                      groups=np.array([0, 1]), max_link_m=2.)
    assert blocked.accepted_edges == 0
    assert connected.accepted_edges == 1
    assert len(connected.coordinates_xyz) > len(blocked.coordinates_xyz)


def test_alternative_slots_from_one_anchor_do_not_bridge() -> None:
    points = np.array([[0., 0., 0.], [1., 0., 0.]])
    scores = np.array([.8, .8])
    features = np.array([[1., 0.], [1., 0.]])
    field = edge_aware_seed_field(points, scores, features, GRID,
                                  groups=np.array([5, 5]), max_link_m=2.)
    assert field.accepted_edges == 0


def test_nearest_keeps_one_cell_and_duplicate_proposals_do_not_add_scores() -> None:
    xyz = np.array([[.03, .02, -.04], [.03, .02, -.04]])
    field = nearest_seed_field(xyz, np.array([.8, .7]), GRID)
    assert len(field.coordinates_xyz) == 1
    assert np.isclose(field.scores[0], .8)


def test_compact_radial_kernels_do_not_bridge_distant_proposals() -> None:
    xyz = np.array([[.03, .07, -.04], [2., 0., 0.]])
    for method in (gaussian_seed_field, inverse_distance_seed_field):
        field = method(xyz, np.array([.8, .8]), GRID, lattice_stride=2)
        centers = field.centers_xyz(GRID)
        assert len(centers) > 2
        assert not np.any((centers[:, 0] > .6) & (centers[:, 0] < 1.4))
        assert np.isclose(field.scores.max(), .8)
        assert np.all((field.scores > 0) & (field.scores <= 1))


def test_local_signed_distance_orients_a_planar_patch_toward_sensor() -> None:
    xyz = np.array([[x, y, -1.] for x in (-.3, 0., .3)
                    for y in (-.3, 0., .3)])
    field = local_signed_distance_field(
        xyz, np.full(len(xyz), .9), GRID, groups=np.arange(len(xyz)),
        normal_radius_m=1., surface_band_m=.2)
    centers = field.centers_xyz(GRID)
    assert field.source_seeds == len(xyz)
    assert field.signed_distance_m is not None
    assert field.signed_distance_m.shape == field.scores.shape
    assert np.any((centers[:, 2] > -1.) & (field.signed_distance_m > 0))
    assert np.any((centers[:, 2] < -1.) & (field.signed_distance_m < 0))
    assert np.all((field.scores > 0) & (field.scores <= 1))


def test_local_signed_distance_abstains_on_a_line() -> None:
    xyz = np.array([[x, 0., -1.] for x in (-.5, -.3, -.1, .1, .3, .5)])
    field = local_signed_distance_field(
        xyz, np.full(len(xyz), .9), GRID, groups=np.arange(len(xyz)))
    assert len(field.coordinates_xyz) == 0


def test_local_signed_distance_does_not_treat_one_anchor_as_a_patch() -> None:
    xyz = np.array([[x, y, -1.] for x in (-.3, 0., .3)
                    for y in (-.3, 0., .3)])
    field = local_signed_distance_field(
        xyz, np.full(len(xyz), .9), GRID, groups=np.zeros(len(xyz), dtype=int))
    assert len(field.coordinates_xyz) == 0
