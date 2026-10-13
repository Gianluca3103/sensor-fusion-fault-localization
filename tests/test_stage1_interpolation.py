"""Behavioral checks for the inference-only region comparison methods."""

import numpy as np

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.interpolation import edge_aware_seed_field, trilinear_seed_field


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
