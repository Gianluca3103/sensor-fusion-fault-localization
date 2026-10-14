"""CPU checks for Stage-II noise, fusion and supervision boundaries."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.bottleneck_support import BottleneckSupportOutput
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage2_bottleneck.attention import LocalBottleneckFusion
from models.radar_lidar_stage2_bottleneck.config import BottleneckDiffusionConfig
from models.radar_lidar_stage2_bottleneck.losses import reconstruction_loss, training_state
from models.radar_lidar_stage2_bottleneck.schedule import NoiseSchedule
from models.radar_lidar_stage2_bottleneck.model import BottleneckSparseDiffusion
from models.radar_lidar_stage2_bottleneck.train import (
    LOSS_NAMES, evaluate, load_frozen_stage1, protect_target,
)
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.coverage import FaultyCoverage
from models.radar_lidar_stage2_residual.targets import ResidualTargets


def small_config(mode="cross_attention"):
    residual = ResidualStage2Config(channels=(8, 12, 16, 24),
                                    query_radii_m=(.75, 1.5, 3., 6.),
                                    conditioning_dim=8, max_neighbors=4)
    return BottleneckDiffusionConfig(residual=residual, channels=(8, 12, 16, 24),
                                     timestep_dim=32, fusion_mode=mode,
                                     attention_heads=4, timesteps=50)


def test_stage1_loader_requires_radar_student_and_freezes_weights():
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(bottleneck_support_only=True,
                                          support_seed_mode="radar_s2")

    student = Student().train()
    saved = {"stage1_interface": "bottleneck_support_v2",
             "data": {"radar_variant": "radar_20frames_verified_doppler_radial"}}
    with patch("models.radar_lidar_stage2_bottleneck.train.load_encoder",
               return_value=(student, saved)):
        loaded, loaded_saved = load_frozen_stage1("unused.ckpt", "cpu",
                                                  "radar_20frames_verified_doppler_radial")
    assert loaded is student and loaded_saved is saved
    assert not loaded.training and not loaded.weight.requires_grad
    with patch("models.radar_lidar_stage2_bottleneck.train.load_encoder",
               return_value=(Student(), saved)):
        with pytest.raises(ValueError, match="radar variant"):
            load_frozen_stage1("unused.ckpt", "cpu", "other_radar")
    invalid = Student()
    invalid.config.support_seed_mode = "coarse_dense"
    with patch("models.radar_lidar_stage2_bottleneck.train.load_encoder",
               return_value=(invalid, saved)):
        with pytest.raises(ValueError, match="radar-seed"):
            load_frozen_stage1("unused.ckpt", "cpu")


def test_validation_counts_frames_without_legacy_frame_key():
    class FakeMetrics:
        def add(self, *args):
            pass

        def summary(self):
            return {"global_missing_f1_0.2m": 0.1}

    class FakeModel(torch.nn.Module):
        def sample(self, *args, **kwargs):
            return object()

    batch = {"frame_id": ["00001", "00002"],
             "clean_lidar": torch.zeros((2, 0, 4)),
             "clean_lidar_valid": torch.zeros((2, 0), dtype=torch.bool),
             "faulty_lidar": torch.zeros((2, 0, 4)),
             "faulty_lidar_valid": torch.zeros((2, 0), dtype=torch.bool),
             "fault_kind": ["fog", "total_loss"]}
    losses = {name: torch.tensor(0.) for name in LOSS_NAMES}
    with (patch("models.radar_lidar_stage2_bottleneck.train.ReconstructionMetrics", FakeMetrics),
          patch("models.radar_lidar_stage2_bottleneck.train.prepare_context",
                return_value=object()),
          patch("models.radar_lidar_stage2_bottleneck.train.make_target",
                return_value=object()),
          patch("models.radar_lidar_stage2_bottleneck.train.denoise_batch",
                return_value=(losses, {}))):
        report = evaluate(FakeModel(), torch.nn.Module(), [batch], small_config(),
                          "cpu", sampling_method="ddim", sampling_steps=10)
    assert report["frames"] == 2
    assert report["sampling"]["full_pipeline_seconds_per_frame"] is not None
    assert report["sampling"]["stage1_encoder_included_in_timing"]


def test_config_roundtrip_and_only_edge_aware_growth(tmp_path):
    config = small_config()
    path = tmp_path / "stage2.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    assert BottleneckDiffusionConfig.from_json(path) == config
    with pytest.raises(ValueError, match="edge-aware"):
        BottleneckDiffusionConfig(residual=ResidualStage2Config(expansion_zyx=(1, 1, 1)))
    with pytest.raises(ValueError, match="capped"):
        BottleneckDiffusionConfig(residual=ResidualStage2Config(max_candidate_sites=100))


def test_forward_noise_and_ddim_reverse_exact_with_perfect_prediction():
    schedule = NoiseSchedule(50, "cosine")
    assert schedule.sampling_timesteps(10, "ddim")[0] == 49
    assert schedule.sampling_timesteps(10, "ddim")[-1] == 0
    with pytest.raises(ValueError, match="DDPM uses every"):
        schedule.sampling_timesteps(10, "ddpm")
    clean = torch.tensor([[1., .2, -.4, .6], [-1., 0., 0., 0.]])
    noise = torch.tensor([[.3, -.7, 1., .2], [-.3, .4, .5, -.8]])
    state = schedule.mix(clean, noise, torch.tensor([49, 49]))
    steps = schedule.sampling_timesteps(10, "ddim")
    for index, t in enumerate(steps):
        previous = steps[index + 1] if index + 1 < len(steps) else -1
        state = schedule.reverse_step(state, noise, t, previous, method="ddim")
    assert torch.allclose(state, clean, atol=2e-4)
    state = schedule.mix(clean, noise, torch.tensor([4, 4]))
    output = schedule.reverse_step(state, noise, 4, 3, method="ddpm",
                                   random=torch.zeros_like(state))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("mode", ["cross_attention", "concat"])
def test_fusion_uses_local_stage1_features_without_stage1_gradients(mode):
    grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (.5, .5, .5))
    query = torch.randn((2, 8), requires_grad=True)
    coords = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]])
    source = torch.randn((3, 4), requires_grad=True)
    centers = torch.tensor([[1.5, 1.5, 1.5], [3., 2., 2.], [1.5, 1.5, 1.5]])
    source_batch = torch.tensor([0, 0, 1])
    fusion = LocalBottleneckFusion(8, 4, heads=2, mode=mode,
                                   radius_m=6., max_neighbors=2)
    output = fusion(query, coords, source, centers, source_batch, grid, 8)
    assert output.shape == query.shape
    output.square().mean().backward()
    assert query.grad is not None
    assert source.grad is None
    assert fusion.output[0].weight.grad is not None if mode == "concat" else fusion.q.weight.grad is not None
    far = centers + 1000
    unchanged = fusion(query.detach(), coords, source.detach(), far,
                       source_batch, grid, 8)
    assert torch.equal(unchanged, query.detach())


def test_loss_masks_unknown_and_preserves_blocked_candidates():
    target = ResidualTargets(
        addition=torch.tensor([True, False, False, False]),
        observed=torch.tensor([False, True, False, False]),
        known_free=torch.tensor([False, False, True, False]),
        offsets_normalized=torch.tensor([[.2, -.1, .3], [0., 0., 0.],
                                         [0., 0., 0.], [0., 0., 0.]]),
        clean=None)
    coverage = FaultyCoverage(features=torch.zeros((4, 7)),
                              measured_voxel=torch.tensor([False, True, False, False]),
                              measured_ray=torch.zeros(4, dtype=torch.bool),
                              in_fault_region=torch.ones(4, dtype=torch.bool))
    state = training_state(target, generator=torch.Generator().manual_seed(9))
    assert state[3, 0].abs() == 1
    noise = torch.randn((4, 4))
    prediction = torch.zeros((4, 4), requires_grad=True)
    logits = torch.zeros(4, requires_grad=True)
    offsets = torch.zeros((4, 3), requires_grad=True)
    losses = reconstruction_loss(prediction, noise, logits, offsets,
                                 target, coverage, small_config())
    assert all(torch.isfinite(value) for key, value in losses.items()
               if key not in {"positive_sites", "free_sites", "observed_sites", "unknown_sites"})
    losses["total"].backward()
    assert torch.equal(prediction.grad[3], torch.zeros(4))
    assert logits.grad[3] == 0
    assert offsets.grad[0].abs().sum() > 0
    assert offsets.grad[1:].abs().sum() == 0


def test_full_conditioning_and_sampling_boundary_without_minkowski_engine():
    class Sparse:
        def __init__(self, *, features, coordinates=None, coordinate_map_key=None,
                     coordinate_manager=None):
            self.F = features
            if coordinates is not None:
                self.coordinate_map_key = "input"
                self.coordinate_manager = {"input": coordinates}
            else:
                self.coordinate_map_key = coordinate_map_key
                self.coordinate_manager = coordinate_manager
            self.C = self.coordinate_manager[self.coordinate_map_key]

    class IdentityUNet(torch.nn.Module):
        def __init__(self, channels, strides, blocks, time_dim):
            super().__init__()
            self.down = torch.nn.Linear(channels[0], channels[-1])
            self.up = torch.nn.Linear(channels[-1], channels[0])

        def forward(self, sparse, time_embedding, bottleneck_fusion):
            manager = sparse.coordinate_manager
            manager["bottleneck"] = torch.zeros((1, 4), dtype=torch.int32,
                                                  device=sparse.C.device)
            bottleneck = Sparse(features=self.down(sparse.F).mean(dim=0, keepdim=True),
                                coordinate_map_key="bottleneck", coordinate_manager=manager)
            fused = bottleneck_fusion(bottleneck)
            decoded = Sparse(features=sparse.F + self.up(fused.F).expand_as(sparse.F),
                             coordinate_map_key=sparse.coordinate_map_key,
                             coordinate_manager=manager)
            return decoded, {"input": len(sparse.F), "output": len(decoded.F)}

    grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))
    coords = torch.tensor([[0, z, y, x] for z in range(1, 5)
                           for y in range(1, 5) for x in range(1, 5)])
    confidence = SparseSites(coords, torch.full((len(coords), 1), .9),
                             grid.scale_shape(2), 2)
    coarse_coords = torch.unique(torch.cat((coords[:, :1], coords[:, 1:] // 4), dim=1),
                                 dim=0, sorted=True)
    source = torch.randn((len(coarse_coords), 4), requires_grad=True)
    bottleneck = SparseSites(coarse_coords, source, grid.scale_shape(8), 8)
    evidence = BottleneckSupportOutput(bottleneck, confidence)
    config = small_config()
    config = BottleneckDiffusionConfig.from_dict({**config.as_dict(), "timesteps": 4})
    fake_me = SimpleNamespace(SparseTensor=Sparse)
    with (patch("models.radar_lidar_stage2_bottleneck.model._require_me"),
          patch("models.radar_lidar_stage2_bottleneck.model.ME", fake_me),
          patch("models.radar_lidar_stage2_bottleneck.unet.ME", fake_me),
          patch("models.radar_lidar_stage2_bottleneck.model.BottleneckDiffusionUNet", IdentityUNet)):
        model = BottleneckSparseDiffusion((4,), config)
        faulty = torch.tensor([[[2.25, 2.25, 2.25, .1]]])
        context = model.prepare(evidence, grid, faulty, torch.ones((1, 1), dtype=torch.bool),
                                [{"full_scan": True}])
        assert len(context.domain.coordinates) > len(confidence.coords)
        clean = torch.tensor([[[2.25, 2.25, 2.25, .1], [2.75, 2.25, 2.75, .2]]])
        from models.radar_lidar_stage2_residual.targets import make_residual_targets
        target = protect_target(make_residual_targets(
            context.domain, clean, torch.ones((1, 2), dtype=torch.bool),
            context.coverage, free_ray_tolerance_m=.15), context.coverage)
        assert torch.equal(target.observed, ~context.coverage.may_add)
        state = training_state(target)
        noise = torch.randn_like(state)
        t = torch.tensor([2])
        mixed = model.schedule.mix(state, noise, t[context.domain.coordinates[:, 0]])
        prediction, logits, offsets, counts = model.predict_noise(context, mixed, t)
        assert counts["input"] == counts["output"] == len(context.domain.coordinates)
        loss = reconstruction_loss(prediction, noise, logits, offsets, target,
                                   context.coverage, config)["total"]
        loss.backward()
        assert model.fusion.output.weight.grad is not None
        assert source.grad is None
        with torch.no_grad():
            first = model.sample(context, frame_seeds=[7])
            repeat = model.sample(context, frame_seeds=[7])
            fast = model.sample(context, frame_seeds=[7], method="ddim", sampling_steps=2)
        assert torch.allclose(first.addition_probability, repeat.addition_probability)
        assert torch.isfinite(fast.reconstructed_points_xyz).all()
        assert len(first.reconstructed_points_xyz) <= int(context.coverage.may_add.sum())
