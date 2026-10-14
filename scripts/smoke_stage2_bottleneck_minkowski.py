"""Synthetic Stage-II v2 GPU forward, gradient, sampling and reload check."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import torch

from models.radar_lidar_stage1.bottleneck_support import BottleneckSupportOutput
from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from models.radar_lidar_stage2_bottleneck.config import BottleneckDiffusionConfig
from models.radar_lidar_stage2_bottleneck.losses import reconstruction_loss, training_state
from models.radar_lidar_stage2_bottleneck.model import BottleneckSparseDiffusion
from models.radar_lidar_stage2_bottleneck.train import protect_target


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(23)
    grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))
    seed_coords = torch.tensor([[0, z, y, x] for z in range(1, 5)
                                for y in range(1, 5) for x in range(1, 5)],
                               dtype=torch.long, device=device)
    confidence = SparseSites(seed_coords,
                             torch.full((len(seed_coords), 1), .9, device=device),
                             grid.scale_shape(2), 2)
    coarse_coords = torch.unique(torch.cat((seed_coords[:, :1], seed_coords[:, 1:] // 4), dim=1),
                                 dim=0, sorted=True)
    features = torch.randn((len(coarse_coords), 4), device=device, requires_grad=True)
    bottleneck = SparseSites(coarse_coords, features, grid.scale_shape(8), 8)
    evidence = BottleneckSupportOutput(bottleneck, confidence)
    residual = ResidualStage2Config(channels=(8, 12, 16, 24), conditioning_dim=8,
                                    query_radii_m=(.75, 1.5, 3., 6.), max_neighbors=4,
                                    confidence_threshold=.2, candidate_method="edge_aware",
                                    expansion_zyx=(0, 0, 0))
    config = BottleneckDiffusionConfig(residual=residual, channels=(8, 12, 16, 24),
                                       timestep_dim=32, attention_heads=4,
                                       attention_max_neighbors=4, timesteps=4)
    model = BottleneckSparseDiffusion((4,), config).to(device).train()
    clean = torch.tensor([[[2.25, 2.25, 2.25, .1], [2.75, 2.25, 2.75, .2]]], device=device)
    faulty = clean[:, :1].clone()
    context = model.prepare(evidence, grid, faulty,
                            torch.ones((1, 1), dtype=torch.bool, device=device),
                            [{"full_scan": True}])
    target = protect_target(make_residual_targets(
        context.domain, clean, torch.ones((1, 2), dtype=torch.bool, device=device),
        context.coverage, free_ray_tolerance_m=.15), context.coverage)
    assert len(context.domain.coordinates) and bool(target.addition.any())
    state = training_state(target)
    timestep = torch.tensor([2], device=device)
    noise = torch.randn_like(state)
    noisy = model.schedule.mix(state, noise,
                               timestep[context.domain.coordinates[:, 0]])
    predicted, logits, offsets, counts = model.predict_noise(context, noisy, timestep)
    assert predicted.shape == state.shape
    assert counts["input"] == counts["output"] == len(context.domain.coordinates)
    loss = reconstruction_loss(predicted, noise, logits, offsets,
                               target, context.coverage, config)["total"]
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert model.fusion.output.weight.grad is not None
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert features.grad is None
    if device == "cuda":
        model.zero_grad(set_to_none=True)
        # Rebuild the trainable conditioning graph for the second backward.
        mixed_context = model.prepare(evidence, grid, faulty,
                                      torch.ones((1, 1), dtype=torch.bool, device=device),
                                      [{"full_scan": True}])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            p2, l2, o2, _ = model.predict_noise(mixed_context, noisy, timestep)
        mixed = reconstruction_loss(p2.float(), noise, l2.float(), o2.float(),
                                    target, mixed_context.coverage, config)["total"]
        mixed.backward()
        assert bool(torch.isfinite(mixed))
    model.eval()
    with torch.no_grad():
        full = model.sample(context, frame_seeds=[41])
        fast = model.sample(context, frame_seeds=[41], method="ddim", sampling_steps=2)
        assert fast.metadata["sampling_steps"] == 2
        assert bool(torch.isfinite(full.addition_probability).all())
        assert bool(torch.isfinite(fast.reconstructed_points_xyz).all())
        assert len(full.reconstructed_points_xyz) <= int(context.coverage.may_add.sum())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage2_v2.pt"
            torch.save(model.state_dict(), path)
            restored = BottleneckSparseDiffusion((4,), config).to(device).eval()
            restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            new_context = restored.prepare(evidence, grid, faulty,
                                           torch.ones((1, 1), dtype=torch.bool, device=device),
                                           [{"full_scan": True}])
            reproduced = restored.sample(new_context, frame_seeds=[41])
            assert torch.allclose(full.addition_probability,
                                  reproduced.addition_probability, atol=1e-6, rtol=1e-5)
    print(json.dumps({"device": device, "candidate_sites": len(context.domain.coordinates),
                      "positive_sites": int(target.addition.sum()),
                      "fusion_gradients": True, "stage1_gradients": False,
                      "ddpm_steps": config.timesteps, "ddim_steps": 2,
                      "checkpoint_reproduces_output": True}, indent=2))


if __name__ == "__main__":
    main()
