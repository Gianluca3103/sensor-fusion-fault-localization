"""Synthetic forward/backward and checkpoint probe for residual Stage II."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.losses import residual_loss
from models.radar_lidar_stage2_residual.model import ResidualRadarLidarStage2
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from scripts.smoke_stage2_minkowski import _stage1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)
    grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (.5, .5, .5))
    config = ResidualStage2Config(channels=(8, 12, 16, 24), conditioning_dim=8,
                                  query_radii_m=(.75, 1.5, 3., 6.), max_neighbors=4,
                                  expansion_zyx=(1, 1, 1), max_candidate_sites=3000)
    evidence = _stage1(grid, device)
    model = ResidualRadarLidarStage2((4, 4, 4, 4), config).to(device).train()
    clean = torch.tensor([[[2.25, 2.25, 2.25, .1], [2.75, 2.25, 2.75, .2]]], device=device)
    faulty = clean[:, :1].clone()
    region = {"x_range": [0., 8.], "y_range": [0., 8.], "min_range_m": 0., "max_range_m": 20.}
    output = model(evidence, grid, faulty, torch.ones((1, 1), device=device, dtype=torch.bool), [region])
    target = make_residual_targets(output.domain, clean,
                                   torch.ones((1, 2), device=device, dtype=torch.bool),
                                   output.coverage, free_ray_tolerance_m=.15)
    losses = residual_loss(output, target, config)
    assert target.addition.sum() >= 1 and target.observed.sum() >= 1
    assert output.metadata["sites"]["input"] == output.metadata["sites"]["output"]
    assert bool(torch.isfinite(losses["total"]))
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert all(sites.features.grad is None for sites in evidence.features.values())
    model.eval()
    with torch.no_grad():
        before = model(evidence, grid, faulty, torch.ones((1, 1), device=device, dtype=torch.bool),
                       [region]).addition_logits
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "residual.pt"
            torch.save(model.state_dict(), path)
            restored = ResidualRadarLidarStage2((4, 4, 4, 4), config).to(device).eval()
            restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            after = restored(evidence, grid, faulty,
                             torch.ones((1, 1), device=device, dtype=torch.bool), [region]).addition_logits
        assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)
    print(json.dumps({"device": device, "candidate_sites": len(output.domain.coordinates),
                      "addition_sites": int(target.addition.sum()),
                      "observed_sites": int(target.observed.sum()),
                      "generated_sites": len(output.reconstructed_points_xyz),
                      "finite_loss": True, "finite_gradients": True,
                      "checkpoint_reproduces_output": True, "diffusion_enabled": False}, indent=2))


if __name__ == "__main__":
    main()
