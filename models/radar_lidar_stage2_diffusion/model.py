"""Denoise additions on a fixed radar-supported sparse 3D domain."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from models.radar_lidar_stage1.model import Stage1Output
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.candidate_domain import CandidateDomain, make_candidates
from models.radar_lidar_stage2.sparse_unet import ME, SparseUNet, _require_me
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning
from models.radar_lidar_stage2.voxel_target import decode_centroids
from models.radar_lidar_stage2_residual.coverage import FaultyCoverage, faulty_coverage
from models.radar_lidar_stage2_residual.model import ResidualOutput

from .config import DiffusionConfig
from .schedule import CosineSchedule


@dataclass(frozen=True)
class DiffusionContext:
    domain: CandidateDomain
    coverage: FaultyCoverage
    condition: torch.Tensor
    batch_size: int


class ResidualSparseDiffusion(nn.Module):
    """The deployed path receives radar and faulty LiDAR, never clean LiDAR."""

    def __init__(self, stage1_channels: tuple[int, ...], config: DiffusionConfig):
        super().__init__()
        _require_me()
        self.config = config
        base = config.residual
        width = base.channels[0]
        self.conditioner = Stage1Conditioning(stage1_channels, base)
        self.faulty_encoder = nn.Sequential(nn.Linear(7, width), nn.LayerNorm(width), nn.SiLU())
        self.time_encoder = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.input_fuse = nn.Sequential(nn.Linear(width * 3 + 4, width), nn.LayerNorm(width), nn.SiLU())
        self.unet = SparseUNet(base.channels)
        self.clean_head = nn.Linear(width, 4)
        self.schedule = CosineSchedule(config.timesteps)

    def prepare(self, stage1: Stage1Output, grid, faulty: torch.Tensor,
                faulty_valid: torch.Tensor, regions: list[dict]) -> DiffusionContext:
        base = self.config.residual
        domain = make_candidates(stage1, grid, confidence_threshold=base.confidence_threshold,
                                 expansion_zyx=base.expansion_zyx, max_sites=base.max_candidate_sites)
        coverage = faulty_coverage(domain, faulty, faulty_valid, regions,
                                   ray_tolerance_m=base.ray_tolerance_m)
        if len(domain.coordinates):
            radar = self.conditioner(stage1, domain)
            observed = self.faulty_encoder(coverage.features)
            condition = torch.cat((radar, observed), dim=1)
        else:
            condition = faulty.new_empty((0, 2 * base.channels[0]))
        return DiffusionContext(domain, coverage, condition, len(regions))

    def _time_features(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        frequency = torch.exp(torch.arange(8, device=t.device, dtype=dtype)
                              * (-math.log(10000.) / 7))
        phase = (t.to(dtype)[:, None] / self.config.timesteps) * frequency[None] * 1000
        return torch.cat((phase.sin(), phase.cos()), dim=1)

    def predict_clean(self, context: DiffusionContext, noisy: torch.Tensor,
                      timestep: torch.Tensor, *,
                      sparse_cache: dict | None = None) -> tuple[torch.Tensor, dict]:
        domain = context.domain
        n = len(domain.coordinates)
        if noisy.shape != (n, 4):
            raise ValueError("Expected occupancy and XYZ state at every candidate")
        if timestep.shape != (context.batch_size,) or (n and
                (sparse_cache is None or "input_key" not in sparse_cache) and
                int(domain.coordinates[:, 0].max()) >= context.batch_size):
            raise ValueError("One diffusion timestep per frame is required")
        if n == 0:
            return (self.clean_head.weight.sum() * 0).expand(0, 4), {"input": 0, "output": 0}
        t_feature = self.time_encoder(self._time_features(timestep, noisy.dtype))
        per_site_time = t_feature[domain.coordinates[:, 0].long()]
        features = self.input_fuse(torch.cat((context.condition, noisy, per_site_time), dim=1))
        if sparse_cache is not None and "input_key" in sparse_cache:
            sparse = ME.SparseTensor(features=features.contiguous(),
                                     coordinate_map_key=sparse_cache["input_key"],
                                     coordinate_manager=sparse_cache["manager"])
        else:
            sparse = ME.SparseTensor(features=features.contiguous(), coordinates=domain.coordinates.int())
            if sparse_cache is not None:
                sparse_cache["input_key"] = sparse.coordinate_map_key
                sparse_cache["manager"] = sparse.coordinate_manager
        decoded, counts = self.unet(sparse)
        if sparse_cache is not None and "order" in sparse_cache:
            if decoded.coordinate_map_key != sparse_cache["output_key"]:
                raise AssertionError("Diffusion changed its sparse output map between timesteps")
            order = sparse_cache["order"]
        else:
            expected = encode_keys(domain.coordinates, domain.grid.shape_zyx)
            actual = encode_keys(decoded.C.long().to(expected.device), domain.grid.shape_zyx)
            order = torch.argsort(actual)
            if not torch.equal(actual[order], expected):
                raise AssertionError("Diffusion changed the candidate coordinates")
            if sparse_cache is not None:
                sparse_cache["order"] = order
                sparse_cache["output_key"] = decoded.coordinate_map_key
        return torch.tanh(self.clean_head(decoded.F[order])), counts

    @torch.no_grad()
    def sample(self, context: DiffusionContext, *, generator: torch.Generator | None = None,
               frame_seeds: list[int] | None = None,
               occupancy_threshold: float | None = None) -> ResidualOutput:
        domain, coverage = context.domain, context.coverage
        n = len(domain.coordinates)
        if not n:
            empty = context.condition.new_empty(0)
            return ResidualOutput(domain, coverage, empty, empty,
                                  empty.reshape(0, 1).expand(0, 3),
                                  empty.reshape(0, 1).expand(0, 3), empty,
                                  {"sites": {"input": 0, "output": 0}, "diffusion_enabled": True})
        if frame_seeds is not None:
            if len(frame_seeds) != context.batch_size:
                raise ValueError("Expected one sampling seed per frame")
            # Keep the reverse diffusion state in FP32. Autocast may make the
            # U-Net prediction BF16, including the final t=0 state.
            state = torch.empty((n, 4), dtype=torch.float32, device=context.condition.device)
            for batch_id, seed in enumerate(frame_seeds):
                selected = domain.coordinates[:, 0] == batch_id
                frame_generator = torch.Generator(device=state.device).manual_seed(seed)
                state[selected] = torch.randn((int(selected.sum()), 4), dtype=state.dtype,
                                              device=state.device, generator=frame_generator)
        else:
            state = torch.randn((n, 4), dtype=torch.float32,
                                device=context.condition.device, generator=generator)
        blocked = ~coverage.may_add
        known_noise = state[blocked].clone()
        known_clean = torch.zeros_like(known_noise)
        known_clean[:, 0] = -1
        sparse_cache = {}
        for step in range(self.config.timesteps - 1, -1, -1):
            alpha = self.schedule.alpha_bar[step].to(state.dtype)
            state[blocked] = alpha.sqrt() * known_clean + (1 - alpha).sqrt() * known_noise
            timestep = torch.full((context.batch_size,), step, dtype=torch.long, device=state.device)
            predicted, counts = self.predict_clean(context, state, timestep,
                                                   sparse_cache=sparse_cache)
            state = self.schedule.reverse_step(state, predicted.float(), step).float()
        # These locations are measured already or outside the injected fault.
        state[blocked] = known_clean
        probability = ((state[:, 0] + 1) * .5).clamp(0, 1)
        logits = torch.logit(probability.clamp(1e-5, 1 - 1e-5))
        offsets = .5 * state[:, 1:].clamp(-1, 1)
        threshold = (self.config.residual.occupancy_threshold if occupancy_threshold is None
                     else occupancy_threshold)
        selected = (probability > threshold) & coverage.may_add
        points = decode_centroids(domain, offsets)[selected]
        return ResidualOutput(domain, coverage, logits, probability, offsets, points,
                              domain.confidence[selected],
                              {"sites": counts, **domain.counts, "selected": int(selected.sum()),
                               "blocked": int(blocked.sum()), "diffusion_enabled": True,
                               "timesteps": self.config.timesteps})
