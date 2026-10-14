"""Radar-conditioned, bottleneck-fused sparse 3D diffusion reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from models.radar_lidar_stage1.bottleneck_support import BottleneckSupportOutput
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.candidate_domain import CandidateDomain, make_candidates
from models.radar_lidar_stage2.sparse_unet import ME, _require_me
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning
from models.radar_lidar_stage2.voxel_target import decode_centroids
from models.radar_lidar_stage2_residual.coverage import FaultyCoverage, faulty_coverage
from models.radar_lidar_stage2_residual.model import ResidualOutput

from .attention import LocalBottleneckFusion
from .config import BottleneckDiffusionConfig
from .schedule import NoiseSchedule
from .unet import BottleneckDiffusionUNet, _replace


@dataclass(frozen=True)
class DiffusionContext:
    domain: CandidateDomain
    coverage: FaultyCoverage
    condition: torch.Tensor
    batch_size: int
    source_features: torch.Tensor
    source_xyz: torch.Tensor
    source_batch: torch.Tensor


class BottleneckSparseDiffusion(nn.Module):
    """All trainable parameters belong to Stage II; clean LiDAR is not an input."""

    def __init__(self, stage1_channels: tuple[int, ...], config: BottleneckDiffusionConfig):
        super().__init__()
        _require_me()
        if len(stage1_channels) != 1:
            raise ValueError("Stage II v2 requires the Stage-I radar-only bottleneck interface")
        self.config = config
        width = config.channels[0]
        self.conditioner = Stage1Conditioning(stage1_channels, config.residual)
        self.faulty_encoder = nn.Sequential(nn.Linear(7, width), nn.LayerNorm(width), nn.SiLU())
        self.input_fuse = nn.Sequential(nn.Linear(2 * width + 4, width),
                                        nn.LayerNorm(width), nn.SiLU())
        self.time_encoder = nn.Sequential(nn.Linear(config.timestep_dim, config.timestep_dim),
                                          nn.SiLU(), nn.Linear(config.timestep_dim, config.timestep_dim))
        self.unet = BottleneckDiffusionUNet(config.channels, config.spatial_strides,
                                            config.blocks_per_level, config.timestep_dim)
        self.fusion = LocalBottleneckFusion(
            config.channels[-1], stage1_channels[0], heads=config.attention_heads,
            mode=config.fusion_mode, radius_m=config.attention_radius_m,
            max_neighbors=config.attention_max_neighbors)
        self.noise_head = nn.Linear(width, 4)
        self.occupancy_head = nn.Linear(width, 1)
        self.offset_head = nn.Linear(width, 3)
        self.schedule = NoiseSchedule(config.timesteps, config.beta_schedule)

    def prepare(self, evidence: BottleneckSupportOutput, grid,
                faulty: torch.Tensor, faulty_valid: torch.Tensor,
                regions: list[dict]) -> DiffusionContext:
        if not isinstance(evidence, BottleneckSupportOutput):
            raise ValueError("Stage II v2 needs radar-only Stage-I bottleneck features")
        base = self.config.residual
        domain = make_candidates(evidence, grid,
                                 confidence_threshold=base.confidence_threshold,
                                 expansion_zyx=base.expansion_zyx,
                                 method=base.candidate_method)
        coverage = faulty_coverage(domain, faulty, faulty_valid, regions,
                                   ray_tolerance_m=base.ray_tolerance_m)
        source = evidence.bottleneck
        if source.stride != 8 or source.shape_zyx != grid.scale_shape(8):
            raise ValueError("Stage-I bottleneck does not share the Stage-II metric grid")
        if len(domain.coordinates):
            radar = self.conditioner(evidence, domain)
            observed = self.faulty_encoder(coverage.features)
            condition = torch.cat((radar, observed), dim=1)
        else:
            condition = faulty.new_empty((0, 2 * base.channels[0]))
        return DiffusionContext(domain, coverage, condition, len(regions),
                                source.features.detach(), source.centers_xyz(grid).detach(),
                                source.coords[:, 0].detach())

    def prepare_cached(self, domain: CandidateDomain, coverage: FaultyCoverage,
                       sources: list[torch.Tensor], indices: list[torch.Tensor],
                       source_centers: list[torch.Tensor], source_batches: list[torch.Tensor],
                       batch_size: int) -> DiffusionContext:
        if not (len(sources) == len(source_centers) == len(source_batches) == len(indices) == 1):
            raise ValueError("The cache must contain one Stage-I bottleneck source")
        if len(domain.coordinates):
            radar = self.conditioner.forward_cached(domain, sources, indices, None,
                                                     source_centers=source_centers)
            observed = self.faulty_encoder(coverage.features)
            condition = torch.cat((radar, observed), dim=1)
        else:
            condition = self.noise_head.weight.new_empty((0, 2 * self.config.channels[0]))
        if len(sources[0]) != len(source_batches[0]):
            raise ValueError("Cached Stage-I source batches differ from source features")
        return DiffusionContext(domain, coverage, condition, batch_size,
                                sources[0].detach(), source_centers[0].detach(),
                                source_batches[0].detach())

    def _time_features(self, timestep: torch.Tensor) -> torch.Tensor:
        dim = self.config.timestep_dim
        half = dim // 2
        frequency = torch.exp(torch.arange(half, device=timestep.device, dtype=torch.float32)
                              * (-math.log(10000.) / max(half - 1, 1)))
        phase = timestep.float()[:, None] * frequency[None]
        encoded = torch.cat((phase.sin(), phase.cos()), dim=1)
        return self.time_encoder(encoded)

    def predict_noise(self, context: DiffusionContext, noisy: torch.Tensor,
                      timestep: torch.Tensor, *, sparse_cache: dict | None = None):
        domain = context.domain
        count = len(domain.coordinates)
        if noisy.shape != (count, 4) or timestep.shape != (context.batch_size,):
            raise ValueError("Diffusion state or per-frame timesteps differ from the candidate domain")
        if context.condition.shape != (count, 2 * self.config.channels[0]):
            raise ValueError("Fine radar/faulty conditioning differs from the candidate domain")
        if count == 0:
            zero = self.noise_head.weight.sum() * 0
            return zero.expand(0, 4), zero.expand(0), zero.expand(0, 3), {"input": 0, "output": 0}
        if int(domain.coordinates[:, 0].max()) >= context.batch_size:
            raise ValueError("Candidate batch index exceeds the supplied diffusion timesteps")
        features = self.input_fuse(torch.cat((context.condition, noisy), dim=1))
        if sparse_cache is not None and "input_key" in sparse_cache:
            sparse = ME.SparseTensor(features=features.contiguous(),
                                     coordinate_map_key=sparse_cache["input_key"],
                                     coordinate_manager=sparse_cache["manager"])
        else:
            sparse = ME.SparseTensor(features=features.contiguous(),
                                     coordinates=domain.coordinates.int())
            if sparse_cache is not None:
                sparse_cache["input_key"] = sparse.coordinate_map_key
                sparse_cache["manager"] = sparse.coordinate_manager

        def fuse(sites):
            stride = math.prod(self.config.spatial_strides)
            if sparse_cache is not None:
                old_key = sparse_cache.get("bottleneck_key")
                if old_key is not None and old_key != sites.coordinate_map_key:
                    raise AssertionError("Reverse diffusion changed the bottleneck map")
                sparse_cache["bottleneck_key"] = sites.coordinate_map_key
            features = self.fusion(
                sites.F, sites.C, context.source_features,
                context.source_xyz, context.source_batch, domain.grid, stride,
                neighbors=(sparse_cache.get("bottleneck_neighbors")
                           if sparse_cache is not None else None), cache=sparse_cache)
            return _replace(sites, features)

        decoded, counts = self.unet(sparse, self._time_features(timestep), fuse)
        if sparse_cache is not None and "order" in sparse_cache:
            if decoded.coordinate_map_key != sparse_cache["output_key"]:
                raise AssertionError("Reverse diffusion changed the output map")
            order = sparse_cache["order"]
        else:
            expected = encode_keys(domain.coordinates, domain.grid.shape_zyx)
            actual = encode_keys(decoded.C.long().to(expected.device), domain.grid.shape_zyx)
            order = torch.argsort(actual)
            if not torch.equal(actual[order], expected):
                raise AssertionError("Stage-II decoder changed the candidate support")
            if sparse_cache is not None:
                sparse_cache["order"] = order
                sparse_cache["output_key"] = decoded.coordinate_map_key
        hidden = decoded.F[order]
        return (self.noise_head(hidden), self.occupancy_head(hidden).squeeze(1),
                self.offset_head(hidden), counts)

    @torch.no_grad()
    def sample(self, context: DiffusionContext, *, generator: torch.Generator | None = None,
               frame_seeds: list[int] | None = None, sampling_steps: int | None = None,
               method: str | None = None, occupancy_threshold: float | None = None) -> ResidualOutput:
        domain, coverage = context.domain, context.coverage
        method = self.config.sampling_method if method is None else method
        steps = self.schedule.sampling_timesteps(sampling_steps, method)
        count = len(domain.coordinates)
        if not count:
            empty = context.condition.new_empty(0)
            return ResidualOutput(domain, coverage, empty, empty,
                                  empty.reshape(0, 1).expand(0, 3),
                                  empty.reshape(0, 1).expand(0, 3), empty,
                                  {"sites": {"input": 0, "output": 0}, "sampling_steps": len(steps)})
        device = context.condition.device
        frame_generators = None
        if frame_seeds is not None:
            if len(frame_seeds) != context.batch_size:
                raise ValueError("One sampling seed is required per frame")
            frame_generators = [torch.Generator(device=device).manual_seed(int(seed))
                                for seed in frame_seeds]

        def random_state():
            if frame_generators is None:
                return torch.randn((count, 4), device=device, dtype=torch.float32,
                                   generator=generator)
            value = torch.empty((count, 4), device=device, dtype=torch.float32)
            for batch_id, rng in enumerate(frame_generators):
                selected = domain.coordinates[:, 0] == batch_id
                value[selected] = torch.randn((int(selected.sum()), 4), device=device,
                                              dtype=torch.float32, generator=rng)
            return value

        state = random_state()
        blocked = ~coverage.may_add
        known_noise = state[blocked].clone()
        known_clean = torch.zeros_like(known_noise)
        known_clean[:, 0] = -1
        sparse_cache = {}
        last_logits = last_offsets = None
        for index, t in enumerate(steps):
            previous = steps[index + 1] if index + 1 < len(steps) else -1
            alpha = self.schedule.alpha_bar[t].to(state.dtype)
            state[blocked] = alpha.sqrt() * known_clean + (1 - alpha).sqrt() * known_noise
            timestep = torch.full((context.batch_size,), t, dtype=torch.long, device=device)
            predicted, last_logits, last_offsets, counts = self.predict_noise(
                context, state, timestep, sparse_cache=sparse_cache)
            random = (random_state() if previous >= 0 and
                      (method == "ddpm" or self.config.ddim_eta > 0) else None)
            state = self.schedule.reverse_step(state, predicted.float(), t, previous,
                                               method=method, random=random,
                                               eta=self.config.ddim_eta).float()
        state[blocked] = known_clean
        probability = torch.sigmoid(last_logits.float())
        logits = last_logits.float()
        offsets = 0.5 * torch.tanh(last_offsets.float())
        threshold = (self.config.residual.occupancy_threshold if occupancy_threshold is None
                     else occupancy_threshold)
        selected = (probability > threshold) & coverage.may_add
        points = decode_centroids(domain, offsets)[selected]
        return ResidualOutput(domain, coverage, logits, probability, offsets,
                              points, domain.confidence[selected],
                              {"sites": counts, **domain.counts,
                               "selected": int(selected.sum()), "blocked": int(blocked.sum()),
                               "sampling_steps": len(steps), "sampling_method": method,
                               "diffusion_enabled": True})
