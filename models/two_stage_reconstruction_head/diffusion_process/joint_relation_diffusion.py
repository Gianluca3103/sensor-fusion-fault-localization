"""Radar-only relation features and absolute-depth ray diffusion.

The paired radar/clean-LiDAR attention is a training-only supervision path.
The diffusion condition is always produced from radar and observed LiDAR, so
the model has exactly the same inputs during training and deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from ..cross_modal_encoders import EncodedGrid, EncoderGrid, RadarGeometryEncoder
from ..range_view.geometry import RangeGeometry
from ..ray_depth_attention import (
    CleanLidarRelationshipTeacher, RayDepthBlueprint, RayDepthCrossAttention,
    _empty_encoded_like,
)
from ..ray_depth_queries import (
    RayDepthQueries, propose_ray_depth_queries, ray_tile_indices,
)
from .diffusion_process import DiffusionProcessConfig, GaussianNoiseSchedule
from .ray_view_diffusion import RadarGatedRangeUNet, project_lidar_tile, _tile_shape


@dataclass(frozen=True)
class RadarRelation:
    """One feature per LiDAR ray, inferred without clean or faulty LiDAR."""

    queries: RayDepthQueries
    features: torch.Tensor       # [B,Q,C]
    support: torch.Tensor        # [B,Q], measured local radar evidence
    depth_hint_m: torch.Tensor   # [B,Q], measured radar range, not a prediction
    candidate_features: torch.Tensor  # [B,Q,D,C]
    candidate_evidence: torch.Tensor  # [B,Q,D,3]


def _relation_tile_shape(queries: RayDepthQueries,
                         shape: tuple[int, int]) -> None:
    height, width = shape
    rows, cols = queries.rows, queries.cols
    if height < 1 or width < 4 or rows.shape[1] != height * width:
        raise ValueError("Expected a rectangular ray tile with width >= 4")
    rr = rows.reshape(-1, height, width)
    cc = cols.reshape(-1, height, width)
    expected_r = rr[:, :1, :1] + torch.arange(
        height, device=rows.device)[None, :, None]
    expected_c = cc[:, :1, :1] + torch.arange(
        width, device=cols.device)[None, None, :]
    if not (bool((rr == expected_r).all()) and bool((cc == expected_c).all())):
        raise ValueError("Ray tile must be row-major, consecutive, and rectangular")


class RadarRelationEncoder(nn.Module):
    """Aggregate local 3D radar attention into range-view relation tokens.

    Candidate depths are radar measurements or geometric anchors. No
    first-return classifier or candidate-depth correction is instantiated.
    The candidates serve only as spatial context for each radar-derived ray.
    """

    def __init__(self, geometry: RangeGeometry,
                 grid: EncoderGrid = EncoderGrid(), width: int = 32,
                 history_scans: int = 20) -> None:
        super().__init__()
        self.geometry = geometry
        self.radar_encoder = RadarGeometryEncoder(grid, width, history_scans)
        self.fusion = RayDepthCrossAttention(
            grid, width, history_scans=history_scans,
            predict_first_return=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.SiLU(), nn.Linear(width, width))

    def encode_radar(self, radar: torch.Tensor,
                     radar_valid: torch.Tensor) -> EncodedGrid:
        return self.radar_encoder(radar, radar_valid)

    def forward_encoded(self, radar: torch.Tensor, radar_valid: torch.Tensor,
                        encoded: EncodedGrid, rows: torch.Tensor,
                        cols: torch.Tensor) -> RadarRelation:
        empty_points = radar.new_empty((len(radar), 0, 4))
        empty_valid = torch.zeros((len(radar), 0), dtype=torch.bool,
                                  device=radar.device)
        queries = propose_ray_depth_queries(
            self.geometry, rows, cols, radar, radar_valid,
            empty_points, empty_valid,
            radar_slots=3, lidar_slots=0, uniform_slots=2)
        candidates, evidence = self.fusion.attend(
            self.geometry, queries, encoded, _empty_encoded_like(encoded), radar)
        measured = queries.valid & (queries.source == 1)
        count = measured.sum(-1)
        feature = (candidates * measured[..., None]).sum(-2)
        feature = self.output(feature / count.clamp_min(1)[..., None])
        feature = feature * (count > 0)[..., None]
        depth_hint = (queries.depths_m * measured).sum(-1)
        depth_hint = depth_hint / count.clamp_min(1)
        return RadarRelation(queries, feature, count > 0,
                             depth_hint, candidates, evidence)

    def forward(self, radar: torch.Tensor, radar_valid: torch.Tensor,
                rows: torch.Tensor, cols: torch.Tensor) -> RadarRelation:
        encoded = self.encode_radar(radar, radar_valid)
        return self.forward_encoded(radar, radar_valid, encoded, rows, cols)


class PairedRadarLidarAttention(nn.Module):
    """Training-only local cross-attention from radar rays to clean features.

    The attended clean features are targets for the radar-only relation path;
    they are never passed to the denoiser. A geometry head anchors the paired
    representation to actual clean first returns instead of letting the
    teacher/student feature spaces collapse together.
    """

    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        if width < 8 or width % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.width, self.heads = width, heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width))
        self.spatial_bias = nn.Parameter(torch.zeros(9))
        self.null_key = nn.Parameter(torch.zeros(heads, width // heads))
        self.null_value = nn.Parameter(torch.zeros(heads, width // heads))
        self.hit_head = nn.Linear(width, 1)
        self.depth_head = nn.Linear(width, 1)

    def forward(self, relation: RadarRelation, teacher: RayDepthBlueprint,
                tile_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = tile_shape
        _tile_shape(teacher, tile_shape)
        if (teacher.features.shape != relation.candidate_features.shape or
                not torch.equal(teacher.queries.depths_m,
                                relation.queries.depths_m)):
            raise ValueError("Paired attention requires matching radar-derived queries")
        batch, rays, slots, channels = teacher.features.shape
        # Unfold 3x3 neighboring rays. LiDAR and radar can reflect from
        # different parts of one object, so exact voxel equality is too strict.
        clean = teacher.features.detach().reshape(batch, height, width, slots, channels)
        clean = clean.permute(0, 3, 4, 1, 2).reshape(batch, slots * channels,
                                                     height, width)
        keys = F.unfold(clean, 3, padding=1)
        keys = keys.reshape(batch, slots, channels, 9, rays)
        keys = keys.permute(0, 4, 3, 1, 2).reshape(batch, rays, 9 * slots, channels)
        clean_support = (teacher.queries.valid &
                         (teacher.evidence_weights[..., 1] > 0))
        clean_support = clean_support.reshape(batch, height, width, slots)
        clean_support = clean_support.permute(0, 3, 1, 2).float()
        available = F.unfold(clean_support, 3, padding=1)
        available = available.reshape(batch, slots, 9, rays)
        available = available.permute(0, 3, 2, 1).reshape(batch, rays, 9 * slots) > 0
        valid = available.any(-1) & relation.support

        head_width = channels // self.heads
        query = self.query(relation.features.detach()).reshape(
            batch, rays, self.heads, head_width)
        key = self.key(keys).reshape(batch, rays, 9 * slots,
                                     self.heads, head_width)
        value = self.value(keys).reshape(batch, rays, 9 * slots,
                                         self.heads, head_width)
        scores = (query[:, :, None] * key).sum(-1) / math.sqrt(head_width)
        scores = scores + self.spatial_bias.repeat_interleave(slots)[None, None, :, None]
        scores = scores.masked_fill(~available[..., None], -1e4)
        null_score = (query * self.null_key).sum(-1) / math.sqrt(head_width)
        scores = torch.cat((scores, null_score[:, :, None]), dim=2)
        weights = scores.softmax(2)
        attended = ((weights[:, :, :-1, :, None] * value).sum(2) +
                    weights[:, :, -1, :, None] * self.null_value)
        paired = self.output(attended.reshape(batch, rays, channels))
        return paired * valid[..., None], valid


def normalize_depth(depth_m: torch.Tensor, geometry: RangeGeometry) -> torch.Tensor:
    """Map physical first-return depth to [-1,1] without a proposal anchor."""
    low, high = geometry.min_range_m, geometry.max_range_m
    return (2 * torch.log(depth_m.clamp(low, high) / low) /
            math.log(high / low) - 1)


def denormalize_depth(normalized: torch.Tensor,
                      geometry: RangeGeometry) -> torch.Tensor:
    low, high = geometry.min_range_m, geometry.max_range_m
    return low * torch.exp((normalized.clamp(-1, 1) + 1) *
                           (math.log(high / low) / 2))


@dataclass(frozen=True)
class RelationCondition:
    relation: RadarRelation
    tile_shape: tuple[int, int]
    observed_depth_m: torch.Tensor
    observed_intensity: torch.Tensor
    observed_mask: torch.Tensor
    proposal_mask: torch.Tensor
    static: torch.Tensor


@dataclass(frozen=True)
class RelationSample:
    geometry: RangeGeometry
    condition: RelationCondition
    depth_m: torch.Tensor
    intensity: torch.Tensor
    return_mask: torch.Tensor
    added_mask: torch.Tensor
    return_probability: torch.Tensor

    def added_points(self, scene: int) -> torch.Tensor:
        selected = self.added_mask[scene].reshape(-1)
        direction = self.condition.relation.queries.directions[scene, selected]
        depth = self.depth_m[scene].reshape(-1)[selected]
        intensity = self.intensity[scene].reshape(-1)[selected]
        return torch.cat((direction * depth[:, None], intensity[:, None]), -1)

    def merge_with_observed(self, observed_lidar: torch.Tensor,
                            observed_valid: torch.Tensor,
                            scene: int) -> torch.Tensor:
        return torch.cat((observed_lidar[scene, observed_valid[scene]],
                          self.added_points(scene)), dim=0)


class RadarRelationDiffusion(nn.Module):
    """Diffuse absolute first-return depth on radar-supported missing rays."""

    def __init__(self, geometry: RangeGeometry, *, relation_width: int = 32,
                 hidden: int = 32, timesteps: int = 200,
                 intensity_scale: float = 10.0) -> None:
        super().__init__()
        self.geometry = geometry
        self.intensity_scale = float(intensity_scale)
        self.denoiser = RadarGatedRangeUNet(relation_width, hidden)
        self.schedule = GaussianNoiseSchedule(DiffusionProcessConfig(
            num_train_timesteps=timesteps))

    def prepare_condition(self, relation: RadarRelation,
                          observed_lidar: torch.Tensor,
                          observed_valid: torch.Tensor,
                          tile_shape: tuple[int, int], *,
                          observed_projection: tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor] | None = None,
                          ) -> RelationCondition:
        _relation_tile_shape(relation.queries, tile_shape)
        batch, rays = relation.support.shape
        height, width = tile_shape
        if observed_projection is None:
            observation = project_lidar_tile(
                self.geometry, relation.queries.rows, relation.queries.cols,
                observed_lidar, observed_valid)
        else:
            observation = observed_projection
        depth, intensity, observed = observation
        if (depth.shape != (batch, rays) or intensity.shape != (batch, rays) or
                observed.shape != (batch, rays) or observed.dtype != torch.bool):
            raise ValueError("Observed projection does not match relation rays")
        def spatial(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, 1, height, width)
        support = spatial(relation.support)
        observed = spatial(observed)
        directions = relation.queries.directions.reshape(batch, height, width, 3)
        features = relation.features.transpose(1, 2).reshape(
            batch, -1, height, width)
        static = torch.cat((
            features,
            normalize_depth(spatial(depth), self.geometry) * observed,
            observed.float(),
            torch.asinh(spatial(intensity) / self.intensity_scale),
            normalize_depth(spatial(relation.depth_hint_m), self.geometry) * support,
            directions.permute(0, 3, 1, 2),
        ), dim=1)
        return RelationCondition(relation, tile_shape, spatial(depth),
                                 spatial(intensity), observed,
                                 support & ~observed, static)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1)

    def training_loss(self, relation: RadarRelation,
                      observed_lidar: torch.Tensor, observed_valid: torch.Tensor,
                      clean_lidar: torch.Tensor, clean_valid: torch.Tensor,
                      tile_shape: tuple[int, int], *,
                      timestep: torch.Tensor | None = None,
                      noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        condition = self.prepare_condition(
            relation, observed_lidar, observed_valid, tile_shape)
        batch, _, height, width = condition.proposal_mask.shape
        depth, intensity, clean_hit = project_lidar_tile(
            self.geometry, relation.queries.rows, relation.queries.cols,
            clean_lidar, clean_valid)
        depth = depth.reshape(batch, 1, height, width)
        intensity = intensity.reshape(batch, 1, height, width)
        clean_hit = clean_hit.reshape(batch, 1, height, width)
        proposal = condition.proposal_mask
        positive = proposal & clean_hit
        target = torch.where(clean_hit,
            normalize_depth(depth, self.geometry), torch.zeros_like(depth))
        if timestep is None:
            timestep = torch.randint(len(self.schedule.alpha_bars), (batch,),
                                     device=depth.device)
        if noise is None:
            noise = torch.randn_like(target)
        if timestep.shape != (batch,) or noise.shape != target.shape:
            raise ValueError("Diffusion timestep or noise has incompatible shape")
        noisy, epsilon_target = self.schedule.add_masked_noise(
            target, noise, timestep, proposal.float())
        epsilon, return_logit, intensity_latent = self.denoiser(
            noisy, condition.static, timestep)
        noise_loss = self._masked_mean(
            (epsilon - epsilon_target).square(), proposal.float())
        return_loss = self._masked_mean(
            F.binary_cross_entropy_with_logits(
                return_logit, clean_hit.float(), reduction="none"),
            proposal.float())
        x0 = self.schedule.predict_x0(noisy, epsilon, timestep)
        depth_loss = self._masked_mean(
            F.smooth_l1_loss(x0, target, reduction="none"), positive.float())
        intensity_target = torch.asinh(intensity / self.intensity_scale)
        intensity_loss = self._masked_mean(
            F.smooth_l1_loss(intensity_latent, intensity_target,
                             reduction="none"), positive.float())
        total = noise_loss + return_loss + depth_loss + 0.05 * intensity_loss
        with torch.no_grad():
            predicted = proposal & (return_logit.sigmoid() >= 0.5)
            correct = predicted & clean_hit
            depth_error = (denormalize_depth(x0.detach(), self.geometry) - depth).abs()
        return {
            "loss": total, "noise": noise_loss, "return": return_loss,
            "depth": depth_loss, "intensity": intensity_loss,
            "supported_rays": proposal.sum().detach(),
            "clean_hits": positive.sum().detach(),
            "predicted_hits": predicted.sum().detach(),
            "true_hits": correct.sum().detach(),
            "depth_error_sum_m": depth_error[positive].sum().detach(),
        }

    @torch.no_grad()
    def sample(self, relation: RadarRelation,
               observed_lidar: torch.Tensor, observed_valid: torch.Tensor,
               tile_shape: tuple[int, int], *, steps: int = 20,
               return_threshold: float = 0.5,
               generator: torch.Generator | None = None,
               observed_projection: tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor] | None = None,
               ) -> RelationSample:
        if steps < 2 or not 0 <= return_threshold <= 1:
            raise ValueError("Invalid DDIM step count or return threshold")
        condition = self.prepare_condition(
            relation, observed_lidar, observed_valid, tile_shape,
            observed_projection=observed_projection)
        proposal = condition.proposal_mask
        probability = torch.zeros_like(condition.observed_depth_m)
        generated_depth = torch.zeros_like(probability)
        generated_intensity = torch.zeros_like(probability)
        added = torch.zeros_like(proposal)
        if bool(proposal.any()):
            state = torch.randn(proposal.shape, device=proposal.device,
                                dtype=condition.static.dtype,
                                generator=generator) * proposal
            times = torch.linspace(len(self.schedule.alpha_bars) - 1, 0,
                min(steps, len(self.schedule.alpha_bars)), device=proposal.device)
            times = torch.unique(times.round().long(), sorted=True).flip(0)
            for index, time_value in enumerate(times):
                timestep = time_value.expand(len(proposal))
                previous = (times[index + 1].expand(len(proposal))
                            if index + 1 < len(times)
                            else torch.full_like(timestep, -1))
                epsilon, return_logit, intensity_latent = self.denoiser(
                    state, condition.static, timestep)
                state, estimate = self.schedule.ddim_step(
                    state, epsilon, timestep, previous, proposal.float(),
                    eta=0.0, generator=generator)
            probability = return_logit.sigmoid()
            added = proposal & (probability >= return_threshold)
            generated_depth = denormalize_depth(estimate, self.geometry)
            generated_intensity = (intensity_latent.clamp(-4, 4).sinh()
                                   * self.intensity_scale)
        depth = torch.where(condition.observed_mask, condition.observed_depth_m,
                            torch.where(added, generated_depth, 0))
        intensity = torch.where(condition.observed_mask, condition.observed_intensity,
                                torch.where(added, generated_intensity, 0))
        batch, _, height, width = depth.shape
        return RelationSample(
            self.geometry, condition, depth.reshape(batch, height, width),
            intensity.reshape(batch, height, width),
            (condition.observed_mask | added).reshape(batch, height, width),
            added.reshape(batch, height, width),
            probability.reshape(batch, height, width),
        )


def joint_relation_loss(
    relation: RadarRelation, paired_model: PairedRadarLidarAttention,
    teacher: CleanLidarRelationshipTeacher,
    clean_lidar: torch.Tensor, clean_valid: torch.Tensor,
    tile_shape: tuple[int, int], *,
    alignment_weight: float = 0.1, paired_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Use clean LiDAR strictly as supervision after radar inference."""
    geometry = teacher.geometry
    with torch.no_grad():
        clean = teacher(relation.queries, clean_lidar, clean_valid)
        clean_depth, _, clean_hit = project_lidar_tile(
            geometry, relation.queries.rows, relation.queries.cols,
            clean_lidar, clean_valid)
    paired, paired_valid = paired_model(relation, clean, tile_shape)
    matched = paired_valid & relation.support & clean_hit
    if bool(matched.any()):
        alignment = F.smooth_l1_loss(
            F.normalize(relation.features[matched], dim=-1),
            F.normalize(paired.detach()[matched], dim=-1))
    else:
        alignment = relation.features.sum() * 0
    supervised = paired_valid & relation.support
    if bool(supervised.any()):
        hit_logits = paired_model.hit_head(paired).squeeze(-1)
        classification = F.binary_cross_entropy_with_logits(
            hit_logits[supervised], clean_hit[supervised].float())
        if bool(matched.any()):
            predicted_depth = paired_model.depth_head(paired).squeeze(-1)
            target_depth = normalize_depth(clean_depth[matched], geometry)
            metric = F.smooth_l1_loss(predicted_depth[matched], target_depth)
        else:
            metric = paired.sum() * 0
        paired_loss = classification + metric
    else:
        paired_loss = paired.sum() * 0
    return {
        "loss": alignment_weight * alignment + paired_weight * paired_loss,
        "alignment": alignment, "paired": paired_loss,
        "aligned_rays": matched.sum().detach(),
        "paired_rays": supervised.sum().detach(),
    }


@torch.inference_mode()
def sample_joint_full_scan(
    relation_model: RadarRelationEncoder, diffusion: RadarRelationDiffusion,
    radar: torch.Tensor, radar_valid: torch.Tensor,
    observed_lidar: torch.Tensor, observed_valid: torch.Tensor, *,
    tile_rows: int = 4, tile_cols: int = 64, steps: int = 20,
    return_threshold: float = 0.5,
    generator: torch.Generator | None = None,
    on_tile: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Return original observed points plus unique radar-supported additions."""
    geometry = diffusion.geometry
    if relation_model.geometry != geometry:
        raise ValueError("Relation and diffusion geometries differ")
    if relation_model.training or diffusion.training:
        raise ValueError("Set relation and diffusion to eval mode before sampling")
    if (radar.ndim != 3 or radar.shape[0] != 1 or radar.shape[-1] != 7 or
            radar_valid.shape != radar.shape[:2] or
            observed_lidar.ndim != 3 or observed_lidar.shape[0] != 1 or
            observed_lidar.shape[-1] != 4 or
            observed_valid.shape != observed_lidar.shape[:2] or
            radar_valid.dtype != torch.bool or observed_valid.dtype != torch.bool or
            radar.device != observed_lidar.device):
        raise ValueError("Expected matching single-scene radar and LiDAR tensors")
    height, width = geometry.shape
    if (not 1 <= tile_rows <= height or not 4 <= tile_cols <= width or
            tile_rows * tile_cols * 5 > relation_model.fusion.max_candidates):
        raise ValueError("Tile dimensions exceed the scan or attention limit")
    original = observed_lidar[0, observed_valid[0]]
    all_rows, all_cols = ray_tile_indices(geometry, device=radar.device)
    observed_depth, observed_intensity, observed_hit = project_lidar_tile(
        geometry, all_rows, all_cols, observed_lidar, observed_valid)
    projection = tuple(value.reshape(1, height, width)
                       for value in (observed_depth, observed_intensity, observed_hit))
    encoded = relation_model.encode_radar(radar, radar_valid)
    additions: list[torch.Tensor] = []
    for row_start in range(0, height, tile_rows):
        row_stop = min(row_start + tile_rows, height)
        for first_col in range(0, width, tile_cols):
            col_stop = min(first_col + tile_cols, width)
            col_start = first_col
            if col_stop - col_start < 4:
                col_start = max(0, col_stop - 4)
            rows, cols = ray_tile_indices(
                geometry, row_start=row_start, row_stop=row_stop,
                col_start=col_start, col_stop=col_stop, device=radar.device)
            relation = relation_model.forward_encoded(
                radar, radar_valid, encoded, rows, cols)
            result = diffusion.sample(
                relation, observed_lidar, observed_valid,
                (row_stop - row_start, col_stop - col_start), steps=steps,
                return_threshold=return_threshold, generator=generator,
                observed_projection=tuple(
                    value[:, row_start:row_stop, col_start:col_stop].reshape(1, -1)
                    for value in projection))
            added = result.added_points(0)
            if first_col != col_start:
                selected_cols = cols[0, result.added_mask[0].reshape(-1)]
                added = added[selected_cols >= first_col]
            additions.append(added)
            if on_tile is not None:
                on_tile()
    return torch.cat((original, *additions), dim=0)
