"""Radar-gated diffusion of LiDAR first-return depth on calibrated scan rays.

The denoiser sees a rectangular range-view tile, but Gaussian noise and all
updates are restricted to missing rays with measured local radar support.
Clean LiDAR is read only by ``training_loss``; the inference mask never uses it.
Observed LiDAR points are preserved exactly when additions are exported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from ..range_view.geometry import RangeGeometry
from ..ray_depth_attention import RayDepthBlueprint, RayDepthBlueprintModel
from ..ray_depth_queries import ray_tile_indices
from .basic_diffusion_unet import SinusoidalTimeEmbedding, TimestepResidualBlock
from .diffusion_process import DiffusionProcessConfig, GaussianNoiseSchedule


def project_lidar_tile(
    geometry: RangeGeometry, rows: torch.Tensor, cols: torch.Tensor,
    points: torch.Tensor, point_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest depth, intensity, and hit mask at specified LiDAR rays.

    Uses the same angular bins and beam tolerances as the range-view geometry.
    An unobserved ray has depth/intensity zero and hit mask false.
    """
    if (points.ndim != 3 or points.shape[-1] != 4 or
            point_valid.shape != points.shape[:2] or point_valid.dtype != torch.bool or
            rows.shape != cols.shape or rows.shape[0] != points.shape[0] or
            rows.device != points.device):
        raise ValueError("Invalid LiDAR points, point mask, or ray indices")
    height, width = geometry.shape
    beams = points.new_tensor(geometry.beam_elevations_rad)
    if geometry.max_beam_error_rad is None:
        gaps = beams[1:] - beams[:-1]
        tolerance = torch.empty_like(beams)
        tolerance[0], tolerance[-1] = gaps[0] / 2, gaps[-1] / 2
        if height > 2:
            tolerance[1:-1] = torch.maximum(gaps[:-1], gaps[1:]) / 2
    else:
        tolerance = beams.new_full((height,), geometry.max_beam_error_rad)
    depth = points.new_zeros(rows.shape)
    intensity = points.new_zeros(rows.shape)
    hit = torch.zeros_like(rows, dtype=torch.bool)
    for scene in range(points.shape[0]):
        cloud = points[scene, point_valid[scene]]
        if not len(cloud):
            continue
        xyz = cloud[:, :3]
        radius = xyz.norm(dim=-1)
        elevation = torch.atan2(xyz[:, 2], xyz[:, :2].norm(dim=-1))
        row = (elevation[:, None] - beams[None]).abs().argmin(-1)
        angle = torch.remainder(
            torch.atan2(xyz[:, 1], xyz[:, 0]) - geometry.azimuth_offset_rad,
            2 * math.pi,
        )
        col = torch.floor(angle * width / geometry.azimuth_span_rad).long()
        keep = (torch.isfinite(cloud).all(-1) &
                (radius >= geometry.min_range_m) & (radius <= geometry.max_range_m) &
                ((elevation - beams[row]).abs() <= tolerance[row]) &
                (angle < geometry.azimuth_span_rad) & (col >= 0) & (col < width))
        if not bool(keep.any()):
            continue
        flat = row[keep] * width + col[keep]
        ranges = radius[keep]
        reflectivity = cloud[keep, 3]
        # Stable sorting selects the closest range for every angular pixel.
        by_depth = torch.argsort(ranges, stable=True)
        by_pixel = torch.argsort(flat[by_depth], stable=True)
        order = by_depth[by_pixel]
        ordered = flat[order]
        first = torch.ones_like(ordered, dtype=torch.bool)
        first[1:] = ordered[1:] != ordered[:-1]
        winner = order[first]
        full_depth = points.new_zeros(height * width)
        full_intensity = points.new_zeros(height * width)
        full_hit = torch.zeros(height * width, dtype=torch.bool, device=points.device)
        full_depth[flat[winner]] = ranges[winner]
        full_intensity[flat[winner]] = reflectivity[winner]
        full_hit[flat[winner]] = True
        lookup = rows[scene] * width + cols[scene]
        depth[scene], intensity[scene], hit[scene] = (
            full_depth[lookup], full_intensity[lookup], full_hit[lookup]
        )
    return depth, intensity, hit


def _tile_shape(blueprint: RayDepthBlueprint, shape: tuple[int, int]) -> None:
    height, width = shape
    rows, cols = blueprint.queries.rows, blueprint.queries.cols
    if height < 1 or width < 4 or rows.shape[1] != height * width:
        raise ValueError("Expected a rectangular ray tile with width >= 4")
    rr = rows.reshape(-1, height, width)
    cc = cols.reshape(-1, height, width)
    expected_r = rr[:, :1, :1] + torch.arange(height, device=rows.device)[None, :, None]
    expected_c = cc[:, :1, :1] + torch.arange(width, device=cols.device)[None, None, :]
    if not (bool((rr == expected_r).all()) and bool((cc == expected_c).all())):
        raise ValueError("Ray tile must be row-major, consecutive, and rectangular")


def calibrate_reliability_threshold(
    scores: torch.Tensor, correctable: torch.Tensor, *,
    minimum_precision: float = 0.9, minimum_predictions: int = 100,
) -> dict[str, float | int]:
    """Choose the widest validation gate meeting a target empirical precision.

    If validation provides too little evidence, threshold 1.0 abstains. This
    calibration describes candidate depth correctness, not detector accuracy.
    """
    if not 0 < minimum_precision <= 1 or minimum_predictions < 1:
        raise ValueError("Invalid calibration precision or sample minimum")
    if scores.ndim != 1 or correctable.shape != scores.shape or correctable.dtype != torch.bool:
        raise ValueError("Expected matching flat scores and boolean labels")
    if not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("Reliability scores must be finite probabilities")
    if len(scores) < minimum_predictions:
        return {"threshold": 1.0, "precision": 0.0, "accepted": 0,
                "evaluated": len(scores)}
    sorted_score, order = scores.detach().float().sort(descending=True)
    sorted_correct = correctable[order].float()
    count = torch.arange(1, len(scores) + 1, device=scores.device)
    precision = sorted_correct.cumsum(0) / count
    last_in_tie = torch.ones_like(correctable)
    last_in_tie[:-1] = sorted_score[:-1] > sorted_score[1:]
    feasible = (count >= minimum_predictions) & (precision >= minimum_precision) & last_in_tie
    if not bool(feasible.any()):
        return {"threshold": 1.0, "precision": 0.0, "accepted": 0,
                "evaluated": len(scores)}
    chosen = int(torch.nonzero(feasible, as_tuple=False)[-1, 0])
    return {"threshold": float(sorted_score[chosen]),
            "precision": float(precision[chosen]),
            "accepted": chosen + 1, "evaluated": len(scores)}


@dataclass(frozen=True)
class RayDiffusionCondition:
    blueprint: RayDepthBlueprint
    tile_shape: tuple[int, int]
    base_depth_m: torch.Tensor       # [B,1,H,W]
    features: torch.Tensor           # [B,C,H,W]
    observed_depth_m: torch.Tensor   # [B,1,H,W]
    observed_intensity: torch.Tensor # [B,1,H,W]
    observed_mask: torch.Tensor      # [B,1,H,W] boolean
    measured_radar: torch.Tensor     # [B,1,H,W] boolean
    proposal_mask: torch.Tensor      # measured radar and missing LiDAR
    reliability_logits: torch.Tensor # [B,1,H,W]
    blueprint_return_probability: torch.Tensor


@dataclass(frozen=True)
class RayDiffusionResult:
    geometry: RangeGeometry
    condition: RayDiffusionCondition
    depth_m: torch.Tensor       # [B,H,W], zero if no return
    intensity: torch.Tensor     # [B,H,W], zero if no return
    return_mask: torch.Tensor   # [B,H,W]
    added_mask: torch.Tensor    # [B,H,W], never overlaps observed rays
    reliability: torch.Tensor   # [B,H,W]

    def added_points(self, scene: int) -> torch.Tensor:
        """Return generated [XYZ, intensity] points for one batch item."""
        if not 0 <= scene < self.depth_m.shape[0]:
            raise IndexError("scene index outside batch")
        selected = self.added_mask[scene].reshape(-1)
        directions = self.condition.blueprint.queries.directions[scene, selected]
        depth = self.depth_m[scene].reshape(-1)[selected]
        intensity = self.intensity[scene].reshape(-1)[selected]
        return torch.cat((directions * depth[:, None], intensity[:, None]), -1)

    def merge_with_observed(self, observed_lidar: torch.Tensor,
                            observed_valid: torch.Tensor,
                            scene: int) -> torch.Tensor:
        """Append additions to the unaltered original points, including intensity."""
        if (observed_lidar.ndim != 3 or observed_lidar.shape[-1] != 4 or
                observed_valid.shape != observed_lidar.shape[:2] or
                observed_valid.dtype != torch.bool or
                observed_lidar.shape[0] != self.depth_m.shape[0]):
            raise ValueError("Expected original [B,N,4] LiDAR and boolean mask")
        return torch.cat((observed_lidar[scene, observed_valid[scene]],
                          self.added_points(scene)), 0)


class RadarGatedRangeUNet(nn.Module):
    """2D range-view U-Net; horizontal downsampling retains every beam row."""

    def __init__(self, blueprint_width: int = 32, hidden: int = 32) -> None:
        super().__init__()
        if hidden < 8 or hidden % 8 or blueprint_width < 1:
            raise ValueError("hidden must be a multiple of eight; width positive")
        time_dim = hidden * 4
        self.time = nn.Sequential(SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(blueprint_width + 8, hidden, 3, padding=1)
        self.enc1 = TimestepResidualBlock(hidden, hidden, time_dim)
        self.down1 = nn.Conv2d(hidden, hidden * 2, 3, stride=(1, 2), padding=1)
        self.enc2 = TimestepResidualBlock(hidden * 2, hidden * 2, time_dim)
        self.down2 = nn.Conv2d(hidden * 2, hidden * 4, 3, stride=(1, 2), padding=1)
        self.middle = TimestepResidualBlock(hidden * 4, hidden * 4, time_dim)
        self.dec2 = TimestepResidualBlock(hidden * 6, hidden * 2, time_dim)
        self.dec1 = TimestepResidualBlock(hidden * 3, hidden, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(),
                                    nn.Conv2d(hidden, 3, 3, padding=1))

    def forward(self, noisy_residual: torch.Tensor,
                static_condition: torch.Tensor,
                timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (noisy_residual.ndim != 4 or noisy_residual.shape[1] != 1 or
                static_condition.shape[:1] != noisy_residual.shape[:1] or
                static_condition.shape[-2:] != noisy_residual.shape[-2:]):
            raise ValueError("Noise and range-view conditions must share [B,H,W]")
        time = self.time(timestep)
        x = self.input(torch.cat((noisy_residual, static_condition), 1))
        ones = lambda value: value.new_ones((len(value), 1, *value.shape[-2:]))
        skip1 = self.enc1(x, time, ones(x))
        down1 = self.down1(skip1)
        skip2 = self.enc2(down1, time, ones(down1))
        middle_input = self.down2(skip2)
        middle = self.middle(middle_input, time, ones(middle_input))
        up2 = F.interpolate(middle, size=skip2.shape[-2:], mode="nearest")
        up2 = self.dec2(torch.cat((up2, skip2), 1), time, ones(skip2))
        up1 = F.interpolate(up2, size=skip1.shape[-2:], mode="nearest")
        up1 = self.dec1(torch.cat((up1, skip1), 1), time, ones(skip1))
        epsilon, return_logit, intensity_latent = self.output(up1).split(1, 1)
        return epsilon, return_logit, intensity_latent


class RadarGatedRayDiffusion(nn.Module):
    """Train and sample a masked depth correction conditioned on a blueprint."""

    def __init__(self, geometry: RangeGeometry, *, blueprint_width: int = 32,
                 hidden: int = 32, timesteps: int = 200,
                 max_correction_m: float = 3.0,
                 intensity_scale: float = 10.0) -> None:
        super().__init__()
        if max_correction_m <= 0 or intensity_scale <= 0:
            raise ValueError("Correction and intensity scales must be positive")
        self.geometry = geometry
        self.max_correction_m = float(max_correction_m)
        self.intensity_scale = float(intensity_scale)
        self.reliability = nn.Sequential(nn.Linear(blueprint_width + 3, hidden),
                                         nn.SiLU(), nn.Linear(hidden, 1))
        self.denoiser = RadarGatedRangeUNet(blueprint_width, hidden)
        self.schedule = GaussianNoiseSchedule(DiffusionProcessConfig(
            num_train_timesteps=timesteps,
        ))

    def prepare_condition(self, blueprint: RayDepthBlueprint,
                          observed_lidar: torch.Tensor,
                          observed_valid: torch.Tensor,
                          tile_shape: tuple[int, int], *,
                          observed_projection: tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor] | None = None,
                          ) -> RayDiffusionCondition:
        """Select radar-supported candidates using inference inputs only."""
        _tile_shape(blueprint, tile_shape)
        batch, rays, slots = blueprint.queries.depths_m.shape
        if blueprint.features.shape[:3] != (batch, rays, slots):
            raise ValueError("Blueprint feature dimensions do not match queries")
        height, width = tile_shape
        measured = blueprint.queries.valid & (blueprint.evidence_weights[..., 0] > 0)
        logits = blueprint.first_return_logits[..., :slots].masked_fill(~measured, -1e4)
        selected = logits.argmax(-1)
        index = selected[..., None]
        gather_features = index[..., None].expand(-1, -1, 1, blueprint.features.shape[-1])
        features = blueprint.features.gather(2, gather_features).squeeze(2)
        base_depth = (blueprint.queries.depths_m + blueprint.depth_residual_m).gather(
            -1, index).squeeze(-1)
        radar_weight = blueprint.evidence_weights[..., 0].gather(-1, index).squeeze(-1)
        candidate_prob = blueprint.first_return_probabilities[..., :slots].gather(
            -1, index).squeeze(-1)
        has_radar = measured.any(-1)
        if observed_projection is None:
            observation, observation_intensity, observed = project_lidar_tile(
                self.geometry, blueprint.queries.rows, blueprint.queries.cols,
                observed_lidar, observed_valid,
            )
        else:
            observation, observation_intensity, observed = observed_projection
            if (observation.shape != (batch, rays) or
                    observation_intensity.shape != (batch, rays) or
                    observed.shape != (batch, rays) or
                    observed.dtype != torch.bool or
                    observation.device != observed_lidar.device or
                    observation_intensity.device != observed_lidar.device or
                    observed.device != observed_lidar.device):
                raise ValueError("Cached observed projection does not match this ray tile")
        reliability_input = torch.cat((features,
            base_depth[..., None] / self.geometry.max_range_m,
            radar_weight[..., None], candidate_prob[..., None]), -1)
        reliability_logits = self.reliability(reliability_input).squeeze(-1)
        def spatial(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, 1, height, width)
        return RayDiffusionCondition(
            blueprint, tile_shape, spatial(base_depth),
            features.transpose(1, 2).reshape(batch, features.shape[-1], height, width),
            spatial(observation), spatial(observation_intensity),
            spatial(observed), spatial(has_radar),
            spatial(has_radar & ~observed), spatial(reliability_logits),
            spatial(candidate_prob),
        )

    def _static_condition(self, condition: RayDiffusionCondition) -> torch.Tensor:
        return torch.cat((
            condition.base_depth_m / self.geometry.max_range_m,
            condition.observed_depth_m / self.geometry.max_range_m,
            condition.observed_mask.float(),
            torch.asinh(condition.observed_intensity / self.intensity_scale),
            condition.measured_radar.float(),
            condition.reliability_logits.sigmoid(),
            condition.blueprint_return_probability,
            condition.features,
        ), 1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1)

    def training_loss(self, blueprint: RayDepthBlueprint,
                      observed_lidar: torch.Tensor, observed_valid: torch.Tensor,
                      clean_lidar: torch.Tensor, clean_valid: torch.Tensor,
                      tile_shape: tuple[int, int], *,
                      timestep: torch.Tensor | None = None,
                      noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Supervise diffusion and reliability without clean-derived input masks."""
        condition = self.prepare_condition(
            blueprint, observed_lidar, observed_valid, tile_shape)
        batch, _, height, width = condition.base_depth_m.shape
        clean_depth, clean_intensity, clean_hit = project_lidar_tile(
            self.geometry, blueprint.queries.rows, blueprint.queries.cols,
            clean_lidar, clean_valid,
        )
        clean_depth = clean_depth.reshape(batch, 1, height, width)
        clean_intensity = clean_intensity.reshape(batch, 1, height, width)
        clean_hit = clean_hit.reshape(batch, 1, height, width)
        proposal = condition.proposal_mask
        correctable = (proposal & clean_hit &
            ((clean_depth - condition.base_depth_m).abs() <= self.max_correction_m))
        target = torch.where(correctable,
            (clean_depth - condition.base_depth_m) / self.max_correction_m,
            torch.zeros_like(clean_depth))
        if timestep is None:
            timestep = torch.randint(len(self.schedule.alpha_bars), (batch,),
                                     device=clean_depth.device)
        if noise is None:
            noise = torch.randn_like(target)
        if timestep.shape != (batch,) or noise.shape != target.shape:
            raise ValueError("timestep or noise has incompatible shape")
        mask = proposal.float()
        noisy, epsilon_target = self.schedule.add_masked_noise(
            target, noise, timestep, mask)
        epsilon, return_logit, intensity_latent = self.denoiser(
            noisy, self._static_condition(condition), timestep)
        reliability_loss = self._masked_mean(
            F.binary_cross_entropy_with_logits(condition.reliability_logits,
                correctable.float(), reduction="none"), mask)
        return_loss = self._masked_mean(
            F.binary_cross_entropy_with_logits(return_logit,
                correctable.float(), reduction="none"), mask)
        epsilon_loss = self._masked_mean((epsilon - epsilon_target).square(), mask)
        predicted_x0 = self.schedule.predict_x0(noisy, epsilon, timestep)
        depth_loss = self._masked_mean(
            F.smooth_l1_loss(predicted_x0, target, reduction="none"),
            correctable.float())
        intensity_target = torch.asinh(clean_intensity / self.intensity_scale)
        intensity_loss = self._masked_mean(
            F.smooth_l1_loss(intensity_latent, intensity_target,
                             reduction="none"), correctable.float())
        total = (epsilon_loss + reliability_loss + return_loss + depth_loss
                 + 0.05 * intensity_loss)
        return {"loss": total, "epsilon": epsilon_loss,
                "reliability": reliability_loss, "return": return_loss,
                "depth": depth_loss, "intensity": intensity_loss,
                "radar_supported_rays": proposal.sum().detach(),
                "correctable_rays": correctable.sum().detach()}

    @torch.no_grad()
    def sample(self, blueprint: RayDepthBlueprint,
               observed_lidar: torch.Tensor, observed_valid: torch.Tensor,
               tile_shape: tuple[int, int], *, steps: int = 20,
               reliability_threshold: float | None = None,
               return_threshold: float = 0.5,
               generator: torch.Generator | None = None,
               observed_projection: tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor] | None = None,
               ) -> RayDiffusionResult:
        """Sample only radar-supported missing rays; preserve all observed ones."""
        if reliability_threshold is None:
            raise ValueError("Supply a reliability threshold calibrated on validation")
        if (steps < 2 or not 0 <= reliability_threshold <= 1 or
                not 0 <= return_threshold <= 1):
            raise ValueError("Invalid sampling steps or probability thresholds")
        condition = self.prepare_condition(
            blueprint, observed_lidar, observed_valid, tile_shape,
            observed_projection=observed_projection)
        probability = condition.reliability_logits.sigmoid()
        eligible = condition.proposal_mask & (probability >= reliability_threshold)
        base = condition.base_depth_m
        observed = condition.observed_mask
        if bool(eligible.any()):
            state = torch.randn(base.shape, device=base.device,
                                dtype=base.dtype, generator=generator) * eligible
            static = self._static_condition(condition)
            times = torch.linspace(len(self.schedule.alpha_bars) - 1, 0,
                min(steps, len(self.schedule.alpha_bars)), device=base.device)
            times = torch.unique(times.round().long(), sorted=True).flip(0)
            for index, time_value in enumerate(times):
                timestep = time_value.expand(len(base))
                previous = (times[index + 1].expand(len(base)) if index + 1 < len(times)
                            else torch.full_like(timestep, -1))
                epsilon, return_logit, intensity_latent = self.denoiser(
                    state, static, timestep)
                state, estimate = self.schedule.ddim_step(
                    state, epsilon, timestep, previous, eligible.float(),
                    eta=0.0, generator=generator)
            added = eligible & (return_logit.sigmoid() >= return_threshold)
            generated_depth = (base + self.max_correction_m * estimate.clamp(-1, 1)).clamp(
                self.geometry.min_range_m, self.geometry.max_range_m)
            generated_intensity = (intensity_latent.clamp(-4, 4).sinh()
                                   * self.intensity_scale)
        else:
            added = torch.zeros_like(eligible)
            generated_depth = torch.zeros_like(base)
            generated_intensity = torch.zeros_like(base)
        final_depth = torch.where(observed, condition.observed_depth_m,
                                  torch.where(added, generated_depth, 0))
        final_intensity = torch.where(observed, condition.observed_intensity,
                                      torch.where(added, generated_intensity, 0))
        final_mask = observed | added
        b, _, h, w = base.shape
        return RayDiffusionResult(
            self.geometry, condition, final_depth.reshape(b, h, w),
            final_intensity.reshape(b, h, w), final_mask.reshape(b, h, w),
            added.reshape(b, h, w), probability.reshape(b, h, w),
        )


@torch.inference_mode()
def sample_full_scan(
    blueprint_model: RayDepthBlueprintModel,
    diffusion: RadarGatedRayDiffusion,
    radar: torch.Tensor, radar_valid: torch.Tensor,
    observed_lidar: torch.Tensor, observed_valid: torch.Tensor,
    *, tile_rows: int = 4, tile_cols: int = 64,
    steps: int = 20, reliability_threshold: float,
    return_threshold: float = 0.5,
    generator: torch.Generator | None = None,
    on_tile: Callable[[], None] | None = None,
) -> torch.Tensor:
    """Stitch nonoverlapping ray tiles into one [N,4] LiDAR scan.

    Sensor encoders run once per scan. Every original faulty LiDAR point is
    copied unchanged, followed by generated points. Batch size is one because
    different scenes can produce different numbers of additions.
    """
    geometry = diffusion.geometry
    if blueprint_model.geometry != geometry:
        raise ValueError("Blueprint and diffusion LiDAR geometries differ")
    if blueprint_model.training or diffusion.training:
        raise ValueError("Set both models to eval mode before sampling")
    if (radar.ndim != 3 or radar.shape[0] != 1 or radar.shape[-1] != 7 or
            radar_valid.shape != radar.shape[:2] or
            observed_lidar.ndim != 3 or observed_lidar.shape[0] != 1 or
            observed_lidar.shape[-1] != 4 or
            observed_valid.shape != observed_lidar.shape[:2] or
            radar_valid.dtype != torch.bool or observed_valid.dtype != torch.bool or
            radar.device != observed_lidar.device):
        raise ValueError("Expected matching single-scene radar and LiDAR tensors")
    height, width = geometry.shape
    if not (1 <= tile_rows <= height and 4 <= tile_cols <= width):
        raise ValueError("Tile dimensions do not fit the calibrated scan")
    if tile_rows * tile_cols * 5 > blueprint_model.fusion.max_candidates:
        raise ValueError("Tile exceeds the blueprint candidate limit")
    if not 0 <= reliability_threshold <= 1:
        raise ValueError("Reliability threshold must be calibrated in [0,1]")
    original = observed_lidar[0, observed_valid[0]]
    # A calibrated threshold of 1.0 means validation did not justify adding
    # any point. Avoid the expensive encoder and attention in that case.
    if reliability_threshold == 1.0:
        return original.clone()
    all_rows, all_cols = ray_tile_indices(geometry, device=radar.device)
    observed_depth, observed_intensity, observed_hit = project_lidar_tile(
        geometry, all_rows, all_cols, observed_lidar, observed_valid)
    projection = tuple(value.reshape(1, height, width) for value in
                       (observed_depth, observed_intensity, observed_hit))
    encoded = blueprint_model.encode_radar(radar, radar_valid)
    additions: list[torch.Tensor] = []
    for row_start in range(0, height, tile_rows):
        row_stop = min(row_start + tile_rows, height)
        for col_start in range(0, width, tile_cols):
            first_new_col = col_start
            col_stop = min(col_start + tile_cols, width)
            # The U-Net downsamples horizontally twice, so pad the final
            # short tile by shifting its start rather than losing ray columns.
            if col_stop - col_start < 4:
                col_start = max(0, col_stop - 4)
            rows, cols = ray_tile_indices(
                geometry, row_start=row_start, row_stop=row_stop,
                col_start=col_start, col_stop=col_stop,
                device=radar.device,
            )
            blueprint = blueprint_model.forward_encoded(
                radar, radar_valid, encoded, rows, cols)
            result = diffusion.sample(
                blueprint, observed_lidar, observed_valid,
                (row_stop - row_start, col_stop - col_start),
                steps=steps, reliability_threshold=reliability_threshold,
                return_threshold=return_threshold, generator=generator,
                observed_projection=tuple(
                    value[:, row_start:row_stop, col_start:col_stop].reshape(1, -1)
                    for value in projection),
            )
            added = result.added_points(0)
            if first_new_col != col_start:
                selected_cols = cols[0, result.added_mask[0].reshape(-1)]
                added = added[selected_cols >= first_new_col]
            additions.append(added)
            if on_tile is not None:
                on_tile()
    return torch.cat((original, *additions), dim=0)
