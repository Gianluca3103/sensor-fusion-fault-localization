"""Selector-local sparse 3D diffusion baseline with a Chamfer loss.

This is intentionally independent of the established 2D BEV refiner.  It
denoises a fixed, sparse lattice of candidate 3D voxels and therefore supports
missing-voxel generation without materialising the full 32x320x320 grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .basic_diffusion_unet import SinusoidalTimeEmbedding
from .sparse_voxel_data import SparseVoxelBatch


@dataclass(frozen=True)
class SparseVoxelDiffusionConfig:
    condition_feature_dim: int = 2
    grid_dimensions_zyx: tuple[int, int, int] = (32, 320, 320)
    hidden_dim: int = 96
    num_blocks: int = 4
    training_timesteps: int = 1000
    lambda_diffusion: float = 1.0
    lambda_bce: float = 1.0
    lambda_chamfer: float = 0.5
    chamfer_temperature_m: float = 0.25
    chamfer_chunk_size: int = 2048
    chamfer_max_points: int | None = None
    metric_distance_m: float = 0.2
    metric_occupancy_threshold: float = 0.5

    def validate(self) -> None:
        if self.condition_feature_dim < 1 or self.hidden_dim < 1:
            raise ValueError("condition_feature_dim and hidden_dim must be positive")
        if self.num_blocks < 1 or self.training_timesteps < 2:
            raise ValueError("num_blocks must be positive and timesteps at least two")
        if len(self.grid_dimensions_zyx) != 3 or any(v < 1 for v in self.grid_dimensions_zyx):
            raise ValueError("grid_dimensions_zyx must contain three positive values")
        if any(weight < 0 for weight in (self.lambda_diffusion, self.lambda_bce, self.lambda_chamfer)):
            raise ValueError("diffusion, BCE, and Chamfer weights must be non-negative")
        if self.chamfer_temperature_m <= 0 or self.metric_distance_m <= 0:
            raise ValueError("Chamfer temperature and metric distance must be positive")
        if not 0 < self.metric_occupancy_threshold < 1:
            raise ValueError("metric occupancy threshold must be in (0, 1)")
        if self.chamfer_chunk_size < 1:
            raise ValueError("chamfer_chunk_size must be positive")
        if self.chamfer_max_points is not None and self.chamfer_max_points < 1:
            raise ValueError("chamfer_max_points must be positive when set")


def _cosine_alpha_bars(timesteps: int) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps + 0.008) / 1.008) * math.pi / 2).square()
    return (alpha_bar / alpha_bar[0])[1:].float().clamp_min(1.0e-8)


class SparseGaussianSchedule(nn.Module):
    """Gaussian schedule for tensors shaped ``[B, N, C]``."""

    def __init__(self, timesteps: int) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least two")
        alpha_bars = _cosine_alpha_bars(timesteps)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1 or timestep.shape[0] != reference.shape[0]:
            raise ValueError("timestep must have shape [B]")
        return values.gather(0, timestep).view(-1, 1, 1).to(reference.dtype)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.shape != noise.shape or mask.shape != clean.shape:
            raise ValueError("clean, noise, and mask must have identical shapes")
        masked_noise = noise * mask
        noisy = self._extract(self.sqrt_alpha_bars, timestep, clean) * clean
        noisy = noisy + self._extract(self.sqrt_one_minus_alpha_bars, timestep, clean) * masked_noise
        return noisy * mask, masked_noise

    def predict_x0(self, noisy: torch.Tensor, epsilon: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.sqrt_alpha_bars, timestep, noisy)
        sigma = self._extract(self.sqrt_one_minus_alpha_bars, timestep, noisy)
        return (noisy - sigma * epsilon) / alpha.clamp_min(1.0e-8)


class _SparseMessageBlock(nn.Module):
    """Sparse 6-neighbour message block on globally indexed ``zyx`` voxels."""

    def __init__(self, hidden_dim: int, time_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.message = nn.Linear(hidden_dim * 2, hidden_dim)
        self.time = nn.Linear(time_dim, hidden_dim)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    @staticmethod
    def _neighbour_mean(hidden: torch.Tensor, coords: torch.Tensor, valid: torch.Tensor, dimensions: tuple[int, int, int]) -> torch.Tensor:
        """Average existing face-neighbours using sort/search, not a dense grid."""

        batch, count, channels = hidden.shape
        z_size, y_size, x_size = dimensions
        keys_per_batch = z_size * y_size * x_size
        batch_ids = torch.arange(batch, device=coords.device)[:, None].expand(batch, count)
        valid_flat = valid[..., 0] > 0.5
        flat_coords = coords.reshape(-1, 3)
        flat_batch = batch_ids.reshape(-1)
        flat_valid = valid_flat.reshape(-1)
        flat_hidden = hidden.reshape(-1, channels)
        result = torch.zeros_like(flat_hidden)
        positions = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
        if positions.numel() == 0:
            return result.reshape(batch, count, channels)
        selected = flat_coords[positions]
        selected_batch = flat_batch[positions]
        keys = selected_batch * keys_per_batch + (selected[:, 0] * y_size + selected[:, 1]) * x_size + selected[:, 2]
        order = keys.argsort()
        sorted_keys = keys[order]
        sorted_positions = positions[order]
        sums = torch.zeros((positions.numel(), channels), dtype=hidden.dtype, device=hidden.device)
        neighbour_counts = torch.zeros((positions.numel(), 1), dtype=hidden.dtype, device=hidden.device)
        shifts = selected.new_tensor(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
        for shift in shifts:
            neighbour = selected + shift
            inside = (
                (neighbour[:, 0] >= 0) & (neighbour[:, 0] < z_size)
                & (neighbour[:, 1] >= 0) & (neighbour[:, 1] < y_size)
                & (neighbour[:, 2] >= 0) & (neighbour[:, 2] < x_size)
            )
            neighbour_keys = selected_batch * keys_per_batch + (neighbour[:, 0] * y_size + neighbour[:, 1]) * x_size + neighbour[:, 2]
            lookup = torch.searchsorted(sorted_keys, neighbour_keys)
            found = inside & (lookup < sorted_keys.numel())
            safe_lookup = lookup.clamp_max(sorted_keys.numel() - 1)
            found &= sorted_keys[safe_lookup] == neighbour_keys
            if torch.any(found):
                neighbour_positions = sorted_positions[safe_lookup[found]]
                sums[found] += flat_hidden[neighbour_positions]
                neighbour_counts[found] += 1.0
        result[positions] = sums / neighbour_counts.clamp_min(1.0)
        return result.reshape(batch, count, channels)

    def forward(self, hidden: torch.Tensor, coords: torch.Tensor, valid: torch.Tensor, timestep_embedding: torch.Tensor, dimensions: tuple[int, int, int]) -> torch.Tensor:
        normalized = self.norm(hidden)
        neighbours = self._neighbour_mean(normalized, coords, valid, dimensions)
        update = self.message(torch.cat((normalized, neighbours), dim=-1))
        update = update + self.time(F.silu(timestep_embedding))[:, None, :]
        return (hidden + self.output(update)) * valid


class SparseVoxelDenoiser(nn.Module):
    """3D sparse message-passing epsilon predictor for the candidate lattice."""

    def __init__(self, config: SparseVoxelDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        time_dim = config.hidden_dim * 2
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.input = nn.Linear(1 + config.condition_feature_dim + 3, config.hidden_dim)
        self.blocks = nn.ModuleList(_SparseMessageBlock(config.hidden_dim, time_dim) for _ in range(config.num_blocks))
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 1))

    def forward(self, noisy_state: torch.Tensor, batch: SparseVoxelBatch, timestep: torch.Tensor) -> torch.Tensor:
        if noisy_state.shape != batch.target_occupancy.shape:
            raise ValueError("noisy_state must align with the sparse batch target")
        if batch.condition_features.shape[-1] != self.config.condition_feature_dim:
            raise ValueError("Sparse batch condition feature width does not match config")
        dimensions = noisy_state.new_tensor(self.config.grid_dimensions_zyx)
        positions = batch.coords_zyx.to(noisy_state.dtype) / dimensions[None, None, :]
        hidden = self.input(torch.cat((noisy_state, batch.condition_features.to(noisy_state.dtype), positions), dim=-1))
        hidden = hidden * batch.valid_mask
        time_embedding = self.time_embedding(timestep)
        for block in self.blocks:
            hidden = block(hidden, batch.coords_zyx, batch.valid_mask, time_embedding, self.config.grid_dimensions_zyx)
        return self.output(hidden) * batch.editable_mask * batch.valid_mask


class SoftVoxelChamferLoss(nn.Module):
    """Differentiable Chamfer surrogate between voxel-centre point sets.

    It weights candidate voxel centres by predicted occupancy and uses a soft
    minimum in the target-to-prediction direction, allowing gradients without
    thresholding or materialising a variable-length predicted point cloud.
    """

    def __init__(
        self, temperature_m: float = 0.25, chunk_size: int = 2048,
        max_points: int | None = None,
    ) -> None:
        super().__init__()
        if temperature_m <= 0 or chunk_size < 1 or (max_points is not None and max_points < 1):
            raise ValueError("temperature_m, chunk_size, and max_points must be positive")
        self.temperature_m = float(temperature_m)
        self.chunk_size = int(chunk_size)
        self.max_points = max_points

    def _one(self, coords: torch.Tensor, probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_coords = coords[target > 0.5]
        if target_coords.numel() == 0:
            # Empty target crops are governed by occupancy/diffusion losses.
            return probabilities.new_zeros(())
        # Chamfer is a soft point-set surrogate.  Large selector components can
        # contain over 100k candidates, making the full pairwise computation
        # impractical.  Sample each set independently; diffusion and BCE still
        # supervise every voxel.  Torch RNG makes the approximation reproducible
        # for a fixed training seed and varies it across optimization steps.
        if self.max_points is not None:
            if len(coords) > self.max_points:
                chosen = torch.randperm(len(coords), device=coords.device)[:self.max_points]
                coords = coords[chosen]
                probabilities = probabilities[chosen]
            if len(target_coords) > self.max_points:
                chosen = torch.randperm(len(target_coords), device=coords.device)[:self.max_points]
                target_coords = target_coords[chosen]
        probabilities = probabilities.clamp(1.0e-6, 1.0 - 1.0e-6)
        weighted_distance = probabilities.new_zeros(())
        probability_sum = probabilities.sum().clamp_min(1.0e-6)
        # Chunk *both* point sets.  A selector crop can contain many clean
        # occupied voxels, so chunking only candidates still materialises a
        # [candidate_chunk, all_targets] CUDA allocation.
        for start in range(0, len(coords), self.chunk_size):
            stop = min(start + self.chunk_size, len(coords))
            nearest = torch.full(
                (stop - start,), torch.inf, dtype=coords.dtype, device=coords.device
            )
            for target_start in range(0, len(target_coords), self.chunk_size):
                target_stop = min(target_start + self.chunk_size, len(target_coords))
                distances = torch.cdist(coords[start:stop], target_coords[target_start:target_stop])
                nearest = torch.minimum(nearest, distances.min(dim=1).values)
            weights = probabilities[start:stop]
            weighted_distance = weighted_distance + (weights * nearest).sum()
        target_to_prediction_total = probabilities.new_zeros(())
        for target_start in range(0, len(target_coords), self.chunk_size):
            target_stop = min(target_start + self.chunk_size, len(target_coords))
            target_chunk = target_coords[target_start:target_stop]
            target_log_support = torch.full(
                (len(target_chunk),),
                -torch.inf,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            for start in range(0, len(coords), self.chunk_size):
                stop = min(start + self.chunk_size, len(coords))
                distances = torch.cdist(coords[start:stop], target_chunk)
                weights = probabilities[start:stop]
                target_log_support = torch.logaddexp(
                    target_log_support,
                    torch.logsumexp(
                        torch.log(weights)[:, None] - distances / self.temperature_m,
                        dim=0,
                    ),
                )
            target_to_prediction_total = target_to_prediction_total + (
                -self.temperature_m * (target_log_support - torch.log(probability_sum))
            ).sum()
        prediction_to_target = weighted_distance / probability_sum
        # Normalise the predicted occupancy weights before the soft minimum.
        # Without this, several candidates near one target could make the
        # log-sum-exp positive and turn a distance term negative.
        target_to_prediction = target_to_prediction_total / len(target_coords)
        return 0.5 * (prediction_to_target + target_to_prediction)

    def forward(self, occupancy_probability: torch.Tensor, batch: SparseVoxelBatch) -> torch.Tensor:
        if occupancy_probability.shape != batch.target_occupancy.shape:
            raise ValueError("occupancy_probability must align with sparse batch")
        losses = []
        for index in range(batch.batch_size):
            valid = batch.valid_mask[index, :, 0] > 0.5
            losses.append(self._one(
                batch.coords_xyz_m[index, valid].float(),
                occupancy_probability[index, valid, 0].float(),
                batch.target_occupancy[index, valid, 0],
            ))
        return torch.stack(losses).mean()


@torch.no_grad()
def voxel_set_metrics_at_distance(
    occupancy_probability: torch.Tensor,
    batch: SparseVoxelBatch,
    *,
    distance_m: float = 0.2,
    occupancy_threshold: float = 0.5,
    chunk_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """Compute tolerance-based precision, recall, F1, and IoU for 3D voxels.

    Only editable selector voxels are scored; trusted context is excluded. A
    prediction is correct when a clean occupied voxel centre lies within the
    requested physical distance.  Small sets use chunked distance matrices;
    large sets use an exact k-d tree lookup to avoid quadratic validation.
    IoU is derived from the symmetric F1 relation, ``F1 / (2 - F1)``.
    """

    if distance_m <= 0 or chunk_size < 1:
        raise ValueError("distance_m and chunk_size must be positive")
    if not 0 < occupancy_threshold < 1:
        raise ValueError("occupancy_threshold must be in (0, 1)")

    def nearest_matches(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not len(source):
            return torch.empty(0, dtype=torch.bool, device=source.device)
        if len(source) * len(target) > 1_000_000:
            # Validation needs exact nearest-neighbour distances, but not a
            # differentiable N x M CUDA matrix.  A CPU k-d tree bounds the
            # work for the large selector components present in VoD.
            tree = cKDTree(target.detach().cpu().numpy())
            distances, _ = tree.query(source.detach().cpu().numpy(), k=1, workers=1)
            return torch.as_tensor(distances <= distance_m + 1.0e-6, device=source.device)
        matches = []
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            nearest = torch.full(
                (stop - start,), torch.inf, dtype=source.dtype, device=source.device
            )
            for target_start in range(0, len(target), chunk_size):
                target_stop = min(target_start + chunk_size, len(target))
                distances = torch.cdist(source[start:stop], target[target_start:target_stop])
                nearest = torch.minimum(nearest, distances.min(dim=1).values)
            matches.append(nearest <= distance_m)
        return torch.cat(matches)

    f1_values, iou_values = [], []
    precision_values, recall_values = [], []
    for index in range(batch.batch_size):
        valid = (batch.valid_mask[index, :, 0] > 0.5) & (
            batch.editable_mask[index, :, 0] > 0.5
        )
        coords = batch.coords_xyz_m[index, valid].float()
        predicted = coords[occupancy_probability[index, valid, 0] >= occupancy_threshold]
        target = coords[batch.target_occupancy[index, valid, 0] > 0.5]
        if not len(predicted) and not len(target):
            precision = recall = coords.new_tensor(1.0)
        elif not len(predicted) or not len(target):
            precision = recall = coords.new_zeros(())
        else:
            precision = nearest_matches(predicted, target).float().mean()
            recall = nearest_matches(target, predicted).float().mean()
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
        iou = f1 / (2.0 - f1).clamp_min(1.0e-8)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        iou_values.append(iou)
    return {
        "precision_at_0_2m": torch.stack(precision_values).mean(),
        "recall_at_0_2m": torch.stack(recall_values).mean(),
        "f1_at_0_2m": torch.stack(f1_values).mean(),
        "iou_at_0_2m": torch.stack(iou_values).mean(),
    }


class SparseVoxelDiffusionBaseline(nn.Module):
    """3D local diffusion model with ``L = L_current_diffusion + λ_CD L_CD``."""

    def __init__(self, config: SparseVoxelDiffusionConfig = SparseVoxelDiffusionConfig()) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.schedule = SparseGaussianSchedule(config.training_timesteps)
        self.denoiser = SparseVoxelDenoiser(config)
        self.chamfer_loss = SoftVoxelChamferLoss(
            config.chamfer_temperature_m, config.chamfer_chunk_size,
            config.chamfer_max_points,
        )

    @staticmethod
    def _target_state(batch: SparseVoxelBatch) -> torch.Tensor:
        # Diffusion acts only on edits. Outside the operation mask the known
        # faulty occupancy is composed back unchanged at the end.
        return (batch.target_occupancy * 2.0 - 1.0) * batch.editable_mask * batch.valid_mask

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, batch: SparseVoxelBatch, timestep: torch.Tensor | None = None, *, compute_metrics: bool = False) -> dict[str, torch.Tensor]:
        device = batch.target_occupancy.device
        if timestep is None:
            timestep = torch.randint(self.config.training_timesteps, (batch.batch_size,), device=device)
        clean = self._target_state(batch)
        noise = torch.randn_like(clean)
        noisy, target_epsilon = self.schedule.add_noise(clean, noise, timestep, batch.editable_mask * batch.valid_mask)
        epsilon_prediction = self.denoiser(noisy, batch, timestep)
        diffusion_loss = self._masked_mean((epsilon_prediction - target_epsilon).square(), batch.editable_mask * batch.valid_mask)
        predicted_state = self.schedule.predict_x0(noisy, epsilon_prediction, timestep)
        editable_probability = torch.sigmoid(2.0 * predicted_state)
        occupancy_probability = torch.where(
            batch.editable_mask > 0.5, editable_probability, batch.faulty_occupancy
        ) * batch.valid_mask
        bce_loss = self._masked_mean(
            F.binary_cross_entropy(occupancy_probability.clamp(1.0e-6, 1.0 - 1.0e-6), batch.target_occupancy, reduction="none"),
            batch.editable_mask * batch.valid_mask,
        )
        current_diffusion_loss = diffusion_loss
        chamfer = (
            self.chamfer_loss(occupancy_probability, batch)
            if self.config.lambda_chamfer > 0
            else diffusion_loss.new_zeros(())
        )
        loss = (
            self.config.lambda_diffusion * diffusion_loss
            + self.config.lambda_bce * bce_loss
            + self.config.lambda_chamfer * chamfer
        )
        output = {
            "loss": loss,
            "current_diffusion_loss": current_diffusion_loss,
            "diffusion_loss": diffusion_loss,
            "bce_loss": bce_loss,
            "chamfer_loss": chamfer,
            "occupancy_probability": occupancy_probability,
            "predicted_state": predicted_state,
            "timestep": timestep,
        }
        if compute_metrics:
            output.update(voxel_set_metrics_at_distance(
                occupancy_probability,
                batch,
                distance_m=self.config.metric_distance_m,
                occupancy_threshold=self.config.metric_occupancy_threshold,
                chunk_size=self.config.chamfer_chunk_size,
            ))
        return output

    @torch.no_grad()
    def sample(
        self,
        batch: SparseVoxelBatch,
        *,
        sampling_steps: int = 25,
        occupancy_threshold: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """DDIM-sample editable voxels and compose trusted evidence unchanged."""

        if not 1 <= sampling_steps <= self.config.training_timesteps:
            raise ValueError("sampling_steps must be in [1, training_timesteps]")
        if not 0.0 < occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must be in (0, 1)")
        device = batch.target_occupancy.device
        edit = batch.editable_mask * batch.valid_mask
        state = torch.randn(
            batch.target_occupancy.shape,
            device=device,
            dtype=batch.target_occupancy.dtype,
            generator=generator,
        ) * edit
        timesteps = torch.linspace(
            self.config.training_timesteps - 1,
            0,
            sampling_steps,
            device=device,
        ).round().long().unique_consecutive()
        predicted_state = state
        for index, current_timestep in enumerate(timesteps):
            timestep = current_timestep.expand(batch.batch_size)
            epsilon = self.denoiser(state, batch, timestep)
            predicted_state = self.schedule.predict_x0(state, epsilon, timestep) * edit
            if index + 1 == len(timesteps):
                state = predicted_state
                continue
            previous_timestep = timesteps[index + 1]
            alpha_previous = self.schedule.alpha_bars[previous_timestep].to(state.dtype)
            state = (
                alpha_previous.sqrt() * predicted_state
                + (1.0 - alpha_previous).sqrt() * epsilon
            ) * edit
        editable_probability = torch.sigmoid(2.0 * predicted_state)
        occupancy_probability = torch.where(
            batch.editable_mask > 0.5,
            editable_probability,
            batch.faulty_occupancy,
        ) * batch.valid_mask
        occupied_mask = (occupancy_probability >= occupancy_threshold) * (batch.valid_mask > 0.5)
        return {
            "occupancy_probability": occupancy_probability,
            "occupied_mask": occupied_mask,
            "candidate_coords_zyx": batch.coords_zyx,
            "candidate_coords_xyz_m": batch.coords_xyz_m,
            "predicted_state": predicted_state,
            "sampling_timesteps": timesteps,
        }
