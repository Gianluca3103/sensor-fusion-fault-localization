"""Gaussian schedules and mask-restricted direct-BEV residual diffusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionProcessConfig:
    num_train_timesteps: int = 1000
    noise_schedule: str = "cosine"
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    prediction_type: str = "epsilon"
    denominator_epsilon: float = 1.0e-8

    def validate(self) -> None:
        if self.num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2")
        if self.noise_schedule not in {"cosine", "linear"}:
            raise ValueError("noise_schedule must be cosine or linear")
        if self.prediction_type != "epsilon":
            raise ValueError("Only epsilon prediction is supported initially")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("Expected 0 < beta_start < beta_end < 1")
        if self.denominator_epsilon <= 0:
            raise ValueError("denominator_epsilon must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


class BEVChannelNormalization(nn.Module):
    """Configurable per-channel scaling; defaults to the repository's identity scale."""

    def __init__(
        self,
        means=(0.0, 0.0, 0.0),
        stds=(1.0, 1.0, 1.0),
        epsilon=1e-8,
        source="existing_uint8_div_255_channel_scaling",
    ):
        super().__init__()
        means = torch.as_tensor(means, dtype=torch.float32)
        stds = torch.as_tensor(stds, dtype=torch.float32)
        if means.ndim != 1 or stds.shape != means.shape:
            raise ValueError("means and stds must be one-dimensional and equal length")
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
            raise ValueError("normalization statistics must be finite")
        if torch.any(stds <= 0) or epsilon <= 0:
            raise ValueError("standard deviations and epsilon must be positive")
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.epsilon = float(epsilon)
        self.source = str(source)

    def _stats(self, tensor):
        if tensor.ndim != 4 or tensor.shape[1] != len(self.means):
            raise ValueError("BEV tensor channel count does not match normalization")
        return self.means[None, :, None, None], self.stds[None, :, None, None]

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        means, stds = self._stats(tensor)
        return (tensor - means) / (stds + self.epsilon)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        means, stds = self._stats(tensor)
        return tensor * (stds + self.epsilon) + means

    def normalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        _means, stds = self._stats(residual)
        return residual / (stds + self.epsilon)

    def denormalize_residual(self, residual: torch.Tensor) -> torch.Tensor:
        _means, stds = self._stats(residual)
        return residual * (stds + self.epsilon)

    def metadata(self) -> dict:
        return {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "epsilon": self.epsilon,
            "source": self.source,
        }


def _cosine_betas(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((steps / timesteps + offset) / (1 + offset)) * math.pi / 2
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999).float()


class GaussianNoiseSchedule(nn.Module):
    def __init__(self, config: DiffusionProcessConfig | None = None):
        super().__init__()
        self.config = config or DiffusionProcessConfig()
        self.config.validate()
        if self.config.noise_schedule == "cosine":
            betas = _cosine_betas(self.config.num_train_timesteps)
        else:
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.config.num_train_timesteps,
                dtype=torch.float32,
            )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        ).clamp_min(1e-20)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1 - alpha_bars).sqrt())
        self.register_buffer("sqrt_recip_alpha_bars", alpha_bars.rsqrt())
        self.register_buffer(
            "sqrt_recipm1_alpha_bars", (1 / alpha_bars - 1).sqrt()
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alpha_bars_previous.sqrt() / (1 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1 - alpha_bars_previous) * alphas.sqrt() / (1 - alpha_bars),
        )

    @staticmethod
    def extract(values: torch.Tensor, timestep: torch.Tensor, reference: torch.Tensor):
        if timestep.ndim != 1 or timestep.shape[0] != reference.shape[0]:
            raise ValueError("timestep must have shape [B]")
        return values.gather(0, timestep).reshape(-1, 1, 1, 1).to(reference.dtype)

    def add_masked_noise(self, residual, noise, timestep, reconstruction_mask):
        if residual.shape != noise.shape:
            raise ValueError("residual and noise must have identical shapes")
        masked_noise = reconstruction_mask * noise
        noisy = (
            self.extract(self.sqrt_alpha_bars, timestep, residual) * residual
            + self.extract(self.sqrt_one_minus_alpha_bars, timestep, residual)
            * masked_noise
        )
        return reconstruction_mask * noisy, masked_noise

    def ddpm_step(
        self,
        residual_t: torch.Tensor,
        epsilon_prediction: torch.Tensor,
        timestep: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pred_x0 = (
            self.extract(self.sqrt_recip_alpha_bars, timestep, residual_t) * residual_t
            - self.extract(self.sqrt_recipm1_alpha_bars, timestep, residual_t)
            * epsilon_prediction
        )
        mean = (
            self.extract(self.posterior_mean_coef1, timestep, residual_t) * pred_x0
            + self.extract(self.posterior_mean_coef2, timestep, residual_t) * residual_t
        )
        noise = torch.randn(
            residual_t.shape,
            device=residual_t.device,
            dtype=residual_t.dtype,
            generator=generator,
        )
        nonzero = (timestep > 0).to(residual_t.dtype).reshape(-1, 1, 1, 1)
        previous = mean + nonzero * self.extract(
            self.posterior_variance.sqrt(), timestep, residual_t
        ) * noise
        return reconstruction_mask * previous

    def predict_x0(
        self,
        residual_t: torch.Tensor,
        epsilon_prediction: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the clean residual estimate from an epsilon prediction."""

        return (
            self.extract(self.sqrt_recip_alpha_bars, timestep, residual_t)
            * residual_t
            - self.extract(self.sqrt_recipm1_alpha_bars, timestep, residual_t)
            * epsilon_prediction
        )

    def ddim_step(
        self,
        residual_t: torch.Tensor,
        epsilon_prediction: torch.Tensor,
        timestep: torch.Tensor,
        previous_timestep: torch.Tensor,
        reconstruction_mask: torch.Tensor,
        *,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one mask-restricted DDIM step and return `(x_prev, x0)`."""

        if eta < 0:
            raise ValueError("DDIM eta must be non-negative")
        if previous_timestep.shape != timestep.shape:
            raise ValueError("previous_timestep must have the same shape as timestep")
        alpha_t = self.extract(self.alpha_bars, timestep, residual_t)
        previous_index = previous_timestep.clamp_min(0)
        alpha_previous = self.extract(
            self.alpha_bars, previous_index, residual_t
        )
        alpha_previous = torch.where(
            (previous_timestep < 0).reshape(-1, 1, 1, 1),
            torch.ones_like(alpha_previous),
            alpha_previous,
        )
        x0 = self.predict_x0(residual_t, epsilon_prediction, timestep)
        sigma = eta * torch.sqrt(
            ((1.0 - alpha_previous) / (1.0 - alpha_t).clamp_min(1.0e-12))
            * (1.0 - alpha_t / alpha_previous.clamp_min(1.0e-12))
        ).clamp_min(0.0)
        direction = torch.sqrt(
            (1.0 - alpha_previous - sigma.square()).clamp_min(0.0)
        ) * epsilon_prediction
        previous = alpha_previous.sqrt() * x0 + direction
        if eta > 0:
            noise = torch.randn(
                residual_t.shape,
                device=residual_t.device,
                dtype=residual_t.dtype,
                generator=generator,
            )
            previous = previous + sigma * noise
        return reconstruction_mask * previous, reconstruction_mask * x0


def residual_target(clean_lidar_bev, coarse_lidar_bev, reconstruction_mask):
    if clean_lidar_bev.shape != coarse_lidar_bev.shape:
        raise ValueError("clean and coarse LiDAR BEVs must have identical shapes")
    return reconstruction_mask * (clean_lidar_bev - coarse_lidar_bev)


class MaskedEpsilonMSELoss(nn.Module):
    def __init__(self, denominator_epsilon: float = 1.0e-8):
        super().__init__()
        if denominator_epsilon <= 0:
            raise ValueError("denominator_epsilon must be positive")
        self.denominator_epsilon = float(denominator_epsilon)

    def forward(self, epsilon_prediction, epsilon, reconstruction_mask):
        if epsilon_prediction.shape != epsilon.shape:
            raise ValueError("epsilon prediction and target must have identical shapes")
        squared_error = reconstruction_mask * (epsilon_prediction - epsilon).square()
        denominator = epsilon.shape[1] * reconstruction_mask.sum()
        return squared_error.sum() / (denominator + self.denominator_epsilon)
