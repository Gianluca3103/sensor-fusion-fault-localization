from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    """Small self-contained DDPM/DDIM schedule for residual diffusion."""

    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self):
        if self.num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2")
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(
        self,
        residual: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1).to(residual.device)
        return alpha_bar.sqrt() * residual + (1.0 - alpha_bar).sqrt() * (mask * noise)

    def reconstruct_x0(self, noisy: torch.Tensor, timestep: torch.Tensor, noise_prediction: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1).to(noisy.device)
        return (noisy - (1.0 - alpha_bar).sqrt() * noise_prediction) / alpha_bar.sqrt().clamp_min(1e-8)

    def inference_timesteps(self, num_inference_steps: int, device: torch.device | str) -> torch.Tensor:
        steps = torch.linspace(self.num_train_timesteps - 1, 0, int(num_inference_steps), device=device)
        return steps.round().long().unique_consecutive()

    @torch.no_grad()
    def ddim_step(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        previous_timestep: torch.Tensor,
        noise_prediction: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1).to(noisy.device)
        previous_alpha_bar = self.alpha_bars[previous_timestep].view(-1, 1, 1, 1).to(noisy.device)
        x0 = self.reconstruct_x0(noisy, timestep, noise_prediction)
        direction = (1.0 - previous_alpha_bar).sqrt() * noise_prediction
        return previous_alpha_bar.sqrt() * x0 + direction

