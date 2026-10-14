"""Gaussian forward process and DDPM/DDIM reverse transitions."""

from __future__ import annotations

import math

import torch
from torch import nn


class NoiseSchedule(nn.Module):
    def __init__(self, timesteps: int, kind: str = "cosine"):
        super().__init__()
        if timesteps < 2 or kind not in {"cosine", "linear"}:
            raise ValueError("Invalid beta schedule")
        if kind == "linear":
            beta = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64)
        else:
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            cumulative = torch.cos(((steps / timesteps + 0.008) / 1.008) * math.pi / 2).square()
            cumulative = cumulative / cumulative[0]
            beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, dim=0).float())

    def mix(self, clean: torch.Tensor, noise: torch.Tensor,
            site_t: torch.Tensor) -> torch.Tensor:
        if clean.shape != noise.shape or site_t.shape != (len(clean),):
            raise ValueError("Diffusion inputs must share the candidate shape")
        alpha = self.alpha_bar[site_t.long()].to(clean.dtype)[:, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def predict_x0(self, noisy: torch.Tensor, predicted_noise: torch.Tensor, t: int) -> torch.Tensor:
        if noisy.shape != predicted_noise.shape or not 0 <= t < len(self.alpha_bar):
            raise ValueError("Invalid clean-state prediction")
        alpha = self.alpha_bar[t].to(noisy.dtype)
        return (noisy - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-8)

    def sampling_timesteps(self, steps: int | None, method: str) -> list[int]:
        count = len(self.alpha_bar)
        if steps is None:
            steps = count
        if not 1 <= steps <= count:
            raise ValueError(f"Sampling steps must be between 1 and {count}")
        if method == "ddpm" and steps != count:
            raise ValueError("DDPM uses every trained step; choose DDIM to skip steps")
        if method not in {"ddpm", "ddim"}:
            raise ValueError("Sampling method must be ddpm or ddim")
        if steps == 1:
            return [count - 1]
        return torch.linspace(count - 1, 0, steps).round().long().tolist()

    def reverse_step(self, noisy: torch.Tensor, predicted_noise: torch.Tensor,
                     t: int, previous_t: int, *, method: str,
                     random: torch.Tensor | None = None, eta: float = 0.0) -> torch.Tensor:
        if (noisy.shape != predicted_noise.shape or not 0 <= t < len(self.alpha_bar)
                or not -1 <= previous_t < t):
            raise ValueError("Invalid reverse diffusion transition")
        clean = self.predict_x0(noisy, predicted_noise, t)
        if previous_t == -1:
            return clean
        current = self.alpha_bar[t].to(noisy.dtype)
        earlier = self.alpha_bar[previous_t].to(noisy.dtype)
        if method == "ddpm":
            if previous_t != t - 1 or random is None or random.shape != noisy.shape:
                raise ValueError("DDPM needs an adjacent timestep and full-sized Gaussian noise")
            beta = self.beta[t].to(noisy.dtype)
            variance = beta * (1 - earlier) / (1 - current)
            mean = (noisy - beta * predicted_noise / (1 - current).sqrt()) / (1 - beta).sqrt()
            return mean + variance.clamp_min(0).sqrt() * random
        if method != "ddim" or not 0 <= eta <= 1:
            raise ValueError("Invalid DDIM settings")
        sigma = eta * ((1 - earlier) / (1 - current)).sqrt() * (1 - current / earlier).clamp_min(0).sqrt()
        direction = (1 - earlier - sigma.square()).clamp_min(0).sqrt() * predicted_noise
        if eta and (random is None or random.shape != noisy.shape):
            raise ValueError("Stochastic DDIM needs full-sized Gaussian noise")
        return earlier.sqrt() * clean + direction + (sigma * random if eta else 0)
