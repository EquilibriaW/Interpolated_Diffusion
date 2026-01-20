import math
from typing import Dict

import torch


def linear_beta_schedule(n_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_timesteps)


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = n_timesteps + 1
    x = torch.linspace(0, n_timesteps, steps)
    alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def make_beta_schedule(name: str, n_timesteps: int) -> torch.Tensor:
    if name == "linear":
        return linear_beta_schedule(n_timesteps)
    if name == "cosine":
        return cosine_beta_schedule(n_timesteps)
    raise ValueError(f"Unknown schedule {name}")


def make_alpha_bars(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
    }
