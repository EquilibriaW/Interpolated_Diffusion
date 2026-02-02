from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTemporalInterpolator(nn.Module):
    """Tiny per-channel temporal conv interpolator for video latents."""

    def __init__(self, data_dim: int, kernel_size: int = 3, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        padding = kernel_size // 2
        layers = []
        for _ in range(n_layers):
            layers.append(
                nn.Conv1d(
                    data_dim,
                    data_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=data_dim,
                )
            )
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,T,D] -> [B,D,T]
        z_t = z.transpose(1, 2)
        out = self.net(z_t)
        out = out.transpose(1, 2)
        return out
