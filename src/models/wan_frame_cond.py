from __future__ import annotations

import torch
from torch import nn


class FrameCondProjector(nn.Module):
    """Project per-frame conditioning features into Wan cross-attention token space.

    Input:  feat [B,T,F]
    Output: tok  [B,T,text_dim]

    This is intentionally per-frame (no temporal mixing). It provides explicit
    inference-time signals (mask/geometry/conf) without changing the latent input.
    """

    def __init__(
        self,
        feat_dim: int,
        *,
        text_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if text_dim <= 0:
            raise ValueError("text_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.feat_dim = int(feat_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(self.feat_dim, self.text_dim, bias=True))
        else:
            layers.append(nn.Linear(self.feat_dim, self.hidden_dim, bias=True))
            layers.append(nn.GELU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
                layers.append(nn.GELU())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, self.text_dim, bias=True))
        self.net = nn.Sequential(*layers)

        # Start from "no effect" to avoid destabilizing a pretrained cross-attn stack.
        last = None
        for m in reversed(self.net):
            if isinstance(m, nn.Linear):
                last = m
                break
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 3:
            raise ValueError("feat must be [B,T,F]")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"feat last dim mismatch: got {feat.shape[-1]} expected {self.feat_dim}")
        return self.net(feat)


__all__ = ["FrameCondProjector"]

