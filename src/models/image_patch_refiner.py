from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerEncoder


def _sincos_1d(n: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.linspace(0.0, 1.0, n, device=device)
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(0, half, device=device) / half)
    args = pos.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _sincos_2d(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    orig_dim = dim
    if dim % 2 == 1:
        dim -= 1
    dim_h = dim // 2
    emb_h = _sincos_1d(h, dim_h, device)
    emb_w = _sincos_1d(w, dim_h, device)
    emb = torch.cat(
        [
            emb_h[:, None, :].expand(h, w, dim_h),
            emb_w[None, :, :].expand(h, w, dim_h),
        ],
        dim=-1,
    ).reshape(h * w, dim)
    if orig_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ImagePatchRefiner(nn.Module):
    """One-step refiner for keypatch interpolation corruption."""

    def __init__(
        self,
        data_dim: int,
        num_classes: int,
        max_levels: int = 8,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_dim = int(data_dim)
        self.num_classes = int(num_classes)
        self.max_levels = int(max_levels)

        self.in_proj = nn.Linear(self.data_dim + 1, d_model)
        self.level_emb = nn.Embedding(self.max_levels + 1, d_model)
        self.class_emb = nn.Embedding(self.num_classes, d_model)
        self.alpha_proj = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.level_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.class_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            cond_dim=d_model,
            causal=False,
        )
        self.out = nn.Linear(d_model, self.data_dim)

    def forward(
        self,
        z_s: torch.Tensor,
        level: torch.Tensor,
        mask: torch.Tensor,
        class_ids: torch.Tensor,
        grid_hw: Tuple[int, int],
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict patch-token correction.

        Args:
            z_s: [B, N, D]
            level: [B] long in [0, max_levels]
            mask: [B, N] bool/float (1=anchor)
            class_ids: [B] long in [0, num_classes)
            grid_hw: patch grid (H_p, W_p)
            alpha: [B] CFG-conditioning scale
        Returns:
            delta: [B, N, D]
        """
        if z_s.dim() != 3:
            raise ValueError("z_s must be [B,N,D]")
        b, n, d = z_s.shape
        if d != self.data_dim:
            raise ValueError(f"data_dim mismatch: expected {self.data_dim}, got {d}")
        if mask.dim() != 2:
            raise ValueError("mask must be [B,N]")
        if mask.shape != (b, n):
            raise ValueError("mask shape mismatch")
        if level.shape[0] != b or class_ids.shape[0] != b:
            raise ValueError("level/class batch mismatch")
        if alpha is None:
            alpha = torch.ones((b,), device=z_s.device, dtype=z_s.dtype)
        if alpha.shape[0] != b:
            raise ValueError("alpha batch mismatch")

        mask_f = mask.float().unsqueeze(-1)
        x = torch.cat([z_s, mask_f], dim=-1)
        h = self.in_proj(x)

        hp, wp = int(grid_hw[0]), int(grid_hw[1])
        pos = _sincos_2d(hp, wp, h.shape[-1], h.device)
        if pos.shape[0] != n:
            raise ValueError(f"grid mismatch: pos={pos.shape[0]} tokens={n}")
        h = h + pos.unsqueeze(0)

        cond = self.level_proj(self.level_emb(level)) + self.class_proj(self.class_emb(class_ids))
        cond = cond + self.alpha_proj(alpha.to(dtype=z_s.dtype).unsqueeze(-1))
        h = h + cond.unsqueeze(1)
        h = self.transformer(h, cond=cond)
        delta = self.out(h)
        return delta


__all__ = ["ImagePatchRefiner"]
