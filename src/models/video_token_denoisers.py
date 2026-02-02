from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MazeConditionEncoder
from .transformer import TransformerEncoder


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=timesteps.device)) * torch.arange(0, half, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


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
        dim = dim - 1
    dim_half = dim // 2
    emb_h = _sincos_1d(h, dim_half, device)
    emb_w = _sincos_1d(w, dim_half, device)
    emb = torch.cat(
        [
            emb_h[:, None, :].expand(h, w, dim_half),
            emb_w[None, :, :].expand(h, w, dim_half),
        ],
        dim=-1,
    )
    emb = emb.view(h * w, dim)
    if orig_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class VideoTokenKeypointDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        d_cond: int = 128,
        use_sdf: bool = False,
        use_start_goal: bool = True,
        data_dim: int = 256,
        cond_encoder: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.d_cond = d_cond
        self.in_proj = nn.Linear(data_dim, d_model)
        self.t_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if cond_encoder is None:
            cond_encoder = MazeConditionEncoder(use_sdf=use_sdf, d_cond=d_cond, use_start_goal=use_start_goal)
        self.cond_enc = cond_encoder
        self.cond_proj = nn.Linear(d_cond, d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            cond_dim=d_cond,
            causal=False,
            use_checkpoint=use_checkpoint,
        )
        self.out = nn.Linear(d_model, data_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        idx: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        T: int,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        # z_t: [B,K,N,D]
        B, K, N, D = z_t.shape
        if D != self.data_dim:
            raise ValueError(f"data_dim mismatch: expected {self.data_dim}, got {D}")
        h = self.in_proj(z_t)

        time_emb_all = _sincos_1d(T, h.shape[-1], z_t.device)
        time_emb = time_emb_all[idx]  # [B,K,d_model]
        H_p, W_p = spatial_shape
        space_emb = _sincos_2d(H_p, W_p, h.shape[-1], z_t.device)
        h = h + time_emb[:, :, None, :] + space_emb[None, None, :, :]

        t_emb = timestep_embedding(t, h.shape[-1])
        t_emb = self.t_embed(t_emb)
        h = h + t_emb[:, None, None, :]

        if cond and self.cond_enc is not None:
            cond_vec = self.cond_enc(cond)
        else:
            cond_vec = torch.zeros((B, self.d_cond), device=z_t.device, dtype=z_t.dtype)
        h = h + self.cond_proj(cond_vec).unsqueeze(1).unsqueeze(2)

        h = h.view(B, K * N, -1)
        h = self.transformer(h, cond=cond_vec)
        out = self.out(h).view(B, K, N, D)
        return out


class VideoTokenInterpLevelDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
        d_cond: int = 128,
        use_sdf: bool = False,
        use_start_goal: bool = True,
        data_dim: int = 256,
        max_levels: int = 8,
        use_checkpoint: bool = False,
        mask_channels: int = 1,
        cond_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.d_cond = d_cond
        self.mask_channels = mask_channels
        self.in_proj = nn.Linear(data_dim + mask_channels, d_model)
        self.level_emb = nn.Embedding(max_levels + 1, d_model)
        self.level_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if cond_encoder is None:
            cond_encoder = MazeConditionEncoder(use_sdf=use_sdf, d_cond=d_cond, use_start_goal=use_start_goal)
        self.cond_enc = cond_encoder
        self.cond_proj = nn.Linear(d_cond, d_model)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            cond_dim=d_cond,
            causal=False,
            use_checkpoint=use_checkpoint,
        )
        self.out = nn.Linear(d_model, data_dim)

    def forward(
        self,
        x_s: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        # x_s: [B,T,N,D], mask: [B,T,N,C]
        B, T, N, D = x_s.shape
        if D != self.data_dim:
            raise ValueError(f"data_dim mismatch: expected {self.data_dim}, got {D}")
        if mask.dim() == 3:
            mask_in = mask.unsqueeze(-1).float()
        else:
            mask_in = mask.float()
        if mask_in.shape[-1] != self.mask_channels:
            raise ValueError(f"mask has {mask_in.shape[-1]} channels, expected {self.mask_channels}")

        x = torch.cat([x_s, mask_in], dim=-1)
        h = self.in_proj(x)

        time_emb = _sincos_1d(T, h.shape[-1], x_s.device)
        H_p, W_p = spatial_shape
        space_emb = _sincos_2d(H_p, W_p, h.shape[-1], x_s.device)
        pos = (time_emb[:, None, :] + space_emb[None, :, :]).view(T * N, -1)
        h = h.view(B, T * N, -1) + pos.unsqueeze(0)

        level = self.level_proj(self.level_emb(s))
        h = h + level.unsqueeze(1)

        if cond and self.cond_enc is not None:
            cond_vec = self.cond_enc(cond)
        else:
            cond_vec = torch.zeros((B, self.d_cond), device=x_s.device, dtype=x_s.dtype)
        h = h + self.cond_proj(cond_vec).unsqueeze(1)

        h = self.transformer(h, cond=cond_vec)
        out = self.out(h).view(B, T, N, D)
        return out


__all__ = ["VideoTokenKeypointDenoiser", "VideoTokenInterpLevelDenoiser"]
