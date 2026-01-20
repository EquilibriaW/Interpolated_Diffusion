from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MazeConditionEncoder
from .transformer import TransformerEncoder


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for diffusion timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=timesteps.device)) * torch.arange(0, half, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def continuous_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous time in [0,1]."""
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(0, half, device=t.device) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class KeypointDenoiser(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
        d_cond: int = 128,
        use_sdf: bool = False,
        data_dim: int = 2,
        pos_dim: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.data_dim = data_dim
        if pos_dim is None:
            pos_dim = d_model // 2
        self.pos_dim = pos_dim
        self.in_proj = nn.Linear(data_dim + pos_dim + 1, d_model)
        self.t_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_enc = MazeConditionEncoder(use_sdf=use_sdf, d_cond=d_cond)
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
        known: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        T: int,
    ) -> torch.Tensor:
        B, K, D = z_t.shape
        pos = idx.float() / max(1.0, float(T - 1))
        pos_emb = continuous_time_embedding(pos, self.pos_dim)
        x = torch.cat([z_t, pos_emb, known.unsqueeze(-1).float()], dim=-1)
        h = self.in_proj(x)
        t_emb = timestep_embedding(t, h.shape[-1])
        t_emb = self.t_embed(t_emb)
        h = h + t_emb.unsqueeze(1)
        cond_vec = self.cond_enc(cond)
        h = h + self.cond_proj(cond_vec).unsqueeze(1)
        h = self.transformer(h, cond=cond_vec)
        return self.out(h)
