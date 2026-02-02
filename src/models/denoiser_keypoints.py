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
        use_start_goal: bool = True,
        data_dim: int = 2,
        pos_dim: Optional[int] = None,
        cond_encoder: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        kp_feat_dim: int = 0,
        maze_channels: tuple[int, ...] = (32, 64),
    ):
        super().__init__()
        self.data_dim = data_dim
        self.d_cond = d_cond
        self.kp_feat_dim = kp_feat_dim
        if pos_dim is None:
            pos_dim = d_model // 2
        self.pos_dim = pos_dim
        self.in_proj = nn.Linear(data_dim + pos_dim + data_dim + kp_feat_dim, d_model)
        self.t_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if cond_encoder is None:
            cond_encoder = MazeConditionEncoder(
                use_sdf=use_sdf, d_cond=d_cond, use_start_goal=use_start_goal, maze_channels=maze_channels
            )
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
        known_mask: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        T: int,
    ) -> torch.Tensor:
        B, K, D = z_t.shape
        pos = idx.float() / max(1.0, float(T - 1))
        pos_emb = continuous_time_embedding(pos, self.pos_dim)
        if self.kp_feat_dim > 0 and cond is not None and "kp_feat" in cond:
            kp_feat = cond["kp_feat"]
            if kp_feat.shape[:2] != (B, K):
                raise ValueError("kp_feat must have shape [B,K,F]")
            if kp_feat.shape[-1] != self.kp_feat_dim:
                raise ValueError("kp_feat_dim mismatch")
        else:
            kp_feat = torch.zeros((B, K, self.kp_feat_dim), device=z_t.device, dtype=z_t.dtype)
        x = torch.cat([z_t, pos_emb, known_mask.float(), kp_feat], dim=-1)
        h = self.in_proj(x)
        t_emb = timestep_embedding(t, h.shape[-1])
        t_emb = self.t_embed(t_emb)
        h = h + t_emb.unsqueeze(1)
        if cond and self.cond_enc is not None:
            cond_vec = self.cond_enc(cond)
        else:
            cond_vec = torch.zeros((B, self.d_cond), device=z_t.device, dtype=z_t.dtype)
        h = h + self.cond_proj(cond_vec).unsqueeze(1)
        h = self.transformer(h, cond=cond_vec)
        return self.out(h)
