from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MazeConditionEncoder
from .transformer import TransformerEncoder


class InterpLevelDenoiser(nn.Module):
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
        max_levels: int = 8,
        use_checkpoint: bool = False,
        mask_channels: int = 1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.mask_channels = mask_channels
        self.in_proj = nn.Linear(data_dim + mask_channels, d_model)
        self.level_emb = nn.Embedding(max_levels + 1, d_model)
        self.level_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_enc = MazeConditionEncoder(use_sdf=use_sdf, d_cond=d_cond, use_start_goal=use_start_goal)
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

    def _positional_embedding(self, T: int, device: torch.device, dim: int) -> torch.Tensor:
        pos = torch.linspace(0.0, 1.0, T, device=device)
        half = dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(0, half, device=device) / half)
        args = pos.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x_s: torch.Tensor, s: torch.Tensor, mask: torch.Tensor, cond: Dict[str, torch.Tensor]):
        B, T, D = x_s.shape
        if mask.dim() == 2:
            mask_in = mask.unsqueeze(-1).float()
        else:
            mask_in = mask.float()
        if mask_in.shape[-1] != self.mask_channels:
            raise ValueError(f"mask has {mask_in.shape[-1]} channels, expected {self.mask_channels}")
        x = torch.cat([x_s, mask_in], dim=-1)
        h = self.in_proj(x)
        pos = self._positional_embedding(T, x_s.device, h.shape[-1])
        h = h + pos.unsqueeze(0)
        level = self.level_proj(self.level_emb(s))
        h = h + level.unsqueeze(1)
        cond_vec = self.cond_enc(cond)
        h = h + self.cond_proj(cond_vec).unsqueeze(1)
        h = self.transformer(h, cond=cond_vec)
        return self.out(h)
