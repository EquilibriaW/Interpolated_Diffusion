from typing import Dict

import torch
import torch.nn as nn

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
        data_dim: int = 2,
        max_levels: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.in_proj = nn.Linear(data_dim + 1, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.level_emb = nn.Embedding(max_levels + 1, d_model)
        self.level_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
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

    def forward(self, x_s: torch.Tensor, s: torch.Tensor, mask: torch.Tensor, cond: Dict[str, torch.Tensor]):
        B, T, D = x_s.shape
        x = torch.cat([x_s, mask.unsqueeze(-1).float()], dim=-1)
        h = self.in_proj(x)
        pos = self.pos_emb(torch.arange(T, device=x_s.device))
        h = h + pos.unsqueeze(0)
        level = self.level_proj(self.level_emb(s))
        h = h + level.unsqueeze(1)
        cond_vec = self.cond_enc(cond)
        h = h + self.cond_proj(cond_vec).unsqueeze(1)
        h = self.transformer(h, cond=cond_vec)
        return self.out(h)
