from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import TextConditionEncoder
from src.models.transformer import TransformerEncoder


def _sinusoidal_1d(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(0, half, device=t.device) / max(1, half)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class VideoKeyframeSelector(nn.Module):
    """Text-conditioned keyframe selector for fixed-length clips.

    Input: cond dict with "text_embed" and optional "level" (scalar per sample).
    Output: logits over time indices [B, T].
    """

    def __init__(
        self,
        *,
        T: int,
        text_dim: int,
        d_model: int = 256,
        d_cond: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        pos_dim: int = 64,
        dropout: float = 0.0,
        use_level: bool = False,
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.text_dim = int(text_dim)
        self.d_model = int(d_model)
        self.d_cond = int(d_cond)
        self.pos_dim = int(pos_dim)
        self.use_level = bool(use_level)

        self.cond_enc = TextConditionEncoder(text_dim=self.text_dim, d_cond=self.d_cond)
        self.level_mlp: Optional[nn.Module] = None
        if self.use_level:
            self.level_mlp = nn.Sequential(nn.Linear(1, self.d_cond), nn.SiLU(), nn.Linear(self.d_cond, self.d_cond))

        self.pos_proj = nn.Linear(self.pos_dim, self.d_model)
        self.time_embed = nn.Parameter(torch.randn(self.T, self.d_model) * 0.02)

        self.tr = TransformerEncoder(
            d_model=self.d_model,
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            dropout=float(dropout),
            cond_dim=self.d_cond,
            causal=False,
            use_checkpoint=False,
        )
        self.out = nn.Linear(self.d_model, 1)

    def forward(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond_vec = self.cond_enc(cond)  # [B,d_cond]
        if self.use_level:
            if "level" not in cond:
                raise ValueError("use_level=True but level missing from cond")
            level = cond["level"]
            if level.dim() == 1:
                level = level.unsqueeze(1)
            cond_vec = cond_vec + self.level_mlp(level)

        t = torch.linspace(0.0, 1.0, self.T, device=cond_vec.device, dtype=cond_vec.dtype)
        pos = _sinusoidal_1d(t, self.pos_dim)
        x = self.pos_proj(pos).unsqueeze(0).expand(cond_vec.shape[0], -1, -1)
        x = x + self.time_embed.to(dtype=x.dtype, device=x.device).unsqueeze(0)
        x = self.tr(x, cond=cond_vec)
        return self.out(x).squeeze(-1)


__all__ = ["VideoKeyframeSelector"]

