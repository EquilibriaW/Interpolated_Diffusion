from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0, cond_dim: Optional[int] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.film1 = nn.Linear(cond_dim, d_model * 2)
            self.film2 = nn.Linear(cond_dim, d_model * 2)
        else:
            self.film1 = None
            self.film2 = None

    def _apply_film(self, x: torch.Tensor, cond: Optional[torch.Tensor], layer: nn.Linear) -> torch.Tensor:
        if cond is None:
            return x
        gamma_beta = layer(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        if self.film1 is not None:
            h = self._apply_film(h, cond, self.film1)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        if self.film2 is not None:
            h = self._apply_film(h, cond, self.film2)
        ff_out = self.ff(h)
        x = x + self.dropout(ff_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,
        causal: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, cond_dim=cond_dim) for _ in range(n_layers)]
        )
        self.causal = causal
        self.use_checkpoint = use_checkpoint

    def _causal_mask(self, T: int, device: torch.device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        attn_mask = None
        if self.causal:
            attn_mask = self._causal_mask(x.shape[1], x.device)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(lambda _x: layer(_x, attn_mask=attn_mask, cond=cond), x)
            else:
                x = layer(x, attn_mask=attn_mask, cond=cond)
        return x
