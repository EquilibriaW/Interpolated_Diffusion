from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoders import MazeConditionEncoder


class SegmentCostPredictor(nn.Module):
    def __init__(
        self,
        d_cond: int = 128,
        seg_feat_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.0,
        cond_encoder: Optional[nn.Module] = None,
        use_sdf: bool = False,
        use_start_goal: bool = True,
        maze_channels: tuple[int, ...] = (32, 64),
    ) -> None:
        super().__init__()
        if cond_encoder is None:
            cond_encoder = MazeConditionEncoder(
                use_sdf=use_sdf, d_cond=d_cond, use_start_goal=use_start_goal, maze_channels=maze_channels
            )
        self.cond_enc = cond_encoder
        self.d_cond = d_cond
        self.seg_feat_dim = seg_feat_dim

        layers = []
        in_dim = d_cond + seg_feat_dim
        for i in range(max(1, n_layers - 1)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cond: Dict[str, torch.Tensor], seg_feat: torch.Tensor) -> torch.Tensor:
        if cond is None:
            raise ValueError("cond is required for SegmentCostPredictor")
        cond_vec = self.cond_enc(cond)  # [B, d_cond]
        if seg_feat.dim() == 2:
            seg_feat = seg_feat.unsqueeze(0).expand(cond_vec.shape[0], -1, -1)
        if seg_feat.dim() != 3:
            raise ValueError("seg_feat must be [S,F] or [B,S,F]")
        if seg_feat.shape[-1] != self.seg_feat_dim:
            raise ValueError("seg_feat_dim mismatch")
        cond_exp = cond_vec.unsqueeze(1).expand(-1, seg_feat.shape[1], -1)
        x = torch.cat([cond_exp, seg_feat], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out
