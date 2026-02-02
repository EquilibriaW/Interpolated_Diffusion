from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MazeConditionEncoder


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(0, half, device=t.device) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.norm1(q)
        attn_out, _ = self.attn(h, kv, kv)
        x = q + self.dropout(attn_out)
        h = self.norm2(x)
        ff_out = self.ff(h)
        return x + self.dropout(ff_out)


class KeypointSelector(nn.Module):
    def __init__(
        self,
        T: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 512,
        n_layers: int = 2,
        pos_dim: int = 64,
        dropout: float = 0.0,
        use_sdf: bool = False,
        use_start_goal: bool = True,
        use_sg_map: bool = True,
        use_sg_token: bool = True,
        use_goal_dist_token: bool = False,
        use_cond_bias: bool = False,
        cond_bias_mode: str = "memory",
        use_level: bool = False,
        level_mode: str = "k_norm",
        sg_map_sigma: float = 1.5,
        maze_channels: tuple[int, ...] = (32, 64),
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.pos_dim = int(pos_dim)
        self.use_sdf = bool(use_sdf)
        self.use_start_goal = bool(use_start_goal)
        self.use_sg_map = bool(use_sg_map)
        self.use_sg_token = bool(use_sg_token)
        self.use_goal_dist_token = bool(use_goal_dist_token)
        self.use_cond_bias = bool(use_cond_bias)
        self.cond_bias_mode = str(cond_bias_mode)
        self.use_level = bool(use_level)
        self.level_mode = str(level_mode)
        self.sg_map_sigma = float(sg_map_sigma)

        in_channels = 1 + (1 if self.use_sdf else 0) + (2 if self.use_sg_map else 0)
        convs = []
        c_in = in_channels
        for c_out in maze_channels:
            convs.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            convs.append(nn.SiLU())
            c_in = c_out
        self.spatial_conv = nn.Sequential(*convs)
        self.spatial_proj = nn.Conv2d(c_in, d_model, kernel_size=1) if c_in != d_model else nn.Identity()

        self.sg_token = None
        if self.use_start_goal and self.use_sg_token:
            self.sg_token = nn.Sequential(nn.Linear(4, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.goal_dist_token = None
        if self.use_goal_dist_token:
            self.goal_dist_token = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        self.time_proj = nn.Linear(pos_dim, d_model)
        self.level_mlp = None
        if self.use_level:
            self.level_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_bias = None
        self.cond_enc = None
        if self.use_cond_bias:
            self.cond_bias = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            if self.cond_bias_mode not in {"memory", "encoder"}:
                raise ValueError(f"cond_bias_mode must be 'memory' or 'encoder', got {self.cond_bias_mode}")
            if self.cond_bias_mode == "encoder":
                self.cond_enc = MazeConditionEncoder(
                    use_sdf=self.use_sdf, d_cond=d_model, use_start_goal=self.use_start_goal, maze_channels=maze_channels
                )
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(max(1, n_layers))]
        )
        self.out = nn.Linear(d_model, 1)

    def _sg_map(self, start_goal: torch.Tensor, H: int, W: int) -> torch.Tensor:
        device = start_goal.device
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        sx = start_goal[:, 0].clamp(0.0, 1.0) * float(W - 1)
        sy = start_goal[:, 1].clamp(0.0, 1.0) * float(H - 1)
        gx = start_goal[:, 2].clamp(0.0, 1.0) * float(W - 1)
        gy = start_goal[:, 3].clamp(0.0, 1.0) * float(H - 1)
        sx = sx.view(-1, 1, 1)
        sy = sy.view(-1, 1, 1)
        gx = gx.view(-1, 1, 1)
        gy = gy.view(-1, 1, 1)

        if self.sg_map_sigma <= 0:
            s_map = torch.zeros((start_goal.shape[0], H, W), device=device)
            g_map = torch.zeros((start_goal.shape[0], H, W), device=device)
            s_xi = sx.round().long().clamp(0, W - 1)
            s_yi = sy.round().long().clamp(0, H - 1)
            g_xi = gx.round().long().clamp(0, W - 1)
            g_yi = gy.round().long().clamp(0, H - 1)
            batch_idx = torch.arange(start_goal.shape[0], device=device)
            s_map[batch_idx, s_yi.squeeze(-1).squeeze(-1), s_xi.squeeze(-1).squeeze(-1)] = 1.0
            g_map[batch_idx, g_yi.squeeze(-1).squeeze(-1), g_xi.squeeze(-1).squeeze(-1)] = 1.0
            return torch.stack([s_map, g_map], dim=1)

        sigma2 = float(self.sg_map_sigma) ** 2
        s_map = torch.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * sigma2))
        g_map = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma2))
        return torch.stack([s_map, g_map], dim=1)

    def forward(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        occ = cond["occ"]
        feats = [occ]
        if self.use_sdf:
            sdf = cond.get("sdf")
            if sdf is None:
                raise ValueError("use_sdf is True but sdf missing from cond")
            feats.append(sdf)
        if self.use_start_goal and self.use_sg_map:
            if "start_goal" not in cond:
                raise ValueError("use_start_goal is True but start_goal missing from cond")
            sg_map = self._sg_map(cond["start_goal"], occ.shape[-2], occ.shape[-1])
            feats.append(sg_map)
        x = torch.cat(feats, dim=1)
        x = self.spatial_conv(x)
        x = self.spatial_proj(x)
        B, C, H, W = x.shape
        spatial = x.flatten(2).transpose(1, 2)
        tokens = [spatial]
        if self.use_start_goal and self.use_sg_token:
            if "start_goal" not in cond:
                raise ValueError("use_start_goal is True but start_goal missing from cond")
            sg_token = self.sg_token(cond["start_goal"]).unsqueeze(1)
            tokens.insert(0, sg_token)
        if self.use_goal_dist_token:
            if "start_goal" not in cond:
                raise ValueError("use_goal_dist_token requires start_goal")
            sg = cond["start_goal"]
            goal_dist = torch.norm(sg[:, :2] - sg[:, 2:], dim=-1, keepdim=True)
            gd_token = self.goal_dist_token(goal_dist).unsqueeze(1)
            tokens.insert(0, gd_token)
        memory = torch.cat(tokens, dim=1)

        t = torch.linspace(0.0, 1.0, self.T, device=memory.device, dtype=memory.dtype)
        time_emb = sinusoidal_time_embedding(t, self.pos_dim)
        q = self.time_proj(time_emb).unsqueeze(0).expand(B, -1, -1)
        if self.use_cond_bias:
            if self.cond_bias_mode == "encoder":
                cond_vec = self.cond_enc(cond)
            else:
                cond_vec = memory.mean(dim=1)
            q = q + self.cond_bias(cond_vec).unsqueeze(1)
        if self.use_level:
            if "level" not in cond:
                raise ValueError("use_level is True but level missing from cond")
            level = cond["level"]
            if level.dim() == 1:
                level = level.unsqueeze(1)
            level_emb = self.level_mlp(level)
            q = q + level_emb.unsqueeze(1)
        for block in self.blocks:
            q = block(q, memory)
        return self.out(q).squeeze(-1)


def select_topk_indices(logits: torch.Tensor, K: int) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("logits must be [B,T]")
    B, T = logits.shape
    if K < 2:
        raise ValueError("K must be >= 2")
    if K > T:
        K = T
    if K == 2:
        idx = torch.zeros((B, 2), device=logits.device, dtype=torch.long)
        idx[:, 1] = T - 1
        return idx
    interior = logits[:, 1:-1]
    topk = torch.topk(interior, K - 2, dim=1).indices + 1
    idx = torch.cat(
        [
            torch.zeros((B, 1), device=logits.device, dtype=torch.long),
            topk,
            torch.full((B, 1), T - 1, device=logits.device, dtype=torch.long),
        ],
        dim=1,
    )
    idx = torch.sort(idx, dim=1).values
    return idx
