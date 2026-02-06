from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MazeEncoder(nn.Module):
    def __init__(self, in_channels: int, d_cond: int = 128, channels: tuple[int, ...] = (32, 64)):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must be non-empty")
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            layers.append(nn.SiLU())
            c_in = c_out
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], d_cond)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class StartGoalEncoder(nn.Module):
    def __init__(self, d_cond: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

    def forward(self, start_goal: torch.Tensor) -> torch.Tensor:
        return self.mlp(start_goal)


class MazeConditionEncoder(nn.Module):
    def __init__(
        self,
        use_sdf: bool = False,
        d_cond: int = 128,
        use_start_goal: bool = True,
        maze_channels: tuple[int, ...] = (32, 64),
    ):
        super().__init__()
        in_channels = 2 if use_sdf else 1
        self.use_sdf = use_sdf
        self.use_start_goal = use_start_goal
        self.maze = MazeEncoder(in_channels, d_cond, channels=maze_channels)
        self.sg = StartGoalEncoder(d_cond) if use_start_goal else None

    def forward(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        occ = cond["occ"]
        if self.use_sdf:
            sdf = cond.get("sdf")
            if sdf is None:
                raise ValueError("use_sdf is True but sdf missing from cond")
            x = torch.cat([occ, sdf], dim=1)
        else:
            x = occ
        maze_emb = self.maze(x)
        if self.use_start_goal:
            if "start_goal" not in cond:
                raise ValueError("use_start_goal is True but start_goal missing from cond")
            sg_emb = self.sg(cond["start_goal"])
            return maze_emb + sg_emb
        return maze_emb


class TextConditionEncoder(nn.Module):
    def __init__(self, text_dim: int, d_cond: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

    def forward(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        text = cond.get("text_embed")
        if text is None:
            raise ValueError("text_embed missing from cond")
        # Wan synth stores text embeddings as sequences [B, L, D]. D_phi / selectors expect a single
        # conditioning vector, so we pool over non-batch, non-feature dims.
        if text.dim() > 2:
            pool_dims = tuple(range(1, text.dim() - 1))
            text = text.mean(dim=pool_dims)
        # Wan synth stores text embeds in bf16; keep encoder in fp32 by default and cast inputs to match.
        param_dtype = next(self.proj.parameters()).dtype
        if text.dtype != param_dtype:
            text = text.to(dtype=param_dtype)
        return self.proj(text)
