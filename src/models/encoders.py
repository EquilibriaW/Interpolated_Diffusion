from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MazeEncoder(nn.Module):
    def __init__(self, in_channels: int, d_cond: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, d_cond)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
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
    def __init__(self, use_sdf: bool = False, d_cond: int = 128, use_start_goal: bool = True):
        super().__init__()
        in_channels = 2 if use_sdf else 1
        self.use_sdf = use_sdf
        self.use_start_goal = use_start_goal
        self.maze = MazeEncoder(in_channels, d_cond)
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
