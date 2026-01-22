from typing import Optional, Tuple

import numpy as np
import torch

from .astar import astar


def _boundary_walls(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> None:
    h, w = occ.shape
    occ[0, :] = 1
    occ[h - 1, :] = 1
    occ[:, 0] = 1
    occ[:, w - 1] = 1
    occ[start] = 0
    occ[goal] = 0


def generate_maze(
    rng: np.random.RandomState,
    h: int = 21,
    w: int = 21,
    p_wall: float = 0.2,
    min_l1: Optional[int] = None,
    max_tries: int = 100,
):
    """Generate a random maze with a valid A* path."""
    min_l1 = min_l1 or (h // 2)
    for _ in range(max_tries):
        occ = (rng.rand(h, w) < p_wall).astype(np.int32)
        # Sample start/goal on free cells and far apart.
        free = np.argwhere(occ == 0)
        if len(free) < 2:
            continue
        start = tuple(free[rng.randint(0, len(free))])
        goal = tuple(free[rng.randint(0, len(free))])
        if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < min_l1:
            continue
        _boundary_walls(occ, start, goal)
        path = astar(occ, start, goal)
        if path is None:
            continue
        return occ, start, goal, path
    raise RuntimeError("Failed to generate a valid maze with path")


def sdf_from_occupancy(occ: np.ndarray, signed: bool = True) -> np.ndarray:
    """Compute grid SDF via torch.cdist (L1 distance)."""
    occ_t = torch.from_numpy(occ.astype(np.float32))
    h, w = occ.shape
    grid_i = torch.arange(h, device=occ_t.device)
    grid_j = torch.arange(w, device=occ_t.device)
    coords = torch.stack(torch.meshgrid(grid_i, grid_j, indexing="ij"), dim=-1).reshape(-1, 2).float()
    wall_mask = occ_t.reshape(-1) > 0.5
    if wall_mask.sum() == 0:
        dist = torch.zeros((h, w), dtype=torch.float32)
    else:
        wall_coords = coords[wall_mask]
        dist = torch.cdist(coords, wall_coords, p=1).min(dim=1).values.view(h, w)
    if signed:
        dist = dist * (1.0 - 2.0 * occ_t)
    return dist.cpu().numpy()
