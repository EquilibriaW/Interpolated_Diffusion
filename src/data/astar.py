import heapq
from typing import List, Optional, Tuple

import numpy as np


def astar(occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* on 4-connected grid. occ=1 is wall."""
    h, w = occ.shape
    si, sj = start
    gi, gj = goal
    if occ[si, sj] == 1 or occ[gi, gj] == 1:
        return None

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        ci, cj = current
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            if ni < 0 or nj < 0 or ni >= h or nj >= w:
                continue
            if occ[ni, nj] == 1:
                continue
            neighbor = (ni, nj)
            tentative = cost + 1
            if tentative < g_score.get(neighbor, 1e9):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                priority = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, tentative, neighbor))

    return None
