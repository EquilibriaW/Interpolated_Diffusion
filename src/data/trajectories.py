from typing import List, Tuple

import numpy as np


def grid_path_to_xy(path: List[Tuple[int, int]], h: int, w: int) -> np.ndarray:
    pts = np.zeros((len(path), 2), dtype=np.float32)
    for idx, (i, j) in enumerate(path):
        pts[idx, 0] = (j + 0.5) / w
        pts[idx, 1] = (i + 0.5) / h
    return pts


def resample_polyline(points: np.ndarray, T: int) -> np.ndarray:
    if points.shape[0] == 1:
        return np.repeat(points, T, axis=0)
    seg = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-8:
        return np.repeat(points[:1], T, axis=0)
    samples = np.linspace(0.0, total, T)
    out = np.zeros((T, 2), dtype=np.float32)
    for i, s in enumerate(samples):
        idx = np.searchsorted(cum, s, side="right") - 1
        idx = min(max(idx, 0), len(seg_len) - 1)
        denom = seg_len[idx]
        if denom <= 1e-8:
            out[i] = points[idx]
        else:
            t = (s - cum[idx]) / denom
            out[i] = points[idx] + t * seg[idx]
    return out


def path_to_trajectory(path: List[Tuple[int, int]], h: int, w: int, T: int, with_velocity: bool = False) -> np.ndarray:
    xy = grid_path_to_xy(path, h, w)
    pos = resample_polyline(xy, T)
    if not with_velocity:
        return pos.astype(np.float32)
    v = np.zeros_like(pos)
    dt = 1.0 / float(T)
    v[:-1] = (pos[1:] - pos[:-1]) / dt
    v[-1] = 0.0
    traj = np.concatenate([pos, v], axis=-1)
    return traj.astype(np.float32)
