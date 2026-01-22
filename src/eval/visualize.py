from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon, Rectangle


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _maze_map_to_occ(maze_map) -> Optional[np.ndarray]:
    if isinstance(maze_map, np.ndarray):
        arr = np.asarray(maze_map)
        if arr.ndim != 2:
            return None
        uniq = np.unique(arr)
        if set(uniq.tolist()).issubset({0, 1}):
            return (arr > 0).astype(np.float32)
        if set(uniq.tolist()).issubset({10, 11, 12}):
            return (arr == 10).astype(np.float32).T
        return (arr > 0).astype(np.float32)
    if isinstance(maze_map, (list, tuple)) and len(maze_map) > 0:
        if isinstance(maze_map[0], (list, tuple, np.ndarray)):
            arr = np.array(maze_map)
            if arr.ndim != 2:
                return None
            uniq = np.unique(arr)
            if set(uniq.tolist()).issubset({0, 1}):
                return (arr > 0).astype(np.float32)
            if set(uniq.tolist()).issubset({10, 11, 12}):
                return (arr == 10).astype(np.float32).T
            return (arr > 0).astype(np.float32)
        if isinstance(maze_map[0], str):
            h = len(maze_map)
            w = len(maze_map[0])
            occ = np.zeros((h, w), dtype=np.float32)
            wall_chars = {"#", "1", "X"}
            for i, row in enumerate(maze_map):
                for j, ch in enumerate(row):
                    if ch in wall_chars:
                        occ[i, j] = 1.0
            return occ
    return None


def _maze_map_to_walls(
    maze_map: Union[np.ndarray, Sequence],
    scale: float,
) -> List[Tuple[float, float, float, float]]:
    if maze_map is None:
        return []
    occ = _maze_map_to_occ(maze_map)
    if occ is None:
        return []
    h, w = occ.shape
    walls = []
    for i in range(h):
        for j in range(w):
            if occ[i, j]:
                walls.append((j * scale, i * scale, scale, scale))
    return walls


def plot_maze2d_trajectories(
    maze_map: Union[np.ndarray, Sequence],
    scale: float,
    trajs: List[np.ndarray],
    labels: List[str],
    colors: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    title: Optional[str] = None,
    footer: Optional[str] = None,
    plot_points: bool = False,
    keypoints: Optional[List[np.ndarray]] = None,
):
    trajs = [_to_numpy(traj) for traj in trajs]
    walls = _maze_map_to_walls(maze_map, scale)
    fig, ax = plt.subplots(figsize=(4, 4))
    for (x, y, w, h) in walls:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="black", edgecolor="black", linewidth=0.0))
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, traj in enumerate(trajs):
        xy = traj[:, :2]
        if plot_points:
            base_color = "0.6" if keypoints is not None and i < len(keypoints) and keypoints[i] is not None else colors[i % len(colors)]
            ax.scatter(xy[:, 0], xy[:, 1], color=base_color, s=6, label=labels[i])
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=colors[i % len(colors)], label=labels[i])
        ax.scatter([xy[0, 0]], [xy[0, 1]], color=colors[i % len(colors)], s=20)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], color=colors[i % len(colors)], s=24, marker="x")
        if keypoints is not None and i < len(keypoints) and keypoints[i] is not None:
            kp = _to_numpy(keypoints[i])
            if kp.size > 0:
                ax.scatter(kp[:, 0], kp[:, 1], color="tab:orange", s=20, edgecolors="black", linewidths=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if bounds is not None:
        (xmin, xmax), (ymin, ymax) = bounds
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    ax.legend(loc="lower right", fontsize=6)
    if title is not None:
        ax.set_title(title, fontsize=8)
    if footer is not None:
        fig.text(0.5, 0.02, footer, ha="center", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_maze2d_geom_walls(
    wall_polys: List[np.ndarray],
    trajs: List[np.ndarray],
    labels: List[str],
    colors: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    title: Optional[str] = None,
    footer: Optional[str] = None,
    plot_points: bool = False,
    keypoints: Optional[List[np.ndarray]] = None,
):
    trajs = [_to_numpy(traj) for traj in trajs]
    fig, ax = plt.subplots(figsize=(4, 4))
    for poly in wall_polys:
        poly_np = _to_numpy(poly)
        ax.add_patch(Polygon(poly_np, closed=True, facecolor="black", edgecolor="black", linewidth=0.0))
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, traj in enumerate(trajs):
        xy = traj[:, :2]
        if plot_points:
            base_color = "0.6" if keypoints is not None and i < len(keypoints) and keypoints[i] is not None else colors[i % len(colors)]
            ax.scatter(xy[:, 0], xy[:, 1], color=base_color, s=6, label=labels[i])
        else:
            ax.plot(xy[:, 0], xy[:, 1], color=colors[i % len(colors)], label=labels[i])
        ax.scatter([xy[0, 0]], [xy[0, 1]], color=colors[i % len(colors)], s=20)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], color=colors[i % len(colors)], s=24, marker="x")
        if keypoints is not None and i < len(keypoints) and keypoints[i] is not None:
            kp = _to_numpy(keypoints[i])
            if kp.size > 0:
                ax.scatter(kp[:, 0], kp[:, 1], color="tab:orange", s=20, edgecolors="black", linewidths=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if bounds is not None:
        (xmin, xmax), (ymin, ymax) = bounds
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    ax.legend(loc="lower right", fontsize=6)
    if title is not None:
        ax.set_title(title, fontsize=8)
    if footer is not None:
        fig.text(0.5, 0.02, footer, ha="center", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_trajectories(
    occ: np.ndarray,
    trajs: List[np.ndarray],
    labels: List[str],
    colors: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    footer: Optional[str] = None,
    plot_points: bool = False,
    keypoints: Optional[List[np.ndarray]] = None,
):
    occ = _to_numpy(occ)
    trajs = [_to_numpy(traj) for traj in trajs]
    h, w = occ.shape
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(occ, cmap="gray_r", origin="upper")
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, traj in enumerate(trajs):
        xy = traj[:, :2]
        x = xy[:, 0] * w
        y = xy[:, 1] * h
        if plot_points:
            base_color = "0.6" if keypoints is not None and i < len(keypoints) and keypoints[i] is not None else colors[i % len(colors)]
            ax.scatter(x, y, color=base_color, s=6, label=labels[i])
        else:
            ax.plot(x, y, color=colors[i % len(colors)], label=labels[i])
        ax.scatter([x[0]], [y[0]], color=colors[i % len(colors)], s=20)
        ax.scatter([x[-1]], [y[-1]], color=colors[i % len(colors)], s=24, marker="x")
        if keypoints is not None and i < len(keypoints) and keypoints[i] is not None:
            kp = _to_numpy(keypoints[i])
            if kp.size > 0:
                ax.scatter(kp[:, 0] * w, kp[:, 1] * h, color="tab:orange", s=20, edgecolors="black", linewidths=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    if title is not None:
        ax.set_title(title, fontsize=8)
    if footer is not None:
        fig.text(0.5, 0.02, footer, ha="center", fontsize=8)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        return fig
