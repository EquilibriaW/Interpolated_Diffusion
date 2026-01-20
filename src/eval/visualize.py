from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def plot_trajectories(
    occ: np.ndarray,
    trajs: List[np.ndarray],
    labels: List[str],
    colors: Optional[List[str]] = None,
    out_path: Optional[str] = None,
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
        ax.plot(x, y, color=colors[i % len(colors)], label=labels[i])
        ax.scatter([x[0]], [y[0]], color=colors[i % len(colors)], s=12)
        ax.scatter([x[-1]], [y[-1]], color=colors[i % len(colors)], s=12, marker="x")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        return fig
