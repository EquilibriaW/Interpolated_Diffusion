import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.eval.visualize import plot_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--plot_points", type=int, default=1)
    parser.add_argument("--flip_y", type=int, default=1)
    parser.add_argument("--show_start_goal", type=int, default=1)
    args = parser.parse_args()

    data = np.load(args.npz)
    idx = args.index
    n_samples = data["interp"].shape[0]
    if idx < 0 or idx >= n_samples:
        raise IndexError(f"index {idx} out of range (n_samples={n_samples})")
    occ = data["occ"]
    if occ.ndim == 2:
        occ = occ
    elif occ.ndim == 3:
        if occ.shape[0] == n_samples:
            occ = occ[idx]
        else:
            # Try to map via difficulty/maze id if available; fallback to first.
            maze_idx = None
            if "difficulty" in data:
                try:
                    maze_idx = int(data["difficulty"][idx])
                except Exception:
                    maze_idx = None
            if maze_idx is not None and 0 <= maze_idx < occ.shape[0]:
                occ = occ[maze_idx]
            else:
                occ = occ[0]
                print(f"[warn] occ has {occ.shape[0]} mazes; using occ[0] for sample {idx}.")
    else:
        raise ValueError(f"Unexpected occ shape: {occ.shape}")
    start_goal = data["start_goal"][idx]

    interp = data["interp"][idx]
    refined = data["refined"][idx]
    oracle = data["gt"][idx] if "gt" in data else None

    trajs = [interp[:, :2], refined[:, :2]]
    labels = ["interp", "refined"]
    if oracle is not None:
        trajs.append(oracle[:, :2])
        labels.append("gt")

    # Use the same plot function used in sampling for consistency.
    if args.show_start_goal:
        fig = plot_trajectories(
            occ,
            trajs,
            labels,
            out_path=None,
            plot_points=bool(args.plot_points),
            flip_y=bool(args.flip_y),
            keypoints=None,
        )
        ax = fig.axes[0]
        h, w = occ.shape
        scale_w = max(w - 1, 1)
        scale_h = max(h - 1, 1)
        ax.scatter([start_goal[0] * scale_w], [start_goal[1] * scale_h], s=30, color="tab:green", marker="o")
        ax.scatter([start_goal[2] * scale_w], [start_goal[3] * scale_h], s=40, color="tab:red", marker="x")
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
    else:
        plot_trajectories(
            occ,
            trajs,
            labels,
            out_path=args.out,
            plot_points=bool(args.plot_points),
            flip_y=bool(args.flip_y),
            keypoints=None,
        )

    print(f"Wrote {args.out}")
    print(f"start_goal: {start_goal}")
    print(f"interp start {interp[0,:2]} goal {interp[-1,:2]}")
    print(f"refined start {refined[0,:2]} goal {refined[-1,:2]}")


if __name__ == "__main__":
    main()
