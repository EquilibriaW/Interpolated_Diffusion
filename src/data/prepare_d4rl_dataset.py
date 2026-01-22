import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from src.data.dataset import D4RLMazeDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="maze2d-large-v1")
    p.add_argument("--out_dir", type=str, default="outputs/d4rl_prepared")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--hard_fraction", type=float, default=0.5)
    p.add_argument("--hard_num", type=int, default=None)
    p.add_argument("--easy_min_goal_dist", type=float, default=None)
    p.add_argument("--easy_min_path_len", type=float, default=None)
    p.add_argument("--easy_min_tortuosity", type=float, default=None)
    p.add_argument("--easy_min_turns", type=int, default=None)
    p.add_argument("--easy_turn_angle_deg", type=float, default=30.0)
    p.add_argument("--shuffle", type=int, default=1)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--d4rl_flip_y", type=int, default=0)
    p.add_argument("--window_mode", type=str, default="episode", choices=["end", "random", "episode"])
    p.add_argument("--goal_mode", type=str, default="window_end", choices=["env", "window_end"])
    p.add_argument("--max_collision_rate", type=float, default=0.0)
    p.add_argument("--max_resample_tries", type=int, default=200)
    p.add_argument("--min_goal_dist", type=float, default=6.0)
    p.add_argument("--min_path_len", type=float, default=12.0)
    p.add_argument("--min_tortuosity", type=float, default=1.8)
    p.add_argument("--min_turns", type=int, default=6)
    p.add_argument("--turn_angle_deg", type=float, default=30.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "dataset.npz")
    meta_path = os.path.join(args.out_dir, "meta.json")

    n_total = int(args.num_samples)
    if args.hard_num is not None:
        n_hard = int(args.hard_num)
    else:
        n_hard = int(round(n_total * float(args.hard_fraction)))
    n_hard = max(0, min(n_total, n_hard))
    n_easy = n_total - n_hard

    hard_ds = D4RLMazeDataset(
        env_id=args.env_id,
        num_samples=max(1, n_hard),
        T=args.T,
        with_velocity=bool(args.with_velocity),
        use_sdf=bool(args.use_sdf),
        seed=args.seed,
        flip_y=bool(args.d4rl_flip_y),
        swap_xy=False,
        max_collision_rate=args.max_collision_rate,
        max_resample_tries=args.max_resample_tries,
        min_goal_dist=args.min_goal_dist,
        min_path_len=args.min_path_len,
        min_tortuosity=args.min_tortuosity,
        min_turns=args.min_turns,
        turn_angle_deg=args.turn_angle_deg,
        window_mode=args.window_mode,
        goal_mode=args.goal_mode,
    )

    easy_ds = D4RLMazeDataset(
        env_id=args.env_id,
        num_samples=max(1, n_easy),
        T=args.T,
        with_velocity=bool(args.with_velocity),
        use_sdf=bool(args.use_sdf),
        seed=args.seed + 1000003,
        flip_y=bool(args.d4rl_flip_y),
        swap_xy=False,
        max_collision_rate=args.max_collision_rate,
        max_resample_tries=args.max_resample_tries,
        min_goal_dist=args.easy_min_goal_dist,
        min_path_len=args.easy_min_path_len,
        min_tortuosity=args.easy_min_tortuosity,
        min_turns=args.easy_min_turns,
        turn_angle_deg=args.easy_turn_angle_deg,
        window_mode=args.window_mode,
        goal_mode=args.goal_mode,
    )

    first = hard_ds[0] if n_hard > 0 else easy_ds[0]
    x0 = first["x"].numpy()
    data_dim = x0.shape[-1]
    x_all = np.zeros((n_total, args.T, data_dim), dtype=np.float32)
    start_goal_all = np.zeros((n_total, 4), dtype=np.float32)
    difficulty = np.zeros((n_total,), dtype=np.int8)
    occ = first["cond"]["occ"].numpy()
    sdf = first["cond"].get("sdf")
    if sdf is not None:
        sdf = sdf.numpy()

    for i in tqdm(range(n_hard), dynamic_ncols=True, desc="hard"):
        sample = hard_ds[i]
        x_all[i] = sample["x"].numpy()
        start_goal_all[i] = sample["cond"]["start_goal"].numpy()
        difficulty[i] = 1
    for i in tqdm(range(n_easy), dynamic_ncols=True, desc="easy"):
        sample = easy_ds[i]
        j = n_hard + i
        x_all[j] = sample["x"].numpy()
        start_goal_all[j] = sample["cond"]["start_goal"].numpy()
        difficulty[j] = 0

    if bool(args.shuffle):
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n_total)
        x_all = x_all[perm]
        start_goal_all = start_goal_all[perm]
        difficulty = difficulty[perm]

    save_kwargs = {
        "x": x_all,
        "start_goal": start_goal_all,
        "occ": occ,
        "difficulty": difficulty,
    }
    if sdf is not None:
        save_kwargs["sdf"] = sdf
    np.savez_compressed(out_path, **save_kwargs)

    meta = {
        "env_id": args.env_id,
        "num_samples": args.num_samples,
        "num_hard": n_hard,
        "num_easy": n_easy,
        "hard_fraction": float(n_hard) / max(1, n_total),
        "T": args.T,
        "data_dim": int(data_dim),
        "with_velocity": bool(args.with_velocity),
        "use_sdf": bool(args.use_sdf),
        "seed": args.seed,
        "d4rl_flip_y": bool(args.d4rl_flip_y),
        "window_mode": args.window_mode,
        "goal_mode": args.goal_mode,
        "max_collision_rate": args.max_collision_rate,
        "max_resample_tries": args.max_resample_tries,
        "min_goal_dist": args.min_goal_dist,
        "min_path_len": args.min_path_len,
        "min_tortuosity": args.min_tortuosity,
        "min_turns": args.min_turns,
        "turn_angle_deg": args.turn_angle_deg,
        "easy_min_goal_dist": args.easy_min_goal_dist,
        "easy_min_path_len": args.easy_min_path_len,
        "easy_min_tortuosity": args.easy_min_tortuosity,
        "easy_min_turns": args.easy_min_turns,
        "easy_turn_angle_deg": args.easy_turn_angle_deg,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
