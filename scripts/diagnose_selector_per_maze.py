import argparse
import hashlib
import os

import numpy as np
import torch

from src.data.dataset import PreparedTrajectoryDataset
from src.models.keypoint_selector import KeypointSelector, select_topk_indices


def _hash_occ(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(arr.tobytes())
    return h.hexdigest()


def build_model(ckpt_path: str) -> tuple[KeypointSelector, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {})
    state = payload.get("model", payload)
    use_cond_bias = bool(meta.get("use_cond_bias", False)) or any(k.startswith("cond_bias.") for k in state)
    cond_bias_mode = meta.get("cond_bias_mode", "memory")
    if any(k.startswith("cond_enc.") for k in state):
        cond_bias_mode = "encoder"
    if cond_bias_mode not in {"memory", "encoder"}:
        cond_bias_mode = "memory"
    model = KeypointSelector(
        T=int(meta.get("T", 128)),
        d_model=int(meta.get("d_model", 256)),
        n_heads=int(meta.get("n_heads", 8)),
        d_ff=int(meta.get("d_ff", 512)),
        n_layers=int(meta.get("n_layers", 2)),
        pos_dim=int(meta.get("pos_dim", 64)),
        dropout=float(meta.get("dropout", 0.0)),
        use_sdf=bool(meta.get("use_sdf", False)),
        use_start_goal=bool(meta.get("cond_start_goal", True)),
        use_sg_map=bool(meta.get("use_sg_map", True)),
        use_sg_token=bool(meta.get("use_sg_token", True)),
        use_goal_dist_token=bool(meta.get("use_goal_dist_token", False)),
        use_cond_bias=use_cond_bias,
        cond_bias_mode=str(cond_bias_mode),
        use_level=bool(meta.get("use_level", False)),
        level_mode=str(meta.get("level_mode", "k_norm")),
        sg_map_sigma=float(meta.get("sg_map_sigma", 1.5)),
        maze_channels=tuple(int(x) for x in str(meta.get("maze_channels", "32,64")).split(",")),
    ).eval()
    model.load_state_dict(state)
    return model, meta


def dist(idx: np.ndarray, T: int) -> np.ndarray:
    counts = np.zeros(T, dtype=int)
    for row in idx:
        for t in row:
            counts[t] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--eval_npz", type=str, required=True)
    parser.add_argument("--batch_per_maze", type=int, default=256)
    parser.add_argument("--max_mazes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    model, meta = build_model(args.ckpt)
    ds = PreparedTrajectoryDataset(args.eval_npz, use_sdf=bool(meta.get("use_sdf", False)))

    occ = np.load(args.eval_npz)["occ"]
    if occ.ndim == 2:
        print("occ is shared across dataset; no per-maze grouping possible.")
        return
    if occ.ndim == 3 and occ.shape[0] != len(ds):
        print("occ appears to be shared or mismatched; no per-maze grouping possible.")
        return

    # Group indices by occ hash.
    maze_to_indices = {}
    for i in range(len(ds)):
        key = _hash_occ(occ[i])
        maze_to_indices.setdefault(key, []).append(i)

    maze_keys = sorted(maze_to_indices.keys(), key=lambda k: len(maze_to_indices[k]), reverse=True)
    maze_keys = maze_keys[: max(1, args.max_mazes)]
    T = int(meta.get("T", 128))
    K = int(meta.get("K", 8))

    for mi, key in enumerate(maze_keys):
        ids = maze_to_indices[key]
        B = min(args.batch_per_maze, len(ids))
        chosen = rng.choice(ids, size=B, replace=False)
        cond = {}
        for k in ds[0]["cond"]:
            cond[k] = torch.stack([ds[i]["cond"][k] for i in chosen], dim=0)
        if "kp_mask_levels" in ds[0]["cond"]:
            levels = int(meta.get("levels", ds[0]["cond"]["kp_mask_levels"].shape[0] - 1))
            s_idx = torch.full((B,), levels, dtype=torch.long)
            kp_mask_levels = torch.stack([ds[i]["cond"]["kp_mask_levels"] for i in chosen], dim=0)
            true_mask = kp_mask_levels[torch.arange(B), s_idx]
            if bool(meta.get("use_level", False)):
                if str(meta.get("level_mode", "k_norm")) == "s_norm":
                    level_val = s_idx.float() / float(max(1, levels))
                else:
                    k_frac = float(meta.get("K", true_mask.sum(dim=1).float().mean().item())) / float(
                        max(1, int(meta.get("T", 128)) - 1)
                    )
                    level_val = torch.full((B,), k_frac, dtype=torch.float32)
                cond = dict(cond)
                cond["level"] = level_val.unsqueeze(1)
            true = []
            for b in range(B):
                idx = torch.nonzero(true_mask[b], as_tuple=False).squeeze(-1).cpu().numpy()
                true.append(idx)
            true = np.stack(true, axis=0)
        else:
            true = np.stack([ds[i]["cond"]["kp_idx"].numpy() for i in chosen], axis=0)

        with torch.no_grad():
            logits = model(cond)
            pred = select_topk_indices(logits, K).cpu().numpy()

        mae = np.abs(pred - true).mean()
        overlap = np.mean([len(set(pred[i]) & set(true[i])) / len(true[i]) for i in range(B)])

        dist_true = dist(true, T)
        dist_pred = dist(pred, T)
        top_true = (np.argsort(-dist_true[1:-1])[:10] + 1).tolist()
        top_pred = (np.argsort(-dist_pred[1:-1])[:10] + 1).tolist()

        print(f"maze[{mi}] n={len(ids)} sample={B} mae={mae:.2f} overlap={overlap:.3f}")
        print(f"  top label idx: {top_true}")
        print(f"  top pred  idx: {top_pred}")


if __name__ == "__main__":
    main()
