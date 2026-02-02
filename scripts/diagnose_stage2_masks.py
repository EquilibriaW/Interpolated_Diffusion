import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch

from src.corruptions.keyframes import (
    _compute_k_schedule,
    build_nested_masks_batch,
    build_nested_masks_from_base,
    build_nested_masks_from_level_logits,
    build_nested_masks_from_logits,
    sample_fixed_k_indices_uniform_batch,
)
from src.data.dataset import PreparedTrajectoryDataset
from src.models.keypoint_selector import KeypointSelector
from src.train.train_interp_levels import build_interp_adjacent_batch


def _load_selector(ckpt_path: str, device: torch.device) -> Tuple[KeypointSelector, Dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    state = payload.get("model", payload)
    use_cond_bias = bool(meta.get("use_cond_bias", False)) or any(k.startswith("cond_bias.") for k in state)
    cond_bias_mode = meta.get("cond_bias_mode", "memory")
    if any(k.startswith("cond_enc.") for k in state):
        cond_bias_mode = "encoder"
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
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, meta


def _build_selector_masks(
    selector: KeypointSelector,
    meta: Dict,
    cond: Dict[str, torch.Tensor],
    T: int,
    K_min: int,
    levels: int,
    k_schedule: str,
    k_geom_gamma: float,
) -> Tuple[torch.Tensor, list]:
    device = next(selector.parameters()).device
    selector_use_level = bool(meta.get("use_level", False))
    selector_level_mode = str(meta.get("level_mode", "k_norm"))
    if selector_use_level:
        k_list = _compute_k_schedule(T, K_min, levels, schedule=k_schedule, geom_gamma=k_geom_gamma)
        logits_levels = []
        for s in range(levels + 1):
            if selector_level_mode == "s_norm":
                level_val = float(s) / float(max(1, levels))
            else:
                level_val = float(k_list[s]) / float(max(1, T - 1))
            cond_sel = dict(cond)
            cond_sel["level"] = torch.full((cond["occ"].shape[0], 1), level_val, device=device)
            with torch.no_grad():
                logits_s = selector(cond_sel)
            logits_levels.append(logits_s)
        logits_levels = torch.stack(logits_levels, dim=1)
        return build_nested_masks_from_level_logits(
            logits_levels,
            K_min,
            levels,
            k_schedule=k_schedule,
            k_geom_gamma=k_geom_gamma,
        )
    with torch.no_grad():
        logits = selector(cond)
    return build_nested_masks_from_logits(
        logits,
        K_min,
        levels,
        k_schedule=k_schedule,
        k_geom_gamma=k_geom_gamma,
    )


def _gap_stats(idx: torch.Tensor) -> Tuple[float, float]:
    gaps = idx[:, 1:] - idx[:, :-1]
    return gaps.float().mean().item(), gaps.float().max(dim=1).values.mean().item()


def _oob_frac(x: torch.Tensor) -> float:
    oob = (x[:, :, :2] < 0.0) | (x[:, :, :2] > 1.0)
    return oob.any(dim=-1).float().mean().item()


def _mean_l2(x: torch.Tensor) -> float:
    return torch.sqrt((x[:, :, :2] ** 2).sum(dim=-1)).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_npz", type=str, required=True)
    parser.add_argument("--selector_ckpt", type=str, default=None)
    parser.add_argument("--baseline", type=str, default="uniform_base", choices=["uniform_base", "random_nested"])
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--K_min", type=int, default=8)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    parser.add_argument("--k_geom_gamma", type=float, default=None)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_sdf", type=int, default=1)
    parser.add_argument("--cond_start_goal", type=int, default=1)
    parser.add_argument("--corrupt_mode", type=str, default="dist", choices=["none", "dist", "gauss"])
    parser.add_argument("--corrupt_sigma_max", type=float, default=0.08)
    parser.add_argument("--corrupt_sigma_min", type=float, default=0.012)
    parser.add_argument("--corrupt_sigma_pow", type=float, default=0.75)
    parser.add_argument("--corrupt_anchor_frac", type=float, default=0.25)
    parser.add_argument("--corrupt_index_jitter_max", type=int, default=0)
    parser.add_argument("--corrupt_index_jitter_prob", type=float, default=0.0)
    parser.add_argument("--corrupt_index_jitter_pow", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rng = np.random.RandomState(args.seed)
    ds = PreparedTrajectoryDataset(args.eval_npz, use_sdf=bool(args.use_sdf))
    B = min(args.batch, len(ds))
    idxs = rng.choice(len(ds), size=B, replace=False)

    cond = {}
    for k in ds[0]["cond"]:
        cond[k] = torch.stack([ds[i]["cond"][k] for i in idxs], dim=0).to(device)
    x0 = torch.stack([ds[i]["x"] for i in idxs], dim=0).to(device)

    methods = []
    if args.selector_ckpt:
        selector, meta = _load_selector(args.selector_ckpt, device)
        methods.append(("selector", meta, selector))
    methods.append((args.baseline, None, None))

    print("method,level,K,mean_gap,max_gap,mean_adj,mean_xs,oob_xs,oob_prev")
    for method, meta, selector in methods:
        if method == "selector":
            masks_levels, idx_levels = _build_selector_masks(
                selector,
                meta,
                cond,
                args.T,
                args.K_min,
                args.levels,
                args.k_schedule,
                args.k_geom_gamma,
            )
        elif method == "uniform_base":
            gen = torch.Generator(device=device).manual_seed(args.seed + 7)
            idx_base, _ = sample_fixed_k_indices_uniform_batch(
                B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True
            )
            masks_levels, idx_levels = build_nested_masks_from_base(
                idx_base,
                args.T,
                args.levels,
                generator=gen,
                device=device,
                k_schedule=args.k_schedule,
                k_geom_gamma=args.k_geom_gamma,
            )
        else:
            gen = torch.Generator(device=device).manual_seed(args.seed + 9)
            masks_levels, idx_levels = build_nested_masks_batch(
                B,
                args.T,
                args.K_min,
                args.levels,
                generator=gen,
                device=device,
                k_schedule=args.k_schedule,
                k_geom_gamma=args.k_geom_gamma,
            )

        for s in range(1, args.levels + 1):
            s_idx = torch.full((B,), s, device=device, dtype=torch.long)
            gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + s)
            x_s, x_prev, _, _, _, _, _ = build_interp_adjacent_batch(
                x0,
                args.K_min,
                args.levels,
                gen,
                recompute_velocity=False,
                x0_override=None,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma_max=args.corrupt_sigma_max,
                corrupt_sigma_min=args.corrupt_sigma_min,
                corrupt_sigma_pow=args.corrupt_sigma_pow,
                corrupt_anchor_frac=args.corrupt_anchor_frac,
                corrupt_index_jitter_max=args.corrupt_index_jitter_max,
                corrupt_index_jitter_prob=args.corrupt_index_jitter_prob,
                corrupt_index_jitter_pow=args.corrupt_index_jitter_pow,
                clamp_endpoints=True,
            )
            idx = idx_levels[s]
            mean_gap, max_gap = _gap_stats(idx)
            mean_adj = _mean_l2(x_prev - x_s)
            mean_xs = _mean_l2(x_s - x0)
            oob_xs = _oob_frac(x_s)
            oob_prev = _oob_frac(x_prev)
            print(
                f"{method},{s},{idx.shape[1]},{mean_gap:.2f},{max_gap:.2f},"
                f"{mean_adj:.4f},{mean_xs:.4f},{oob_xs:.4f},{oob_prev:.4f}"
            )


if __name__ == "__main__":
    main()
