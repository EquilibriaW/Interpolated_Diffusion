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
from src.models.denoiser_interp_levels import InterpLevelDenoiser
from src.models.keypoint_selector import KeypointSelector
from src.train.train_interp_levels import (
    _anneal_conf,
    _build_anchor_conf,
    build_interp_adjacent_batch,
    build_interp_level_batch,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_npz", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--selector_ckpt", type=str, default=None)
    parser.add_argument("--baseline", type=str, default="uniform_base", choices=["uniform_base", "random_nested"])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rng = np.random.RandomState(args.seed)

    payload = torch.load(args.stage2_ckpt, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    state = payload.get("model", payload)

    T = int(meta.get("T", 128))
    K_min = int(meta.get("K_min", 8))
    levels = int(meta.get("levels", 8))
    k_schedule = str(meta.get("k_schedule", "geom"))
    k_geom_gamma = meta.get("k_geom_gamma", None)
    stage2_mode = str(meta.get("stage2_mode", "adj"))
    anchor_conf = bool(meta.get("anchor_conf", True))
    anchor_conf_teacher = float(meta.get("anchor_conf_teacher", 0.95))
    anchor_conf_student = float(meta.get("anchor_conf_student", 0.5))
    anchor_conf_endpoints = float(meta.get("anchor_conf_endpoints", 1.0))
    anchor_conf_missing = float(meta.get("anchor_conf_missing", 0.0))
    anchor_conf_anneal = bool(meta.get("anchor_conf_anneal", True))
    anchor_conf_anneal_mode = str(meta.get("anchor_conf_anneal_mode", "linear"))

    use_sdf = bool(meta.get("use_sdf", False))
    cond_start_goal = bool(meta.get("cond_start_goal", True))
    maze_channels = meta.get("s2_maze_channels", meta.get("maze_channels", "32,64"))
    d_model = int(meta.get("s2_d_model", meta.get("d_model", 256)))
    n_layers = int(meta.get("s2_n_layers", meta.get("n_layers", 8)))
    n_heads = int(meta.get("s2_n_heads", meta.get("n_heads", 8)))
    d_ff = int(meta.get("s2_d_ff", meta.get("d_ff", 1024)))
    d_cond = int(meta.get("s2_d_cond", meta.get("d_cond", 128)))
    mask_channels = int(meta.get("mask_channels", 3 if anchor_conf else 2))

    model = InterpLevelDenoiser(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=float(meta.get("dropout", 0.0)),
        d_cond=d_cond,
        use_sdf=use_sdf,
        use_start_goal=cond_start_goal,
        data_dim=2,
        max_levels=levels,
        mask_channels=mask_channels,
        maze_channels=tuple(int(x) for x in str(maze_channels).split(",")),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    ds = PreparedTrajectoryDataset(args.eval_npz, use_sdf=use_sdf)
    B = min(args.batch, len(ds))
    idxs = rng.choice(len(ds), size=B, replace=False)
    cond = {}
    for k in ds[0]["cond"]:
        cond[k] = torch.stack([ds[i]["cond"][k] for i in idxs], dim=0).to(device)
    x0 = torch.stack([ds[i]["x"] for i in idxs], dim=0).to(device)

    methods = []
    if args.selector_ckpt:
        selector, sel_meta = _load_selector(args.selector_ckpt, device)
        methods.append(("selector", sel_meta, selector))
    methods.append((args.baseline, None, None))

    print("method,level,K,mean_gap,max_gap,loss")
    for method, sel_meta, selector in methods:
        if method == "selector":
            masks_levels, idx_levels = _build_selector_masks(
                selector, sel_meta, cond, T, K_min, levels, k_schedule, k_geom_gamma
            )
        elif method == "uniform_base":
            gen = torch.Generator(device=device).manual_seed(args.seed + 7)
            idx_base, _ = sample_fixed_k_indices_uniform_batch(
                B, T, K_min, generator=gen, device=device, ensure_endpoints=True
            )
            masks_levels, idx_levels = build_nested_masks_from_base(
                idx_base, T, levels, generator=gen, device=device, k_schedule=k_schedule, k_geom_gamma=k_geom_gamma
            )
        else:
            gen = torch.Generator(device=device).manual_seed(args.seed + 9)
            masks_levels, idx_levels = build_nested_masks_batch(
                B, T, K_min, levels, generator=gen, device=device, k_schedule=k_schedule, k_geom_gamma=k_geom_gamma
            )

        for s in range(1, levels + 1):
            s_idx = torch.full((B,), s, device=device, dtype=torch.long)
            gen = torch.Generator(device=device).manual_seed(args.seed + 1000 + s)
            if stage2_mode == "adj":
                x_s, x_prev, mask_s, mask_prev, _, _, _ = build_interp_adjacent_batch(
                    x0,
                    K_min,
                    levels,
                    gen,
                    recompute_velocity=False,
                    masks_levels=masks_levels,
                    idx_levels=idx_levels,
                    s_idx=s_idx,
                    corrupt_mode=meta.get("corrupt_mode", "dist"),
                    corrupt_sigma_max=float(meta.get("corrupt_sigma_max", 0.08)),
                    corrupt_sigma_min=float(meta.get("corrupt_sigma_min", 0.012)),
                    corrupt_sigma_pow=float(meta.get("corrupt_sigma_pow", 0.75)),
                    corrupt_anchor_frac=float(meta.get("corrupt_anchor_frac", 0.25)),
                    corrupt_index_jitter_max=int(meta.get("corrupt_index_jitter_max", 0)),
                    corrupt_index_jitter_prob=float(meta.get("corrupt_index_jitter_prob", 0.0)),
                    corrupt_index_jitter_pow=float(meta.get("corrupt_index_jitter_pow", 1.0)),
                    clamp_endpoints=True,
                )
                target = x_prev - x_s
                if anchor_conf:
                    conf_s = _build_anchor_conf(
                        mask_s,
                        None,
                        anchor_conf_teacher,
                        anchor_conf_student,
                        anchor_conf_endpoints,
                        anchor_conf_missing,
                        True,
                    )
                    conf_prev = _build_anchor_conf(
                        mask_prev,
                        None,
                        anchor_conf_teacher,
                        anchor_conf_student,
                        anchor_conf_endpoints,
                        anchor_conf_missing,
                        True,
                    )
                    if anchor_conf_anneal:
                        conf_prev = _anneal_conf(conf_prev, s_idx - 1, levels, anchor_conf_anneal_mode)
                    mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
                    weight_mask = conf_prev
                else:
                    mask_in = torch.stack([mask_s, mask_prev], dim=-1)
                    weight_mask = mask_prev
            else:
                x_s, mask_s, _, _, _ = build_interp_level_batch(
                    x0,
                    K_min,
                    levels,
                    gen,
                    recompute_velocity=False,
                    masks_levels=masks_levels,
                    idx_levels=idx_levels,
                    s_idx=s_idx,
                    corrupt_mode=meta.get("corrupt_mode", "dist"),
                    corrupt_sigma_max=float(meta.get("corrupt_sigma_max", 0.08)),
                    corrupt_sigma_min=float(meta.get("corrupt_sigma_min", 0.012)),
                    corrupt_sigma_pow=float(meta.get("corrupt_sigma_pow", 0.75)),
                    corrupt_anchor_frac=float(meta.get("corrupt_anchor_frac", 0.25)),
                    corrupt_index_jitter_max=int(meta.get("corrupt_index_jitter_max", 0)),
                    corrupt_index_jitter_prob=float(meta.get("corrupt_index_jitter_prob", 0.0)),
                    corrupt_index_jitter_pow=float(meta.get("corrupt_index_jitter_pow", 1.0)),
                    clamp_endpoints=True,
                )
                target = x0 - x_s
                if anchor_conf:
                    conf_s = _build_anchor_conf(
                        mask_s,
                        None,
                        anchor_conf_teacher,
                        anchor_conf_student,
                        anchor_conf_endpoints,
                        anchor_conf_missing,
                        True,
                    )
                    if anchor_conf_anneal:
                        conf_s = _anneal_conf(conf_s, s_idx, levels, anchor_conf_anneal_mode)
                    mask_in = torch.stack([mask_s.float(), conf_s], dim=-1)
                    weight_mask = conf_s
                else:
                    mask_in = mask_s
                    weight_mask = mask_s

            with torch.no_grad():
                delta_hat = model(x_s, s_idx, mask_in, cond)
            diff = (delta_hat - target) ** 2
            diff = diff.sum(dim=-1)
            loss = (diff * weight_mask).sum() / (weight_mask.sum() * x0.shape[-1] + 1e-8)

            mean_gap, max_gap = _gap_stats(idx_levels[s])
            print(f"{method},{s},{idx_levels[s].shape[1]},{mean_gap:.2f},{max_gap:.2f},{loss.item():.6f}")


if __name__ == "__main__":
    main()
