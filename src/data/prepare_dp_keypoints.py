import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from src.models.segment_cost import SegmentCostPredictor
from src.selection.epiplexity_dp import (
    build_cost_matrix_from_segments,
    build_cost_matrix_from_segments_batch,
    build_kp_feat,
    build_kp_feat_batch,
    build_segment_precompute,
    build_segment_features,
    build_snr_weights,
    compute_segment_costs_batch,
    dp_select_indices,
    dp_select_indices_batch,
    sample_timesteps_log_snr,
)
from src.corruptions.keyframes import _compute_k_schedule


def _parse_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_npz", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--snr_min", type=float, default=0.1)
    p.add_argument("--snr_max", type=float, default=10.0)
    p.add_argument("--snr_gamma", type=float, default=1.0)
    p.add_argument("--t_steps", type=int, default=16)
    p.add_argument("--segment_cost_samples", type=int, default=16)
    p.add_argument("--subset_frac", type=float, default=0.1)
    p.add_argument("--subset_num", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--store_kp_feat", type=int, default=1)
    p.add_argument("--store_kp_mask_levels", type=int, default=0)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--per_sample", type=int, default=0)
    p.add_argument("--kp_idx_from", type=str, default=None)
    p.add_argument("--cost_source", type=str, default="gt", choices=["gt", "dphi"])
    p.add_argument("--dphi_ckpt", type=str, default=None)
    args = p.parse_args()

    in_dir = os.path.dirname(args.in_npz)
    src_meta_path = os.path.join(in_dir, "meta.json")
    src_meta = None
    if os.path.exists(src_meta_path):
        try:
            with open(src_meta_path, "r", encoding="utf-8") as f:
                src_meta = json.load(f)
        except Exception:
            src_meta = None

    data = np.load(args.in_npz)
    x = data["x"].astype(np.float32)
    n, T, D = x.shape
    idx = None
    kp_feat_src = None
    kp_feat = None
    kp_mask_levels = None
    meta = {
        "source": args.in_npz,
        "K": args.K,
        "schedule": args.schedule,
        "N_train": args.N_train,
        "snr_min": args.snr_min,
        "snr_max": args.snr_max,
        "snr_gamma": args.snr_gamma,
        "t_steps": int(args.t_steps),
        "segment_cost_samples": int(args.segment_cost_samples),
        "subset_frac": float(args.subset_frac),
        "subset_num": int(args.subset_num),
        "seed": int(args.seed),
        "per_sample": bool(args.per_sample),
        "kp_idx_from": args.kp_idx_from,
        "store_kp_mask_levels": bool(args.store_kp_mask_levels),
        "levels": int(args.levels),
        "k_schedule": str(args.k_schedule),
        "k_geom_gamma": None if args.k_geom_gamma is None else float(args.k_geom_gamma),
    }
    if src_meta is not None and "d4rl_flip_y" in src_meta:
        meta["d4rl_flip_y"] = bool(src_meta.get("d4rl_flip_y"))
    if args.kp_idx_from:
        src = np.load(args.kp_idx_from)
        if "kp_idx" not in src:
            raise ValueError(f"kp_idx_from missing kp_idx: {args.kp_idx_from}")
        idx = src["kp_idx"]
        if idx.ndim == 2:
            idx = idx[0]
        if idx.ndim != 1:
            raise ValueError("kp_idx_from must contain [K] or [N,K] kp_idx")
        if idx.shape[0] != args.K:
            raise ValueError(f"kp_idx_from K mismatch: expected {args.K}, got {idx.shape[0]}")
        if "kp_feat" in src:
            kp_feat_src = src["kp_feat"]
            if kp_feat_src.ndim == 3:
                kp_feat_src = kp_feat_src[0]
    else:
        device = _parse_device(args.device)
        precomp = build_segment_precompute(T, args.segment_cost_samples, device)
        if args.cost_source == "gt":
            snr, weights = build_snr_weights(args.schedule, args.N_train, args.snr_min, args.snr_max, args.snr_gamma)
            t_idx = sample_timesteps_log_snr(snr, args.t_steps)
            weight_scale = float(weights[t_idx].sum().item())
            meta["t_idx"] = t_idx.cpu().numpy().tolist()
            meta["weight_scale"] = weight_scale
        else:
            weight_scale = 1.0

        dphi = None
        dphi_meta = None
        seg_feat = None
        if args.cost_source == "dphi":
            if args.dphi_ckpt is None:
                raise ValueError("cost_source=dphi requires --dphi_ckpt")
            payload = torch.load(args.dphi_ckpt, map_location="cpu")
            dphi_meta = payload.get("meta", {})
            if dphi_meta.get("stage") != "segment_cost":
                raise ValueError("dphi_ckpt does not appear to be a segment_cost checkpoint")
            maze_channels = tuple(int(p.strip()) for p in str(dphi_meta.get("maze_channels", "32,64")).split(",") if p.strip())
            dphi = SegmentCostPredictor(
                d_cond=int(dphi_meta.get("d_cond", 128)),
                seg_feat_dim=int(dphi_meta.get("seg_feat_dim", 3)),
                hidden_dim=int(dphi_meta.get("hidden_dim", 256)),
                n_layers=int(dphi_meta.get("n_layers", 3)),
                dropout=float(dphi_meta.get("dropout", 0.0)),
                use_sdf=bool(dphi_meta.get("use_sdf", False)),
                use_start_goal=bool(dphi_meta.get("cond_start_goal", True)),
                maze_channels=maze_channels,
            ).to(device)
            dphi.load_state_dict(payload.get("model", payload))
            dphi.eval()
            seg_feat = build_segment_features(T, precomp.seg_i, precomp.seg_j).to(device)
            meta["dphi_ckpt"] = args.dphi_ckpt
            meta["dphi_meta"] = dphi_meta
        meta["cost_source"] = args.cost_source

        if bool(args.per_sample):
            idx_list = []
            kp_feat_list = []
            kp_mask_levels_list = []
            k_list = _compute_k_schedule(
                T, args.K, args.levels, schedule=args.k_schedule, geom_gamma=args.k_geom_gamma
            )
            for start in range(0, n, args.batch):
                end = min(n, start + args.batch)
                xb = torch.from_numpy(x[start:end]).to(device)
                if args.cost_source == "gt":
                    cost_seg = compute_segment_costs_batch(xb[:, :, :2], precomp, weight_scale)
                else:
                    # Build cond for Dphi.
                    occ = data["occ"][start:end].astype(np.float32)
                    if occ.ndim == 2:
                        occ = occ[None, ...]
                    if occ.ndim == 3:
                        occ = occ[:, None, ...]
                    cond = {
                        "occ": torch.from_numpy(occ).to(device),
                        "start_goal": torch.from_numpy(data["start_goal"][start:end].astype(np.float32)).to(device),
                    }
                    if dphi_meta and dphi_meta.get("use_sdf", False) and "sdf" in data:
                        sdf = data["sdf"][start:end].astype(np.float32)
                        if sdf.ndim == 2:
                            sdf = sdf[None, ...]
                        if sdf.ndim == 3:
                            sdf = sdf[:, None, ...]
                        cond["sdf"] = torch.from_numpy(sdf).to(device)
                    with torch.no_grad():
                        cost_seg = dphi(cond, seg_feat)
                        if dphi_meta and dphi_meta.get("normalize_targets", False):
                            mean = float(dphi_meta.get("target_mean", 0.0))
                            std = float(dphi_meta.get("target_std", 1.0))
                            cost_seg = cost_seg * std + mean
                C = build_cost_matrix_from_segments_batch(cost_seg, precomp, T)
                idx_batch = dp_select_indices_batch(C, args.K).cpu().numpy()
                idx_list.append(idx_batch)
                if bool(args.store_kp_feat):
                    feat_batch = build_kp_feat_batch(torch.from_numpy(idx_batch), T).cpu().numpy()
                    kp_feat_list.append(feat_batch)
                if bool(args.store_kp_mask_levels):
                    masks_levels = torch.zeros(
                        (idx_batch.shape[0], args.levels + 1, T), dtype=torch.bool, device=device
                    )
                    for s, K_s in enumerate(k_list):
                        idx_s = dp_select_indices_batch(C, int(K_s))
                        mask_s = torch.zeros((idx_s.shape[0], T), dtype=torch.bool, device=device)
                        mask_s.scatter_(1, idx_s, True)
                        masks_levels[:, s] = mask_s
                    kp_mask_levels_list.append(masks_levels.cpu().numpy())
            idx = np.concatenate(idx_list, axis=0)
            if kp_feat_list:
                kp_feat = np.concatenate(kp_feat_list, axis=0)
            if kp_mask_levels_list:
                kp_mask_levels = np.concatenate(kp_mask_levels_list, axis=0)
        else:
            subset_n = int(round(float(args.subset_frac) * n)) if args.subset_num <= 0 else int(args.subset_num)
            subset_n = max(1, min(n, subset_n))
            meta["subset_num"] = int(subset_n)

            rng = np.random.RandomState(args.seed)
            subset_idx = rng.choice(n, size=subset_n, replace=False)
            cost_sum = torch.zeros((precomp.seg_i.shape[0],), device=device)
            count = 0

            for start in range(0, subset_n, args.batch):
                end = min(subset_n, start + args.batch)
                batch_idx = subset_idx[start:end]
                if args.cost_source == "gt":
                    xb = torch.from_numpy(x[batch_idx]).to(device)
                    x_pos = xb[:, :, :2]
                    cost_seg = compute_segment_costs_batch(x_pos, precomp, weight_scale)
                else:
                    occ = data["occ"][batch_idx].astype(np.float32)
                    if occ.ndim == 2:
                        occ = occ[None, ...]
                    if occ.ndim == 3:
                        occ = occ[:, None, ...]
                    cond = {
                        "occ": torch.from_numpy(occ).to(device),
                        "start_goal": torch.from_numpy(data["start_goal"][batch_idx].astype(np.float32)).to(device),
                    }
                    if dphi_meta and dphi_meta.get("use_sdf", False) and "sdf" in data:
                        sdf = data["sdf"][batch_idx].astype(np.float32)
                        if sdf.ndim == 2:
                            sdf = sdf[None, ...]
                        if sdf.ndim == 3:
                            sdf = sdf[:, None, ...]
                        cond["sdf"] = torch.from_numpy(sdf).to(device)
                    with torch.no_grad():
                        cost_seg = dphi(cond, seg_feat)
                        if dphi_meta and dphi_meta.get("normalize_targets", False):
                            mean = float(dphi_meta.get("target_mean", 0.0))
                            std = float(dphi_meta.get("target_std", 1.0))
                            cost_seg = cost_seg * std + mean
                cost_sum += cost_seg.sum(dim=0)
                count += cost_seg.shape[0]

            cost_avg = cost_sum / max(1, count)
            C = build_cost_matrix_from_segments(cost_avg, precomp, T).cpu()
            idx = dp_select_indices(C, args.K).cpu().numpy()
            if bool(args.store_kp_mask_levels):
                k_list = _compute_k_schedule(
                    T, args.K, args.levels, schedule=args.k_schedule, geom_gamma=args.k_geom_gamma
                )
                masks_levels = torch.zeros((args.levels + 1, T), dtype=torch.bool)
                for s, K_s in enumerate(k_list):
                    idx_s = dp_select_indices(C, int(K_s))
                    mask_s = torch.zeros((T,), dtype=torch.bool)
                    mask_s.scatter_(0, idx_s, True)
                    masks_levels[s] = mask_s
                kp_mask_levels = masks_levels.cpu().numpy()

    if idx.ndim == 1:
        kp_idx = np.repeat(idx[None, :], n, axis=0).astype(np.int64)
    else:
        kp_idx = idx.astype(np.int64)
    out = {k: data[k] for k in data.files}
    out["kp_idx"] = kp_idx
    if bool(args.store_kp_feat):
        if kp_feat is not None:
            feat = kp_feat.astype(np.float32)
        elif kp_feat_src is not None:
            feat = kp_feat_src.astype(np.float32)
        else:
            feat = build_kp_feat(torch.from_numpy(idx), T).cpu().numpy().astype(np.float32)
        if feat.ndim == 2:
            out["kp_feat"] = np.repeat(feat[None, ...], n, axis=0)
        else:
            out["kp_feat"] = feat
    if bool(args.store_kp_mask_levels):
        if kp_mask_levels is None:
            raise ValueError("store_kp_mask_levels requested but kp_mask_levels not computed")
        out["kp_mask_levels"] = kp_mask_levels

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, **out)

    meta_path = os.path.splitext(args.out_npz)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if src_meta is not None:
        out_meta_path = os.path.join(os.path.dirname(args.out_npz), "meta.json")
        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(src_meta, f, indent=2)

    print(f"Wrote {args.out_npz}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
