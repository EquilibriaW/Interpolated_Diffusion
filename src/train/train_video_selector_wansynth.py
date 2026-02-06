import argparse
import math
import os
import sys
import time
from typing import Dict, Tuple

import torch
from tqdm import tqdm

from src.corruptions.keyframes import _compute_k_schedule
from src.data.wan_synth import create_wan_synth_dataloader
from src.models.encoders import TextConditionEncoder
from src.models.segment_cost import SegmentCostPredictor
from src.models.video_selector import VideoKeyframeSelector
from src.selection.epiplexity_dp import (
    build_cost_matrix_from_segments_batch,
    build_segment_features,
    build_segment_precompute,
    build_snr_weights,
    dp_select_indices_batch,
    sample_timesteps_log_snr,
)
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_device
from src.utils.logging import create_writer
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train_pattern", type=str, required=True)
    p.add_argument("--val_pattern", type=str, default="")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)

    # D_phi checkpoint (for DP labels).
    p.add_argument("--dphi_ckpt", type=str, required=True)

    # Cost integration over noise steps (defaults to match D_phi meta if present).
    p.add_argument("--schedule", type=str, default="")
    p.add_argument("--N_train", type=int, default=0)
    p.add_argument("--snr_min", type=float, default=0.0)
    p.add_argument("--snr_max", type=float, default=0.0)
    p.add_argument("--snr_gamma", type=float, default=0.0)
    p.add_argument("--t_steps", type=int, default=0)

    # Selector model.
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_cond", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--pos_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_level", type=int, default=1)
    p.add_argument("--level_mode", type=str, default="k_norm", choices=["k_norm", "s_norm"])

    # Training.
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--sel_kl_weight", type=float, default=0.02)
    p.add_argument("--sel_tau_start", type=float, default=1.0)
    p.add_argument("--sel_tau_end", type=float, default=0.3)
    p.add_argument("--sel_tau_frac", type=float, default=0.8)
    p.add_argument("--sel_tau_anneal", type=str, default="cosine", choices=["none", "linear", "cosine"])

    # Logging.
    p.add_argument("--log_dir", type=str, default="runs/video_selector_wansynth")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/video_selector_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--resume", type=str, default="")

    # Loader perf.
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=200)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--resampled", type=int, default=1)
    return p


def _anneal_tau(step: int, total_steps: int, start: float, end: float, frac: float, mode: str) -> float:
    if mode == "none":
        return float(start)
    horizon = max(1, int(total_steps * max(0.0, min(1.0, frac))))
    t = min(step / float(horizon), 1.0)
    if mode == "linear":
        return float(start + (end - start) * t)
    if mode == "cosine":
        return float(end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t)))
    return float(start)


@torch.no_grad()
def _integrated_cost_matrix(
    dphi: torch.nn.Module,
    cond: Dict[str, torch.Tensor],
    seg_feat_base: torch.Tensor,
    precomp,
    *,
    t_idx: torch.Tensor,
    t_weights: torch.Tensor,
    N_train: int,
) -> torch.Tensor:
    # seg_feat_base: [S,3] for all (i,j).
    B = cond["text_embed"].shape[0]
    S = seg_feat_base.shape[0]
    acc = None
    for k in range(int(t_idx.numel())):
        t = int(t_idx[k].item())
        w = float(t_weights[k].item())
        t_norm = float(t) / float(max(1, N_train - 1))
        t_col = torch.full((S, 1), t_norm, device=seg_feat_base.device, dtype=seg_feat_base.dtype)
        seg_feat = torch.cat([seg_feat_base, t_col], dim=-1)  # [S,4]
        pred = dphi(cond, seg_feat).float()  # [B,S]
        if acc is None:
            acc = pred * w
        else:
            acc = acc + pred * w
    if acc is None:
        raise RuntimeError("empty t_idx")
    return build_cost_matrix_from_segments_batch(acc, precomp, int(precomp.seg_id.shape[0]))


@torch.no_grad()
def _eval(
    selector: torch.nn.Module,
    dphi: torch.nn.Module,
    seg_feat_base: torch.Tensor,
    precomp,
    args,
    *,
    t_idx: torch.Tensor,
    t_weights: torch.Tensor,
    N_train: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    if not args.val_pattern:
        return float("nan"), float("nan")
    selector_was_training = selector.training
    selector.eval()
    loader = create_wan_synth_dataloader(
        args.val_pattern,
        batch_size=args.batch,
        num_workers=max(0, int(args.num_workers)),
        shuffle_buffer=max(1, int(args.shuffle_buffer)),
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=False,
        pin_memory=False,
        shuffle=False,
        shardshuffle=False,
        keep_text_embed=True,
        keep_text=False,
        resampled=False,
        seed=args.seed + 999,
    )
    it = iter(loader)

    k_list = _compute_k_schedule(args.T, args.K_min, args.levels, schedule=str(args.k_schedule), geom_gamma=args.k_geom_gamma)
    K_eval = int(k_list[args.levels])
    overlaps = []
    overlaps_int = []
    maes = []
    for _ in range(int(args.val_batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        text = batch["text_embed"].to(device, non_blocking=True)
        cond = {"text_embed": text}
        if bool(args.use_level):
            if args.level_mode == "s_norm":
                level_val = torch.full((text.shape[0], 1), float(args.levels) / float(max(1, args.levels)), device=device)
            else:
                level_val = torch.full((text.shape[0], 1), float(K_eval) / float(max(1, args.T - 1)), device=device)
            cond["level"] = level_val

        C = _integrated_cost_matrix(dphi, cond={"text_embed": text}, seg_feat_base=seg_feat_base, precomp=precomp, t_idx=t_idx, t_weights=t_weights, N_train=N_train)
        idx = dp_select_indices_batch(C, K_eval)  # [B,K]
        target = torch.zeros((text.shape[0], args.T), device=device, dtype=torch.bool)
        target.scatter_(1, idx, True)

        logits = selector(cond)
        pred_idx = torch.topk(logits[:, 1:-1], k=max(0, K_eval - 2), dim=1).indices + 1 if K_eval > 2 else torch.empty((text.shape[0], 0), device=device, dtype=torch.long)
        pred_full = torch.cat(
            [
                torch.zeros((text.shape[0], 1), device=device, dtype=torch.long),
                pred_idx,
                torch.full((text.shape[0], 1), args.T - 1, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        pred_full = torch.sort(pred_full, dim=1).values
        pred_mask = torch.zeros((text.shape[0], args.T), device=device, dtype=torch.bool)
        pred_mask.scatter_(1, pred_full, True)

        overlap = (pred_mask & target).float().sum(dim=1) / torch.clamp(target.float().sum(dim=1), min=1.0)
        overlaps.append(overlap.mean().item())
        # Interior overlap excludes endpoints, which are always included and can hide failures at low-K.
        t_int = target[:, 1:-1]
        p_int = pred_mask[:, 1:-1]
        denom_int = torch.clamp(t_int.float().sum(dim=1), min=1.0)
        overlaps_int.append(((t_int & p_int).float().sum(dim=1) / denom_int).mean().item())

        # MAE on the sorted index sequences (same K).
        mae = torch.mean(torch.abs(pred_full.float() - idx.float()))
        maes.append(mae.item())

    if selector_was_training:
        selector.train()
    overlap = float(sum(overlaps) / max(1, len(overlaps)))
    overlap_int = float(sum(overlaps_int) / max(1, len(overlaps_int))) if overlaps_int else float("nan")
    mae = float(sum(maes) / max(1, len(maes)))
    return overlap, overlap_int, mae


def _load_dphi(ckpt: str, *, device: torch.device) -> Tuple[SegmentCostPredictor, dict]:
    payload = torch.load(ckpt, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if str(meta.get("stage")) not in {"segment_cost_wansynth", "segment_cost"}:
        raise ValueError("dphi_ckpt meta.stage is not segment_cost_wansynth")
    d_cond = int(meta.get("d_cond", 256))
    seg_feat_dim = int(meta.get("seg_feat_dim", 4))
    hidden_dim = int(meta.get("hidden_dim", 512))
    n_layers = int(meta.get("n_layers", 3))
    dropout = float(meta.get("dropout", 0.0))
    text_dim = int(meta.get("text_dim", 1024))
    cond_enc = TextConditionEncoder(text_dim=text_dim, d_cond=d_cond)
    model = SegmentCostPredictor(
        d_cond=d_cond,
        seg_feat_dim=seg_feat_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        cond_encoder=cond_enc,
    )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = get_device(None)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth selector training.")

    loader = create_wan_synth_dataloader(
        args.train_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
        shuffle=True,
        shardshuffle=True,
        keep_text_embed=True,
        keep_text=False,
        resampled=bool(args.resampled),
        seed=args.seed,
    )
    it = iter(loader)

    # Prime for text dim.
    batch0 = next(it)
    text0 = batch0.get("text_embed")
    if text0 is None:
        raise RuntimeError("text_embed missing from dataset")
    text_dim = int(text0.shape[-1])
    if int(batch0["latents"].shape[1]) != int(args.T):
        raise ValueError(f"T mismatch: batch={batch0['latents'].shape[1]} args={args.T}")

    dphi, meta_dphi = _load_dphi(args.dphi_ckpt, device=device)

    # Determine integration timesteps/weights (match D_phi meta by default).
    schedule = str(meta_dphi.get("schedule", "cosine"))
    N_train = int(meta_dphi.get("N_train", 1000))
    snr_min = float(meta_dphi.get("snr_min", 0.1))
    snr_max = float(meta_dphi.get("snr_max", 10.0))
    snr_gamma = float(meta_dphi.get("snr_gamma", 1.0))
    t_steps = int(meta_dphi.get("t_steps", 16))
    if args.schedule:
        schedule = str(args.schedule)
    if args.N_train:
        N_train = int(args.N_train)
    if args.snr_min:
        snr_min = float(args.snr_min)
    if args.snr_max:
        snr_max = float(args.snr_max)
    if args.snr_gamma:
        snr_gamma = float(args.snr_gamma)
    if args.t_steps:
        t_steps = int(args.t_steps)
    snr, weights = build_snr_weights(schedule, N_train, snr_min, snr_max, snr_gamma)
    t_idx = sample_timesteps_log_snr(snr.to(device), t_steps).to(device)
    t_weights = weights.to(device)[t_idx].float()

    # Segment precompute for all (i,j).
    precomp = build_segment_precompute(int(args.T), samples_per_seg=1, device=device)
    seg_feat_base = build_segment_features(int(args.T), precomp.seg_i, precomp.seg_j).to(device=device, dtype=torch.float32)

    selector = VideoKeyframeSelector(
        T=int(args.T),
        text_dim=text_dim,
        d_model=int(args.d_model),
        d_cond=int(args.d_cond),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        pos_dim=int(args.pos_dim),
        dropout=float(args.dropout),
        use_level=bool(args.use_level),
    ).to(device=device)

    opt = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, selector, opt, ema=None, map_location=device)

    k_list = _compute_k_schedule(args.T, args.K_min, args.levels, schedule=str(args.k_schedule), geom_gamma=args.k_geom_gamma)
    k_list_t = torch.tensor(k_list, device=device, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 23)

    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        step_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        text = batch["text_embed"].to(device, non_blocking=True)
        B = text.shape[0]
        cond_text = {"text_embed": text}

        # Sample a level (and thus K) per sample, build DP labels from D_phi costs.
        s_idx = torch.randint(1, int(args.levels) + 1, (B,), generator=gen, device=device)
        K_s = k_list_t[s_idx]

        C = _integrated_cost_matrix(
            dphi,
            cond=cond_text,
            seg_feat_base=seg_feat_base,
            precomp=precomp,
            t_idx=t_idx,
            t_weights=t_weights,
            N_train=N_train,
        )

        # DP per-sample with varying K is awkward; group by unique K values.
        target = torch.zeros((B, int(args.T)), device=device, dtype=torch.float32)
        for K_val in torch.unique(K_s).tolist():
            K_val = int(K_val)
            m = (K_s == float(K_val))
            if not bool(m.any()):
                continue
            idx_sel = dp_select_indices_batch(C[m], K_val)
            target[m].scatter_(1, idx_sel, 1.0)

        cond = {"text_embed": text}
        if bool(args.use_level):
            if args.level_mode == "s_norm":
                level_val = s_idx.float() / float(max(1, int(args.levels)))
            else:
                level_val = K_s / float(max(1, int(args.T - 1)))
            cond["level"] = level_val.unsqueeze(1)

        logits = selector(cond)

        pos_weight = (float(args.T) - K_s) / torch.clamp(K_s, min=1.0)
        loss = criterion(logits, target)
        weights_pos = 1.0 + (pos_weight.unsqueeze(1) - 1.0) * target
        loss = (loss * weights_pos).mean()

        tau = _anneal_tau(
            step,
            int(args.steps),
            float(args.sel_tau_start),
            float(args.sel_tau_end),
            float(args.sel_tau_frac),
            str(args.sel_tau_anneal),
        )
        if float(args.sel_kl_weight) > 0.0:
            logits_i = logits[:, 1:-1] / max(1e-6, float(tau))
            logp = torch.log_softmax(logits_i, dim=-1)
            p_i = logp.exp()
            kl = (p_i * (logp + math.log(max(1, args.T - 2)))).sum(dim=-1).mean()
            loss = loss + float(args.sel_kl_weight) * kl
        else:
            kl = None

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), float(args.grad_clip))
        opt.step()

        step_time = time.perf_counter() - step_start
        if step % 50 == 0:
            pbar.set_description(f"loss {loss.item():.4f} {step_time:.2f}s")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/sel_tau", tau, step)
            if kl is not None:
                writer.add_scalar("train/sel_kl", kl.item(), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            writer.add_scalar("train/samples_per_sec", float(B) / max(step_time, 1e-8), step)

        if args.val_pattern and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            overlap, overlap_int, mae = _eval(
                selector,
                dphi,
                seg_feat_base,
                precomp,
                args,
                t_idx=t_idx,
                t_weights=t_weights,
                N_train=N_train,
                device=device,
            )
            writer.add_scalar("val/overlap", overlap, step)
            writer.add_scalar("val/overlap_int", overlap_int, step)
            writer.add_scalar("val/mae", mae, step)
            print(
                f"[VAL step={step}] overlap={overlap:.4f} overlap_int={overlap_int:.4f} mae={mae:.3f}",
                file=sys.stderr,
                flush=True,
            )

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "video_selector_wansynth",
                "T": int(args.T),
                "K_min": int(args.K_min),
                "levels": int(args.levels),
                "k_schedule": str(args.k_schedule),
                "k_geom_gamma": None if args.k_geom_gamma is None else float(args.k_geom_gamma),
                "dphi_ckpt": args.dphi_ckpt,
                "schedule": schedule,
                "N_train": int(N_train),
                "snr_min": float(snr_min),
                "snr_max": float(snr_max),
                "snr_gamma": float(snr_gamma),
                "t_steps": int(t_steps),
                "t_idx": t_idx.detach().cpu().tolist(),
                "d_model": int(args.d_model),
                "d_cond": int(args.d_cond),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "d_ff": int(args.d_ff),
                "pos_dim": int(args.pos_dim),
                "dropout": float(args.dropout),
                "use_level": bool(args.use_level),
                "level_mode": str(args.level_mode),
            }
            save_checkpoint(ckpt_path, selector, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "video_selector_wansynth",
        "T": int(args.T),
        "K_min": int(args.K_min),
        "levels": int(args.levels),
        "k_schedule": str(args.k_schedule),
        "k_geom_gamma": None if args.k_geom_gamma is None else float(args.k_geom_gamma),
        "dphi_ckpt": args.dphi_ckpt,
        "schedule": schedule,
        "N_train": int(N_train),
        "snr_min": float(snr_min),
        "snr_max": float(snr_max),
        "snr_gamma": float(snr_gamma),
        "t_steps": int(t_steps),
        "t_idx": t_idx.detach().cpu().tolist(),
        "d_model": int(args.d_model),
        "d_cond": int(args.d_cond),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "d_ff": int(args.d_ff),
        "pos_dim": int(args.pos_dim),
        "dropout": float(args.dropout),
        "use_level": bool(args.use_level),
        "level_mode": str(args.level_mode),
    }
    save_checkpoint(final_path, selector, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
