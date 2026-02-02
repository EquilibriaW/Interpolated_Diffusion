import argparse
import os
import sys

import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PreparedTrajectoryDataset
from src.models.keypoint_selector import KeypointSelector
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_device
from src.utils.logging import create_writer
from src.utils.run_config import write_run_config
from src.utils.seed import get_seed_from_env, set_seed
from src.corruptions.keyframes import _compute_k_schedule


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared_path", type=str, required=True)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--cond_start_goal", type=int, default=1)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--pos_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--maze_channels", type=str, default="32,64,128,128")
    p.add_argument("--use_sg_map", type=int, default=1)
    p.add_argument("--use_sg_token", type=int, default=1)
    p.add_argument("--use_goal_dist_token", type=int, default=0)
    p.add_argument("--use_cond_bias", type=int, default=0)
    p.add_argument("--cond_bias_mode", type=str, default="memory", choices=["memory", "encoder"])
    p.add_argument("--use_level", type=int, default=0)
    p.add_argument("--level_mode", type=str, default="k_norm", choices=["k_norm", "s_norm"])
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--sg_map_sigma", type=float, default=1.5)
    p.add_argument("--sel_kl_weight", type=float, default=0.02)
    p.add_argument("--sel_tau_start", type=float, default=1.0)
    p.add_argument("--sel_tau_end", type=float, default=0.3)
    p.add_argument("--sel_tau_anneal", type=str, default="cosine", choices=["none", "linear", "cosine"])
    p.add_argument("--sel_tau_frac", type=float, default=0.8)
    p.add_argument("--log_dir", type=str, default="runs/keypoint_selector")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/keypoint_selector")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def _parse_int_list(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise ValueError("empty int list")
    return tuple(int(p) for p in parts)


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


def main():
    args = build_argparser().parse_args()
    seed = args.seed if args.seed is not None else get_seed_from_env()
    set_seed(seed, deterministic=True)

    device = get_device(args.device)
    dataset = PreparedTrajectoryDataset(args.prepared_path, use_sdf=bool(args.use_sdf))
    if dataset.kp_idx is None:
        raise ValueError("Prepared dataset missing kp_idx for selector training.")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    it = iter(loader)

    maze_channels = _parse_int_list(args.maze_channels)
    model = KeypointSelector(
        T=int(args.T),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        n_layers=int(args.n_layers),
        pos_dim=int(args.pos_dim),
        dropout=float(args.dropout),
        use_sdf=bool(args.use_sdf),
        use_start_goal=bool(args.cond_start_goal),
        use_sg_map=bool(args.use_sg_map),
        use_sg_token=bool(args.use_sg_token),
        use_goal_dist_token=bool(args.use_goal_dist_token),
        use_cond_bias=bool(args.use_cond_bias),
        cond_bias_mode=str(args.cond_bias_mode),
        use_level=bool(args.use_level),
        level_mode=str(args.level_mode),
        sg_map_sigma=float(args.sg_map_sigma),
        maze_channels=maze_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)
    write_run_config(args.log_dir, args, writer=writer, prepared_path=args.prepared_path, extra={"stage": "selector"})

    k_list = _compute_k_schedule(args.T, args.K, args.levels, schedule=args.k_schedule, geom_gamma=args.k_geom_gamma)
    k_list_t = torch.tensor(k_list, device=device, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, map_location=device)

    model.train()
    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}
        B = cond["occ"].shape[0]
        if "kp_mask_levels" in batch["cond"]:
            kp_mask_levels = batch["cond"]["kp_mask_levels"].to(device)
            if kp_mask_levels.dim() != 3:
                raise ValueError("kp_mask_levels must be [B, levels+1, T]")
            s_idx = torch.randint(1, args.levels + 1, (B,), device=device)
            target = kp_mask_levels[torch.arange(B, device=device), s_idx]
            if bool(args.use_level):
                if args.level_mode == "s_norm":
                    level_val = s_idx.float() / float(max(1, args.levels))
                else:
                    level_val = k_list_t[s_idx] / float(max(1, args.T - 1))
                cond = dict(cond)
                cond["level"] = level_val.unsqueeze(1)
            K_s = k_list_t[s_idx]
        else:
            kp_idx = batch["cond"]["kp_idx"].to(device)
            target = torch.zeros((B, args.T), device=device, dtype=torch.float32)
            target.scatter_(1, kp_idx, 1.0)
            K_s = torch.full((B,), float(args.K), device=device)
            if bool(args.use_level):
                if args.level_mode == "s_norm":
                    level_val = torch.full((B,), float(args.levels) / float(max(1, args.levels)), device=device)
                else:
                    level_val = torch.full((B,), float(args.K) / float(max(1, args.T - 1)), device=device)
                cond = dict(cond)
                cond["level"] = level_val.unsqueeze(1)

        logits = model(cond)
        pos_weight = (args.T - K_s) / torch.clamp(K_s, min=1.0)
        loss = criterion(logits, target)
        weights = 1.0 + (pos_weight.unsqueeze(1) - 1.0) * target
        loss = (loss * weights).mean()
        kl = None
        entropy = None
        tau = _anneal_tau(
            step,
            args.steps,
            float(args.sel_tau_start),
            float(args.sel_tau_end),
            float(args.sel_tau_frac),
            str(args.sel_tau_anneal),
        )
        if args.sel_kl_weight > 0.0:
            logits_i = logits[:, 1:-1] / max(1e-6, float(tau))
            logp = torch.log_softmax(logits_i, dim=-1)
            p = logp.exp()
            kl = (p * (logp + math.log(max(1, args.T - 2)))).sum(dim=-1).mean()
            entropy = -(p * logp).sum(dim=-1).mean()
            loss = loss + float(args.sel_kl_weight) * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/sel_tau", tau, step)
            if kl is not None:
                writer.add_scalar("train/sel_kl", kl.item(), step)
            if entropy is not None:
                writer.add_scalar("train/sel_entropy", entropy.item(), step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "selector",
                "T": int(args.T),
                "K": int(args.K),
                "d_model": int(args.d_model),
                "n_heads": int(args.n_heads),
                "d_ff": int(args.d_ff),
                "pos_dim": int(args.pos_dim),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "use_sdf": bool(args.use_sdf),
                "cond_start_goal": bool(args.cond_start_goal),
                "use_sg_map": bool(args.use_sg_map),
                "use_sg_token": bool(args.use_sg_token),
                "use_goal_dist_token": bool(args.use_goal_dist_token),
                "use_cond_bias": bool(args.use_cond_bias),
                "cond_bias_mode": str(args.cond_bias_mode),
                "use_level": bool(args.use_level),
                "level_mode": str(args.level_mode),
                "levels": int(args.levels),
                "k_schedule": str(args.k_schedule),
                "k_geom_gamma": None if args.k_geom_gamma is None else float(args.k_geom_gamma),
                "sg_map_sigma": float(args.sg_map_sigma),
                "maze_channels": args.maze_channels,
                "sel_kl_weight": float(args.sel_kl_weight),
                "sel_tau_start": float(args.sel_tau_start),
                "sel_tau_end": float(args.sel_tau_end),
                "sel_tau_anneal": str(args.sel_tau_anneal),
                "sel_tau_frac": float(args.sel_tau_frac),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "selector",
        "T": int(args.T),
        "K": int(args.K),
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "d_ff": int(args.d_ff),
        "pos_dim": int(args.pos_dim),
        "n_layers": int(args.n_layers),
        "dropout": float(args.dropout),
        "use_sdf": bool(args.use_sdf),
        "cond_start_goal": bool(args.cond_start_goal),
        "use_sg_map": bool(args.use_sg_map),
        "use_sg_token": bool(args.use_sg_token),
        "use_goal_dist_token": bool(args.use_goal_dist_token),
        "use_cond_bias": bool(args.use_cond_bias),
        "cond_bias_mode": str(args.cond_bias_mode),
        "use_level": bool(args.use_level),
        "level_mode": str(args.level_mode),
        "levels": int(args.levels),
        "k_schedule": str(args.k_schedule),
        "k_geom_gamma": None if args.k_geom_gamma is None else float(args.k_geom_gamma),
        "sg_map_sigma": float(args.sg_map_sigma),
        "maze_channels": args.maze_channels,
        "sel_kl_weight": float(args.sel_kl_weight),
        "sel_tau_start": float(args.sel_tau_start),
        "sel_tau_end": float(args.sel_tau_end),
        "sel_tau_anneal": str(args.sel_tau_anneal),
        "sel_tau_frac": float(args.sel_tau_frac),
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
