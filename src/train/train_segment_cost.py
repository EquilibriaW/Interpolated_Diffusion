import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PreparedTrajectoryDataset
from src.models.segment_cost import SegmentCostPredictor
from src.selection.epiplexity_dp import (
    build_segment_features,
    build_segment_precompute,
    build_snr_weights,
    compute_segment_costs_batch,
    sample_timesteps_log_snr,
)
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.logging import create_writer
from src.utils.run_config import write_run_config
from src.utils.seed import get_seed_from_env, set_seed


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared_path", type=str, required=True)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--cond_start_goal", type=int, default=1)
    p.add_argument("--d_cond", type=int, default=128)
    p.add_argument("--seg_feat_dim", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--maze_channels", type=str, default="32,64,128,128")
    p.add_argument("--segment_cost_samples", type=int, default=16)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--snr_min", type=float, default=0.1)
    p.add_argument("--snr_max", type=float, default=10.0)
    p.add_argument("--snr_gamma", type=float, default=1.0)
    p.add_argument("--t_steps", type=int, default=16)
    p.add_argument("--normalize_targets", type=int, default=1)
    p.add_argument("--stats_subset", type=int, default=512)
    p.add_argument("--stats_batch", type=int, default=64)
    p.add_argument("--log_dir", type=str, default="runs/segment_cost")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/segment_cost")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def _estimate_cost_stats(
    x: np.ndarray,
    precomp,
    weight_scale: float,
    subset: int,
    batch: int,
    device: torch.device,
) -> Tuple[float, float]:
    n = x.shape[0]
    subset = max(1, min(n, subset))
    rng = np.random.RandomState(123)
    idx = rng.choice(n, size=subset, replace=False)
    total = 0.0
    total_sq = 0.0
    count = 0
    for start in range(0, subset, batch):
        end = min(subset, start + batch)
        xb = torch.from_numpy(x[idx[start:end]]).to(device)
        cost = compute_segment_costs_batch(xb[:, :, :2], precomp, weight_scale)
        total += float(cost.sum().item())
        total_sq += float((cost * cost).sum().item())
        count += int(cost.numel())
    mean = total / max(1, count)
    var = max(1e-8, total_sq / max(1, count) - mean * mean)
    std = float(np.sqrt(var))
    return float(mean), std


def main():
    args = build_argparser().parse_args()
    seed = args.seed if args.seed is not None else get_seed_from_env()
    set_seed(seed, deterministic=True)

    device = get_device(args.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dataset = PreparedTrajectoryDataset(args.prepared_path, use_sdf=bool(args.use_sdf))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    it = iter(loader)

    maze_channels = tuple(int(p.strip()) for p in str(args.maze_channels).split(",") if p.strip())
    model = SegmentCostPredictor(
        d_cond=int(args.d_cond),
        seg_feat_dim=int(args.seg_feat_dim),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        use_sdf=bool(args.use_sdf),
        use_start_goal=bool(args.cond_start_goal),
        maze_channels=maze_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)
    write_run_config(args.log_dir, args, writer=writer, prepared_path=args.prepared_path, extra={"stage": "segment_cost"})

    snr, weights = build_snr_weights(args.schedule, args.N_train, args.snr_min, args.snr_max, args.snr_gamma)
    t_idx = sample_timesteps_log_snr(snr, args.t_steps)
    weight_scale = float(weights[t_idx].sum().item())

    precomp = build_segment_precompute(args.T, args.segment_cost_samples, device)
    seg_feat = build_segment_features(args.T, precomp.seg_i, precomp.seg_j).to(device)

    target_mean, target_std = 0.0, 1.0
    if bool(args.normalize_targets):
        target_mean, target_std = _estimate_cost_stats(
            dataset.x, precomp, weight_scale, args.stats_subset, args.stats_batch, device
        )

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
        x0 = batch["x"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}

        with torch.no_grad():
            target = compute_segment_costs_batch(x0[:, :, :2], precomp, weight_scale)
            if bool(args.normalize_targets):
                target = (target - target_mean) / max(1e-6, target_std)

        pred = model(cond, seg_feat)
        loss = torch.mean((pred - target) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "segment_cost",
                "T": args.T,
                "d_cond": int(args.d_cond),
                "seg_feat_dim": int(args.seg_feat_dim),
                "hidden_dim": int(args.hidden_dim),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "use_sdf": bool(args.use_sdf),
                "cond_start_goal": bool(args.cond_start_goal),
                "schedule": args.schedule,
                "N_train": int(args.N_train),
                "snr_min": float(args.snr_min),
                "snr_max": float(args.snr_max),
                "snr_gamma": float(args.snr_gamma),
                "t_steps": int(args.t_steps),
                "t_idx": t_idx.cpu().numpy().tolist(),
                "weight_scale": weight_scale,
                "segment_cost_samples": int(args.segment_cost_samples),
                "maze_channels": args.maze_channels,
                "normalize_targets": bool(args.normalize_targets),
                "target_mean": float(target_mean),
                "target_std": float(target_std),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "segment_cost",
        "T": args.T,
        "d_cond": int(args.d_cond),
        "seg_feat_dim": int(args.seg_feat_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "dropout": float(args.dropout),
        "use_sdf": bool(args.use_sdf),
        "cond_start_goal": bool(args.cond_start_goal),
        "schedule": args.schedule,
        "N_train": int(args.N_train),
        "snr_min": float(args.snr_min),
        "snr_max": float(args.snr_max),
        "snr_gamma": float(args.snr_gamma),
        "t_steps": int(args.t_steps),
        "t_idx": t_idx.cpu().numpy().tolist(),
        "weight_scale": weight_scale,
        "segment_cost_samples": int(args.segment_cost_samples),
        "maze_channels": args.maze_channels,
        "normalize_targets": bool(args.normalize_targets),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
