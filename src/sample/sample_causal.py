import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.eval.metrics import compute_metrics
from src.eval.visualize import plot_trajectories
from src.models.denoiser_causal import CausalDenoiser
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/causal")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=5)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    return p


def _make_y_segment(left: torch.Tensor, right: torch.Tensor, L: int, with_velocity: bool, dt: float) -> torch.Tensor:
    w = torch.linspace(0.0, 1.0, L + 1, device=left.device).unsqueeze(-1)
    seg = left + w * (right - left)
    if not with_velocity:
        return seg
    v = torch.zeros_like(seg)
    v[:-1] = (seg[1:] - seg[:-1]) / dt
    v[-1] = 0.0
    return torch.cat([seg, v], dim=-1)


def _heuristic_right(left: torch.Tensor, goal: torch.Tensor, L: int, remaining: int) -> torch.Tensor:
    frac = min(1.0, float(L) / max(1, remaining))
    return left + frac * (goal - left)


def ddim_sample_segment(model, schedule, y, mask, cond, steps):
    device = y.device
    B, T, D = y.shape
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps)
    rt = torch.randn((B, T, D), device=device)
    rt = rt * (~mask).unsqueeze(-1)
    for i in range(times.numel() - 1):
        t_int = int(times[i].item())
        t_prev_int = int(times[i + 1].item())
        t = torch.full((B, T), t_int, device=device, dtype=torch.long)
        t_prev = torch.full((B, T), t_prev_int, device=device, dtype=torch.long)
        eps = model(rt, t, y, mask, cond)
        rt = ddim_step(rt, eps, t, t_prev, schedule, eta=0.0)
        rt = rt * (~mask).unsqueeze(-1)
    return rt


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2

    model = CausalDenoiser(data_dim=data_dim, use_sdf=bool(args.use_sdf)).to(device)
    if args.use_ema:
        from src.utils.ema import EMA

        ema = EMA(model.parameters())
    else:
        ema = None
    load_checkpoint(args.ckpt, model, optimizer=None, ema=ema, map_location=device)
    if ema is not None:
        ema.copy_to(model.parameters())
    model.eval()

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    if args.dataset == "d4rl":
        dataset = D4RLMazeDataset(
            env_id=args.env_id,
            num_samples=args.n_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            seed=1234,
            flip_y=bool(args.d4rl_flip_y),
        )
    else:
        dataset = ParticleMazeDataset(
            num_samples=args.n_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
        )

    metrics_all: List[Dict[str, float]] = []

    with torch.no_grad():
        for idx in tqdm(range(args.n_samples), dynamic_ncols=True):
            sample = dataset[idx]
            x_gt_t = sample["x"].to(device)
            cond = sample["cond"]
            cond_t = {k: v.unsqueeze(0).to(device) for k, v in cond.items()}
            start_goal_t = cond_t["start_goal"][0]
            start_t = start_goal_t[:2]
            goal_t = start_goal_t[2:]

            x_gen = torch.zeros((args.T, 2), device=device)
            x_gen[0] = start_t
            cur = 1
            dt = 1.0 / args.T
            while cur < args.T:
                end = min(args.T - 1, cur + args.chunk - 1)
                L = end - cur + 1
                remaining = args.T - cur
                left = x_gen[cur - 1]
                right = goal_t if end == args.T - 1 else _heuristic_right(left, goal_t, L, remaining)
                y_seg = _make_y_segment(left, right, L, with_velocity=bool(args.with_velocity), dt=dt)

                mask = torch.zeros((1, L + 1), dtype=torch.bool, device=device)
                mask[0, 0] = True
                mask[0, -1] = True
                y = y_seg.unsqueeze(0)

                r_hat = ddim_sample_segment(model, schedule, y, mask, cond_t, steps=args.ddim_steps)
                x_seg = (y + r_hat).squeeze(0)
                x_gen[cur : end + 1] = x_seg[1:, :2]
                cur = end + 1

            t_lin = torch.linspace(0.0, 1.0, args.T, device=device).unsqueeze(-1)
            baseline_t = start_t + t_lin * (goal_t - start_t)
            occ_t = cond_t["occ"][0]
            gt_t = x_gt_t[:, :2]

            m_pred = compute_metrics(occ_t, x_gen, goal_t, gt_t)
            m_base = compute_metrics(occ_t, baseline_t, goal_t, gt_t)
            metrics_all.append({
                "baseline_collision": m_base["collision_rate"],
                "pred_collision": m_pred["collision_rate"],
                "pred_success": m_pred["success"],
                "pred_goal_dist": m_pred["goal_dist"],
            })

            occ = occ_t.detach().cpu().numpy()
            baseline = baseline_t.detach().cpu().numpy()
            x_gt = x_gt_t.detach().cpu().numpy()
            x_gen_np = x_gen.detach().cpu().numpy()
            out_path = os.path.join(args.out_dir, f"sample_{idx:04d}.png")
            plot_trajectories(occ, [x_gt[:, :2], baseline, x_gen_np], ["gt", "baseline", "pred"], out_path=out_path)

    summary = {}
    for k in metrics_all[0].keys():
        summary[k] = float(sum(m[k] for m in metrics_all) / len(metrics_all))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
