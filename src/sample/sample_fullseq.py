import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import interpolate_keyframes, sample_keyframe_mask
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.diffusion.ddpm import ddpm_sample, ddim_sample
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.eval.metrics import compute_metrics_batch
from src.eval.visualize import plot_trajectories
from src.models.denoiser_fullseq import FullSeqDenoiser
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/fullseq")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--recompute_vel", type=int, default=1)
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    return p


def build_batch_masks(x: torch.Tensor, generator: torch.Generator, recompute_velocity: bool = False):
    B, T, D = x.shape
    masks = torch.zeros((B, T), dtype=torch.bool, device=x.device)
    y = torch.zeros_like(x)
    for b in range(B):
        mask, _ = sample_keyframe_mask(T, mode="mixed", generator=generator, device=x.device)
        masks[b] = mask
        y[b] = interpolate_keyframes(x[b], mask, recompute_velocity=recompute_velocity)
    return y, masks


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2

    model = FullSeqDenoiser(data_dim=data_dim, use_sdf=bool(args.use_sdf)).to(device)
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
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    metrics_all: List[Dict[str, float]] = []
    idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True):
            x = batch["x"].to(device)
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            y, mask = build_batch_masks(x, gen, recompute_velocity=bool(args.recompute_vel))
            if args.sampler == "ddpm":
                r_hat = ddpm_sample(model, schedule, y, mask, cond)
            else:
                r_hat = ddim_sample(model, schedule, y, mask, cond, steps=args.ddim_steps)
            x_hat = y + r_hat

            occ_b = cond["occ"][:, 0]
            goal_b = cond["start_goal"][:, 2:]
            gt_b = x[:, :, :2]
            interp_b = y[:, :, :2]
            pred_b = x_hat[:, :, :2]

            m_interp_b = compute_metrics_batch(occ_b, interp_b, goal_b, gt_b)
            m_pred_b = compute_metrics_batch(occ_b, pred_b, goal_b, gt_b)

            for b in range(x.shape[0]):
                metrics_all.append({
                    "interp_collision": float(m_interp_b["collision_rate"][b].item()),
                    "interp_success": float(m_interp_b["success"][b].item()),
                    "pred_collision": float(m_pred_b["collision_rate"][b].item()),
                    "pred_success": float(m_pred_b["success"][b].item()),
                    "pred_goal_dist": float(m_pred_b["goal_dist"][b].item()),
                })

                occ = occ_b[b].detach().cpu().numpy()
                gt = gt_b[b].detach().cpu().numpy()
                interp = interp_b[b].detach().cpu().numpy()
                pred = pred_b[b].detach().cpu().numpy()
                out_path = os.path.join(args.out_dir, f"sample_{idx:04d}.png")
                plot_trajectories(occ, [gt, interp, pred], ["gt", "interp", "pred"], out_path=out_path)
                idx += 1

    # Aggregate metrics
    summary = {}
    for k in metrics_all[0].keys():
        summary[k] = float(sum(m[k] for m in metrics_all) / len(metrics_all))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
