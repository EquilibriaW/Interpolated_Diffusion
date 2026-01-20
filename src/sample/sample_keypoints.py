import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import sample_fixed_k_mask
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_keypoints import KeypointDenoiser
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/keypoints")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default="particle", choices=["particle", "synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    return p


def _build_known_values(idx: torch.Tensor, cond: dict, D: int, T: int) -> torch.Tensor:
    B, K = idx.shape
    known_values = torch.zeros((B, K, D), device=idx.device, dtype=torch.float32)
    if "start_goal" in cond and D >= 2:
        start = cond["start_goal"][:, :2]
        goal = cond["start_goal"][:, 2:]
        start_pos = start.unsqueeze(1).expand(B, K, 2)
        goal_pos = goal.unsqueeze(1).expand(B, K, 2)
        mask_start = (idx == 0).unsqueeze(-1)
        mask_goal = (idx == T - 1).unsqueeze(-1)
        known_values[:, :, :2] = torch.where(mask_start, start_pos, known_values[:, :, :2])
        known_values[:, :, :2] = torch.where(mask_goal, goal_pos, known_values[:, :, :2])
    return known_values


def _sample_ddim(
    model,
    schedule,
    idx: torch.Tensor,
    known: torch.Tensor,
    known_values: torch.Tensor,
    cond: dict,
    steps: int,
    T: int,
):
    device = idx.device
    B, K = idx.shape
    D = known_values.shape[-1]
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps)
    z = torch.randn((B, K, D), device=device)
    z = torch.where(known.unsqueeze(-1), known_values, z)
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(z, t, idx, known, cond, T)
        z = ddim_step(z, eps, t, t_prev, schedule, eta=0.0)
        z = torch.where(known.unsqueeze(-1), known_values, z)
    return z


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2

    model = KeypointDenoiser(data_dim=data_dim, use_sdf=bool(args.use_sdf)).to(device)
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

    dataset_name = "synthetic" if args.dataset == "synthetic" else args.dataset
    if dataset_name == "d4rl":
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

    all_samples = []
    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True):
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            B = cond["start_goal"].shape[0]
            idx = torch.zeros((B, args.K), dtype=torch.long, device=device)
            for b in range(B):
                mask = sample_fixed_k_mask(args.T, args.K, device=device, ensure_endpoints=True)
                idx[b] = torch.where(mask)[0]
            known = (idx == 0) | (idx == args.T - 1)
            known_values = _build_known_values(idx, cond, data_dim, args.T)
            z_hat = _sample_ddim(model, schedule, idx, known, known_values, cond, args.ddim_steps, args.T)
            all_samples.append({"idx": idx.detach().cpu(), "z": z_hat.detach().cpu()})

    torch.save(all_samples, os.path.join(args.out_dir, "keypoints.pt"))
    print(f"saved {len(all_samples)} batches to {os.path.join(args.out_dir, 'keypoints.pt')}")


if __name__ == "__main__":
    main()
