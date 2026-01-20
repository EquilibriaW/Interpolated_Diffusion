import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import build_nested_masks, interpolate_from_mask
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.models.denoiser_interp_levels_causal import InterpLevelCausalDenoiser
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype, get_device
from src.utils.ema import EMA
from src.utils.logging import create_writer
from src.utils.seed import get_seed_from_env, set_seed


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K_min", type=int, default=8)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--recompute_vel", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default="particle", choices=["particle", "synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="runs/interp_levels_causal")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/interp_levels_causal")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_checkpoint", type=int, default=0)
    p.add_argument("--w_anchor", type=float, default=0.1)
    p.add_argument("--w_missing", type=float, default=1.0)
    return p


def build_interp_level_batch(
    x0: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    recompute_velocity: bool = False,
):
    B, T, D = x0.shape
    device = x0.device
    x_s = torch.zeros_like(x0)
    mask_s = torch.zeros((B, T), dtype=torch.bool, device=device)
    s_idx = torch.zeros((B,), dtype=torch.long, device=device)
    for b in range(B):
        masks = build_nested_masks(T, K_min, levels, generator=generator, device=device)
        s = int(torch.randint(1, levels + 1, (1,), generator=generator, device=device).item())
        m = masks[s]
        xs = interpolate_from_mask(x0[b], m, recompute_velocity=recompute_velocity)
        x_s[b] = xs
        mask_s[b] = m
        s_idx[b] = s
    return x_s, mask_s, s_idx


def main():
    args = build_argparser().parse_args()
    seed = args.seed if args.seed is not None else get_seed_from_env()
    set_seed(seed)

    device = get_device(args.device)
    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    dataset_name = "synthetic" if args.dataset == "synthetic" else args.dataset
    if dataset_name == "d4rl":
        dataset = D4RLMazeDataset(
            env_id=args.env_id,
            num_samples=args.num_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            seed=seed,
            flip_y=bool(args.d4rl_flip_y),
        )
    else:
        dataset = ParticleMazeDataset(
            num_samples=args.num_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            cache_dir=args.cache_dir,
        )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    it = iter(loader)

    data_dim = 4 if args.with_velocity else 2
    model = InterpLevelCausalDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        max_levels=args.levels,
        use_checkpoint=bool(args.use_checkpoint),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 29)

    model.train()
    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x0 = batch["x"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}

        x_s, mask_s, s_idx = build_interp_level_batch(
            x0, args.K_min, args.levels, gen, recompute_velocity=bool(args.recompute_vel)
        )
        target = x0 - x_s

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            delta_hat = model(x_s, s_idx, mask_s, cond)
            diff = (delta_hat - target) ** 2
            diff = diff.sum(dim=-1)
            w = torch.where(mask_s, torch.tensor(args.w_anchor, device=device), torch.tensor(args.w_missing, device=device))
            loss = (diff * w).sum() / (w.sum() * x0.shape[-1] + 1e-8)
            loss = loss / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model.parameters())

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item() * args.grad_accum, step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, ema)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    save_checkpoint(final_path, model, optimizer, args.steps, ema)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
