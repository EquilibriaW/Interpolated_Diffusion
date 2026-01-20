import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import interpolate_keyframes, sample_keyframe_mask
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.diffusion.ddpm import q_sample
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_causal import CausalDenoiser
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype, get_device
from src.utils.ema import EMA
from src.utils.logging import create_writer
from src.utils.seed import get_seed_from_env, set_seed


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
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
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="runs/causal")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/causal")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_checkpoint", type=int, default=0)
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


def build_per_t(generator: torch.Generator, B: int, T: int, N_train: int, device: torch.device):
    t = torch.zeros((B, T), dtype=torch.long, device=device)
    p = torch.randint(1, T, (B,), generator=generator, device=device)
    idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    mask = idx >= p.unsqueeze(1)
    t_rand = torch.randint(1, N_train, (B, T), generator=generator, device=device)
    t = torch.where(mask, t_rand, t)
    return t


def main():
    args = build_argparser().parse_args()
    seed = args.seed if args.seed is not None else get_seed_from_env()
    set_seed(seed)

    device = get_device(args.device)
    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    if args.dataset == "d4rl":
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
    model = CausalDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        use_checkpoint=bool(args.use_checkpoint),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 123)

    model.train()
    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["x"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}

        y, mask = build_batch_masks(x, gen, recompute_velocity=bool(args.recompute_vel))
        r0 = x - y
        r0 = r0 * (~mask).unsqueeze(-1)

        t = build_per_t(gen, x.shape[0], args.T, args.N_train, device)
        r_t, eps = q_sample(r0, t, schedule)
        clean = t == 0
        r_t = torch.where(clean.unsqueeze(-1), r0, r_t)
        eps = torch.where(clean.unsqueeze(-1), torch.zeros_like(eps), eps)

        r_t = r_t * (~mask).unsqueeze(-1)
        eps = eps * (~mask).unsqueeze(-1)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            eps_hat = model(r_t, t, y, mask, cond)
            diff = (eps_hat - eps) ** 2
            diff = diff.sum(dim=-1)
            valid = ((t > 0) & (~mask)).float()
            loss = (diff * valid).sum() / (valid.sum() * x.shape[-1] + 1e-8)
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
