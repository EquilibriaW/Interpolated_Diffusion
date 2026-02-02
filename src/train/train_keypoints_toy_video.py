import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.toy_video import MovingShapesVideoDataset
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_keypoints import KeypointDenoiser
from src.utils.ema import EMA
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.seed import set_seed
from src.utils.logging import create_writer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--latent_size", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N_train", type=int, default=200)
    p.add_argument("--schedule", type=str, default="cosine")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--logit_space", type=int, default=0)
    p.add_argument("--logit_eps", type=float, default=1e-5)
    p.add_argument("--clamp_endpoints", type=int, default=1)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--allow_tf32", type=int, default=1)
    p.add_argument("--enable_flash_sdp", type=int, default=1)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/keypoints_toy_video")
    p.add_argument("--log_dir", type=str, default="runs/keypoints_toy_video")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    return p


def _build_known_mask_values(
    idx: torch.Tensor, start_goal: torch.Tensor, D: int, T: int, clamp_endpoints: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    B, K = idx.shape
    known_mask = torch.zeros((B, K, D), device=idx.device, dtype=torch.bool)
    known_values = torch.zeros((B, K, D), device=idx.device)
    if clamp_endpoints:
        start = start_goal[:, :D]
        goal = start_goal[:, D:]
        mask_start = (idx == 0).unsqueeze(-1)
        mask_goal = (idx == T - 1).unsqueeze(-1)
        start_vals = start.unsqueeze(1).expand(B, K, D)
        goal_vals = goal.unsqueeze(1).expand(B, K, D)
        known_mask[:] = mask_start | mask_goal
        known_values = torch.where(mask_start, start_vals, known_values)
        known_values = torch.where(mask_goal, goal_vals, known_values)
    return known_mask, known_values


def _build_keypoint_batch(
    x0: torch.Tensor, start_goal: torch.Tensor, K: int, gen: torch.Generator, clamp_endpoints: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = x0.shape
    idx, _ = sample_fixed_k_indices_uniform_batch(B, T, K, generator=gen, device=x0.device, ensure_endpoints=True)
    z0 = x0.gather(1, idx.unsqueeze(-1).expand(-1, K, D)).clone()
    known_mask, known_values = _build_known_mask_values(idx, start_goal, D, T, clamp_endpoints)
    return z0, idx, known_mask, known_values


def sample_fixed_k_indices_uniform_batch(B: int, T: int, K: int, generator, device, ensure_endpoints: bool = True):
    from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch as _sample

    return _sample(B, T, K, generator=generator, device=device, ensure_endpoints=ensure_endpoints)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=bool(args.deterministic))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for toy keypoints training.")
    print(f"[toy keypoints] device={device}")
    print(f"[toy keypoints] cuda={torch.cuda.get_device_name(0)}")
    if device.type == "cuda" and not bool(args.deterministic):
        torch.backends.cudnn.benchmark = True
        if bool(args.allow_tf32):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        try:
            torch.backends.cuda.enable_flash_sdp(bool(args.enable_flash_sdp))
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    dataset = MovingShapesVideoDataset(T=args.T, n_samples=args.num_samples, seed=args.seed, latent_size=args.latent_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)

    data_dim = dataset.data_dim
    model = KeypointDenoiser(
        data_dim=data_dim,
        use_sdf=False,
        use_start_goal=False,
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
    gen.manual_seed(args.seed + 11)

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
        cond = batch.get("cond", {})
        start_goal = cond.get("start_goal", None)
        if start_goal is None:
            raise ValueError("Toy video batch missing cond['start_goal']")
        start_goal = start_goal.to(device)

        z0, idx, known_mask, known_values = _build_keypoint_batch(x0, start_goal, args.K, gen, bool(args.clamp_endpoints))
        if args.logit_space:
            z0 = torch.logit(z0.clamp(args.logit_eps, 1 - args.logit_eps))

        B = z0.shape[0]
        t = torch.randint(0, args.N_train, (B,), generator=gen, device=device, dtype=torch.long)
        eps = torch.randn_like(z0)
        z_t = schedule["sqrt_alpha_bar"][t].view(B, 1, 1) * z0 + schedule["sqrt_one_minus_alpha_bar"][t].view(
            B, 1, 1
        ) * eps
        z_t = torch.where(known_mask, known_values, z_t)
        eps = eps * (~known_mask)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            eps_hat = model(z_t, t, idx, known_mask, {}, args.T)
            valid = (~known_mask).float()
            diff = (eps_hat - eps) ** 2
            loss = (diff * valid).sum() / (valid.sum() + 1e-8)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

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
            writer.add_scalar("train/loss", loss.item(), step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "keypoints",
                "T": args.T,
                "K": args.K,
                "data_dim": data_dim,
                "N_train": args.N_train,
                "schedule": args.schedule,
                "logit_space": bool(args.logit_space),
                "logit_eps": float(args.logit_eps),
                "clamp_endpoints": bool(args.clamp_endpoints),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "keypoints",
        "T": args.T,
        "K": args.K,
        "data_dim": data_dim,
        "N_train": args.N_train,
        "schedule": args.schedule,
        "logit_space": bool(args.logit_space),
        "logit_eps": float(args.logit_eps),
        "clamp_endpoints": bool(args.clamp_endpoints),
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
