import argparse
import os
import subprocess
import time
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.didemo import DiDeMoVideoDataset
from src.data.didemo_cache import CachedDiDeMoDataset
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.clip_text import CLIPTextEncoder
from src.models.video_token_denoisers import VideoTokenKeypointDenoiser
from src.models.encoders import TextConditionEncoder
from src.models.frame_vae import FrameAutoencoderKL
from src.utils.video_tokens import patchify_latents
from src.utils.ema import EMA
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.seed import set_seed
from src.utils.logging import create_writer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/didemo")
    p.add_argument("--video_dir", type=str, default="data/didemo/videos")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=256)
    p.add_argument("--clip_seconds", type=float, default=5.0)
    p.add_argument("--single_segment_only", type=int, default=1)
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--use_cached_text", type=int, default=1)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N_train", type=int, default=200)
    p.add_argument("--schedule", type=str, default="cosine")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_cond", type=int, default=128)
    p.add_argument("--text_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--vae_scale", type=float, default=0.18215)
    p.add_argument("--latent_channels", type=int, default=4)
    p.add_argument("--latent_downsample", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--log_gpu", type=int, default=1)
    p.add_argument("--log_gpu_every", type=int, default=50)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/keypoints_didemo")
    p.add_argument("--log_dir", type=str, default="runs/keypoints_didemo")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    return p


def _sample_fixed_k_indices(B: int, T: int, K: int, generator, device):
    from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch

    idx, _ = sample_fixed_k_indices_uniform_batch(
        B,
        T,
        K,
        generator=generator,
        device=device,
        ensure_endpoints=False,
    )
    return idx


def _query_gpu_stats() -> dict | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = out.split(",")
    if len(parts) < 3:
        return None
    try:
        util = float(parts[0].strip())
        mem_used = float(parts[1].strip())
        mem_total = float(parts[2].strip())
    except Exception:
        return None
    return {"util": util, "mem_used": mem_used, "mem_total": mem_total}


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for DiDeMo keypoints training.")

    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    if args.cache_dir:
        dataset = CachedDiDeMoDataset(cache_dir=args.cache_dir, split=args.split)
        use_cache = True
    else:
        dataset = DiDeMoVideoDataset(
            data_dir=args.data_dir,
            video_dir=args.video_dir,
            split=args.split,
            T=args.T,
            frame_size=args.frame_size,
            clip_seconds=args.clip_seconds,
            single_segment_only=bool(args.single_segment_only),
            max_items=args.max_items or None,
        )
        use_cache = False
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)

    if args.frame_size % args.latent_downsample != 0:
        raise ValueError("--frame_size must be divisible by --latent_downsample")
    latent_hw = args.frame_size // args.latent_downsample
    if latent_hw % args.patch_size != 0:
        raise ValueError("--patch_size must divide latent spatial size")
    data_dim = args.latent_channels * args.patch_size * args.patch_size
    text_encoder = None
    text_dim = None
    if not use_cache or not bool(args.use_cached_text):
        text_encoder = CLIPTextEncoder(model_name=args.text_model, device=device, dtype=autocast_dtype)
        text_dim = int(text_encoder.text_model.config.hidden_size)
    else:
        # Placeholder until we read cached embeddings.
        text_dim = 512
    cond_encoder = TextConditionEncoder(text_dim=text_dim, d_cond=args.d_cond).to(device)
    model = VideoTokenKeypointDenoiser(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        d_cond=args.d_cond,
        use_sdf=False,
        use_start_goal=False,
        data_dim=data_dim,
        cond_encoder=cond_encoder,
    ).to(device)

    vae = None
    if not use_cache:
        vae = FrameAutoencoderKL(
            model_name=args.vae_model,
            device=device,
            dtype=autocast_dtype,
            scale=args.vae_scale,
            use_mean=True,
            freeze=True,
        )

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
        step_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        if use_cache:
            latents = batch["latents"].to(device)
            if "text_embed" in batch and bool(args.use_cached_text):
                text_embed = batch["text_embed"].to(device)
            else:
                texts: List[str] = list(batch.get("text", []))
                if text_encoder is None:
                    text_encoder = CLIPTextEncoder(model_name=args.text_model, device=device, dtype=autocast_dtype)
                text_embed = text_encoder(texts)
        else:
            frames = batch["frames"].to(device)
            texts = list(batch["text"])
            if vae is None:
                raise RuntimeError("VAE missing for non-cached dataset")
            with torch.no_grad():
                latents = vae.encode(frames)
            if text_encoder is None:
                text_encoder = CLIPTextEncoder(model_name=args.text_model, device=device, dtype=autocast_dtype)
            text_embed = text_encoder(texts)
        if latents.shape[2] != args.latent_channels:
            raise ValueError(f"latent channels mismatch: expected {args.latent_channels}, got {latents.shape[2]}")
        tokens, spatial_shape = patchify_latents(latents, args.patch_size)
        B, T, N, D = tokens.shape
        if D != data_dim:
            raise ValueError(f"latent token dim mismatch: expected {data_dim}, got {D}")
        idx = _sample_fixed_k_indices(B, T, args.K, gen, device)
        z0_k = tokens.gather(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)).clone()

        t = torch.randint(0, args.N_train, (B,), generator=gen, device=device, dtype=torch.long)
        eps = torch.randn_like(z0_k)
        z_t = schedule["sqrt_alpha_bar"][t].view(B, 1, 1, 1) * z0_k + schedule["sqrt_one_minus_alpha_bar"][t].view(
            B, 1, 1, 1
        ) * eps

        if args.cond_drop_prob > 0.0:
            drop = torch.rand((B,), generator=gen, device=device) < float(args.cond_drop_prob)
            if torch.any(drop):
                text_embed = text_embed.clone()
                text_embed[drop] = 0.0
        cond = {"text_embed": text_embed}

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            eps_hat = model(z_t, t, idx, cond, args.T, spatial_shape)
            diff = (eps_hat - eps) ** 2
            loss = diff.mean()

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

        step_time = time.perf_counter() - step_start
        if step % args.log_every == 0:
            pbar.set_description(f"loss {loss.item():.4f} step {step_time:.3f}s")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
        if bool(args.log_gpu) and step % args.log_gpu_every == 0:
            stats = _query_gpu_stats()
            if stats is not None:
                writer.add_scalar("gpu/util", stats["util"], step)
                writer.add_scalar("gpu/mem_used", stats["mem_used"], step)
                writer.add_scalar("gpu/mem_total", stats["mem_total"], step)
                pbar.write(
                    f"gpu util {stats['util']:.0f}% mem {stats['mem_used']:.0f}/{stats['mem_total']:.0f} MiB"
                )

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "keypoints_didemo",
                "T": args.T,
                "K": args.K,
                "data_dim": data_dim,
                "patch_size": args.patch_size,
                "N_train": args.N_train,
                "schedule": args.schedule,
                "frame_size": args.frame_size,
                "clip_seconds": args.clip_seconds,
                "text_model": args.text_model,
                "vae_model": args.vae_model,
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "keypoints_didemo",
        "T": args.T,
        "K": args.K,
        "data_dim": data_dim,
        "patch_size": args.patch_size,
        "N_train": args.N_train,
        "schedule": args.schedule,
        "frame_size": args.frame_size,
        "clip_seconds": args.clip_seconds,
        "text_model": args.text_model,
        "vae_model": args.vae_model,
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
