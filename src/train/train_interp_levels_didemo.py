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
from src.corruptions.keyframes import build_nested_masks_batch
from src.corruptions.video_keyframes import (
    build_video_token_interp_adjacent_batch,
    build_video_token_interp_level_batch,
)
from src.models.clip_text import CLIPTextEncoder
from src.models.video_token_denoisers import VideoTokenInterpLevelDenoiser
from src.models.encoders import TextConditionEncoder
from src.models.frame_vae import FrameAutoencoderKL
from src.models.video_interpolator import TinyTemporalInterpolator
from src.utils.video_tokens import patchify_latents
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.ema import EMA
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
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_cond", type=int, default=128)
    p.add_argument("--stage2_mode", type=str, default="adj", choices=["adj", "x0"])
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--anchor_conf", type=int, default=1)
    p.add_argument("--anchor_conf_teacher", type=float, default=0.95)
    p.add_argument("--anchor_conf_student", type=float, default=0.5)
    p.add_argument("--anchor_conf_endpoints", type=float, default=1.0)
    p.add_argument("--anchor_conf_missing", type=float, default=0.0)
    p.add_argument("--anchor_conf_anneal", type=int, default=1)
    p.add_argument("--anchor_conf_anneal_mode", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--w_anchor", type=float, default=0.3)
    p.add_argument("--w_missing", type=float, default=1.0)
    p.add_argument("--corrupt_mode", type=str, default="dist", choices=["none", "dist", "gauss"])
    p.add_argument("--corrupt_sigma", type=float, default=0.02)
    p.add_argument("--corrupt_anchor_frac", type=float, default=0.25)
    p.add_argument("--student_replace_prob", type=float, default=0.5)
    p.add_argument("--student_noise_std", type=float, default=0.02)
    p.add_argument("--video_interp_mode", type=str, default="linear", choices=["linear", "smooth", "learned", "flow"])
    p.add_argument("--video_interp_ckpt", type=str, default="")
    p.add_argument("--video_interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
    p.add_argument("--flow_model", type=str, default="raft_large", choices=["raft_large", "raft_small"])
    p.add_argument("--flow_conf_sigma", type=float, default=1.0)
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
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/interp_levels_didemo")
    p.add_argument("--log_dir", type=str, default="runs/interp_levels_didemo")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    return p


def _anneal_conf(conf: torch.Tensor, s_idx: torch.Tensor, levels: int, mode: str) -> torch.Tensor:
    if conf is None or mode == "none" or levels <= 0:
        return conf
    frac = s_idx.float() / float(levels)
    if mode == "linear":
        lam = 1.0 - frac
    elif mode == "cosine":
        lam = 0.5 * (1.0 + torch.cos(torch.pi * frac))
    else:
        lam = torch.zeros_like(frac)
    lam = lam.view(-1, 1)
    return conf + (1.0 - conf) * lam


def _sample_level_indices(B: int, levels: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)


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
        raise RuntimeError("CUDA is required for DiDeMo interp-level training.")

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

    text_encoder = None
    text_dim = None
    if not use_cache or not bool(args.use_cached_text):
        text_encoder = CLIPTextEncoder(model_name=args.text_model, device=device, dtype=autocast_dtype)
        text_dim = int(text_encoder.text_model.config.hidden_size)
    else:
        text_dim = 512

    if args.frame_size % args.latent_downsample != 0:
        raise ValueError("--frame_size must be divisible by --latent_downsample")
    latent_hw = args.frame_size // args.latent_downsample
    if latent_hw % args.patch_size != 0:
        raise ValueError("--patch_size must divide latent spatial size")
    data_dim = args.latent_channels * args.patch_size * args.patch_size
    cond_encoder = TextConditionEncoder(text_dim=text_dim, d_cond=args.d_cond).to(device)

    vae = None
    if args.video_interp_mode == "flow" or not use_cache:
        vae = FrameAutoencoderKL(
            model_name=args.vae_model,
            device=device,
            dtype=autocast_dtype,
            scale=args.vae_scale,
            use_mean=True,
            freeze=True,
        )

    video_interp_model = None
    flow_warper = None
    smooth_kernel = None
    if args.video_interp_mode == "smooth":
        smooth_kernel = torch.tensor([float(x) for x in args.video_interp_smooth_kernel.split(",")], dtype=torch.float32)
    elif args.video_interp_mode == "learned":
        if not args.video_interp_ckpt:
            raise ValueError("--video_interp_ckpt is required for video_interp_mode=learned")
        video_interp_model = TinyTemporalInterpolator(data_dim=data_dim)
        payload = torch.load(args.video_interp_ckpt, map_location="cpu")
        state = payload.get("model", payload)
        video_interp_model.load_state_dict(state)
        video_interp_model.to(device)
        video_interp_model.eval()
    elif args.video_interp_mode == "flow":
        from src.models.flow_warp import FlowWarpInterpolator, RAFTFlowEstimator

        flow_variant = "large" if args.flow_model == "raft_large" else "small"
        flow_estimator = RAFTFlowEstimator(variant=flow_variant, device=device)
        flow_warper = FlowWarpInterpolator(
            flow_estimator=flow_estimator,
            vae=vae,
            frame_size=args.frame_size,
            latent_downsample=args.latent_downsample,
            conf_sigma=args.flow_conf_sigma,
        )

    mask_channels = (2 if args.stage2_mode == "adj" else 1) + (1 if args.anchor_conf else 0)
    model = VideoTokenInterpLevelDenoiser(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        d_cond=args.d_cond,
        data_dim=data_dim,
        max_levels=args.levels,
        use_sdf=False,
        use_start_goal=False,
        mask_channels=mask_channels,
        cond_encoder=cond_encoder,
    ).to(device)

    if vae is None and not use_cache:
        raise RuntimeError("VAE missing for non-cached dataset")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 23)

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
        if tokens.shape[-1] != data_dim:
            raise ValueError(f"latent token dim mismatch: expected {data_dim}, got {tokens.shape[-1]}")

        masks_levels, idx_levels = build_nested_masks_batch(
            tokens.shape[0],
            args.T,
            args.K_min,
            args.levels,
            generator=gen,
            device=device,
            k_schedule=args.k_schedule,
            k_geom_gamma=args.k_geom_gamma,
        )
        s_idx = _sample_level_indices(tokens.shape[0], args.levels, gen, device)

        if args.stage2_mode == "adj":
            z_s, z_prev, mask_s, mask_prev, s_idx, _, _, conf_s, conf_prev = build_video_token_interp_adjacent_batch(
                tokens,
                args.K_min,
                args.levels,
                gen,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma=args.corrupt_sigma,
                anchor_noise_frac=args.corrupt_anchor_frac,
                student_replace_prob=args.student_replace_prob,
                student_noise_std=args.student_noise_std,
                conf_anchor=args.anchor_conf_teacher,
                conf_student=args.anchor_conf_student,
                conf_endpoints=args.anchor_conf_endpoints,
                conf_missing=args.anchor_conf_missing,
                clamp_endpoints=False,
                interp_mode=args.video_interp_mode,
                interp_model=video_interp_model,
                smooth_kernel=smooth_kernel,
                flow_warper=flow_warper,
                patch_size=args.patch_size,
                spatial_shape=spatial_shape,
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
                prev_idx = torch.clamp(s_idx - 1, min=0)
                conf_prev = _anneal_conf(conf_prev, prev_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
            else:
                mask_in = torch.stack([mask_s, mask_prev], dim=-1)
            target = z_prev - z_s
            weight_mask = conf_prev if args.anchor_conf else mask_prev
        else:
            z_s, mask_s, s_idx, _, _, conf_s = build_video_token_interp_level_batch(
                tokens,
                args.K_min,
                args.levels,
                gen,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma=args.corrupt_sigma,
                anchor_noise_frac=args.corrupt_anchor_frac,
                student_replace_prob=args.student_replace_prob,
                student_noise_std=args.student_noise_std,
                conf_anchor=args.anchor_conf_teacher,
                conf_student=args.anchor_conf_student,
                conf_endpoints=args.anchor_conf_endpoints,
                conf_missing=args.anchor_conf_missing,
                clamp_endpoints=False,
                interp_mode=args.video_interp_mode,
                interp_model=video_interp_model,
                smooth_kernel=smooth_kernel,
                flow_warper=flow_warper,
                patch_size=args.patch_size,
                spatial_shape=spatial_shape,
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), conf_s], dim=-1)
            else:
                mask_in = mask_s
            target = tokens - z_s
            weight_mask = conf_s if args.anchor_conf else mask_s

        if args.cond_drop_prob > 0.0:
            drop = torch.rand((tokens.shape[0],), generator=gen, device=device) < float(args.cond_drop_prob)
            if torch.any(drop):
                text_embed = text_embed.clone()
                text_embed[drop] = 0.0
        cond = {"text_embed": text_embed}

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            delta_hat = model(z_s, s_idx, mask_in, cond, spatial_shape)
            diff = (delta_hat - target).pow(2).sum(dim=-1)
            if args.anchor_conf:
                w = float(args.w_missing) + (float(args.w_anchor) - float(args.w_missing)) * weight_mask
            else:
                w = torch.where(
                    weight_mask,
                    torch.tensor(args.w_anchor, device=device),
                    torch.tensor(args.w_missing, device=device),
                )
            loss = (diff * w).sum() / (w.sum() * tokens.shape[-1] + 1e-8)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
                "stage": "interp_levels_didemo",
                "T": args.T,
                "K_min": args.K_min,
                "levels": args.levels,
                "data_dim": data_dim,
                "patch_size": args.patch_size,
                "stage2_mode": args.stage2_mode,
                "mask_channels": mask_channels,
                "k_schedule": args.k_schedule,
                "k_geom_gamma": args.k_geom_gamma,
                "frame_size": args.frame_size,
                "clip_seconds": args.clip_seconds,
                "text_model": args.text_model,
                "vae_model": args.vae_model,
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "interp_levels_didemo",
        "T": args.T,
        "K_min": args.K_min,
        "levels": args.levels,
        "data_dim": data_dim,
        "patch_size": args.patch_size,
        "stage2_mode": args.stage2_mode,
        "mask_channels": mask_channels,
        "k_schedule": args.k_schedule,
        "k_geom_gamma": args.k_geom_gamma,
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
