import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.video_token_denoisers import VideoTokenKeypointDenoiser
from src.models.encoders import TextConditionEncoder
from src.models.wan_backbone import load_wan_transformer, resolve_dtype
from src.utils.video_tokens import patchify_latents
from src.utils.ema import EMA
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.seed import set_seed
from src.utils.logging import create_writer
from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch
from src.corruptions.video_keyframes import interpolate_video_from_indices
from src.utils.video_tokens import unpatchify_tokens


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_cond", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str, default="attn,ffn")
    p.add_argument("--use_wan", type=int, default=0)
    p.add_argument("--wan_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--wan_subfolder", type=str, default="transformer")
    p.add_argument("--wan_dtype", type=str, default="")
    p.add_argument("--wan_attn", type=str, default="default", choices=["default", "sla", "sagesla"])
    p.add_argument("--sla_topk", type=float, default=0.1)
    p.add_argument("--grad_ckpt", type=int, default=1)
    p.add_argument("--video_interp_mode", type=str, default="smooth", choices=["linear", "smooth", "flow", "sinkhorn"])
    p.add_argument("--video_interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
    p.add_argument("--flow_interp_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_win", type=int, default=5)
    p.add_argument("--sinkhorn_angles", type=str, default="-10,-5,0,5,10")
    p.add_argument("--sinkhorn_shift", type=int, default=4)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_tau", type=float, default=0.05)
    p.add_argument("--sinkhorn_dustbin", type=float, default=-2.0)
    p.add_argument("--sinkhorn_d_match", type=int, default=0)
    p.add_argument("--sinkhorn_straightener_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_straightener_dtype", type=str, default="")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/keypoints_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/keypoints_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=1000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth keypoints training.")

    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shuffle=True,
        shardshuffle=True,
    )
    it = iter(loader)

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 17)

    # Prime one batch to build the model.
    batch0 = next(it)
    latents0 = batch0["latents"].to(device)
    text_embed0 = batch0.get("text_embed")
    if text_embed0 is None:
        raise RuntimeError("text_embed missing from Wan synth dataset")
    text_embed0 = text_embed0.to(device)
    tokens0, spatial_shape0 = patchify_latents(latents0, args.patch_size)
    B0, T0, N0, D0 = tokens0.shape
    C0 = latents0.shape[2]
    if T0 != args.T:
        raise ValueError(f"T mismatch: batch={T0} args={args.T}")
    cond_encoder = TextConditionEncoder(text_dim=text_embed0.shape[-1], d_cond=args.d_cond).to(device)
    smooth_kernel = None
    flow_warper = None
    sinkhorn_warper = None
    if args.video_interp_mode == "smooth":
        smooth_kernel = torch.tensor([float(x) for x in args.video_interp_smooth_kernel.split(",")], dtype=torch.float32)
    elif args.video_interp_mode == "flow":
        if not args.flow_interp_ckpt:
            raise ValueError("--flow_interp_ckpt is required for video_interp_mode=flow")
        from src.models.latent_flow_interpolator import load_latent_flow_interpolator
        flow_dtype = resolve_dtype(args.wan_dtype) or get_autocast_dtype()
        flow_warper, _ = load_latent_flow_interpolator(args.flow_interp_ckpt, device=device, dtype=flow_dtype)
    elif args.video_interp_mode == "sinkhorn":
        from src.models.sinkhorn_warp import SinkhornWarpInterpolator
        from src.models.latent_straightener import load_latent_straightener

        straightener = None
        if args.sinkhorn_straightener_ckpt:
            s_dtype = resolve_dtype(args.sinkhorn_straightener_dtype) or get_autocast_dtype()
            straightener, _ = load_latent_straightener(args.sinkhorn_straightener_ckpt, device=device, dtype=s_dtype)
        angles = [float(x) for x in args.sinkhorn_angles.split(",") if x.strip()]
        sinkhorn_warper = SinkhornWarpInterpolator(
            in_channels=C0,
            patch_size=args.patch_size,
            win_size=args.sinkhorn_win,
            angles_deg=angles,
            shift_range=args.sinkhorn_shift,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_tau=args.sinkhorn_tau,
            dustbin_logit=args.sinkhorn_dustbin,
            d_match=args.sinkhorn_d_match,
            straightener=straightener,
        ).to(device=device, dtype=get_autocast_dtype())

    use_wan = bool(args.use_wan)

    if use_wan:
        wan_dtype = resolve_dtype(args.wan_dtype)
        model = load_wan_transformer(
            args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device
        )
        if int(args.grad_ckpt) == 1 and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        if args.wan_attn != "default":
            from src.models.wan_sla import apply_wan_sla

            use_bf16 = wan_dtype == torch.bfloat16 if wan_dtype is not None else True
            replaced = apply_wan_sla(
                model,
                topk=float(args.sla_topk),
                attention_type=str(args.wan_attn),
                use_bf16=use_bf16,
            )
            writer.add_scalar("train/sla_layers", replaced, 0)
    else:
        model = VideoTokenKeypointDenoiser(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            d_cond=args.d_cond,
            use_sdf=False,
            use_start_goal=False,
            data_dim=D0,
            cond_encoder=cond_encoder,
        ).to(device)

    if args.lora_rank > 0:
        from src.models.lora import LoRAConfig, inject_lora, mark_only_lora_trainable

        config = LoRAConfig(
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
            targets=tuple([t.strip() for t in str(args.lora_targets).split(",") if t.strip()]),
        )
        replaced = inject_lora(model, config)
        trainable = mark_only_lora_trainable(model)
        if replaced:
            writer.add_scalar("train/lora_linear_layers", len(replaced), 0)
            writer.add_scalar("train/lora_trainable_params", trainable, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None
    model_dtype = getattr(model, "dtype", None)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        step_start = time.perf_counter()
        log_this = step % args.log_every == 0
        if log_this:
            torch.cuda.reset_peak_memory_stats()
        if step == start_step:
            batch = batch0
        else:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

        if use_wan and model_dtype is not None:
            latents = batch["latents"].to(device, dtype=model_dtype, non_blocking=True)
        else:
            latents = batch["latents"].to(device, non_blocking=True)
        text_embed = batch.get("text_embed")
        if text_embed is None:
            raise RuntimeError("text_embed missing from Wan synth dataset")
        if use_wan and model_dtype is not None:
            text_embed = text_embed.to(device, dtype=model_dtype, non_blocking=True)
        else:
            text_embed = text_embed.to(device, non_blocking=True)

        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        tokens, spatial_shape = patchify_latents(latents, args.patch_size)
        B, T, N, D = tokens.shape
        if T != args.T:
            raise ValueError(f"T mismatch: batch={T} args={args.T}")

        idx, _ = sample_fixed_k_indices_uniform_batch(B, T, args.K, generator=gen, device=device, ensure_endpoints=False)
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

        model.train()
        if use_wan:
            if args.video_interp_mode in ("flow", "sinkhorn"):
                warper = flow_warper if args.video_interp_mode == "flow" else sinkhorn_warper
                if warper is None:
                    raise ValueError(f"{args.video_interp_mode} interpolator requested but not loaded")
                z_seq = torch.zeros((B, T, N, D), device=device, dtype=z_t.dtype)
                z_seq.scatter_(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), z_t)
                latents_seq = unpatchify_tokens(z_seq, args.patch_size, spatial_shape)
                try:
                    flow_dtype = next(warper.parameters()).dtype
                except StopIteration:
                    flow_dtype = latents_seq.dtype
                latents_seq_dtype = latents_seq.dtype
                latents_seq = latents_seq.to(dtype=flow_dtype)
                latents_interp, _ = warper.interpolate(latents_seq, idx)
                latents_interp = latents_interp.to(dtype=latents_seq_dtype)
                latents_t = latents_interp.permute(0, 2, 1, 3, 4)
            else:
                idx_rep = idx.repeat_interleave(N, dim=0)
                vals_rep = z_t.permute(0, 2, 1, 3).reshape(B * N, idx.shape[1], D)
                z_interp_flat = interpolate_video_from_indices(
                    idx_rep, vals_rep, T, mode=args.video_interp_mode, smooth_kernel=smooth_kernel
                )
                z_interp = z_interp_flat.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
                z_interp = z_interp.scatter(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), z_t)
                latents_t = unpatchify_tokens(z_interp, args.patch_size, spatial_shape)
                latents_t = latents_t.permute(0, 2, 1, 3, 4)
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                pred_latents = model(latents_t, t, text_embed).sample
            pred_latents = pred_latents.permute(0, 2, 1, 3, 4)
            pred_tokens, _ = patchify_latents(pred_latents, args.patch_size)
            pred_key = pred_tokens.gather(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D))
            loss = (pred_key - eps).pow(2).mean()
        else:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                eps_hat = model(z_t, t, idx, cond, args.T, spatial_shape)
                loss = (eps_hat - eps).pow(2).mean()

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model.parameters())

        step_time = time.perf_counter() - step_start
        if step % args.log_every == 0:
            pbar.set_description(f"loss {loss.item():.4f} step {step_time:.3f}s")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            writer.add_scalar("train/samples_per_sec", float(B) / max(step_time, 1e-8), step)
            writer.add_scalar("train/frames_per_sec", float(B * T) / max(step_time, 1e-8), step)
            if log_this:
                max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                writer.add_scalar("train/max_mem_gb", max_mem, step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "keypoints_wansynth",
                "T": args.T,
                "K": args.K,
                "data_dim": D,
                "patch_size": args.patch_size,
                "N_train": args.N_train,
                "schedule": args.schedule,
                "text_dim": int(text_embed.shape[-1]),
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_ff": args.d_ff,
                "d_cond": args.d_cond,
                "use_wan": bool(args.use_wan),
                "wan_repo": args.wan_repo,
                "wan_subfolder": args.wan_subfolder,
                "wan_dtype": args.wan_dtype,
                "wan_attn": args.wan_attn,
                "sla_topk": float(args.sla_topk),
                "video_interp_mode": args.video_interp_mode,
                "flow_interp_ckpt": args.flow_interp_ckpt,
                "sinkhorn_win": args.sinkhorn_win,
                "sinkhorn_angles": args.sinkhorn_angles,
                "sinkhorn_shift": args.sinkhorn_shift,
                "sinkhorn_iters": args.sinkhorn_iters,
                "sinkhorn_tau": args.sinkhorn_tau,
                "sinkhorn_dustbin": args.sinkhorn_dustbin,
                "sinkhorn_d_match": args.sinkhorn_d_match,
                "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
                "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "lora_targets": str(args.lora_targets),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "keypoints_wansynth",
        "T": args.T,
        "K": args.K,
        "data_dim": D,
        "patch_size": args.patch_size,
        "N_train": args.N_train,
        "schedule": args.schedule,
        "text_dim": int(text_embed.shape[-1]),
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "d_cond": args.d_cond,
        "use_wan": bool(args.use_wan),
        "wan_repo": args.wan_repo,
        "wan_subfolder": args.wan_subfolder,
        "wan_dtype": args.wan_dtype,
        "wan_attn": args.wan_attn,
        "sla_topk": float(args.sla_topk),
        "video_interp_mode": args.video_interp_mode,
        "flow_interp_ckpt": args.flow_interp_ckpt,
        "sinkhorn_win": args.sinkhorn_win,
        "sinkhorn_angles": args.sinkhorn_angles,
        "sinkhorn_shift": args.sinkhorn_shift,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tau": args.sinkhorn_tau,
        "sinkhorn_dustbin": args.sinkhorn_dustbin,
        "sinkhorn_d_match": args.sinkhorn_d_match,
        "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
        "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": str(args.lora_targets),
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
