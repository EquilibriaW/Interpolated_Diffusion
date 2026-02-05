import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_anchor_dataloader, create_wan_synth_dataloader
from src.corruptions.keyframes import build_nested_masks_batch, build_nested_masks_from_base
from src.corruptions.video_keyframes import (
    build_video_token_interp_adjacent_batch,
    build_video_token_interp_level_batch,
)
from src.models.video_token_denoisers import VideoTokenInterpLevelDenoiser
from src.models.encoders import TextConditionEncoder
from src.models.wan_backbone import load_wan_transformer, resolve_dtype
from src.utils.video_tokens import patchify_latents
from src.utils.video_tokens import unpatchify_tokens
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.ema import EMA
from src.utils.seed import set_seed
from src.utils.logging import create_writer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--anchor_pattern", type=str, default="", help="Optional anchor shard pattern")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_cond", type=int, default=256)
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
    p.add_argument("--video_interp_mode", type=str, default="smooth", choices=["linear", "smooth", "learned", "flow", "sinkhorn"])
    p.add_argument("--video_interp_ckpt", type=str, default="")
    p.add_argument("--flow_interp_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_win", type=int, default=5)
    p.add_argument("--sinkhorn_angles", type=str, default="-10,-5,0,5,10")
    p.add_argument("--sinkhorn_shift", type=int, default=4)
    p.add_argument("--sinkhorn_global_mode", type=str, default="phasecorr", choices=["se2", "phasecorr", "none"])
    p.add_argument("--sinkhorn_warp_space", type=str, default="s", choices=["z", "s"])
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_tau", type=float, default=0.05)
    p.add_argument("--sinkhorn_dustbin", type=float, default=-2.0)
    p.add_argument("--sinkhorn_d_match", type=int, default=0)
    p.add_argument("--sinkhorn_straightener_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_straightener_dtype", type=str, default="")
    p.add_argument("--flow_uncertainty_mode", type=str, default="replace", choices=["none", "add", "replace"])
    p.add_argument("--flow_uncertainty_weight", type=float, default=1.0)
    p.add_argument("--flow_uncertainty_power", type=float, default=1.0)
    p.add_argument("--video_interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
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
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/interp_levels_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/interp_levels_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=1000)
    p.add_argument("--anchor_join", type=int, default=1)
    p.add_argument("--anchor_key_buffer", type=int, default=2000)
    p.add_argument("--anchor_allow_missing", type=int, default=0)
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
    if conf is not None and conf.dim() > 1:
        lam = lam.view(-1, *([1] * (conf.dim() - 1)))
    else:
        lam = lam.view(-1, 1)
    return conf + (1.0 - conf) * lam


def _sample_level_indices(B: int, levels: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth interp-level training.")

    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    if args.anchor_pattern:
        loader = create_wan_synth_anchor_dataloader(
            args.data_pattern,
            args.anchor_pattern,
            batch_size=args.batch,
            num_workers=args.num_workers,
            shuffle_buffer=args.shuffle_buffer,
            shuffle=True,
            join_by_key=bool(args.anchor_join),
            max_key_buffer=args.anchor_key_buffer,
            allow_missing=bool(args.anchor_allow_missing),
        )
    else:
        loader = create_wan_synth_dataloader(
            args.data_pattern,
            batch_size=args.batch,
            num_workers=args.num_workers,
            shuffle_buffer=args.shuffle_buffer,
            shuffle=True,
            shardshuffle=True,
        )
    it = iter(loader)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 23)

    # Prime one batch to build model.
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

    video_interp_model = None
    flow_warper = None
    sinkhorn_warper = None
    smooth_kernel = None
    if args.video_interp_mode == "smooth":
        smooth_kernel = torch.tensor([float(x) for x in args.video_interp_smooth_kernel.split(",")], dtype=torch.float32)
    elif args.video_interp_mode == "learned":
        if not args.video_interp_ckpt:
            raise ValueError("--video_interp_ckpt is required for video_interp_mode=learned")
        from src.models.video_interpolator import TinyTemporalInterpolator

        video_interp_model = TinyTemporalInterpolator(data_dim=D0)
        payload = torch.load(args.video_interp_ckpt, map_location="cpu")
        state = payload.get("model", payload)
        video_interp_model.load_state_dict(state)
        video_interp_model.to(device)
        video_interp_model.eval()
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
            global_mode=args.sinkhorn_global_mode,
            angles_deg=angles,
            shift_range=args.sinkhorn_shift,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_tau=args.sinkhorn_tau,
            dustbin_logit=args.sinkhorn_dustbin,
            d_match=args.sinkhorn_d_match,
            straightener=straightener,
            warp_space=args.sinkhorn_warp_space,
        ).to(device=device, dtype=get_autocast_dtype())

    mask_channels = (2 if args.stage2_mode == "adj" else 1) + (1 if args.anchor_conf else 0)
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
        model = VideoTokenInterpLevelDenoiser(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            d_cond=args.d_cond,
            data_dim=D0,
            max_levels=args.levels,
            use_sdf=False,
            use_start_goal=False,
            mask_channels=mask_channels,
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
        anchor_values = batch.get("anchor")
        anchor_idx = batch.get("anchor_idx")
        if anchor_values is not None:
            if use_wan and model_dtype is not None:
                anchor_values = anchor_values.to(device, dtype=model_dtype, non_blocking=True)
            else:
                anchor_values = anchor_values.to(device, non_blocking=True)
        if anchor_idx is not None:
            anchor_idx = anchor_idx.to(device, non_blocking=True)

        tokens, spatial_shape = patchify_latents(latents, args.patch_size)
        B, T, N, D = tokens.shape
        if T != args.T:
            raise ValueError(f"T mismatch: batch={T} args={args.T}")

        if anchor_idx is not None:
            masks_levels, idx_levels = build_nested_masks_from_base(
                anchor_idx, T, args.levels, generator=gen, device=device, k_schedule=args.k_schedule, k_geom_gamma=args.k_geom_gamma
            )
        else:
            masks_levels, idx_levels = build_nested_masks_batch(
                B, T, args.K_min, args.levels, generator=gen, device=device, k_schedule=args.k_schedule, k_geom_gamma=args.k_geom_gamma
            )
        s_idx = _sample_level_indices(B, args.levels, gen, device)

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
                anchor_values=anchor_values,
                anchor_idx=anchor_idx,
                conf_anchor=args.anchor_conf_teacher,
                conf_student=args.anchor_conf_student,
                conf_endpoints=args.anchor_conf_endpoints,
                conf_missing=args.anchor_conf_missing,
                clamp_endpoints=False,
                interp_mode=args.video_interp_mode,
                interp_model=video_interp_model,
                smooth_kernel=smooth_kernel,
                flow_warper=flow_warper,
                sinkhorn_warper=sinkhorn_warper,
                patch_size=args.patch_size,
                spatial_shape=spatial_shape,
                uncertainty_mode=args.flow_uncertainty_mode,
                uncertainty_weight=args.flow_uncertainty_weight,
                uncertainty_power=args.flow_uncertainty_power,
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
                anchor_values=anchor_values,
                anchor_idx=anchor_idx,
                conf_anchor=args.anchor_conf_teacher,
                conf_student=args.anchor_conf_student,
                conf_endpoints=args.anchor_conf_endpoints,
                conf_missing=args.anchor_conf_missing,
                clamp_endpoints=False,
                interp_mode=args.video_interp_mode,
                interp_model=video_interp_model,
                smooth_kernel=smooth_kernel,
                flow_warper=flow_warper,
                sinkhorn_warper=sinkhorn_warper,
                patch_size=args.patch_size,
                spatial_shape=spatial_shape,
                uncertainty_mode=args.flow_uncertainty_mode,
                uncertainty_weight=args.flow_uncertainty_weight,
                uncertainty_power=args.flow_uncertainty_power,
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), conf_s], dim=-1)
            else:
                mask_in = mask_s
            target = tokens - z_s
            weight_mask = conf_s if args.anchor_conf else mask_s

        cond = {"text_embed": text_embed}
        if use_wan:
            latents_in = unpatchify_tokens(z_s, args.patch_size, spatial_shape)
            latents_in = latents_in.permute(0, 2, 1, 3, 4)
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                pred_latents = model(latents_in, s_idx, text_embed).sample
            pred_latents = pred_latents.permute(0, 2, 1, 3, 4)
            pred_tokens, _ = patchify_latents(pred_latents, args.patch_size)
            diff = (pred_tokens - target).pow(2).sum(dim=-1)
        else:
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
            writer.add_scalar("train/anchor_replace_prob", float(args.student_replace_prob), step)
            if log_this:
                max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                writer.add_scalar("train/max_mem_gb", max_mem, step)
            if anchor_idx is not None:
                mask_t = mask_s.any(dim=2)
                anchor_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
                anchor_mask.scatter_(1, anchor_idx, True)
                overlap = (mask_t & anchor_mask).float().sum() / (mask_t.float().sum() + 1e-6)
                writer.add_scalar("train/anchor_overlap", overlap.item(), step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "interp_levels_wansynth",
                "T": args.T,
                "K_min": args.K_min,
                "levels": args.levels,
                "data_dim": D,
                "patch_size": args.patch_size,
                "stage2_mode": args.stage2_mode,
                "mask_channels": mask_channels,
                "k_schedule": args.k_schedule,
                "k_geom_gamma": args.k_geom_gamma,
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
                "video_interp_ckpt": args.video_interp_ckpt,
                "flow_interp_ckpt": args.flow_interp_ckpt,
                "sinkhorn_win": args.sinkhorn_win,
                "sinkhorn_angles": args.sinkhorn_angles,
                "sinkhorn_shift": args.sinkhorn_shift,
                "sinkhorn_global_mode": args.sinkhorn_global_mode,
                "sinkhorn_warp_space": args.sinkhorn_warp_space,
                "sinkhorn_iters": args.sinkhorn_iters,
                "sinkhorn_tau": args.sinkhorn_tau,
                "sinkhorn_dustbin": args.sinkhorn_dustbin,
                "sinkhorn_d_match": args.sinkhorn_d_match,
                "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
                "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
                "flow_uncertainty_mode": args.flow_uncertainty_mode,
                "flow_uncertainty_weight": float(args.flow_uncertainty_weight),
                "flow_uncertainty_power": float(args.flow_uncertainty_power),
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "lora_targets": str(args.lora_targets),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "interp_levels_wansynth",
        "T": args.T,
        "K_min": args.K_min,
        "levels": args.levels,
        "data_dim": D,
        "patch_size": args.patch_size,
        "stage2_mode": args.stage2_mode,
        "mask_channels": mask_channels,
        "k_schedule": args.k_schedule,
        "k_geom_gamma": args.k_geom_gamma,
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
        "video_interp_ckpt": args.video_interp_ckpt,
        "flow_interp_ckpt": args.flow_interp_ckpt,
        "sinkhorn_win": args.sinkhorn_win,
        "sinkhorn_angles": args.sinkhorn_angles,
        "sinkhorn_shift": args.sinkhorn_shift,
        "sinkhorn_global_mode": args.sinkhorn_global_mode,
        "sinkhorn_warp_space": args.sinkhorn_warp_space,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tau": args.sinkhorn_tau,
        "sinkhorn_dustbin": args.sinkhorn_dustbin,
        "sinkhorn_d_match": args.sinkhorn_d_match,
        "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
        "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
        "flow_uncertainty_mode": args.flow_uncertainty_mode,
        "flow_uncertainty_weight": float(args.flow_uncertainty_weight),
        "flow_uncertainty_power": float(args.flow_uncertainty_power),
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
