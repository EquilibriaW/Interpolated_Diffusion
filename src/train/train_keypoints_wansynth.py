import argparse
import os
import time

import psutil
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
from src.utils.optim import create_optimizer
from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch
from src.corruptions.video_keyframes import interpolate_video_from_indices
from src.utils.video_tokens import unpatchify_tokens
from src.utils.frame_features import frame_features_from_mask


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--K", type=int, default=4)
    p.add_argument(
        "--phase1_input_mode",
        type=str,
        default="full",
        choices=["full", "short_anchors", "short_midpoints", "short_meanpool"],
        help="How to run Phase-1 when use_wan=1. "
        "'full' = run Wan on length-T sequence with interpolated missing frames (current behavior). "
        "'short_anchors' = run Wan on length-K anchors only. "
        "'short_midpoints' = run Wan on length-(2K-1) anchors+midpoints, then discard midpoints downstream. "
        "'short_meanpool' = run Wan on length-(2K-1) anchors+pooled segment summaries (mean of in-between frames).",
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_nesterov", type=int, default=0)
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
    p.add_argument("--wan_frame_cond", type=int, default=0)
    p.add_argument("--wan_frame_cond_hidden", type=int, default=256)
    p.add_argument("--wan_frame_cond_layers", type=int, default=2)
    p.add_argument("--wan_frame_cond_dropout", type=float, default=0.0)
    p.add_argument("--video_interp_mode", type=str, default="smooth", choices=["linear", "smooth", "flow", "sinkhorn"])
    p.add_argument("--video_interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
    p.add_argument("--flow_interp_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_win", type=int, default=5)
    p.add_argument("--sinkhorn_stride", type=int, default=0)
    # Accept negative values without requiring the `--arg=-10` style by using a numeric
    # argument list (argparse treats negative numbers as values for numeric types).
    p.add_argument("--sinkhorn_angles", type=float, nargs="+", default=[-10.0, -5.0, 0.0, 5.0, 10.0])
    p.add_argument("--sinkhorn_shift", type=int, default=4)
    p.add_argument("--sinkhorn_global_mode", type=str, default="phasecorr", choices=["se2", "phasecorr", "none"])
    p.add_argument(
        "--sinkhorn_phasecorr_mode",
        type=str,
        default="multi",
        choices=["mean", "multi"],
        help="When using sinkhorn_global_mode=phasecorr, run phase correlation on a mean score-map or multi-channel features.",
    )
    p.add_argument(
        "--sinkhorn_phasecorr_level",
        type=str,
        default="token",
        choices=["token", "latent"],
        help="When using sinkhorn_global_mode=phasecorr, compute the global shift on token maps or full latent maps.",
    )
    p.add_argument("--sinkhorn_warp_space", type=str, default="s", choices=["z", "s"])
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_tau", type=float, default=0.05)
    p.add_argument("--sinkhorn_dustbin", type=float, default=-2.0)
    p.add_argument("--sinkhorn_spatial_gamma", type=float, default=0.0)
    p.add_argument("--sinkhorn_spatial_radius", type=int, default=0)
    p.add_argument("--sinkhorn_fb_sigma", type=float, default=0.0)
    p.add_argument("--sinkhorn_d_match", type=int, default=0)
    p.add_argument("--sinkhorn_straightener_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_straightener_dtype", type=str, default="")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/keypoints_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/keypoints_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument(
        "--save_optimizer",
        type=int,
        default=0,
        help="Whether to include optimizer state in checkpoints. "
        "For Wan-scale models this can create very large checkpoints and spike CPU RAM.",
    )
    p.add_argument("--save_final", type=int, default=1)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=64)
    p.add_argument(
        "--val_pattern",
        type=str,
        default="",
        help="Optional shard pattern for validation. If set, runs periodic validation on this pattern.",
    )
    p.add_argument("--val_every", type=int, default=0, help="Validate every N steps (0 disables).")
    p.add_argument("--val_batches", type=int, default=10, help="Number of val batches per validation run.")
    p.add_argument("--val_num_workers", type=int, default=0, help="Validation dataloader workers.")
    p.add_argument(
        "--max_cpu_mem_percent",
        type=float,
        default=98.0,
        help="Abort before the machine OOMs (common failure mode with WebDataset shuffles).",
    )
    return p


def _midpoint_indices(idx_base: torch.Tensor) -> torch.Tensor:
    # idx_base: [B,K] sorted
    if idx_base.dim() != 2:
        raise ValueError("idx_base must be [B,K]")
    left = idx_base[:, :-1]
    right = idx_base[:, 1:]
    gap = right - left
    if torch.any(gap < 2):
        raise ValueError("midpoints require all anchor gaps >= 2; reduce K or use phase1_input_mode=short_anchors")
    mid = (left + right) // 2
    mid = torch.minimum(torch.maximum(mid, left + 1), right - 1)
    return mid


def _meanpool_between_anchors(tokens: torch.Tensor, idx_base: torch.Tensor) -> torch.Tensor:
    # tokens: [B,T,N,D], idx_base: [B,K] sorted
    if tokens.dim() != 4:
        raise ValueError("tokens must be [B,T,N,D]")
    if idx_base.dim() != 2:
        raise ValueError("idx_base must be [B,K]")
    B, T, N, D = tokens.shape
    left = idx_base[:, :-1]
    right = idx_base[:, 1:]
    gap = right - left
    if torch.any(gap < 2):
        raise ValueError("meanpool requires all anchor gaps >= 2; reduce K or use short_anchors/full")
    # Sum over (left+1 .. right-1) inclusive using prefix sums: sum = csum[right-1] - csum[left]
    tokens_f = tokens.float()
    csum = tokens_f.cumsum(dim=1)
    idx_r = (right - 1).clamp(0, T - 1)
    gather_shape = (B, idx_r.shape[1], N, D)
    sum_r = csum.gather(1, idx_r.unsqueeze(-1).unsqueeze(-1).expand(gather_shape))
    sum_l = csum.gather(1, left.unsqueeze(-1).unsqueeze(-1).expand(gather_shape))
    seg_sum = sum_r - sum_l
    count = (gap - 1).clamp(min=1).to(dtype=seg_sum.dtype).view(B, -1, 1, 1)
    seg_mean = seg_sum / count
    return seg_mean.to(dtype=tokens.dtype)


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
    val_loader = None
    if args.val_pattern:
        val_loader = create_wan_synth_dataloader(
            args.val_pattern,
            batch_size=args.batch,
            num_workers=int(args.val_num_workers),
            shuffle_buffer=0,
            shuffle=False,
            shardshuffle=False,
        )

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)
    proc = psutil.Process(os.getpid())

    if args.num_workers > 0 and (int(args.num_workers) * int(args.shuffle_buffer)) >= 1000:
        # Each worker holds its own shuffle buffer; with ~8MB/sample this can exhaust CPU RAM fast.
        writer.add_text(
            "warn/loader",
            f"Large shuffle setting: num_workers={args.num_workers} shuffle_buffer={args.shuffle_buffer}. "
            "CPU RAM can spike: consider reducing shuffle_buffer or num_workers.",
            0,
        )

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 17)
    val_gen = torch.Generator(device=device)
    val_gen.manual_seed(args.seed + 1017)

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
    sinkhorn_angles = [float(x) for x in args.sinkhorn_angles]
    sinkhorn_angles_str = ",".join(str(a) for a in sinkhorn_angles)
    # Only needed when Phase-1 runs Wan on full-length T and needs to synthesize missing frames.
    if args.phase1_input_mode == "full":
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
            angles = sinkhorn_angles
            sinkhorn_warper = SinkhornWarpInterpolator(
                in_channels=C0,
                patch_size=args.patch_size,
                win_size=args.sinkhorn_win,
                win_stride=args.sinkhorn_stride,
                global_mode=args.sinkhorn_global_mode,
                phasecorr_mode=str(args.sinkhorn_phasecorr_mode),
                phasecorr_level=str(args.sinkhorn_phasecorr_level),
                angles_deg=angles,
                shift_range=args.sinkhorn_shift,
                sinkhorn_iters=args.sinkhorn_iters,
                sinkhorn_tau=args.sinkhorn_tau,
                dustbin_logit=args.sinkhorn_dustbin,
                spatial_gamma=args.sinkhorn_spatial_gamma,
                spatial_radius=args.sinkhorn_spatial_radius,
                fb_sigma=args.sinkhorn_fb_sigma,
                d_match=args.sinkhorn_d_match,
                straightener=straightener,
                warp_space=args.sinkhorn_warp_space,
            ).to(device=device, dtype=get_autocast_dtype())

    use_wan = bool(args.use_wan)

    if use_wan:
        wan_dtype = resolve_dtype(args.wan_dtype)
        model = load_wan_transformer(
            args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device
        )
        # Patch Wan RoPE to support absolute-time indices for short temporal inputs.
        from src.models.wan_abs_rope import enable_wan_absolute_time_rope

        enable_wan_absolute_time_rope(model)
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

    if use_wan and int(args.wan_frame_cond) == 1:
        from src.models.wan_frame_cond import FrameCondProjector

        text_dim = int(text_embed0.shape[-1])
        feat_dim = 5  # frame_features_from_mask(..., include_time=True)
        # Diffusers models may keep some params in fp32 even when loaded in bf16/fp16.
        # `model.dtype` returns the "effective" dtype (skipping keep-in-fp32 modules), which
        # matches our runtime casting of `text_embed`/`latents`.
        proj_dtype = getattr(model, "dtype", None)
        if proj_dtype is None:
            proj_dtype = next(model.parameters()).dtype
        model.frame_cond_proj = FrameCondProjector(
            feat_dim=feat_dim,
            text_dim=text_dim,
            hidden_dim=int(args.wan_frame_cond_hidden),
            n_layers=int(args.wan_frame_cond_layers),
            dropout=float(args.wan_frame_cond_dropout),
        ).to(device=device, dtype=proj_dtype)
    optimizer = create_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_momentum=float(args.muon_momentum),
        muon_nesterov=bool(int(args.muon_nesterov)),
    )
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None
    model_dtype = getattr(model, "dtype", None)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        step_start = time.perf_counter()
        vm = psutil.virtual_memory()
        if float(vm.percent) >= float(args.max_cpu_mem_percent):
            raise RuntimeError(
                f"CPU RAM usage {vm.percent:.1f}% exceeded max_cpu_mem_percent={args.max_cpu_mem_percent:.1f}. "
                "Reduce --shuffle_buffer and/or --num_workers."
            )
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

        idx_base, _ = sample_fixed_k_indices_uniform_batch(
            B, T, args.K, generator=gen, device=device, ensure_endpoints=False
        )
        gaps = (idx_base[:, 1:] - idx_base[:, :-1]).to(dtype=torch.float32)
        if args.phase1_input_mode == "short_midpoints":
            idx_mid = _midpoint_indices(idx_base)
            idx_in = torch.cat([idx_base, idx_mid], dim=1)
            idx_in = torch.sort(idx_in, dim=1).values
        elif args.phase1_input_mode == "short_meanpool":
            idx_mid = _midpoint_indices(idx_base)
            idx_in = torch.cat([idx_base, idx_mid], dim=1)
            idx_in = torch.sort(idx_in, dim=1).values
        elif args.phase1_input_mode == "short_anchors":
            idx_in = idx_base
        else:
            idx_in = idx_base
        z0_in = tokens.gather(1, idx_in.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)).clone()
        if args.phase1_input_mode == "short_meanpool":
            # Replace midpoint slots with pooled segment summaries (mean of in-between frames).
            pooled = _meanpool_between_anchors(tokens, idx_base)  # [B,K-1,N,D]
            pos_mid = torch.searchsorted(idx_in.contiguous(), idx_mid.contiguous())
            z0_in.scatter_(1, pos_mid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), pooled)

        t = torch.randint(0, args.N_train, (B,), generator=gen, device=device, dtype=torch.long)
        eps = torch.randn_like(z0_in)
        z_t = schedule["sqrt_alpha_bar"][t].view(B, 1, 1, 1) * z0_in + schedule["sqrt_one_minus_alpha_bar"][t].view(
            B, 1, 1, 1
        ) * eps

        if args.cond_drop_prob > 0.0:
            drop = torch.rand((B,), generator=gen, device=device) < float(args.cond_drop_prob)
            if torch.any(drop):
                text_embed = text_embed.clone()
                text_embed[drop] = 0.0
        if use_wan and hasattr(model, "frame_cond_proj"):
            mask = torch.zeros((B, T), device=device, dtype=torch.bool)
            mask.scatter_(1, idx_base, True)
            frame_feat_all = frame_features_from_mask(mask, include_time=True)  # [B,T,5]
            if args.phase1_input_mode == "full":
                frame_feat = frame_feat_all
            else:
                frame_feat = frame_feat_all.gather(
                    1, idx_in.unsqueeze(-1).expand(-1, -1, frame_feat_all.shape[-1])
                )
            frame_feat = frame_feat.to(device=device, dtype=text_embed.dtype)
            frame_tokens = model.frame_cond_proj(frame_feat).to(dtype=text_embed.dtype)
            text_embed_cond = torch.cat([text_embed, frame_tokens], dim=1)
        else:
            text_embed_cond = text_embed
        cond = {"text_embed": text_embed_cond}

        model.train()
        if use_wan:
            from src.models.wan_abs_rope import set_wan_frame_indices

            if args.phase1_input_mode == "full":
                set_wan_frame_indices(model, None)
            else:
                # Absolute-time RoPE: pass the true frame indices for each slot in the short sequence.
                set_wan_frame_indices(model, idx_in)
            if args.phase1_input_mode == "full":
                if args.video_interp_mode in ("flow", "sinkhorn"):
                    warper = flow_warper if args.video_interp_mode == "flow" else sinkhorn_warper
                    if warper is None:
                        raise ValueError(f"{args.video_interp_mode} interpolator requested but not loaded")
                    z_seq = torch.zeros((B, T, N, D), device=device, dtype=z_t.dtype)
                    z_seq.scatter_(1, idx_base.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), z_t)
                    latents_seq = unpatchify_tokens(z_seq, args.patch_size, spatial_shape)
                    try:
                        flow_dtype = next(warper.parameters()).dtype
                    except StopIteration:
                        flow_dtype = latents_seq.dtype
                    latents_seq_dtype = latents_seq.dtype
                    latents_seq = latents_seq.to(dtype=flow_dtype)
                    latents_interp, _ = warper.interpolate(latents_seq, idx_base)
                    latents_interp = latents_interp.to(dtype=latents_seq_dtype)
                    latents_t = latents_interp.permute(0, 2, 1, 3, 4)
                else:
                    idx_rep = idx_base.repeat_interleave(N, dim=0)
                    vals_rep = z_t.permute(0, 2, 1, 3).reshape(B * N, idx_base.shape[1], D)
                    z_interp_flat = interpolate_video_from_indices(
                        idx_rep, vals_rep, T, mode=args.video_interp_mode, smooth_kernel=smooth_kernel
                    )
                    z_interp = z_interp_flat.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
                    z_interp = z_interp.scatter(1, idx_base.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), z_t)
                    latents_t = unpatchify_tokens(z_interp, args.patch_size, spatial_shape)
                    latents_t = latents_t.permute(0, 2, 1, 3, 4)
            else:
                # Short-sequence Phase-1: Wan runs only on selected slots (anchors or anchors+midpoints).
                latents_in = unpatchify_tokens(z_t, args.patch_size, spatial_shape)  # [B,L,C,H,W]
                latents_t = latents_in.permute(0, 2, 1, 3, 4)  # [B,C,L,H,W]
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                pred_latents = model(latents_t, t, text_embed_cond).sample
            pred_latents = pred_latents.permute(0, 2, 1, 3, 4)  # [B,L,C,H,W] or [B,T,C,H,W]
            pred_tokens, _ = patchify_latents(pred_latents, args.patch_size)
            if args.phase1_input_mode == "full":
                pred_sel = pred_tokens.gather(1, idx_base.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D))
                loss = (pred_sel - eps).pow(2).mean()
                loss_anchor = loss.detach()
                loss_summary = None
            else:
                loss = (pred_tokens - eps).pow(2).mean()
                # Slot-level diagnostics: anchors vs summary slots (midpoints or pooled summaries).
                is_anchor_slot = (idx_in.unsqueeze(-1) == idx_base.unsqueeze(1)).any(dim=-1)  # [B,L]
                mse_slot = (pred_tokens - eps).pow(2).mean(dim=(2, 3))  # [B,L]
                loss_anchor = mse_slot.masked_select(is_anchor_slot).mean().detach()
                loss_summary = (
                    mse_slot.masked_select(~is_anchor_slot).mean().detach()
                    if bool((~is_anchor_slot).any())
                    else None
                )
        else:
            if args.phase1_input_mode != "full":
                raise ValueError("phase1_input_mode != 'full' currently requires --use_wan 1")
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                eps_hat = model(z_t, t, idx_base, cond, args.T, spatial_shape)
                loss = (eps_hat - eps).pow(2).mean()
            loss_anchor = loss.detach()
            loss_summary = None

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
            writer.add_scalar("train/loss_anchor", float(loss_anchor.item()), step)
            if loss_summary is not None:
                writer.add_scalar("train/loss_summary", float(loss_summary.item()), step)
                writer.add_scalar("train/loss_summary_over_anchor", float(loss_summary.item() / (loss_anchor.item() + 1e-12)), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            writer.add_scalar("train/samples_per_sec", float(B) / max(step_time, 1e-8), step)
            wan_T = int(T if args.phase1_input_mode == "full" else idx_in.shape[1])
            writer.add_scalar("train/wan_frames_per_sec", float(B * wan_T) / max(step_time, 1e-8), step)
            writer.add_scalar("train/fullT_frames_per_sec", float(B * T) / max(step_time, 1e-8), step)
            writer.add_scalar("train/avg_gap", float(gaps.mean().item()), step)
            writer.add_scalar("train/min_gap", float(gaps.min().item()), step)
            writer.add_scalar("train/max_gap", float(gaps.max().item()), step)
            writer.add_scalar("train/slots_L", float(wan_T), step)
            rss_gb = proc.memory_info().rss / (1024**3)
            writer.add_scalar("train/cpu_mem_percent", float(vm.percent), step)
            writer.add_scalar("train/cpu_rss_gb", float(rss_gb), step)
            if log_this:
                max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                writer.add_scalar("train/max_mem_gb", max_mem, step)

        if (
            val_loader is not None
            and int(args.val_every) > 0
            and step % int(args.val_every) == 0
            and step != start_step
        ):
            model.eval()
            losses = []
            losses_anchor = []
            losses_summary = []
            val_it = iter(val_loader)
            for _ in range(int(args.val_batches)):
                try:
                    vbatch = next(val_it)
                except StopIteration:
                    val_it = iter(val_loader)
                    vbatch = next(val_it)
                if use_wan and model_dtype is not None:
                    vlatents = vbatch["latents"].to(device, dtype=model_dtype, non_blocking=True)
                    vtext_embed = vbatch["text_embed"].to(device, dtype=model_dtype, non_blocking=True)
                else:
                    vlatents = vbatch["latents"].to(device, non_blocking=True)
                    vtext_embed = vbatch["text_embed"].to(device, non_blocking=True)
                vtokens, vspatial_shape = patchify_latents(vlatents, args.patch_size)
                VB, VT, VN, VD = vtokens.shape
                vidx_base, _ = sample_fixed_k_indices_uniform_batch(
                    VB, VT, args.K, generator=val_gen, device=device, ensure_endpoints=False
                )
                if args.phase1_input_mode == "short_midpoints":
                    vidx_mid = _midpoint_indices(vidx_base)
                    vidx_in = torch.cat([vidx_base, vidx_mid], dim=1)
                    vidx_in = torch.sort(vidx_in, dim=1).values
                elif args.phase1_input_mode == "short_meanpool":
                    vidx_mid = _midpoint_indices(vidx_base)
                    vidx_in = torch.cat([vidx_base, vidx_mid], dim=1)
                    vidx_in = torch.sort(vidx_in, dim=1).values
                elif args.phase1_input_mode == "short_anchors":
                    vidx_in = vidx_base
                else:
                    vidx_in = vidx_base

                vz0_in = vtokens.gather(1, vidx_in.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, VN, VD)).clone()
                if args.phase1_input_mode == "short_meanpool":
                    pooled = _meanpool_between_anchors(vtokens, vidx_base)  # [B,K-1,N,D]
                    pos_mid = torch.searchsorted(vidx_in.contiguous(), vidx_mid.contiguous())
                    vz0_in.scatter_(1, pos_mid.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, VN, VD), pooled)

                vt = torch.randint(0, args.N_train, (VB,), generator=val_gen, device=device, dtype=torch.long)
                veps = torch.randn_like(vz0_in)
                vz_t = schedule["sqrt_alpha_bar"][vt].view(VB, 1, 1, 1) * vz0_in + schedule[
                    "sqrt_one_minus_alpha_bar"
                ][vt].view(VB, 1, 1, 1) * veps

                if use_wan and hasattr(model, "frame_cond_proj"):
                    vmask = torch.zeros((VB, VT), device=device, dtype=torch.bool)
                    vmask.scatter_(1, vidx_base, True)
                    vframe_feat_all = frame_features_from_mask(vmask, include_time=True)  # [B,T,5]
                    if args.phase1_input_mode == "full":
                        vframe_feat = vframe_feat_all
                    else:
                        vframe_feat = vframe_feat_all.gather(
                            1, vidx_in.unsqueeze(-1).expand(-1, -1, vframe_feat_all.shape[-1])
                        )
                    vframe_feat = vframe_feat.to(device=device, dtype=vtext_embed.dtype)
                    vframe_tokens = model.frame_cond_proj(vframe_feat).to(dtype=vtext_embed.dtype)
                    vtext_embed_cond = torch.cat([vtext_embed, vframe_tokens], dim=1)
                else:
                    vtext_embed_cond = vtext_embed

                if use_wan:
                    from src.models.wan_abs_rope import set_wan_frame_indices

                    if args.phase1_input_mode == "full":
                        set_wan_frame_indices(model, None)
                        raise ValueError("val for phase1_input_mode='full' not implemented in this helper yet")
                    else:
                        set_wan_frame_indices(model, vidx_in)
                        vlatents_in = unpatchify_tokens(vz_t, args.patch_size, vspatial_shape)  # [B,L,C,H,W]
                        vlatents_t = vlatents_in.permute(0, 2, 1, 3, 4)  # [B,C,L,H,W]
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        vpred_latents = model(vlatents_t, vt, vtext_embed_cond).sample
                    vpred_latents = vpred_latents.permute(0, 2, 1, 3, 4)
                    vpred_tokens, _ = patchify_latents(vpred_latents, args.patch_size)
                    vloss = (vpred_tokens - veps).pow(2).mean()

                    is_anchor_slot = (vidx_in.unsqueeze(-1) == vidx_base.unsqueeze(1)).any(dim=-1)  # [B,L]
                    mse_slot = (vpred_tokens - veps).pow(2).mean(dim=(2, 3))  # [B,L]
                    vloss_anchor = mse_slot.masked_select(is_anchor_slot).mean()
                    vloss_summary = (
                        mse_slot.masked_select(~is_anchor_slot).mean() if bool((~is_anchor_slot).any()) else None
                    )
                else:
                    raise ValueError("val requires --use_wan 1")

                losses.append(float(vloss.detach().item()))
                losses_anchor.append(float(vloss_anchor.detach().item()))
                if vloss_summary is not None:
                    losses_summary.append(float(vloss_summary.detach().item()))

            writer.add_scalar("val/loss", float(sum(losses) / max(len(losses), 1)), step)
            writer.add_scalar("val/loss_anchor", float(sum(losses_anchor) / max(len(losses_anchor), 1)), step)
            if losses_summary:
                writer.add_scalar("val/loss_summary", float(sum(losses_summary) / max(len(losses_summary), 1)), step)
            model.train()

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "keypoints_wansynth",
                "T": args.T,
                "K": args.K,
                "phase1_input_mode": str(args.phase1_input_mode),
                "data_dim": D,
                "patch_size": args.patch_size,
                "N_train": args.N_train,
                "schedule": args.schedule,
                "optimizer": str(args.optimizer),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "muon_momentum": float(args.muon_momentum),
                "muon_nesterov": int(args.muon_nesterov),
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
                "wan_frame_cond": int(args.wan_frame_cond),
                "wan_frame_cond_feat_dim": 5,
                "wan_frame_cond_hidden": int(args.wan_frame_cond_hidden),
                "wan_frame_cond_layers": int(args.wan_frame_cond_layers),
                "wan_frame_cond_dropout": float(args.wan_frame_cond_dropout),
                "video_interp_mode": args.video_interp_mode,
                "flow_interp_ckpt": args.flow_interp_ckpt,
                "sinkhorn_win": args.sinkhorn_win,
                "sinkhorn_stride": int(args.sinkhorn_stride),
                "sinkhorn_angles": sinkhorn_angles_str,
                "sinkhorn_shift": args.sinkhorn_shift,
                "sinkhorn_global_mode": args.sinkhorn_global_mode,
                "sinkhorn_phasecorr_mode": str(args.sinkhorn_phasecorr_mode),
                "sinkhorn_phasecorr_level": str(args.sinkhorn_phasecorr_level),
                "sinkhorn_warp_space": args.sinkhorn_warp_space,
                "sinkhorn_iters": args.sinkhorn_iters,
                "sinkhorn_tau": args.sinkhorn_tau,
                "sinkhorn_dustbin": args.sinkhorn_dustbin,
                "sinkhorn_spatial_gamma": float(args.sinkhorn_spatial_gamma),
                "sinkhorn_spatial_radius": int(args.sinkhorn_spatial_radius),
                "sinkhorn_fb_sigma": float(args.sinkhorn_fb_sigma),
                "sinkhorn_d_match": args.sinkhorn_d_match,
                "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
                "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
                "lora_targets": str(args.lora_targets),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta, save_optimizer=bool(args.save_optimizer))

    if int(args.save_final) == 1:
        final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
        meta = {
            "stage": "keypoints_wansynth",
            "T": args.T,
            "K": args.K,
            "phase1_input_mode": str(args.phase1_input_mode),
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
            "wan_frame_cond": int(args.wan_frame_cond),
            "wan_frame_cond_feat_dim": 5,
            "wan_frame_cond_hidden": int(args.wan_frame_cond_hidden),
            "wan_frame_cond_layers": int(args.wan_frame_cond_layers),
            "wan_frame_cond_dropout": float(args.wan_frame_cond_dropout),
            "video_interp_mode": args.video_interp_mode,
            "flow_interp_ckpt": args.flow_interp_ckpt,
            "sinkhorn_win": args.sinkhorn_win,
            "sinkhorn_stride": int(args.sinkhorn_stride),
            "sinkhorn_angles": sinkhorn_angles_str,
            "sinkhorn_shift": args.sinkhorn_shift,
            "sinkhorn_global_mode": args.sinkhorn_global_mode,
            "sinkhorn_phasecorr_mode": str(args.sinkhorn_phasecorr_mode),
            "sinkhorn_phasecorr_level": str(args.sinkhorn_phasecorr_level),
            "sinkhorn_warp_space": args.sinkhorn_warp_space,
            "sinkhorn_iters": args.sinkhorn_iters,
            "sinkhorn_tau": args.sinkhorn_tau,
            "sinkhorn_dustbin": args.sinkhorn_dustbin,
            "sinkhorn_spatial_gamma": float(args.sinkhorn_spatial_gamma),
            "sinkhorn_spatial_radius": int(args.sinkhorn_spatial_radius),
            "sinkhorn_fb_sigma": float(args.sinkhorn_fb_sigma),
            "sinkhorn_d_match": args.sinkhorn_d_match,
            "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
            "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_targets": str(args.lora_targets),
        }
        save_checkpoint(
            final_path, model, optimizer, args.steps, ema, meta=meta, save_optimizer=bool(args.save_optimizer)
        )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
