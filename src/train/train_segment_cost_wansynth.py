import argparse
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.encoders import TextConditionEncoder
from src.models.latent_straightener import load_latent_straightener
from src.models.segment_cost import SegmentCostPredictor
from src.models.wan_backbone import load_wan_transformer, resolve_dtype
from src.selection.epiplexity_dp import build_snr_weights, sample_timesteps_log_snr
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.logging import create_writer
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train_pattern", type=str, required=True)
    p.add_argument("--val_pattern", type=str, default="")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    # Diffusion schedule (teacher noise).
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--snr_min", type=float, default=0.1)
    p.add_argument("--snr_max", type=float, default=10.0)
    p.add_argument("--snr_gamma", type=float, default=1.0)
    p.add_argument("--t_steps", type=int, default=16)

    # Segment sampling.
    p.add_argument("--min_gap", type=int, default=2)

    # Interpolator used to create the student clean sequence.
    p.add_argument("--interp_mode", type=str, default="straight_lerp", choices=["lerp", "straight_lerp"])
    p.add_argument("--straightener_ckpt", type=str, default="")
    p.add_argument("--straightener_dtype", type=str, default="")

    # Teacher model (frozen).
    p.add_argument("--wan_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--wan_subfolder", type=str, default="transformer")
    p.add_argument("--wan_dtype", type=str, default="bf16")
    p.add_argument("--wan_attn", type=str, default="sagesla", choices=["default", "sla", "sagesla"])
    p.add_argument("--sla_topk", type=float, default=0.07)

    # D_phi model.
    p.add_argument("--d_cond", type=int, default=256)
    p.add_argument(
        "--seg_feat_dim",
        type=int,
        default=4,
        help="Segment feature dim; default is [i_norm, j_norm, gap_norm, t_norm].",
    )
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # I/O and logging.
    p.add_argument("--log_dir", type=str, default="runs/segment_cost_wansynth")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/segment_cost_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--val_batches", type=int, default=25)
    p.add_argument("--resume", type=str, default="")

    # Loader perf.
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=200)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--resampled", type=int, default=1)
    return p


def _sample_segments(B: int, T: int, min_gap: int, gen: torch.Generator, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if T < 3:
        raise ValueError("T must be >= 3")
    if min_gap < 2:
        min_gap = 2
    t0 = torch.randint(0, T - 1, (B,), generator=gen, device=device)
    t1 = torch.randint(0, T - 1, (B,), generator=gen, device=device)
    lo = torch.minimum(t0, t1)
    hi = torch.maximum(t0, t1)
    gap = hi - lo
    while True:
        bad = gap < min_gap
        if not bool(bad.any()):
            break
        n_bad = int(bad.sum().item())
        a = torch.randint(0, T - 1, (n_bad,), generator=gen, device=device)
        b = torch.randint(0, T - 1, (n_bad,), generator=gen, device=device)
        lo_new = torch.minimum(a, b)
        hi_new = torch.maximum(a, b)
        gap_new = hi_new - lo_new
        lo[bad] = lo_new
        hi[bad] = hi_new
        gap[bad] = gap_new
    return lo, hi


def _build_student_latents_straight_lerp(
    latents: torch.Tensor,
    t0: torch.Tensor,
    t1: torch.Tensor,
    *,
    straightener: torch.nn.Module,
) -> torch.Tensor:
    # latents: [B,T,C,H,W]
    B, T, C, H, W = latents.shape
    idx = torch.arange(B, device=latents.device)
    z0 = latents[idx, t0]  # [B,C,H,W]
    z1 = latents[idx, t1]
    with torch.no_grad():
        s0 = straightener.encode(z0)
        s1 = straightener.encode(z1)

        # Broadcast alphas across time.
        denom = (t1 - t0).float().clamp_min(1.0).view(B, 1, 1, 1, 1)
        tt = torch.arange(T, device=latents.device, dtype=torch.float32).view(1, T, 1, 1, 1)
        alpha = (tt - t0.float().view(B, 1, 1, 1, 1)) / denom
        alpha = alpha.clamp(0.0, 1.0).to(dtype=s0.dtype)

        s0_rep = s0.unsqueeze(1)  # [B,1,C,H,W]
        s1_rep = s1.unsqueeze(1)
        s_interp = (1.0 - alpha) * s0_rep + alpha * s1_rep  # [B,T,C,H,W]
        z_interp = straightener.decode(s_interp.view(B * T, C, H, W)).view(B, T, C, H, W)

    t_grid = torch.arange(T, device=latents.device).view(1, T)
    interior = (t_grid > t0.view(B, 1)) & (t_grid < t1.view(B, 1))  # [B,T]
    mask = interior.view(B, T, 1, 1, 1)
    return torch.where(mask, z_interp.to(dtype=latents.dtype), latents)


def _build_student_latents_lerp(latents: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    B, T, C, H, W = latents.shape
    idx = torch.arange(B, device=latents.device)
    z0 = latents[idx, t0]
    z1 = latents[idx, t1]
    denom = (t1 - t0).float().clamp_min(1.0).view(B, 1, 1, 1, 1)
    tt = torch.arange(T, device=latents.device, dtype=torch.float32).view(1, T, 1, 1, 1)
    alpha = (tt - t0.float().view(B, 1, 1, 1, 1)) / denom
    alpha = alpha.clamp(0.0, 1.0).to(dtype=latents.dtype)
    z0_rep = z0.unsqueeze(1)
    z1_rep = z1.unsqueeze(1)
    z_interp = (1.0 - alpha) * z0_rep + alpha * z1_rep
    t_grid = torch.arange(T, device=latents.device).view(1, T)
    interior = (t_grid > t0.view(B, 1)) & (t_grid < t1.view(B, 1))
    mask = interior.view(B, T, 1, 1, 1)
    return torch.where(mask, z_interp, latents)


@torch.no_grad()
def _compute_teacher_student_cost(
    teacher,
    latents: torch.Tensor,
    student_latents: torch.Tensor,
    text_embed: torch.Tensor,
    t_noise: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
    *,
    t0: torch.Tensor,
    t1: torch.Tensor,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    # latents/student_latents: [B,T,C,H,W]
    B, T, C, H, W = latents.shape
    eps = torch.randn((B, T, C, H, W), device=latents.device, dtype=torch.float32)
    sqrt_ab = schedule["sqrt_alpha_bar"][t_noise].view(B, 1, 1, 1, 1).float()
    sqrt_omb = schedule["sqrt_one_minus_alpha_bar"][t_noise].view(B, 1, 1, 1, 1).float()

    x_teacher = sqrt_ab * latents.float() + sqrt_omb * eps
    x_student = sqrt_ab * student_latents.float() + sqrt_omb * eps

    x_cat = torch.cat([x_teacher, x_student], dim=0).permute(0, 2, 1, 3, 4).contiguous()
    t_cat = torch.cat([t_noise, t_noise], dim=0)
    txt_cat = torch.cat([text_embed, text_embed], dim=0)

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        pred = teacher(x_cat, t_cat, txt_cat).sample  # [2B,C,T,H,W]

    pred_teacher = pred[:B]
    pred_student = pred[B:]
    diff = (pred_teacher - pred_student).float().pow(2)  # [B,C,T,H,W]

    # Mask interior frames only.
    t_grid = torch.arange(T, device=latents.device).view(1, T)
    interior = (t_grid > t0.view(B, 1)) & (t_grid < t1.view(B, 1))  # [B,T]
    mask = interior.view(B, 1, T, 1, 1).to(dtype=diff.dtype)
    diff = diff * mask
    # Important: cost should be additive across segments for DP. We therefore scale with the number
    # of interior frames instead of averaging it out.
    denom = float(C * H * W)
    cost = diff.sum(dim=(1, 2, 3, 4)) / denom
    return cost  # [B]


@torch.no_grad()
def _eval(
    dphi: torch.nn.Module,
    teacher,
    straightener: Optional[torch.nn.Module],
    args,
    schedule: Dict[str, torch.Tensor],
    t_candidates: torch.Tensor,
    t_probs: torch.Tensor,
    *,
    autocast_dtype: torch.dtype,
    device: torch.device,
) -> float:
    if not args.val_pattern:
        return float("nan")
    dphi_was_training = dphi.training
    dphi.eval()
    loader = create_wan_synth_dataloader(
        args.val_pattern,
        batch_size=args.batch,
        num_workers=max(0, int(args.num_workers)),
        shuffle_buffer=max(1, int(args.shuffle_buffer)),
        prefetch_factor=max(1, int(args.prefetch_factor)),
        persistent_workers=False,
        pin_memory=False,
        shuffle=False,
        shardshuffle=False,
        keep_text_embed=True,
        keep_text=False,
        resampled=False,
        seed=args.seed + 999,
    )
    it = iter(loader)
    losses = []
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 12345)
    for _ in range(int(args.val_batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        latents = batch["latents"].to(device, non_blocking=True)
        text_embed = batch["text_embed"].to(device, non_blocking=True)
        B, T, _, _, _ = latents.shape
        t0, t1 = _sample_segments(B, T, int(args.min_gap), gen, device)
        if args.interp_mode == "straight_lerp":
            assert straightener is not None
            student_latents = _build_student_latents_straight_lerp(latents, t0, t1, straightener=straightener)
        else:
            student_latents = _build_student_latents_lerp(latents, t0, t1)
        t_noise = t_candidates[torch.multinomial(t_probs, B, replacement=True, generator=gen)]
        cost = _compute_teacher_student_cost(
            teacher,
            latents,
            student_latents,
            text_embed,
            t_noise,
            schedule,
            t0=t0,
            t1=t1,
            autocast_dtype=autocast_dtype,
        )
        denomT = float(max(1, args.T - 1))
        t_norm = t_noise.float() / float(max(1, int(args.N_train) - 1))
        seg_feat = torch.stack(
            [
                t0.float() / denomT,
                t1.float() / denomT,
                (t1 - t0).float() / denomT,
                t_norm,
            ],
            dim=-1,
        ).to(device=device, dtype=torch.float32)
        seg_feat = seg_feat.unsqueeze(1)  # [B,1,4]
        pred = dphi({"text_embed": text_embed}, seg_feat).squeeze(1).float()
        loss = torch.mean((pred - cost.float()) ** 2)
        losses.append(loss.item())
    if dphi_was_training:
        dphi.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth segment-cost training.")

    autocast_dtype = get_autocast_dtype()

    # Data loader.
    loader = create_wan_synth_dataloader(
        args.train_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
        shuffle=True,
        shardshuffle=True,
        keep_text_embed=True,
        keep_text=False,
        resampled=bool(args.resampled),
        seed=args.seed,
    )
    it = iter(loader)

    # Prime one batch for dims.
    batch0 = next(it)
    lat0 = batch0["latents"]
    txt0 = batch0.get("text_embed")
    if txt0 is None:
        raise RuntimeError("text_embed missing from dataset")
    if lat0.dim() != 5:
        raise ValueError("latents must be [B,T,C,H,W]")
    if int(lat0.shape[1]) != int(args.T):
        raise ValueError(f"T mismatch: batch={lat0.shape[1]} args={args.T}")
    text_dim = int(txt0.shape[-1])

    # Frozen teacher.
    wan_dtype = resolve_dtype(args.wan_dtype)
    teacher = load_wan_transformer(args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device)
    if args.wan_attn != "default":
        from src.models.wan_sla import apply_wan_sla

        use_bf16 = wan_dtype == torch.bfloat16 if wan_dtype is not None else True
        apply_wan_sla(teacher, topk=float(args.sla_topk), attention_type=str(args.wan_attn), use_bf16=use_bf16)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Optional straightener for straight-LERP interpolation.
    straightener = None
    straightener_meta = {}
    if args.interp_mode == "straight_lerp":
        if not args.straightener_ckpt:
            raise ValueError("interp_mode=straight_lerp requires --straightener_ckpt")
        s_dtype = resolve_dtype(args.straightener_dtype) or wan_dtype
        straightener, straightener_meta = load_latent_straightener(args.straightener_ckpt, device=device, dtype=s_dtype)

    # D_phi model (trainable).
    cond_encoder = TextConditionEncoder(text_dim=text_dim, d_cond=int(args.d_cond))
    dphi = SegmentCostPredictor(
        d_cond=int(args.d_cond),
        seg_feat_dim=int(args.seg_feat_dim),
        hidden_dim=int(args.hidden_dim),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        cond_encoder=cond_encoder,
    ).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(dphi.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Schedule + timestep sampling distribution.
    betas = make_beta_schedule(args.schedule, int(args.N_train)).to(device)
    sched = make_alpha_bars(betas)
    snr, weights = build_snr_weights(args.schedule, int(args.N_train), args.snr_min, args.snr_max, args.snr_gamma)
    t_idx = sample_timesteps_log_snr(snr.to(device), int(args.t_steps)).to(device)
    w = weights.to(device)[t_idx].float()
    t_probs = (w / w.sum().clamp_min(1e-8)).to(device)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, dphi, opt, ema=None, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 17)

    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        step_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        latents = batch["latents"].to(device, non_blocking=True)
        text_embed = batch["text_embed"].to(device, non_blocking=True)
        B, T, _, _, _ = latents.shape
        if T != args.T:
            raise ValueError(f"T mismatch: batch={T} args={args.T}")

        t0, t1 = _sample_segments(B, T, int(args.min_gap), gen, device)
        if args.interp_mode == "straight_lerp":
            assert straightener is not None
            student_latents = _build_student_latents_straight_lerp(latents, t0, t1, straightener=straightener)
        else:
            student_latents = _build_student_latents_lerp(latents, t0, t1)

        # Sample diffusion step from the SNR-weighted distribution.
        t_noise = t_idx[torch.multinomial(t_probs, B, replacement=True)]
        cost = _compute_teacher_student_cost(
            teacher,
            latents,
            student_latents,
            text_embed,
            t_noise,
            sched,
            t0=t0,
            t1=t1,
            autocast_dtype=autocast_dtype,
        )

        denomT = float(max(1, args.T - 1))
        t_norm = t_noise.float() / float(max(1, int(args.N_train) - 1))
        seg_feat = torch.stack(
            [
                t0.float() / denomT,
                t1.float() / denomT,
                (t1 - t0).float() / denomT,
                t_norm,
            ],
            dim=-1,
        ).to(device=device, dtype=torch.float32)
        seg_feat = seg_feat.unsqueeze(1)

        pred = dphi({"text_embed": text_embed}, seg_feat).squeeze(1).float()
        loss = torch.mean((pred - cost.float()) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dphi.parameters(), float(args.grad_clip))
        opt.step()

        step_time = time.perf_counter() - step_start
        if step % 50 == 0:
            pbar.set_description(f"loss {loss.item():.4f} cost {cost.mean().item():.4f} {step_time:.2f}s")
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/cost_mean", cost.mean().item(), step)
            writer.add_scalar("train/pred_mean", pred.mean().item(), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            writer.add_scalar("train/samples_per_sec", float(B) / max(step_time, 1e-8), step)

        if args.val_pattern and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            val_loss = _eval(
                dphi,
                teacher,
                straightener,
                args,
                sched,
                t_idx,
                t_probs,
                autocast_dtype=autocast_dtype,
                device=device,
            )
            writer.add_scalar("val/loss", val_loss, step)
            print(f"[VAL step={step}] loss={val_loss:.6f}", file=sys.stderr, flush=True)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "segment_cost_wansynth",
                "T": int(args.T),
                "interp_mode": str(args.interp_mode),
                "cost_mode": "sum_over_interior_frames",
                "straightener_ckpt": args.straightener_ckpt,
                "straightener_meta": straightener_meta,
                "wan_repo": args.wan_repo,
                "wan_subfolder": args.wan_subfolder,
                "wan_dtype": args.wan_dtype,
                "wan_attn": str(args.wan_attn),
                "sla_topk": float(args.sla_topk),
                "schedule": str(args.schedule),
                "N_train": int(args.N_train),
                "snr_min": float(args.snr_min),
                "snr_max": float(args.snr_max),
                "snr_gamma": float(args.snr_gamma),
                "t_steps": int(args.t_steps),
                "t_idx": t_idx.detach().cpu().tolist(),
                "d_cond": int(args.d_cond),
                "seg_feat_dim": int(args.seg_feat_dim),
                "hidden_dim": int(args.hidden_dim),
                "n_layers": int(args.n_layers),
                "dropout": float(args.dropout),
                "text_dim": int(text_dim),
                "min_gap": int(args.min_gap),
            }
            save_checkpoint(ckpt_path, dphi, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "segment_cost_wansynth",
        "T": int(args.T),
        "interp_mode": str(args.interp_mode),
        "cost_mode": "sum_over_interior_frames",
        "straightener_ckpt": args.straightener_ckpt,
        "straightener_meta": straightener_meta,
        "wan_repo": args.wan_repo,
        "wan_subfolder": args.wan_subfolder,
        "wan_dtype": args.wan_dtype,
        "wan_attn": str(args.wan_attn),
        "sla_topk": float(args.sla_topk),
        "schedule": str(args.schedule),
        "N_train": int(args.N_train),
        "snr_min": float(args.snr_min),
        "snr_max": float(args.snr_max),
        "snr_gamma": float(args.snr_gamma),
        "t_steps": int(args.t_steps),
        "t_idx": t_idx.detach().cpu().tolist(),
        "d_cond": int(args.d_cond),
        "seg_feat_dim": int(args.seg_feat_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "dropout": float(args.dropout),
        "text_dim": int(text_dim),
        "min_gap": int(args.min_gap),
    }
    save_checkpoint(final_path, dphi, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
