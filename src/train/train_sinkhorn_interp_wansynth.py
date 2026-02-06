import argparse
import os
import sys
import time
from typing import Optional

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_straightener import LatentStraightener
from src.models.sinkhorn_warp import SinkhornWarpInterpolator
from src.models.wan_backbone import resolve_dtype
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.logging import create_writer
from src.utils.seed import set_seed
from src.utils.video_tokens import patchify_latents


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--train_pattern", type=str, default="", help="Optional train shard pattern override")
    p.add_argument("--val_pattern", type=str, default="", help="Optional validation shard pattern")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--model_dtype", type=str, default="")
    p.add_argument("--min_gap", type=int, default=2)

    # Straightener (S/R).
    p.add_argument("--init_straightener_ckpt", type=str, default="", help="Optional init checkpoint for straightener")
    p.add_argument("--hidden_channels", type=int, default=448)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--use_residual", type=int, default=1)
    p.add_argument("--freeze_straightener", type=int, default=0, help="If set, do not update straightener weights")

    # Sinkhorn matcher hyperparams.
    p.add_argument("--sinkhorn_win", type=int, default=5)
    p.add_argument("--sinkhorn_stride", type=int, default=0, help="Token-window stride (0 uses sinkhorn_win)")
    p.add_argument("--sinkhorn_angles", type=str, default="-10,-5,0,5,10")
    p.add_argument("--sinkhorn_shift", type=int, default=4)
    p.add_argument("--sinkhorn_global_mode", type=str, default="phasecorr", choices=["se2", "phasecorr", "none"])
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_tau", type=float, default=0.05)
    p.add_argument("--sinkhorn_dustbin", type=float, default=-2.0)
    p.add_argument("--sinkhorn_d_match", type=int, default=64)
    p.add_argument("--sinkhorn_spatial_gamma", type=float, default=0.0, help="Spatial prior weight in Sinkhorn logits")
    p.add_argument("--sinkhorn_spatial_radius", type=int, default=0, help="Hard spatial radius (token units) for matching; 0 disables")
    p.add_argument("--sinkhorn_fb_sigma", type=float, default=0.0, help="Forward-backward consistency sigma (token units); 0 disables")

    # Loss weights.
    p.add_argument("--interp_weight", type=float, default=1.0)
    p.add_argument("--recon_weight", type=float, default=0.1)
    p.add_argument("--lin_weight", type=float, default=0.0)

    # Logging/checkpoint.
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/sinkhorn_interp_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/sinkhorn_interp_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")

    # Loader perf.
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=200)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--resampled", type=int, default=1)
    # Terminal output hygiene: tqdm updates can be extremely spammy and can
    # destabilize long foreground runs when stdout/stderr are captured.
    p.add_argument("--tqdm_interval", type=float, default=5.0, help="Minimum seconds between tqdm refreshes")
    return p


def _sample_triplets(B: int, T: int, min_gap: int, gen: torch.Generator, device: torch.device):
    if T <= 2:
        raise ValueError("T must be >= 3 to sample triplets")
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
    t0 = lo
    t1 = hi
    t = t0 + 1 + (torch.rand((B,), generator=gen, device=device) * (gap - 1).float()).floor().long()
    alpha = (t - t0).float() / gap.float().clamp(min=1)
    return t0, t1, t, alpha


def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    # flow: [B,2,H,W] in pixels
    bsz, _, h, w = flow.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=-1).to(dtype=flow.dtype)
    base = base.unsqueeze(0).expand(bsz, -1, -1, -1)
    grid = base + flow.permute(0, 2, 3, 1)
    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1).to(dtype=flow.dtype)


def _warp_fp32(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # Warp in float32 for stability; caller can cast afterward.
    x_f = x.float()
    flow_f = flow.float()
    grid = _flow_to_grid(flow_f)
    out = F.grid_sample(x_f, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out


def _token_features(
    s: torch.Tensor,
    *,
    patch_size: int,
    d_match: int,
) -> tuple[torch.Tensor, int, int]:
    # s: [B,C,H,W] -> [B,Hp,Wp,Dfeat]
    tokens, (hp, wp) = patchify_latents(s.unsqueeze(1), patch_size)
    tok = tokens[:, 0].float()  # [B,N,D]
    B, N, D = tok.shape
    if d_match > 0 and d_match < D:
        if D % d_match != 0:
            raise ValueError(f"sinkhorn_d_match={d_match} must divide token dim {D}")
        tok = tok.view(B, N, d_match, D // d_match).mean(dim=-1)
    tok = F.normalize(tok, dim=-1, eps=1e-6)
    feats = tok.view(B, hp, wp, -1)
    return feats, hp, wp


def _load_straightener_trainable(
    ckpt_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LatentStraightener, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    in_channels = int(meta.get("in_channels", 16))
    hidden_channels = int(meta.get("hidden_channels", 64))
    blocks = int(meta.get("blocks", 2))
    kernel_size = int(meta.get("kernel_size", 3))
    use_residual = bool(meta.get("use_residual", True))
    model = LatentStraightener(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        blocks=blocks,
        kernel_size=kernel_size,
        use_residual=use_residual,
    )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, meta


@torch.no_grad()
def _eval_model(
    model: LatentStraightener,
    matcher: SinkhornWarpInterpolator,
    data_pattern: str,
    *,
    T: int,
    batch: int,
    patch_size: int,
    d_match: int,
    num_batches: int,
    min_gap: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    model_dtype: torch.dtype,
    num_workers: int,
    shuffle_buffer: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    resampled: bool,
    seed: int,
) -> dict:
    model_was_training = model.training
    model.eval()
    loader = create_wan_synth_dataloader(
        data_pattern,
        batch_size=batch,
        num_workers=num_workers,
        shuffle_buffer=shuffle_buffer,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        shuffle=False,
        shardshuffle=False,
        keep_text_embed=False,
        keep_text=False,
        resampled=resampled,
        seed=seed,
    )
    it = iter(loader)
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)

    err_sinkhorn = []
    err_lerp = []
    err_straight = []
    for _ in range(int(num_batches)):
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(loader)
            batch_data = next(it)
        latents = batch_data["latents"].to(device, dtype=model_dtype, non_blocking=True)
        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T0, _, H, W = latents.shape
        if T0 != T:
            raise ValueError(f"T mismatch: batch={T0} args={T}")

        t0, t1, t, alpha = _sample_triplets(B, T, min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]
        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=torch.float32)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            s0 = model.encode(z0)
            s1 = model.encode(z1)
            s_lerp = (1.0 - alpha4.to(dtype=s0.dtype)) * s0 + alpha4.to(dtype=s0.dtype) * s1
        f0, hp, wp = _token_features(s0, patch_size=patch_size, d_match=d_match)
        f1, _, _ = _token_features(s1, patch_size=patch_size, d_match=d_match)
        flow01_tok, flow10_tok, conf01_tok, conf10_tok, _, _, _, _ = matcher.compute_bidirectional_flow_and_confs_batch(f0, f1)

        flow01 = (
            F.interpolate(flow01_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
            * float(patch_size)
        )
        flow10 = (
            F.interpolate(flow10_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
            * float(patch_size)
        )
        conf01 = F.interpolate(conf01_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(0.0, 1.0)
        conf10 = F.interpolate(conf10_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(0.0, 1.0)

        # Warp and blend in straightened space; decode back to z.
        s0_w = _warp_fp32(s0, -flow01 * alpha4)
        s1_w = _warp_fp32(s1, -flow10 * (1.0 - alpha4))
        conf0_w = _warp_fp32(conf01, -flow01 * alpha4)
        conf1_w = _warp_fp32(conf10, -flow10 * (1.0 - alpha4))
        w0 = (1.0 - alpha4) * conf0_w
        w1 = alpha4 * conf1_w
        denom = w0 + w1
        s_mix = (w0 * s0_w + w1 * s1_w) / denom.clamp_min(1e-6)
        s_lin = (1.0 - alpha4) * s0_w + alpha4 * s1_w
        s_t = torch.where(denom > 1e-6, s_mix, s_lin)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            z_hat = model.decode(s_t.to(dtype=s0.dtype))
            z_straight = model.decode(s_lerp)

        z_hat_f = z_hat.float()
        zt_f = zt.float()
        z_lerp = z0.float() * (1.0 - alpha4) + z1.float() * alpha4
        err_sinkhorn.append(((z_hat_f - zt_f) ** 2).mean(dim=(1, 2, 3)).cpu())
        err_lerp.append(((z_lerp - zt_f) ** 2).mean(dim=(1, 2, 3)).cpu())
        err_straight.append(((z_straight.float() - zt_f) ** 2).mean(dim=(1, 2, 3)).cpu())

    if model_was_training:
        model.train()
    mse_sink = torch.cat(err_sinkhorn, dim=0).mean().item()
    mse_lerp = torch.cat(err_lerp, dim=0).mean().item()
    mse_straight = torch.cat(err_straight, dim=0).mean().item()
    return {
        "val_sinkhorn_mse": float(mse_sink),
        "val_lerp_mse": float(mse_lerp),
        "val_straight_lerp_mse": float(mse_straight),
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth sinkhorn interpolator training.")

    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    train_pattern = args.train_pattern or args.data_pattern
    loader = create_wan_synth_dataloader(
        train_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.persistent_workers),
        pin_memory=bool(args.pin_memory),
        shuffle=True,
        shardshuffle=True,
        keep_text_embed=False,
        keep_text=False,
        resampled=bool(args.resampled),
        seed=args.seed,
    )
    it = iter(loader)

    model_dtype = resolve_dtype(args.model_dtype) or get_autocast_dtype()

    # Prime one batch to infer channel dims.
    batch0 = next(it)
    latents0 = batch0["latents"].to(device, dtype=model_dtype, non_blocking=True)
    if latents0.dim() != 5:
        raise ValueError("latents must be [B,T,C,H,W]")
    _, T0, C0, _, _ = latents0.shape
    if T0 != args.T:
        raise ValueError(f"T mismatch: batch={T0} args={args.T}")

    if args.init_straightener_ckpt:
        model, init_meta = _load_straightener_trainable(args.init_straightener_ckpt, device=device, dtype=model_dtype)
        C0 = int(init_meta.get("in_channels", C0))
    else:
        model = LatentStraightener(
            in_channels=C0,
            hidden_channels=args.hidden_channels,
            blocks=args.blocks,
            kernel_size=args.kernel_size,
            use_residual=bool(args.use_residual),
        ).to(device=device, dtype=model_dtype)
        init_meta = {}

    if bool(args.freeze_straightener):
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    angles = [float(x) for x in str(args.sinkhorn_angles).split(",") if x.strip()]
    matcher = SinkhornWarpInterpolator(
        in_channels=C0,
        patch_size=args.patch_size,
        win_size=args.sinkhorn_win,
        win_stride=args.sinkhorn_stride,
        global_mode=args.sinkhorn_global_mode,
        angles_deg=angles,
        shift_range=args.sinkhorn_shift,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tau=args.sinkhorn_tau,
        dustbin_logit=args.sinkhorn_dustbin,
        spatial_gamma=args.sinkhorn_spatial_gamma,
        spatial_radius=args.sinkhorn_spatial_radius,
        fb_sigma=args.sinkhorn_fb_sigma,
        d_match=0,
        straightener=None,
        warp_space="z",
    ).to(device=device)

    train_params = [p for p in model.parameters() if p.requires_grad]
    if not train_params:
        raise RuntimeError("No trainable parameters. Disable --freeze_straightener or add trainable components.")
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt, ema=None, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 23)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Use a conservative refresh cadence to avoid huge log spam when output is captured.
    pbar = tqdm(
        range(start_step, args.steps),
        dynamic_ncols=True,
        mininterval=max(0.5, float(args.tqdm_interval)),
        file=sys.stderr,
    )
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
        latents = batch["latents"].to(device, dtype=model_dtype, non_blocking=True)
        B, T, _, H, W = latents.shape
        t0, t1, t, alpha = _sample_triplets(B, T, args.min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]
        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=torch.float32)

        # Encode endpoints/target.
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            s0 = model.encode(z0)
            s1 = model.encode(z1)
            st = model.encode(zt)

        # Compute correspondence in float32 (features + sinkhorn).
        with torch.cuda.amp.autocast(enabled=False):
            f0, hp, wp = _token_features(s0, patch_size=args.patch_size, d_match=args.sinkhorn_d_match)
            f1, _, _ = _token_features(s1, patch_size=args.patch_size, d_match=args.sinkhorn_d_match)
            flow01_tok, flow10_tok, conf01_tok, conf10_tok, conf01_dust, conf10_dust, fb_err01, fb_err10 = (
                matcher.compute_bidirectional_flow_and_confs_batch(f0, f1)
            )

            flow01 = (
                F.interpolate(flow01_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
                * float(args.patch_size)
            )
            flow10 = (
                F.interpolate(flow10_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
                * float(args.patch_size)
            )
            conf01 = F.interpolate(conf01_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(0.0, 1.0)
            conf10 = F.interpolate(conf10_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(0.0, 1.0)

            # Warp and blend in straightened space; decode back to z.
            s0_w = _warp_fp32(s0, -flow01 * alpha4)
            s1_w = _warp_fp32(s1, -flow10 * (1.0 - alpha4))
            conf0_w = _warp_fp32(conf01, -flow01 * alpha4)
            conf1_w = _warp_fp32(conf10, -flow10 * (1.0 - alpha4))
            w0 = (1.0 - alpha4) * conf0_w
            w1 = alpha4 * conf1_w
            denom = w0 + w1
            s_mix = (w0 * s0_w + w1 * s1_w) / denom.clamp_min(1e-6)
            s_lin = (1.0 - alpha4) * s0_w + alpha4 * s1_w
            s_t = torch.where(denom > 1e-6, s_mix, s_lin)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            z_hat = model.decode(s_t.to(dtype=s0.dtype))
            z_recon = model.decode(st)

        # Losses in float32.
        interp_loss = F.mse_loss(z_hat.float(), zt.float())
        recon_loss = F.mse_loss(z_recon.float(), zt.float())
        lin_loss = F.mse_loss(((1.0 - alpha4) * s0.float() + alpha4 * s1.float()), st.float())

        loss = float(args.interp_weight) * interp_loss + float(args.recon_weight) * recon_loss + float(args.lin_weight) * lin_loss

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if log_this:
            pbar.set_description(f"loss {loss.item():.4f}")
            step_time = time.perf_counter() - step_start
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/interp_mse", interp_loss.item(), step)
            writer.add_scalar("train/recon_mse", recon_loss.item(), step)
            writer.add_scalar("train/lin_mse", lin_loss.item(), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            writer.add_scalar("train/samples_per_sec", float(B) / max(step_time, 1e-8), step)
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            writer.add_scalar("train/max_mem_gb", max_mem, step)
            with torch.no_grad():
                # Baseline for this batch: plain z-space LERP between endpoints.
                z_lerp = z0.float() * (1.0 - alpha4) + z1.float() * alpha4
                lerp_mse = F.mse_loss(z_lerp, zt.float()).item()
                writer.add_scalar("train/lerp_mse", lerp_mse, step)
                writer.add_scalar("train/sinkhorn_minus_lerp_mse", float(interp_loss.item()) - lerp_mse, step)

                # Baseline in straightened space: decode((1-a)S(z0)+aS(z1)).
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    s_lerp = (1.0 - alpha4.to(dtype=s0.dtype)) * s0.detach() + alpha4.to(dtype=s0.dtype) * s1.detach()
                    z_straight = model.decode(s_lerp)
                straight_mse = F.mse_loss(z_straight.float(), zt.float()).item()
                writer.add_scalar("train/straight_lerp_mse", straight_mse, step)
                writer.add_scalar(
                    "train/sinkhorn_minus_straight_mse",
                    float(interp_loss.item()) - straight_mse,
                    step,
                )

                # Matching diagnostics. conf_* are (1 - dustbin_mass) on the token grid.
                writer.add_scalar("match/conf01_tok_mean", float(conf01_tok.mean().item()), step)
                writer.add_scalar("match/conf10_tok_mean", float(conf10_tok.mean().item()), step)
                writer.add_scalar("match/conf01_dust_mean", float(conf01_dust.mean().item()), step)
                writer.add_scalar("match/conf10_dust_mean", float(conf10_dust.mean().item()), step)
                # Approximate fb factor as combined / dust, guarded.
                fb01_fac = torch.where(conf01_dust > 1e-6, conf01_tok / conf01_dust, torch.zeros_like(conf01_tok))
                fb10_fac = torch.where(conf10_dust > 1e-6, conf10_tok / conf10_dust, torch.zeros_like(conf10_tok))
                writer.add_scalar("match/conf01_fb_mean", float(fb01_fac.mean().item()), step)
                writer.add_scalar("match/conf10_fb_mean", float(fb10_fac.mean().item()), step)
                writer.add_scalar("match/fb_err01_tok_mean", float(fb_err01.mean().item()), step)
                writer.add_scalar("match/fb_err10_tok_mean", float(fb_err10.mean().item()), step)

                conf_min_tok = torch.minimum(conf01_tok, conf10_tok)
                writer.add_scalar("match/conf_min_tok_mean", float(conf_min_tok.mean().item()), step)
                writer.add_scalar("match/dustbin01_tok_mean", float((1.0 - conf01_tok).mean().item()), step)
                writer.add_scalar("match/dustbin10_tok_mean", float((1.0 - conf10_tok).mean().item()), step)
                writer.add_scalar("match/conf_min_tok_frac_lt_0p1", float((conf_min_tok < 0.1).float().mean().item()), step)

                # Flow magnitude on token grid and latent grid (pixels).
                flow01_tok_mag = torch.linalg.norm(flow01_tok.float(), dim=-1)  # [B,Hp,Wp]
                flow10_tok_mag = torch.linalg.norm(flow10_tok.float(), dim=-1)
                writer.add_scalar("match/flow01_tok_mag_mean", float(flow01_tok_mag.mean().item()), step)
                writer.add_scalar("match/flow10_tok_mag_mean", float(flow10_tok_mag.mean().item()), step)
                flow01_lat_mag = torch.linalg.norm(flow01.float(), dim=1)  # [B,H,W]
                flow10_lat_mag = torch.linalg.norm(flow10.float(), dim=1)
                writer.add_scalar("match/flow01_lat_mag_mean", float(flow01_lat_mag.mean().item()), step)
                writer.add_scalar("match/flow10_lat_mag_mean", float(flow10_lat_mag.mean().item()), step)

                # Blend/fallback behavior.
                denom_mean = float(denom.mean().item())
                denom_good_frac = float((denom > 1e-6).float().mean().item())
                writer.add_scalar("blend/denom_mean", denom_mean, step)
                writer.add_scalar("blend/denom_gt_eps_frac", denom_good_frac, step)

                # Global alignment diagnostics (non-differentiable); useful to detect “always zero” motion.
                try:
                    if args.sinkhorn_global_mode == "phasecorr":
                        f0_score = f0.mean(dim=-1).detach()
                        f1_score = f1.mean(dim=-1).detach()
                        theta_dbg, dx_dbg, dy_dbg = matcher._phasecorr_se2_batch(f0_score, f1_score)
                    elif args.sinkhorn_global_mode == "se2":
                        theta_dbg, dx_dbg, dy_dbg = matcher._estimate_global_se2_batch(f0.detach(), f1.detach())
                    else:
                        theta_dbg = torch.zeros((B,), device=device, dtype=torch.float32)
                        dx_dbg = torch.zeros((B,), device=device, dtype=torch.float32)
                        dy_dbg = torch.zeros((B,), device=device, dtype=torch.float32)
                    theta_deg = theta_dbg.float() * (180.0 / math.pi)
                    writer.add_scalar("global/theta_deg_abs_mean", float(theta_deg.abs().mean().item()), step)
                    writer.add_scalar("global/dx_abs_mean", float(dx_dbg.float().abs().mean().item()), step)
                    writer.add_scalar("global/dy_abs_mean", float(dy_dbg.float().abs().mean().item()), step)
                except Exception:
                    # Diagnostics should never crash training.
                    pass

        if args.val_pattern and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            metrics = _eval_model(
                model,
                matcher,
                args.val_pattern,
                T=args.T,
                batch=args.batch,
                patch_size=args.patch_size,
                d_match=args.sinkhorn_d_match,
                num_batches=args.val_batches,
                min_gap=args.min_gap,
                device=device,
                autocast_dtype=autocast_dtype,
                model_dtype=model_dtype,
                num_workers=max(0, int(args.num_workers)),
                shuffle_buffer=max(1, int(args.shuffle_buffer)),
                prefetch_factor=max(1, int(args.prefetch_factor)),
                persistent_workers=False,
                pin_memory=False,
                resampled=False,
                seed=args.seed,
            )
            writer.add_scalar("val/sinkhorn_mse", metrics["val_sinkhorn_mse"], step)
            writer.add_scalar("val/lerp_mse", metrics["val_lerp_mse"], step)
            writer.add_scalar("val/straight_lerp_mse", metrics["val_straight_lerp_mse"], step)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "sinkhorn_interp_wansynth",
                "T": args.T,
                "patch_size": args.patch_size,
                "min_gap": args.min_gap,
                "in_channels": int(model.in_channels),
                "hidden_channels": int(model.hidden_channels),
                "blocks": int(model.blocks),
                "kernel_size": int(model.kernel_size),
                "use_residual": bool(model.use_residual),
                "interp_weight": float(args.interp_weight),
                "recon_weight": float(args.recon_weight),
                "lin_weight": float(args.lin_weight),
                "sinkhorn_win": args.sinkhorn_win,
                "sinkhorn_stride": int(args.sinkhorn_stride),
                "sinkhorn_angles": args.sinkhorn_angles,
                "sinkhorn_shift": args.sinkhorn_shift,
                "sinkhorn_global_mode": args.sinkhorn_global_mode,
                "sinkhorn_iters": args.sinkhorn_iters,
                "sinkhorn_tau": float(args.sinkhorn_tau),
                "sinkhorn_dustbin": float(args.sinkhorn_dustbin),
                "sinkhorn_spatial_gamma": float(args.sinkhorn_spatial_gamma),
                "sinkhorn_spatial_radius": int(args.sinkhorn_spatial_radius),
                "sinkhorn_fb_sigma": float(args.sinkhorn_fb_sigma),
                "sinkhorn_d_match": int(args.sinkhorn_d_match),
                "model_dtype": str(model_dtype).replace("torch.", ""),
                "init_straightener_ckpt": args.init_straightener_ckpt,
                "init_meta": init_meta,
            }
            save_checkpoint(ckpt_path, model, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "sinkhorn_interp_wansynth",
        "T": args.T,
        "patch_size": args.patch_size,
        "min_gap": args.min_gap,
        "in_channels": int(model.in_channels),
        "hidden_channels": int(model.hidden_channels),
        "blocks": int(model.blocks),
        "kernel_size": int(model.kernel_size),
        "use_residual": bool(model.use_residual),
        "interp_weight": float(args.interp_weight),
        "recon_weight": float(args.recon_weight),
        "lin_weight": float(args.lin_weight),
        "sinkhorn_win": args.sinkhorn_win,
        "sinkhorn_stride": int(args.sinkhorn_stride),
        "sinkhorn_angles": args.sinkhorn_angles,
        "sinkhorn_shift": args.sinkhorn_shift,
        "sinkhorn_global_mode": args.sinkhorn_global_mode,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tau": float(args.sinkhorn_tau),
        "sinkhorn_dustbin": float(args.sinkhorn_dustbin),
        "sinkhorn_spatial_gamma": float(args.sinkhorn_spatial_gamma),
        "sinkhorn_spatial_radius": int(args.sinkhorn_spatial_radius),
        "sinkhorn_fb_sigma": float(args.sinkhorn_fb_sigma),
        "sinkhorn_d_match": int(args.sinkhorn_d_match),
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "init_straightener_ckpt": args.init_straightener_ckpt,
        "init_meta": init_meta,
    }
    save_checkpoint(final_path, model, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
