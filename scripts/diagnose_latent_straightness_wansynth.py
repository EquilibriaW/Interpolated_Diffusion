import argparse
import math

import torch
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_straightener import load_latent_straightener
from src.models.wan_backbone import resolve_dtype


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True)
    p.add_argument("--train_pattern", type=str, default="")
    p.add_argument("--val_pattern", type=str, default="")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=200)
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--model_dtype", type=str, default="bf16")
    p.add_argument("--straightener_ckpt", type=str, default="")
    p.add_argument("--straightener_dtype", type=str, default="")
    p.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=1000)
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


def _bucket_stats(gaps: torch.Tensor, errs: torch.Tensor, buckets: list[tuple[int, int]]):
    out = []
    for lo, hi in buckets:
        mask = (gaps >= lo) & (gaps <= hi)
        if mask.any():
            out.append((lo, hi, float(errs[mask].mean().item()), int(mask.sum().item())))
        else:
            out.append((lo, hi, math.nan, 0))
    return out


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for this diagnostic.")

    model_dtype = resolve_dtype(args.model_dtype) or torch.bfloat16
    straightener = None
    if args.straightener_ckpt:
        straightener_dtype = resolve_dtype(args.straightener_dtype) or model_dtype
        straightener, _ = load_latent_straightener(args.straightener_ckpt, device=device, dtype=straightener_dtype)
        straightener.eval()
    data_pattern = args.val_pattern or args.train_pattern or args.data_pattern
    loader = create_wan_synth_dataloader(
        data_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shuffle=True,
        shardshuffle=True,
    )
    it = iter(loader)
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)

    lerp_err_all = []
    copy_err_all = []
    s_lerp_err_all = []
    z_from_s_err_all = []
    gap_all = []
    curv_all = []
    curv_ratio_all = []
    s_curv_all = []
    s_curv_ratio_all = []

    pbar = tqdm(range(args.num_batches), dynamic_ncols=True, desc="straightness")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        latents = batch["latents"].to(device, dtype=model_dtype, non_blocking=True)
        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T, C, H, W = latents.shape
        if T != args.T:
            raise ValueError(f"T mismatch: batch={T} args={args.T}")

        # Temporal curvature on true sequences: z_{t+1} - 2 z_t + z_{t-1}
        z_prev = latents[:, :-2]
        z_mid = latents[:, 1:-1]
        z_next = latents[:, 2:]
        if args.loss_type == "l2":
            curv = (z_next - 2.0 * z_mid + z_prev).pow(2).mean(dim=(2, 3, 4)).sqrt()
            span = (z_next - z_prev).pow(2).mean(dim=(2, 3, 4)).sqrt()
        else:
            curv = (z_next - 2.0 * z_mid + z_prev).abs().mean(dim=(2, 3, 4))  # [B, T-2]
            span = (z_next - z_prev).abs().mean(dim=(2, 3, 4))
        curv_ratio = curv / (span + 1e-8)
        curv_all.append(curv.mean(dim=1).detach().cpu())
        curv_ratio_all.append(curv_ratio.mean(dim=1).detach().cpu())
        if straightener is not None:
            with torch.no_grad():
                s_prev = straightener.encode(z_prev.reshape(-1, C, H, W)).reshape(B, T - 2, C, H, W)
                s_mid = straightener.encode(z_mid.reshape(-1, C, H, W)).reshape(B, T - 2, C, H, W)
                s_next = straightener.encode(z_next.reshape(-1, C, H, W)).reshape(B, T - 2, C, H, W)
            if args.loss_type == "l2":
                s_curv = (s_next - 2.0 * s_mid + s_prev).pow(2).mean(dim=(2, 3, 4)).sqrt()
                s_span = (s_next - s_prev).pow(2).mean(dim=(2, 3, 4)).sqrt()
            else:
                s_curv = (s_next - 2.0 * s_mid + s_prev).abs().mean(dim=(2, 3, 4))
                s_span = (s_next - s_prev).abs().mean(dim=(2, 3, 4))
            s_curv_ratio = s_curv / (s_span + 1e-8)
            s_curv_all.append(s_curv.mean(dim=1).detach().cpu())
            s_curv_ratio_all.append(s_curv_ratio.mean(dim=1).detach().cpu())

        # Barycentric linearity via random triplets
        t0, t1, t, alpha = _sample_triplets(B, T, args.min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]
        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=z0.dtype)
        z_lerp = (1.0 - alpha4) * z0 + alpha4 * z1
        if args.loss_type == "l2":
            err_lerp = (z_lerp - zt).pow(2).mean(dim=(1, 2, 3)).sqrt()
            err_copy = torch.minimum(
                (z0 - zt).pow(2).mean(dim=(1, 2, 3)).sqrt(),
                (z1 - zt).pow(2).mean(dim=(1, 2, 3)).sqrt(),
            )
        else:
            err_lerp = (z_lerp - zt).abs().mean(dim=(1, 2, 3))
            err_copy = torch.minimum(
                (z0 - zt).abs().mean(dim=(1, 2, 3)),
                (z1 - zt).abs().mean(dim=(1, 2, 3)),
            )
        if straightener is not None:
            with torch.no_grad():
                s0 = straightener.encode(z0)
                s1 = straightener.encode(z1)
                st = straightener.encode(zt)
                s_lerp = (1.0 - alpha4) * s0 + alpha4 * s1
                z_from_s = straightener.decode(s_lerp)
            if args.loss_type == "l2":
                err_s_lerp = (s_lerp - st).pow(2).mean(dim=(1, 2, 3)).sqrt()
                err_z_from_s = (z_from_s - zt).pow(2).mean(dim=(1, 2, 3)).sqrt()
            else:
                err_s_lerp = (s_lerp - st).abs().mean(dim=(1, 2, 3))
                err_z_from_s = (z_from_s - zt).abs().mean(dim=(1, 2, 3))
        lerp_err_all.append(err_lerp.detach().cpu())
        copy_err_all.append(err_copy.detach().cpu())
        if straightener is not None:
            s_lerp_err_all.append(err_s_lerp.detach().cpu())
            z_from_s_err_all.append(err_z_from_s.detach().cpu())
        gap_all.append((t1 - t0).float().detach().cpu())

    lerp_err = torch.cat(lerp_err_all, dim=0)
    copy_err = torch.cat(copy_err_all, dim=0)
    gaps = torch.cat(gap_all, dim=0)
    curv = torch.cat(curv_all, dim=0)
    curv_ratio = torch.cat(curv_ratio_all, dim=0)
    s_lerp_err = torch.cat(s_lerp_err_all, dim=0) if s_lerp_err_all else None
    z_from_s_err = torch.cat(z_from_s_err_all, dim=0) if z_from_s_err_all else None
    s_curv = torch.cat(s_curv_all, dim=0) if s_curv_all else None
    s_curv_ratio = torch.cat(s_curv_ratio_all, dim=0) if s_curv_ratio_all else None

    buckets = [(2, 3), (4, 6), (7, 10), (11, 20)]
    bucket_stats = _bucket_stats(gaps, lerp_err, buckets)
    bucket_stats_copy = _bucket_stats(gaps, copy_err, buckets)

    print("\n=== Latent Straightness Diagnostics (Raw VAE Latents) ===")
    print(f"samples (triplets): {lerp_err.numel()}")
    label = "L2" if args.loss_type == "l2" else "L1"
    print(f"LERP {label} (mean): {lerp_err.mean().item():.6f}")
    print(f"Copy-endpoint {label} (mean): {copy_err.mean().item():.6f}")
    print(f"LERP improvement vs copy: {(copy_err.mean()-lerp_err.mean()).item():.3f}")
    print(f"Temporal curvature {label} (mean over t): {curv.mean().item():.6f}")
    print(f"Temporal curvature ratio (mean): {curv_ratio.mean().item():.6f}")
    if s_lerp_err is not None:
        print("\n--- Straightened space ---")
        print(f"S-space LERP {label} (mean): {s_lerp_err.mean().item():.6f}")
        print(f"Z from S-LERP {label} (mean): {z_from_s_err.mean().item():.6f}")
        print(f"S-space curvature {label} (mean over t): {s_curv.mean().item():.6f}")
        print(f"S-space curvature ratio (mean): {s_curv_ratio.mean().item():.6f}")
    print(f"\nLERP {label} by gap bucket:")
    for lo, hi, val, n in bucket_stats:
        print(f"  gap {lo:02d}-{hi:02d}: {val:.6f} (n={n})")
    print(f"Copy {label} by gap bucket:")
    for lo, hi, val, n in bucket_stats_copy:
        print(f"  gap {lo:02d}-{hi:02d}: {val:.6f} (n={n})")


if __name__ == "__main__":
    main()
