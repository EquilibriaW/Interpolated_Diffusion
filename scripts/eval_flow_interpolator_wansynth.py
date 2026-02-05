import argparse
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_flow_interpolator import load_latent_flow_interpolator
from src.models.latent_lerp_interpolator import load_latent_lerp_interpolator
from src.models.latent_straightener import load_latent_straightener
from src.models.wan_backbone import resolve_dtype


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--ckpt", type=str, required=True, help="Flow interpolator checkpoint")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=200)
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--model_dtype", type=str, default="bf16")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=1000)
    p.add_argument("--train_pattern", type=str, default="")
    p.add_argument("--val_pattern", type=str, default="")
    p.add_argument("--straightener_ckpt", type=str, default="")
    p.add_argument("--straightener_dtype", type=str, default="")
    p.add_argument("--ms_scales", type=str, default="2,4")
    return p


def _sample_triplets(B: int, T: int, min_gap: int, gen: torch.Generator, device: torch.device):
    t0 = torch.empty((B,), dtype=torch.long, device=device)
    t1 = torch.empty((B,), dtype=torch.long, device=device)
    t = torch.empty((B,), dtype=torch.long, device=device)
    for i in range(B):
        while True:
            a = int(torch.randint(0, T - 1, (1,), generator=gen, device=device).item())
            b = int(torch.randint(0, T - 1, (1,), generator=gen, device=device).item())
            lo = min(a, b)
            hi = max(a, b)
            if hi - lo >= min_gap:
                break
        t0[i] = lo
        t1[i] = hi
        t[i] = int(torch.randint(lo + 1, hi, (1,), generator=gen, device=device).item())
    alpha = (t - t0).float() / (t1 - t0).float()
    return t0, t1, t, alpha


def _bucket_stats(gaps: torch.Tensor, errs: torch.Tensor, buckets: List[Tuple[int, int]]):
    out = []
    for lo, hi in buckets:
        mask = (gaps >= lo) & (gaps <= hi)
        if mask.any():
            out.append((lo, hi, float(errs[mask].mean().item()), int(mask.sum().item())))
        else:
            out.append((lo, hi, math.nan, 0))
    return out


def _gradient_error(z_hat: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    dx_hat = z_hat[..., 1:] - z_hat[..., :-1]
    dx_gt = zt[..., 1:] - zt[..., :-1]
    dy_hat = z_hat[..., 1:, :] - z_hat[..., :-1, :]
    dy_gt = zt[..., 1:, :] - zt[..., :-1, :]
    err = (dx_hat - dx_gt).abs().mean(dim=(1, 2, 3)) + (dy_hat - dy_gt).abs().mean(dim=(1, 2, 3))
    return err


def _multiscale_l1(z_hat: torch.Tensor, zt: torch.Tensor, scales: list[int]) -> torch.Tensor:
    errs = []
    for s in scales:
        if s <= 1:
            continue
        if z_hat.shape[-2] < s or z_hat.shape[-1] < s:
            continue
        z_hat_s = F.avg_pool2d(z_hat, s, stride=s)
        zt_s = F.avg_pool2d(zt, s, stride=s)
        errs.append((z_hat_s - zt_s).abs().mean(dim=(1, 2, 3)))
    if not errs:
        return torch.zeros(z_hat.shape[0], device=z_hat.device)
    return sum(errs) / float(len(errs))


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for flow interpolator eval.")

    model_dtype = resolve_dtype(args.model_dtype) or torch.bfloat16
    meta = {}
    try:
        payload = torch.load(args.ckpt, map_location="cpu")
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    except Exception:
        meta = {}
    interp_mode = str(meta.get("interp_mode", "flow"))
    if interp_mode == "lerp_residual":
        model, _ = load_latent_lerp_interpolator(args.ckpt, device=device, dtype=model_dtype)
    else:
        model, _ = load_latent_flow_interpolator(args.ckpt, device=device, dtype=model_dtype)
    model.eval()

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

    all_err = []
    all_err_lerp = []
    all_err_lerp_s = []
    all_unc = []
    all_gap = []
    all_grad = []
    all_grad_lerp = []
    all_grad_lerp_s = []
    all_ms = []
    all_ms_lerp = []
    all_ms_lerp_s = []

    ms_scales = [int(x) for x in str(args.ms_scales).split(",") if x.strip()]

    pbar = tqdm(range(args.num_batches), dynamic_ncols=True, desc="eval")
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

        t0, t1, t, alpha = _sample_triplets(B, T, args.min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]

        gap = (t1 - t0).float()
        gap_norm = gap / float(max(T - 1, 1))

        with torch.no_grad():
            if straightener is not None:
                s0 = straightener.encode(z0)
                s1 = straightener.encode(z1)
                st = straightener.encode(zt)
                z0_in, z1_in, target = s0, s1, st
            else:
                z0_in, z1_in, target = z0, z1, zt
            if getattr(model, "gap_cond", False):
                z_hat, uncertainty = model.interpolate_pair(z0_in, z1_in, alpha, gap=gap_norm)
            else:
                z_hat, uncertainty = model.interpolate_pair(z0_in, z1_in, alpha, gap=None)
            if straightener is not None:
                z_hat = straightener.decode(z_hat)
                zt_eval = zt
            else:
                zt_eval = zt

        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=z0.dtype)
        z_lerp = z0 * (1.0 - alpha4) + z1 * alpha4
        if straightener is not None:
            s_lerp = s0 * (1.0 - alpha4) + s1 * alpha4
            z_lerp_s = straightener.decode(s_lerp)
        else:
            z_lerp_s = None

        err = (z_hat - zt_eval).abs().mean(dim=(1, 2, 3))
        err_lerp = (z_lerp - zt_eval).abs().mean(dim=(1, 2, 3))
        if z_lerp_s is not None:
            err_lerp_s = (z_lerp_s - zt_eval).abs().mean(dim=(1, 2, 3))
        else:
            err_lerp_s = None
        unc = uncertainty.mean(dim=(1, 2, 3)).float()
        gap = (t1 - t0).float()

        all_err.append(err.detach().cpu())
        all_err_lerp.append(err_lerp.detach().cpu())
        if err_lerp_s is not None:
            all_err_lerp_s.append(err_lerp_s.detach().cpu())
        all_unc.append(unc.detach().cpu())
        all_gap.append(gap.detach().cpu())

        grad = _gradient_error(z_hat.float(), zt_eval.float())
        grad_lerp = _gradient_error(z_lerp.float(), zt_eval.float())
        all_grad.append(grad.detach().cpu())
        all_grad_lerp.append(grad_lerp.detach().cpu())
        if z_lerp_s is not None:
            grad_lerp_s = _gradient_error(z_lerp_s.float(), zt_eval.float())
            all_grad_lerp_s.append(grad_lerp_s.detach().cpu())

        ms = _multiscale_l1(z_hat.float(), zt_eval.float(), ms_scales)
        ms_lerp = _multiscale_l1(z_lerp.float(), zt_eval.float(), ms_scales)
        all_ms.append(ms.detach().cpu())
        all_ms_lerp.append(ms_lerp.detach().cpu())
        if z_lerp_s is not None:
            ms_lerp_s = _multiscale_l1(z_lerp_s.float(), zt_eval.float(), ms_scales)
            all_ms_lerp_s.append(ms_lerp_s.detach().cpu())

    err = torch.cat(all_err, dim=0)
    err_lerp = torch.cat(all_err_lerp, dim=0)
    err_lerp_s = torch.cat(all_err_lerp_s, dim=0) if all_err_lerp_s else None
    unc = torch.cat(all_unc, dim=0)
    gaps = torch.cat(all_gap, dim=0)
    grad = torch.cat(all_grad, dim=0)
    grad_lerp = torch.cat(all_grad_lerp, dim=0)
    grad_lerp_s = torch.cat(all_grad_lerp_s, dim=0) if all_grad_lerp_s else None
    ms = torch.cat(all_ms, dim=0)
    ms_lerp = torch.cat(all_ms_lerp, dim=0)
    ms_lerp_s = torch.cat(all_ms_lerp_s, dim=0) if all_ms_lerp_s else None

    # Pearson correlation (uncertainty vs error)
    unc_center = unc - unc.mean()
    err_center = err - err.mean()
    corr = float((unc_center * err_center).mean() / (unc_center.std() * err_center.std() + 1e-8))

    q25 = torch.quantile(unc, 0.25)
    q75 = torch.quantile(unc, 0.75)
    err_low = float(err[unc <= q25].mean().item())
    err_high = float(err[unc >= q75].mean().item())

    buckets = [(2, 3), (4, 6), (7, 10), (11, 20)]
    bucket_stats = _bucket_stats(gaps, err, buckets)
    bucket_stats_lerp = _bucket_stats(gaps, err_lerp, buckets)
    bucket_stats_lerp_s = _bucket_stats(gaps, err_lerp_s, buckets) if err_lerp_s is not None else None

    print("\n=== Flow Interpolator Eval ===")
    print(f"samples: {err.numel()}")
    print(f"mean L1 (flow) : {err.mean().item():.6f}")
    print(f"mean L1 (lerp) : {err_lerp.mean().item():.6f}")
    if err_lerp_s is not None:
        print(f"mean L1 (lerp straight): {err_lerp_s.mean().item():.6f}")
        print(
            "rel improve vs lerp raw: "
            f"{(err_lerp.mean()-err.mean()).item()/max(err_lerp.mean().item(), 1e-8):.3f}"
        )
        print(
            "rel improve vs lerp straight: "
            f"{(err_lerp_s.mean()-err.mean()).item()/max(err_lerp_s.mean().item(), 1e-8):.3f}"
        )
    else:
        print(f"rel improve    : {(err_lerp.mean()-err.mean()).item()/max(err_lerp.mean().item(), 1e-8):.3f}")
    print(f"uncertainty corr (pearson): {corr:.3f}")
    print(f"err @ low-unc (<=25%) : {err_low:.6f}")
    print(f"err @ high-unc (>=75%): {err_high:.6f}")
    print(f"grad L1 (flow): {grad.mean().item():.6f}")
    print(f"grad L1 (lerp): {grad_lerp.mean().item():.6f}")
    if grad_lerp_s is not None:
        print(f"grad L1 (lerp straight): {grad_lerp_s.mean().item():.6f}")
    print(f"ms L1 (flow, scales={ms_scales}): {ms.mean().item():.6f}")
    print(f"ms L1 (lerp): {ms_lerp.mean().item():.6f}")
    if ms_lerp_s is not None:
        print(f"ms L1 (lerp straight): {ms_lerp_s.mean().item():.6f}")
    print("\nL1 by gap bucket (flow):")
    for lo, hi, val, n in bucket_stats:
        print(f"  gap {lo:02d}-{hi:02d}: {val:.6f} (n={n})")
    print("L1 by gap bucket (lerp):")
    for lo, hi, val, n in bucket_stats_lerp:
        print(f"  gap {lo:02d}-{hi:02d}: {val:.6f} (n={n})")
    if bucket_stats_lerp_s is not None:
        print("L1 by gap bucket (lerp straight):")
        for lo, hi, val, n in bucket_stats_lerp_s:
            print(f"  gap {lo:02d}-{hi:02d}: {val:.6f} (n={n})")


if __name__ == "__main__":
    main()
