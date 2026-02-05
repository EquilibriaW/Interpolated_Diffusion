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
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--model_dtype", type=str, default="bf16")
    p.add_argument("--vae_dtype", type=str, default="bf16")
    p.add_argument("--vae_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--vae_subfolder", type=str, default="vae")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=1000)
    p.add_argument("--train_pattern", type=str, default="")
    p.add_argument("--val_pattern", type=str, default="")
    p.add_argument("--straightener_ckpt", type=str, default="")
    p.add_argument("--straightener_dtype", type=str, default="")
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


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # x, y: [N,C,H,W] in [0,1]
    C1 = 0.01**2
    C2 = 0.03**2
    C = x.shape[1]
    window = _gaussian_window(window_size, sigma, C, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean(dim=(1, 2, 3))


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = (x - y).pow(2).mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse + 1e-10)


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for flow interpolator eval.")

    torch.backends.cudnn.benchmark = True

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

    from diffusers import AutoencoderKLWan

    vae_dtype = resolve_dtype(args.vae_dtype) or torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.vae_repo, subfolder=args.vae_subfolder, torch_dtype=vae_dtype)
    vae.to(device)
    vae.eval()

    latents_mean = (
        torch.tensor(vae.config.latents_mean, device=device, dtype=vae_dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std, device=device, dtype=vae_dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )

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

    all_psnr = []
    all_psnr_lerp = []
    all_psnr_lerp_s = []
    all_ssim = []
    all_ssim_lerp = []
    all_ssim_lerp_s = []

    pbar = tqdm(range(args.num_batches), dynamic_ncols=True, desc="eval_rgb")
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
                z0_in, z1_in = s0, s1
            else:
                z0_in, z1_in = z0, z1
            if getattr(model, "gap_cond", False):
                z_hat, _unc = model.interpolate_pair(z0_in, z1_in, alpha, gap=gap_norm)
            else:
                z_hat, _unc = model.interpolate_pair(z0_in, z1_in, alpha, gap=None)
            if straightener is not None:
                z_hat = straightener.decode(z_hat)

        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=z0.dtype)
        z_lerp = z0 * (1.0 - alpha4) + z1 * alpha4
        if straightener is not None:
            s_lerp = s0 * (1.0 - alpha4) + s1 * alpha4
            z_lerp_s = straightener.decode(s_lerp)
        else:
            z_lerp_s = None

        # Decode flow, lerp, and gt in one batch for efficiency.
        if z_lerp_s is not None:
            z_all = torch.cat([z_hat, z_lerp, z_lerp_s, zt], dim=0).to(dtype=vae_dtype)
        else:
            z_all = torch.cat([z_hat, z_lerp, zt], dim=0).to(dtype=vae_dtype)
        z_all = z_all.unsqueeze(2)  # [NB, C, 1, H, W]
        z_all = z_all * latents_std + latents_mean
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=vae_dtype):
            rgb_all = vae.decode(z_all, return_dict=False)[0]
        rgb_all = rgb_all.squeeze(2).clamp(-1.0, 1.0)
        rgb_all = (rgb_all + 1.0) / 2.0
        rgb_all = rgb_all.clamp(0.0, 1.0)

        if z_lerp_s is not None:
            rgb_hat, rgb_lerp, rgb_lerp_s, rgb_gt = (
                rgb_all[:B],
                rgb_all[B : 2 * B],
                rgb_all[2 * B : 3 * B],
                rgb_all[3 * B :],
            )
        else:
            rgb_hat, rgb_lerp, rgb_gt = rgb_all[:B], rgb_all[B : 2 * B], rgb_all[2 * B :]

        rgb_hat_f = rgb_hat.float()
        rgb_lerp_f = rgb_lerp.float()
        rgb_gt_f = rgb_gt.float()
        rgb_lerp_s_f = rgb_lerp_s.float() if z_lerp_s is not None else None

        psnr_flow = psnr(rgb_hat_f, rgb_gt_f)
        psnr_lerp = psnr(rgb_lerp_f, rgb_gt_f)
        ssim_flow = ssim(rgb_hat_f, rgb_gt_f)
        ssim_lerp = ssim(rgb_lerp_f, rgb_gt_f)
        if rgb_lerp_s_f is not None:
            psnr_lerp_s = psnr(rgb_lerp_s_f, rgb_gt_f)
            ssim_lerp_s = ssim(rgb_lerp_s_f, rgb_gt_f)

        all_psnr.append(psnr_flow.detach().cpu())
        all_psnr_lerp.append(psnr_lerp.detach().cpu())
        all_ssim.append(ssim_flow.detach().cpu())
        all_ssim_lerp.append(ssim_lerp.detach().cpu())
        if rgb_lerp_s_f is not None:
            all_psnr_lerp_s.append(psnr_lerp_s.detach().cpu())
            all_ssim_lerp_s.append(ssim_lerp_s.detach().cpu())

    psnr_flow = torch.cat(all_psnr, dim=0)
    psnr_lerp = torch.cat(all_psnr_lerp, dim=0)
    ssim_flow = torch.cat(all_ssim, dim=0)
    ssim_lerp = torch.cat(all_ssim_lerp, dim=0)
    psnr_lerp_s = torch.cat(all_psnr_lerp_s, dim=0) if all_psnr_lerp_s else None
    ssim_lerp_s = torch.cat(all_ssim_lerp_s, dim=0) if all_ssim_lerp_s else None

    print("\n=== Flow Interpolator Eval (RGB decode) ===")
    print(f"samples: {psnr_flow.numel()}")
    print(f"PSNR (flow): {psnr_flow.mean().item():.4f} dB")
    print(f"PSNR (lerp): {psnr_lerp.mean().item():.4f} dB")
    if psnr_lerp_s is not None:
        print(f"PSNR (lerp straight): {psnr_lerp_s.mean().item():.4f} dB")
        print(f"PSNR delta (flow - lerp): {(psnr_flow.mean() - psnr_lerp.mean()).item():.4f} dB")
        print(f"PSNR delta (flow - lerp straight): {(psnr_flow.mean() - psnr_lerp_s.mean()).item():.4f} dB")
    else:
        print(f"PSNR delta : {(psnr_flow.mean() - psnr_lerp.mean()).item():.4f} dB")
    print(f"SSIM (flow): {ssim_flow.mean().item():.5f}")
    print(f"SSIM (lerp): {ssim_lerp.mean().item():.5f}")
    if ssim_lerp_s is not None:
        print(f"SSIM (lerp straight): {ssim_lerp_s.mean().item():.5f}")
        print(f"SSIM delta (flow - lerp): {(ssim_flow.mean() - ssim_lerp.mean()).item():.5f}")
        print(f"SSIM delta (flow - lerp straight): {(ssim_flow.mean() - ssim_lerp_s.mean()).item():.5f}")
    else:
        print(f"SSIM delta : {(ssim_flow.mean() - ssim_lerp.mean()).item():.5f}")


if __name__ == "__main__":
    main()
