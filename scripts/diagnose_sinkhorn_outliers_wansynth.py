import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_straightener import LatentStraightener
from src.models.sinkhorn_warp import SinkhornWarpInterpolator
from src.models.wan_backbone import resolve_dtype

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Joint sinkhorn interpolator checkpoint (model+matcher)")
    p.add_argument("--data_pattern", type=str, required=True, help="Wan2.1-synth shard glob for eval")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_batches", type=int, default=30)
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_dtype", type=str, default="bf16")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shuffle_buffer", type=int, default=1)

    p.add_argument("--warp_space", type=str, default="z", choices=["z", "s"])
    p.add_argument(
        "--phasecorr_mode",
        type=str,
        default="",
        choices=["", "mean", "multi"],
        help="Override phasecorr mode used by matcher (for analyzing older checkpoints).",
    )
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--out_dir", type=str, default="tmp_sinkhorn_outliers")
    p.add_argument("--save_tensors", type=int, default=1)
    return p


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
    x_f = x.float()
    flow_f = flow.float()
    grid = _flow_to_grid(flow_f)
    return F.grid_sample(x_f, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _sample_triplets(B: int, T: int, min_gap: int, gen: torch.Generator, device: torch.device):
    if T <= 2:
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
    t0 = lo
    t1 = hi
    t = t0 + 1 + (torch.rand((B,), generator=gen, device=device) * (gap - 1).float()).floor().long()
    alpha = (t - t0).float() / gap.float().clamp(min=1)
    return t0, t1, t, alpha


@dataclass
class Case:
    key: str
    url: str
    t0: int
    t1: int
    t: int
    alpha: float
    gap: int
    sinkhorn_mse: float
    lerp_mse: float
    straight_lerp_mse: float
    delta_vs_lerp: float
    delta_vs_straight: float
    flow01_tok_mag_mean: float
    flow01_tok_mag_max: float
    conf01_tok_mean: float
    conf10_tok_mean: float
    conf01_dust_mean: float
    conf10_dust_mean: float
    fb_err01_tok_mean: float
    fb_err10_tok_mean: float
    theta_deg: float
    dx_tok: float
    dy_tok: float


def _load_joint(
    ckpt: str, *, device: torch.device, dtype: torch.dtype, phasecorr_mode_override: str = ""
) -> Tuple[LatentStraightener, SinkhornWarpInterpolator, dict]:
    payload = torch.load(ckpt, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    model = LatentStraightener(
        in_channels=int(meta.get("in_channels", 16)),
        hidden_channels=int(meta.get("hidden_channels", 448)),
        blocks=int(meta.get("blocks", 5)),
        kernel_size=int(meta.get("kernel_size", 3)),
        use_residual=bool(meta.get("use_residual", True)),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    angles = [float(x) for x in str(meta.get("sinkhorn_angles", "-10,-5,0,5,10")).split(",") if x.strip()]
    phasecorr_mode = str(meta.get("sinkhorn_phasecorr_mode", "mean"))
    if phasecorr_mode_override:
        phasecorr_mode = str(phasecorr_mode_override)

    matcher = SinkhornWarpInterpolator(
        in_channels=int(meta.get("in_channels", 16)),
        patch_size=int(meta.get("patch_size", 4)),
        win_size=int(meta.get("sinkhorn_win", 5)),
        win_stride=int(meta.get("sinkhorn_stride", 0)),
        global_mode=str(meta.get("sinkhorn_global_mode", "phasecorr")),
        phasecorr_mode=phasecorr_mode,
        angles_deg=angles,
        shift_range=int(meta.get("sinkhorn_shift", 4)),
        se2_chunk=int(meta.get("se2_chunk", 64)),
        sinkhorn_iters=int(meta.get("sinkhorn_iters", 20)),
        sinkhorn_tau=float(meta.get("sinkhorn_tau", 0.05)),
        dustbin_logit=float(meta.get("sinkhorn_dustbin", -2.0)),
        spatial_gamma=float(meta.get("sinkhorn_spatial_gamma", 0.0)),
        spatial_radius=int(meta.get("sinkhorn_spatial_radius", 0)),
        fb_sigma=float(meta.get("sinkhorn_fb_sigma", 0.0)),
        d_match=int(meta.get("sinkhorn_d_match", 64)),
        proj_mode=str(meta.get("sinkhorn_proj_mode", "linear")),
        learn_tau=bool(meta.get("sinkhorn_learn_tau", True)),
        learn_dustbin=bool(meta.get("sinkhorn_learn_dustbin", True)),
        tau_min=float(meta.get("sinkhorn_tau_min", 1e-3)),
        straightener=None,
        warp_space="z",
    )
    matcher.load_state_dict(payload.get("matcher", {}), strict=True)
    matcher.to(device=device)
    matcher.eval()
    for p in matcher.parameters():
        p.requires_grad_(False)
    return model, matcher, meta


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    model_dtype = resolve_dtype(args.model_dtype) or torch.bfloat16
    model, matcher, meta = _load_joint(
        args.ckpt, device=device, dtype=model_dtype, phasecorr_mode_override=str(args.phasecorr_mode)
    )

    os.makedirs(args.out_dir, exist_ok=True)
    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=args.batch,
        num_workers=int(args.num_workers),
        shuffle_buffer=int(args.shuffle_buffer),
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        shuffle=False,
        shardshuffle=False,
        return_keys=True,
        keep_text_embed=False,
        keep_text=False,
        resampled=False,
        seed=int(args.seed),
    )
    it = iter(loader)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 1234)

    cases: List[Case] = []
    pbar = tqdm(range(int(args.num_batches)), dynamic_ncols=True, desc="scan")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        keys = batch.get("key", [""] * args.batch)
        urls = batch.get("url", [""] * args.batch)
        lat = batch["latents"].to(device=device, dtype=model_dtype, non_blocking=True)
        if lat.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T, C, H, W = lat.shape
        if T != int(args.T):
            raise ValueError(f"T mismatch: batch={T} args={args.T}")

        t0, t1, t, alpha = _sample_triplets(B, T, int(args.min_gap), gen, device)
        idx = torch.arange(B, device=device)
        z0 = lat[idx, t0]
        z1 = lat[idx, t1]
        zt = lat[idx, t]
        alpha4 = alpha.view(-1, 1, 1, 1).float()

        # Straightened endpoints/target.
        s0 = model.encode(z0)
        s1 = model.encode(z1)
        st = model.encode(zt)

        # Correspondences in fp32.
        with torch.cuda.amp.autocast(enabled=False):
            f0, hp, wp = matcher.token_features(s0, assume_straightened=True)
            f1, _, _ = matcher.token_features(s1, assume_straightened=True)
            flow01_tok, flow10_tok, conf01_tok, conf10_tok, conf01_dust, conf10_dust, fb_err01, fb_err10 = (
                matcher.compute_bidirectional_flow_and_confs_batch(f0, f1)
            )
            flow01 = (
                F.interpolate(flow01_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
                * float(meta.get("patch_size", 4))
            )
            flow10 = (
                F.interpolate(flow10_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
                * float(meta.get("patch_size", 4))
            )
            conf01 = F.interpolate(conf01_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(
                0.0, 1.0
            )
            conf10 = F.interpolate(conf10_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(
                0.0, 1.0
            )

            if args.warp_space == "s":
                s0_w = _warp_fp32(s0, -flow01 * alpha4)
                s1_w = _warp_fp32(s1, -flow10 * (1.0 - alpha4))
                conf0_w = _warp_fp32(conf01, -flow01 * alpha4)
                conf1_w = _warp_fp32(conf10, -flow10 * (1.0 - alpha4))
                w0 = (1.0 - alpha4) * conf0_w
                w1 = alpha4 * conf1_w
                denom = w0 + w1
                s_mix = (w0 * s0_w + w1 * s1_w) / denom.clamp_min(1e-6)
                s_lin = (1.0 - alpha4) * s0_w + alpha4 * s1_w
                s_hat = torch.where(denom > 1e-6, s_mix, s_lin)
                z_hat = model.decode(s_hat.to(dtype=s0.dtype))
            else:
                z0_w = _warp_fp32(z0, -flow01 * alpha4)
                z1_w = _warp_fp32(z1, -flow10 * (1.0 - alpha4))
                conf0_w = _warp_fp32(conf01, -flow01 * alpha4)
                conf1_w = _warp_fp32(conf10, -flow10 * (1.0 - alpha4))
                w0 = (1.0 - alpha4) * conf0_w
                w1 = alpha4 * conf1_w
                denom = w0 + w1
                z_mix = (w0 * z0_w + w1 * z1_w) / denom.clamp_min(1e-6)
                z_lin = (1.0 - alpha4) * z0_w + alpha4 * z1_w
                z_hat = torch.where(denom > 1e-6, z_mix, z_lin)

        # Baselines.
        z_lerp = z0.float() * (1.0 - alpha4) + z1.float() * alpha4
        s_lerp = (1.0 - alpha4.to(dtype=s0.dtype)) * s0 + alpha4.to(dtype=s0.dtype) * s1
        z_straight = model.decode(s_lerp).float()

        # Per-sample MSE.
        sink_mse = (z_hat.float() - zt.float()).pow(2).mean(dim=(1, 2, 3))
        lerp_mse = (z_lerp - zt.float()).pow(2).mean(dim=(1, 2, 3))
        straight_mse = (z_straight - zt.float()).pow(2).mean(dim=(1, 2, 3))

        # Global params (for interpretability).
        if getattr(matcher, "global_mode", "phasecorr") == "phasecorr":
            if getattr(matcher, "phasecorr_mode", "mean") == "multi":
                theta, dx, dy = matcher._phasecorr_se2_multi_batch(f0, f1)
            else:
                theta, dx, dy = matcher._phasecorr_se2_batch(f0.mean(dim=-1), f1.mean(dim=-1))
        elif getattr(matcher, "global_mode", "phasecorr") == "se2":
            theta, dx_i, dy_i = matcher._estimate_global_se2_batch(f0, f1)
            dx = dx_i.to(dtype=theta.dtype)
            dy = dy_i.to(dtype=theta.dtype)
        else:
            theta = torch.zeros((B,), device=device, dtype=torch.float32)
            dx = torch.zeros((B,), device=device, dtype=torch.float32)
            dy = torch.zeros((B,), device=device, dtype=torch.float32)

        theta_deg = (theta.float() * (180.0 / math.pi)).detach().cpu()

        flow01_mag = torch.linalg.norm(flow01_tok.float(), dim=-1)  # [B,Hp,Wp]
        for i in range(B):
            k = keys[i] if isinstance(keys, list) else str(keys[i])
            u = urls[i] if isinstance(urls, list) else str(urls[i])
            cases.append(
                Case(
                    key=str(k),
                    url=str(u),
                    t0=int(t0[i].item()),
                    t1=int(t1[i].item()),
                    t=int(t[i].item()),
                    alpha=float(alpha[i].item()),
                    gap=int((t1[i] - t0[i]).item()),
                    sinkhorn_mse=float(sink_mse[i].item()),
                    lerp_mse=float(lerp_mse[i].item()),
                    straight_lerp_mse=float(straight_mse[i].item()),
                    delta_vs_lerp=float((sink_mse[i] - lerp_mse[i]).item()),
                    delta_vs_straight=float((sink_mse[i] - straight_mse[i]).item()),
                    flow01_tok_mag_mean=float(flow01_mag[i].mean().item()),
                    flow01_tok_mag_max=float(flow01_mag[i].amax().item()),
                    conf01_tok_mean=float(conf01_tok[i].mean().item()),
                    conf10_tok_mean=float(conf10_tok[i].mean().item()),
                    conf01_dust_mean=float(conf01_dust[i].mean().item()),
                    conf10_dust_mean=float(conf10_dust[i].mean().item()),
                    fb_err01_tok_mean=float(fb_err01[i].mean().item()),
                    fb_err10_tok_mean=float(fb_err10[i].mean().item()),
                    theta_deg=float(theta_deg[i].item()),
                    dx_tok=float(dx[i].float().item()),
                    dy_tok=float(dy[i].float().item()),
                )
            )

    # Sort by how much worse than LERP.
    cases_sorted = sorted(cases, key=lambda c: c.delta_vs_lerp, reverse=True)
    worst = cases_sorted[: int(args.topk)]
    best = sorted(cases, key=lambda c: c.delta_vs_lerp)[: int(args.topk)]

    # Save jsonl summaries.
    summary_path = os.path.join(args.out_dir, "cases.jsonl")
    with open(summary_path, "w", encoding="utf-8") as f:
        for c in cases_sorted:
            f.write(json.dumps(asdict(c)) + "\n")

    def _print_block(name: str, block: List[Case]) -> None:
        print(f"\n=== {name} (by sinkhorn_mse - lerp_mse) ===")
        for c in block:
            print(
                f"key={c.key} t0={c.t0} t1={c.t1} t={c.t} gap={c.gap} a={c.alpha:.3f} "
                f"delta={c.delta_vs_lerp:+.6f} sink={c.sinkhorn_mse:.6f} lerp={c.lerp_mse:.6f} "
                f"straight={c.straight_lerp_mse:.6f} "
                f"theta={c.theta_deg:+.1f} dx={c.dx_tok:+.2f} dy={c.dy_tok:+.2f} "
                f"flowTokMean={c.flow01_tok_mag_mean:.3f} conf01={c.conf01_tok_mean:.3f} fbErr={c.fb_err01_tok_mean:.3f}"
            )

    _print_block("WORST", worst)
    _print_block("BEST", best)
    print(f"\nWrote: {summary_path}")

    if not bool(args.save_tensors):
        return

    # Re-run and save tensors for selected cases (worst+best) only.
    selected = {("worst", i, c.key, c.t0, c.t1, c.t, round(c.alpha, 6)) for i, c in enumerate(worst)}
    selected |= {("best", i, c.key, c.t0, c.t1, c.t, round(c.alpha, 6)) for i, c in enumerate(best)}

    # Create a lookup to match cases quickly.
    sel_lookup = {(k, t0, t1, t, round(a, 6)): (tag, rank) for (tag, rank, k, t0, t1, t, a) in selected}

    # Scan again and dump per-case tensors. (num_batches is small; avoid holding full dataset in RAM.)
    it = iter(loader)
    gen.manual_seed(int(args.seed) + 1234)
    pbar = tqdm(range(int(args.num_batches)), dynamic_ncols=True, desc="dump")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        keys = batch.get("key", [""] * args.batch)
        urls = batch.get("url", [""] * args.batch)
        lat = batch["latents"].to(device=device, dtype=model_dtype, non_blocking=True)
        B, T, C, H, W = lat.shape

        t0, t1, t, alpha = _sample_triplets(B, T, int(args.min_gap), gen, device)
        idx = torch.arange(B, device=device)
        z0 = lat[idx, t0]
        z1 = lat[idx, t1]
        zt = lat[idx, t]
        alpha4 = alpha.view(-1, 1, 1, 1).float()

        s0 = model.encode(z0)
        s1 = model.encode(z1)
        st = model.encode(zt)

        with torch.cuda.amp.autocast(enabled=False):
            f0, hp, wp = matcher.token_features(s0, assume_straightened=True)
            f1, _, _ = matcher.token_features(s1, assume_straightened=True)
            flow01_tok, flow10_tok, conf01_tok, conf10_tok, conf01_dust, conf10_dust, fb_err01, fb_err10 = (
                matcher.compute_bidirectional_flow_and_confs_batch(f0, f1)
            )
            flow01 = (
                F.interpolate(flow01_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
                * float(meta.get("patch_size", 4))
            )
            flow10 = (
                F.interpolate(flow10_tok.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=True)
                * float(meta.get("patch_size", 4))
            )
            conf01 = F.interpolate(conf01_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(
                0.0, 1.0
            )
            conf10 = F.interpolate(conf10_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True).clamp(
                0.0, 1.0
            )
            if args.warp_space == "s":
                s0_w = _warp_fp32(s0, -flow01 * alpha4)
                s1_w = _warp_fp32(s1, -flow10 * (1.0 - alpha4))
                conf0_w = _warp_fp32(conf01, -flow01 * alpha4)
                conf1_w = _warp_fp32(conf10, -flow10 * (1.0 - alpha4))
                w0 = (1.0 - alpha4) * conf0_w
                w1 = alpha4 * conf1_w
                denom = w0 + w1
                s_mix = (w0 * s0_w + w1 * s1_w) / denom.clamp_min(1e-6)
                s_lin = (1.0 - alpha4) * s0_w + alpha4 * s1_w
                s_hat = torch.where(denom > 1e-6, s_mix, s_lin)
                z_hat = model.decode(s_hat.to(dtype=s0.dtype))
            else:
                z0_w = _warp_fp32(z0, -flow01 * alpha4)
                z1_w = _warp_fp32(z1, -flow10 * (1.0 - alpha4))
                conf0_w = _warp_fp32(conf01, -flow01 * alpha4)
                conf1_w = _warp_fp32(conf10, -flow10 * (1.0 - alpha4))
                w0 = (1.0 - alpha4) * conf0_w
                w1 = alpha4 * conf1_w
                denom = w0 + w1
                z_mix = (w0 * z0_w + w1 * z1_w) / denom.clamp_min(1e-6)
                z_lin = (1.0 - alpha4) * z0_w + alpha4 * z1_w
                z_hat = torch.where(denom > 1e-6, z_mix, z_lin)

        z_lerp = z0.float() * (1.0 - alpha4) + z1.float() * alpha4
        s_lerp = (1.0 - alpha4.to(dtype=s0.dtype)) * s0 + alpha4.to(dtype=s0.dtype) * s1
        z_straight = model.decode(s_lerp).float()

        for i in range(B):
            key = keys[i] if isinstance(keys, list) else str(keys[i])
            tag = sel_lookup.get((str(key), int(t0[i].item()), int(t1[i].item()), int(t[i].item()), round(float(alpha[i].item()), 6)))
            if tag is None:
                continue
            split, rank = tag
            case_dir = os.path.join(args.out_dir, split, f"{rank:03d}_{key}_t{int(t0[i])}-{int(t[i])}-{int(t1[i])}")
            os.makedirs(case_dir, exist_ok=True)
            blob: Dict[str, Any] = {
                "key": str(key),
                "url": urls[i] if isinstance(urls, list) else str(urls[i]),
                "t0": int(t0[i].item()),
                "t1": int(t1[i].item()),
                "t": int(t[i].item()),
                "alpha": float(alpha[i].item()),
                "z0": z0[i].detach().cpu(),
                "z1": z1[i].detach().cpu(),
                "zt": zt[i].detach().cpu(),
                "z_hat": z_hat[i].detach().cpu(),
                "z_lerp": z_lerp[i].detach().cpu(),
                "z_straight": z_straight[i].detach().cpu(),
                "s0": s0[i].detach().cpu(),
                "s1": s1[i].detach().cpu(),
                "st": st[i].detach().cpu(),
                "flow01_tok": flow01_tok[i].detach().cpu(),
                "flow10_tok": flow10_tok[i].detach().cpu(),
                "conf01_tok": conf01_tok[i].detach().cpu(),
                "conf10_tok": conf10_tok[i].detach().cpu(),
                "conf01_dust": conf01_dust[i].detach().cpu(),
                "conf10_dust": conf10_dust[i].detach().cpu(),
                "fb_err01_tok": fb_err01[i].detach().cpu(),
                "fb_err10_tok": fb_err10[i].detach().cpu(),
                "flow01_lat": flow01[i].detach().cpu(),
                "flow10_lat": flow10[i].detach().cpu(),
                "conf0_w": conf0_w[i].detach().cpu(),
                "conf1_w": conf1_w[i].detach().cpu(),
            }
            torch.save(blob, os.path.join(case_dir, "case.pt"))

    print(f"Wrote tensors under: {args.out_dir}/worst and {args.out_dir}/best")


if __name__ == "__main__":
    main()
