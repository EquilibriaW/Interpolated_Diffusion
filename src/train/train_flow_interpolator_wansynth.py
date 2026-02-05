import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_flow_interpolator import LatentFlowInterpolator
from src.models.wan_backbone import resolve_dtype
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.logging import create_writer
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--train_pattern", type=str, default="", help="Optional train shard pattern override")
    p.add_argument("--val_pattern", type=str, default="", help="Optional validation shard pattern")
    p.add_argument("--teacher_pattern", type=str, default="", help="Optional LDMVFI teacher shard pattern")
    p.add_argument("--T", type=int, default=21)
    p.add_argument(
        "--interp_mode",
        type=str,
        default="flow",
        choices=["flow", "lerp_residual"],
        help="Interpolation backbone to train",
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--max_flow", type=float, default=20.0)
    p.add_argument("--residual_blocks", type=int, default=2)
    p.add_argument("--residual_channels", type=int, default=32)
    p.add_argument("--time_mask", type=int, default=0, help="Enable time-dependent blend mask")
    p.add_argument("--gap_cond", type=int, default=0, help="Condition flow/mask on gap length")
    p.add_argument("--cost_volume", type=int, default=0, help="Enable low-res cost volume input")
    p.add_argument("--cv_radius", type=int, default=2)
    p.add_argument("--cv_downscale", type=int, default=2)
    p.add_argument("--cv_norm", type=int, default=1)
    p.add_argument("--model_dtype", type=str, default="")
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--teacher_weight", type=float, default=1.0)
    p.add_argument("--gt_weight", type=float, default=0.0)
    p.add_argument("--uncertainty_loss_weight", type=float, default=0.1)
    p.add_argument("--uncertainty_scale", type=float, default=0.5)
    p.add_argument("--flow_smooth_weight", type=float, default=0.0)
    p.add_argument("--edge_weight", type=float, default=0.0)
    p.add_argument("--ms_weight", type=float, default=0.0)
    p.add_argument("--ms_scales", type=str, default="2,4")
    p.add_argument("--endpoint_weight", type=float, default=0.0)
    p.add_argument("--flow_consistency_weight", type=float, default=0.0)
    p.add_argument("--gap_weight", type=float, default=0.0, help="Strength of gap-weighted L1 (0=off)")
    p.add_argument("--gap_gamma", type=float, default=1.0, help="Exponent for gap weighting")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=200)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/latent_flow_interp_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/latent_flow_interp_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=1000)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--prefetch_to_gpu", type=int, default=1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--straightener_ckpt", type=str, default="")
    p.add_argument("--straightener_dtype", type=str, default="")
    p.add_argument("--resampled", type=int, default=1)
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
    # Resample invalid gaps until all meet min_gap.
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
    # Sample t uniformly in (t0, t1).
    t = t0 + 1 + (torch.rand((B,), generator=gen, device=device) * (gap - 1).float()).floor().long()
    alpha = (t - t0).float() / gap.float().clamp(min=1)
    return t0, t1, t, alpha


def _flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    # flow: [B,2,H,W]
    dx = (flow[..., 1:] - flow[..., :-1]).abs().mean()
    dy = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
    return dx + dy


def _gradient_loss(z_hat: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
    dx_hat = z_hat[..., 1:] - z_hat[..., :-1]
    dx_gt = zt[..., 1:] - zt[..., :-1]
    dy_hat = z_hat[..., 1:, :] - z_hat[..., :-1, :]
    dy_gt = zt[..., 1:, :] - zt[..., :-1, :]
    return (dx_hat - dx_gt).abs().mean() + (dy_hat - dy_gt).abs().mean()


def _multiscale_l1(z_hat: torch.Tensor, zt: torch.Tensor, scales: list[int]) -> torch.Tensor:
    losses = []
    for s in scales:
        if s <= 1:
            continue
        if z_hat.shape[-2] < s or z_hat.shape[-1] < s:
            continue
        z_hat_s = F.avg_pool2d(z_hat, s, stride=s)
        zt_s = F.avg_pool2d(zt, s, stride=s)
        losses.append(F.l1_loss(z_hat_s, zt_s))
    if not losses:
        return torch.tensor(0.0, device=z_hat.device)
    return sum(losses) / float(len(losses))


def _eval_model(
    model: LatentFlowInterpolator,
    data_pattern: str,
    *,
    T: int,
    batch: int,
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
    straightener=None,
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
        shuffle=True,
        shardshuffle=True,
        keep_text_embed=False,
        keep_text=False,
        resampled=resampled,
        seed=seed,
    )
    it = iter(loader)
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)

    err_all = []
    err_lerp_all = []
    unc_all = []
    for _ in range(num_batches):
        try:
            batch_data = next(it)
        except StopIteration:
            it = iter(loader)
            batch_data = next(it)
        latents = batch_data["latents"].to(device, dtype=model_dtype, non_blocking=True)
        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T0, _, _, _ = latents.shape
        if T0 != T:
            raise ValueError(f"T mismatch: batch={T0} args={T}")
        t0, t1, t, alpha = _sample_triplets(B, T, min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]
        gap = (t1 - t0).float()
        gap_norm = gap / float(max(T - 1, 1))
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
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
        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=z0.dtype)
        z_lerp = z0 * (1.0 - alpha4) + z1 * alpha4
        err = (z_hat - zt).abs().mean(dim=(1, 2, 3))
        err_lerp = (z_lerp - zt).abs().mean(dim=(1, 2, 3))
        unc = uncertainty.mean(dim=(1, 2, 3)).float()
        err_all.append(err.detach().cpu())
        err_lerp_all.append(err_lerp.detach().cpu())
        unc_all.append(unc.detach().cpu())

    err = torch.cat(err_all, dim=0)
    err_lerp = torch.cat(err_lerp_all, dim=0)
    unc = torch.cat(unc_all, dim=0)
    unc_center = unc - unc.mean()
    err_center = err - err.mean()
    corr = float((unc_center * err_center).mean() / (unc_center.std() * err_center.std() + 1e-8))
    q25 = torch.quantile(unc, 0.25)
    q75 = torch.quantile(unc, 0.75)
    err_low = float(err[unc <= q25].mean().item())
    err_high = float(err[unc >= q75].mean().item())

    if model_was_training:
        model.train()

    return {
        "val_l1": float(err.mean().item()),
        "val_lerp_l1": float(err_lerp.mean().item()),
        "val_unc_corr": corr,
        "val_unc_low": err_low,
        "val_unc_high": err_high,
    }


def _move_batch_to_device(batch: dict, device: torch.device, model_dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k in {"latents", "teacher"}:
                out[k] = v.to(device, dtype=model_dtype, non_blocking=True)
            elif k in {"teacher_idx"}:
                out[k] = v.to(device, dtype=torch.long, non_blocking=True)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


class _CUDAPrefetcher:
    def __init__(self, it, device: torch.device, model_dtype: torch.dtype):
        self.it = it
        self.device = device
        self.model_dtype = model_dtype
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.it)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            batch = _move_batch_to_device(batch, self.device, self.model_dtype)
        self.next_batch = batch

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            return None
        self._preload()
        return batch


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth flow interpolator training.")
    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    train_pattern = args.train_pattern or args.data_pattern
    if args.teacher_pattern:
        from src.data.wan_synth import create_wan_synth_teacher_dataloader

        loader = create_wan_synth_teacher_dataloader(
            train_pattern,
            teacher_path_pattern=args.teacher_pattern,
            batch_size=args.batch,
            num_workers=args.num_workers,
            shuffle_buffer=args.shuffle_buffer,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=bool(args.persistent_workers),
            pin_memory=bool(args.pin_memory),
            shuffle=True,
            join_by_key=True,
            allow_missing=True,
            keep_text_embed=False,
            keep_text=False,
            resampled=bool(args.resampled),
            seed=args.seed,
        )
    else:
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

    batch0 = next(it)
    latents0 = batch0["latents"].to(device, dtype=model_dtype, non_blocking=True)
    if latents0.dim() != 5:
        raise ValueError("latents must be [B,T,C,H,W]")
    _, T0, C0, _, _ = latents0.shape
    if T0 != args.T:
        raise ValueError(f"T mismatch: batch={T0} args={args.T}")

    if args.interp_mode == "lerp_residual":
        from src.models.latent_lerp_interpolator import LatentLerpResidualInterpolator

        model = LatentLerpResidualInterpolator(
            in_channels=C0,
            base_channels=args.base_channels,
            residual_channels=args.residual_channels,
            residual_blocks=args.residual_blocks,
            gap_cond=bool(args.gap_cond),
        ).to(device=device, dtype=model_dtype)
    else:
        model = LatentFlowInterpolator(
            in_channels=C0,
            base_channels=args.base_channels,
            max_flow=args.max_flow,
            residual_channels=args.residual_channels,
            residual_blocks=args.residual_blocks,
            time_mask=bool(args.time_mask),
            gap_cond=bool(args.gap_cond),
            cost_volume=bool(args.cost_volume),
            cv_radius=args.cv_radius,
            cv_downscale=args.cv_downscale,
            cv_norm=bool(args.cv_norm),
        ).to(device=device, dtype=model_dtype)
    straightener = None
    if args.straightener_ckpt:
        from src.models.latent_straightener import load_latent_straightener

        straightener_dtype = resolve_dtype(args.straightener_dtype) or model_dtype
        straightener, _ = load_latent_straightener(
            args.straightener_ckpt, device=device, dtype=straightener_dtype
        )
        straightener.eval()
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt, ema=None, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 11)
    ms_scales = [int(x) for x in str(args.ms_scales).split(",") if x.strip()]

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    prefetcher = None
    if args.prefetch_to_gpu:
        prefetcher = _CUDAPrefetcher(it, device, model_dtype)

    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        step_start = time.perf_counter()
        log_this = step % args.log_every == 0
        if log_this:
            torch.cuda.reset_peak_memory_stats()
        if step == start_step:
            batch = _move_batch_to_device(batch0, device, model_dtype) if args.prefetch_to_gpu else batch0
        else:
            if prefetcher is not None:
                batch = prefetcher.next()
                if batch is None:
                    it = iter(loader)
                    prefetcher = _CUDAPrefetcher(it, device, model_dtype)
                    batch = prefetcher.next()
            else:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)

        if args.prefetch_to_gpu:
            latents = batch["latents"]
        else:
            latents = batch["latents"].to(device, dtype=model_dtype, non_blocking=True)
        B, T, C, H, W = latents.shape

        teacher_latent = None
        teacher_idx = None
        use_teacher = False
        if args.teacher_pattern:
            teacher_latent = batch.get("teacher")
            teacher_idx = batch.get("teacher_idx")
            if teacher_latent is None or teacher_idx is None:
                raise RuntimeError("teacher_pattern provided but teacher fields missing in batch")
            use_teacher = True
        else:
            # Self-contained teacher shards may store raw keys like "teacher.pth".
            teacher_latent = batch.get("teacher")
            teacher_idx = batch.get("teacher_idx")
            if teacher_latent is None:
                teacher_latent = batch.get("teacher.pth")
            if teacher_idx is None:
                teacher_idx = batch.get("teacher_idx.pth")
            if teacher_latent is not None and teacher_idx is not None:
                use_teacher = True

        if use_teacher:
            if not args.prefetch_to_gpu:
                teacher_latent = teacher_latent.to(device, dtype=model_dtype, non_blocking=True)
            if teacher_idx.dim() == 1:
                teacher_idx = teacher_idx.view(-1, 3)
            if teacher_idx.shape[-1] != 3:
                raise ValueError(f"teacher_idx must have shape [B,3], got {teacher_idx.shape}")
            if args.prefetch_to_gpu:
                t0 = teacher_idx[:, 0]
                t1 = teacher_idx[:, 1]
                t = teacher_idx[:, 2]
            else:
                t0 = teacher_idx[:, 0].to(device=device, dtype=torch.long)
                t1 = teacher_idx[:, 1].to(device=device, dtype=torch.long)
                t = teacher_idx[:, 2].to(device=device, dtype=torch.long)
            alpha = (t - t0).float() / (t1 - t0).float().clamp(min=1)
        else:
            t0, t1, t, alpha = _sample_triplets(B, T, args.min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]

        gap = (t1 - t0).float()
        gap_norm = gap / float(max(T - 1, 1))

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            if straightener is not None:
                with torch.no_grad():
                    z0_s = straightener.encode(z0)
                    z1_s = straightener.encode(z1)
                    zt_s = straightener.encode(zt)
                    if use_teacher:
                        teacher_latent = straightener.encode(teacher_latent)
                z0_in, z1_in, target = z0_s, z1_s, zt_s
            else:
                z0_in, z1_in, target = z0, z1, zt
            if getattr(model, "gap_cond", False):
                z_hat, uncertainty = model.interpolate_pair(z0_in, z1_in, alpha, gap=gap_norm)
            else:
                z_hat, uncertainty = model.interpolate_pair(z0_in, z1_in, alpha, gap=None)

            if use_teacher:
                target = teacher_latent
            err_per = (z_hat - target).abs().mean(dim=(1, 2, 3))
            if args.gap_weight > 0.0:
                gap = (t1 - t0).float()
                weight = (gap / float(args.min_gap)).clamp(min=1.0) ** float(args.gap_gamma)
                weight = weight / weight.mean().clamp(min=1e-6)
                weight = 1.0 + float(args.gap_weight) * (weight - 1.0)
                recon_loss = (err_per * weight).mean()
            else:
                recon_loss = err_per.mean()

            if use_teacher and args.gt_weight > 0.0:
                gt_target = zt_s if straightener is not None else zt
                gt_loss = F.l1_loss(z_hat, gt_target)
            else:
                gt_loss = torch.tensor(0.0, device=device)
            err_map = (z_hat - target).abs().mean(dim=1, keepdim=True)
            u_target = err_map / (err_map + float(args.uncertainty_scale))
            unc_loss = F.l1_loss(uncertainty, u_target)
            loss = recon_loss + float(args.uncertainty_loss_weight) * unc_loss
            if use_teacher:
                loss = loss * float(args.teacher_weight)
                if args.gt_weight > 0.0:
                    loss = loss + float(args.gt_weight) * gt_loss

            if args.edge_weight > 0.0:
                edge_loss = _gradient_loss(z_hat.float(), target.float())
                loss = loss + float(args.edge_weight) * edge_loss
            else:
                edge_loss = torch.tensor(0.0, device=device)

            if args.ms_weight > 0.0:
                ms_loss = _multiscale_l1(z_hat.float(), target.float(), ms_scales)
                loss = loss + float(args.ms_weight) * ms_loss
            else:
                ms_loss = torch.tensor(0.0, device=device)

            if args.flow_smooth_weight > 0.0 and args.interp_mode == "flow":
                flow01, flow10, _, _, _ = model.predict_flow(z0_in, z1_in, gap=gap_norm if args.gap_cond else None)
                smooth_loss = _flow_smoothness(flow01) + _flow_smoothness(flow10)
                loss = loss + float(args.flow_smooth_weight) * smooth_loss
            else:
                smooth_loss = torch.tensor(0.0, device=device)

            if args.endpoint_weight > 0.0 and args.interp_mode == "flow":
                flow01, flow10, mask_a, mask_b, _ = model.predict_flow(
                    z0_in, z1_in, gap=gap_norm if args.gap_cond else None
                )
                alpha0 = torch.zeros((B,), device=device, dtype=z0.dtype)
                alpha1 = torch.ones((B,), device=device, dtype=z0.dtype)
                z0_hat = model.blend_from_flow(
                    z0_in, z1_in, alpha0, flow01, flow10, mask_a, mask_b, gap=gap_norm if args.gap_cond else None
                )
                z1_hat = model.blend_from_flow(
                    z0_in, z1_in, alpha1, flow01, flow10, mask_a, mask_b, gap=gap_norm if args.gap_cond else None
                )
                z0_ref = z0_in if straightener is not None else z0
                z1_ref = z1_in if straightener is not None else z1
                endpoint_loss = F.l1_loss(z0_hat, z0_ref) + F.l1_loss(z1_hat, z1_ref)
                loss = loss + float(args.endpoint_weight) * endpoint_loss
            else:
                endpoint_loss = torch.tensor(0.0, device=device)

            if args.flow_consistency_weight > 0.0 and args.interp_mode == "flow":
                flow01, flow10, _, _, _ = model.predict_flow(z0_in, z1_in, gap=gap_norm if args.gap_cond else None)
                from src.models.latent_flow_interpolator import _warp

                z0_to_1 = _warp(z0_in, -flow01)
                z1_to_0 = _warp(z1_in, -flow10)
                z0_ref = z0_in if straightener is not None else z0
                z1_ref = z1_in if straightener is not None else z1
                flow_consistency = F.l1_loss(z0_to_1, z1_ref) + F.l1_loss(z1_to_0, z0_ref)
                loss = loss + float(args.flow_consistency_weight) * flow_consistency
            else:
                flow_consistency = torch.tensor(0.0, device=device)

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
            writer.add_scalar("train/recon_loss", recon_loss.item(), step)
            if use_teacher:
                writer.add_scalar("train/gt_loss", gt_loss.item(), step)
            writer.add_scalar("train/uncertainty_loss", unc_loss.item(), step)
            writer.add_scalar("train/edge_loss", edge_loss.item(), step)
            writer.add_scalar("train/ms_loss", ms_loss.item(), step)
            writer.add_scalar("train/flow_smooth_loss", smooth_loss.item(), step)
            writer.add_scalar("train/endpoint_loss", endpoint_loss.item(), step)
            writer.add_scalar("train/flow_consistency_loss", flow_consistency.item(), step)
            writer.add_scalar("train/uncertainty_mean", uncertainty.mean().item(), step)
            if straightener is not None:
                with torch.no_grad():
                    z_hat_orig = straightener.decode(z_hat)
                    straight_l1 = F.l1_loss(z_hat_orig, zt)
                writer.add_scalar("train/straight_l1", straight_l1.item(), step)
            writer.add_scalar("train/step_time_sec", step_time, step)
            writer.add_scalar("train/samples_per_sec", float(B) / max(step_time, 1e-8), step)
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            writer.add_scalar("train/max_mem_gb", max_mem, step)

        if args.val_pattern and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            metrics = _eval_model(
                model,
                args.val_pattern,
                T=args.T,
                batch=args.batch,
                num_batches=args.val_batches,
                min_gap=args.min_gap,
                device=device,
                autocast_dtype=autocast_dtype,
                model_dtype=model_dtype,
                num_workers=args.num_workers,
                shuffle_buffer=args.shuffle_buffer,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=bool(args.persistent_workers),
                pin_memory=bool(args.pin_memory),
                resampled=bool(args.resampled),
                seed=args.seed,
                straightener=straightener,
            )
            writer.add_scalar("val/l1", metrics["val_l1"], step)
            writer.add_scalar("val/lerp_l1", metrics["val_lerp_l1"], step)
            writer.add_scalar("val/unc_corr", metrics["val_unc_corr"], step)
            writer.add_scalar("val/unc_low", metrics["val_unc_low"], step)
            writer.add_scalar("val/unc_high", metrics["val_unc_high"], step)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "latent_flow_interpolator_wansynth",
                "T": args.T,
                "in_channels": C0,
                "base_channels": args.base_channels,
                "max_flow": args.max_flow,
                "residual_blocks": int(args.residual_blocks),
                "residual_channels": int(args.residual_channels),
                "min_gap": args.min_gap,
                "uncertainty_scale": args.uncertainty_scale,
                "uncertainty_loss_weight": args.uncertainty_loss_weight,
                "interp_mode": args.interp_mode,
                "time_mask": bool(args.time_mask),
                "gap_cond": bool(args.gap_cond),
                "cost_volume": bool(args.cost_volume),
                "cv_radius": int(args.cv_radius),
                "cv_downscale": int(args.cv_downscale),
                "cv_norm": bool(args.cv_norm),
                "endpoint_weight": args.endpoint_weight,
                "flow_consistency_weight": args.flow_consistency_weight,
                "gap_weight": args.gap_weight,
                "gap_gamma": args.gap_gamma,
                "edge_weight": args.edge_weight,
                "ms_weight": args.ms_weight,
                "ms_scales": args.ms_scales,
                "teacher_pattern": args.teacher_pattern,
                "teacher_weight": args.teacher_weight,
                "gt_weight": args.gt_weight,
                "straightener_ckpt": args.straightener_ckpt,
                "straightener_dtype": args.straightener_dtype,
                "model_dtype": str(model_dtype).replace("torch.", ""),
            }
            save_checkpoint(ckpt_path, model, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "latent_flow_interpolator_wansynth",
        "T": args.T,
        "in_channels": C0,
        "base_channels": args.base_channels,
        "max_flow": args.max_flow,
        "residual_blocks": int(args.residual_blocks),
        "residual_channels": int(args.residual_channels),
        "min_gap": args.min_gap,
        "uncertainty_scale": args.uncertainty_scale,
        "uncertainty_loss_weight": args.uncertainty_loss_weight,
        "interp_mode": args.interp_mode,
        "time_mask": bool(args.time_mask),
        "gap_cond": bool(args.gap_cond),
        "cost_volume": bool(args.cost_volume),
        "cv_radius": int(args.cv_radius),
        "cv_downscale": int(args.cv_downscale),
        "cv_norm": bool(args.cv_norm),
        "endpoint_weight": args.endpoint_weight,
        "flow_consistency_weight": args.flow_consistency_weight,
        "gap_weight": args.gap_weight,
        "gap_gamma": args.gap_gamma,
        "edge_weight": args.edge_weight,
        "ms_weight": args.ms_weight,
        "ms_scales": args.ms_scales,
        "teacher_pattern": args.teacher_pattern,
        "teacher_weight": args.teacher_weight,
        "gt_weight": args.gt_weight,
        "straightener_ckpt": args.straightener_ckpt,
        "straightener_dtype": args.straightener_dtype,
        "model_dtype": str(model_dtype).replace("torch.", ""),
    }
    save_checkpoint(final_path, model, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
