import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_straightener import LatentStraightener, LatentStraightenerTokenTransformer
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
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--arch", type=str, default="conv", choices=["conv", "token_transformer"])
    # conv straightener
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--kernel_size", type=int, default=3)
    # token-grid transformer straightener
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use_residual", type=int, default=1)
    p.add_argument("--model_dtype", type=str, default="")
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--lin_weight", type=float, default=1.0)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--interp_weight", type=float, default=1.0)
    p.add_argument("--id_weight", type=float, default=0.0)
    p.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--iso_weight", type=float, default=0.0)
    p.add_argument("--iso_mode", type=str, default="mean", choices=["mean", "unit"])
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=200)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/latent_straightener_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/latent_straightener_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=200)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--prefetch_to_gpu", type=int, default=1)
    p.add_argument("--compile", type=int, default=0)
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


def _move_batch_to_device(batch: dict, device: torch.device, model_dtype: torch.dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k == "latents":
                out[k] = v.to(device, dtype=model_dtype, non_blocking=True)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l2":
        return F.mse_loss(pred, target)
    return F.l1_loss(pred, target)


def _iso_loss(s: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    # s: [B,C,H,W] -> compute channel covariance across batch+spatial
    B, C, H, W = s.shape
    s_flat = s.permute(1, 0, 2, 3).reshape(C, B * H * W).float()
    s_flat = s_flat - s_flat.mean(dim=1, keepdim=True)
    denom = max(s_flat.shape[1] - 1, 1)
    cov = (s_flat @ s_flat.T) / float(denom)
    if mode == "unit":
        target = torch.eye(C, device=s.device, dtype=cov.dtype)
    else:
        mean_var = cov.diag().mean()
        target = torch.eye(C, device=s.device, dtype=cov.dtype) * mean_var
    diff = cov - target
    return (diff * diff).mean()


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


def _eval_model(
    model: LatentStraightener,
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

    err_lerp_all = []
    err_straight_all = []
    err_recon_all = []
    err_lin_all = []
    log_every = max(1, int(num_batches // 10))
    for i in range(num_batches):
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
        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=z0.dtype)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                s0 = model.encode(z0)
                s1 = model.encode(z1)
                st = model.encode(zt)
                s_lerp = (1.0 - alpha4) * s0 + alpha4 * s1
                z_hat = model.decode(s_lerp)
                z_recon = model.decode(st)

        z_lerp = (1.0 - alpha4) * z0 + alpha4 * z1
        err_lerp = (z_lerp - zt).abs().mean(dim=(1, 2, 3))
        err_straight = (z_hat - zt).abs().mean(dim=(1, 2, 3))
        err_recon = (z_recon - zt).abs().mean(dim=(1, 2, 3))
        err_lin = (s_lerp - st).abs().mean(dim=(1, 2, 3))
        err_lerp_all.append(err_lerp.detach().cpu())
        err_straight_all.append(err_straight.detach().cpu())
        err_recon_all.append(err_recon.detach().cpu())
        err_lin_all.append(err_lin.detach().cpu())

        if (i + 1) % log_every == 0 or (i + 1) == num_batches:
            print(f"[eval] batch {i + 1}/{num_batches}", flush=True)

    err_lerp = torch.cat(err_lerp_all, dim=0)
    err_straight = torch.cat(err_straight_all, dim=0)
    err_recon = torch.cat(err_recon_all, dim=0)
    err_lin = torch.cat(err_lin_all, dim=0)

    if model_was_training:
        model.train()

    return {
        "val_lerp_l1": float(err_lerp.mean().item()),
        "val_straight_l1": float(err_straight.mean().item()),
        "val_recon_l1": float(err_recon.mean().item()),
        "val_lin_l1": float(err_lin.mean().item()),
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth straightener training.")

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

    batch0 = next(it)
    latents0 = batch0["latents"].to(device, dtype=model_dtype, non_blocking=True)
    if latents0.dim() != 5:
        raise ValueError("latents must be [B,T,C,H,W]")
    _, T0, C0, _, _ = latents0.shape
    if T0 != args.T:
        raise ValueError(f"T mismatch: batch={T0} args={args.T}")

    if str(args.arch) == "token_transformer":
        model = LatentStraightenerTokenTransformer(
            in_channels=C0,
            patch_size=int(args.patch_size),
            d_model=int(args.d_model),
            n_layers=int(args.n_layers),
            n_heads=int(args.n_heads),
            d_ff=int(args.d_ff),
            dropout=float(args.dropout),
            use_residual=bool(args.use_residual),
        ).to(device=device, dtype=model_dtype)
    else:
        model = LatentStraightener(
            in_channels=C0,
            hidden_channels=args.hidden_channels,
            blocks=args.blocks,
            kernel_size=args.kernel_size,
            use_residual=bool(args.use_residual),
        ).to(device=device, dtype=model_dtype)
    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

        B, T, _, _, _ = latents.shape
        t0, t1, t, alpha = _sample_triplets(B, T, args.min_gap, gen, device)
        z0 = latents[torch.arange(B, device=device), t0]
        z1 = latents[torch.arange(B, device=device), t1]
        zt = latents[torch.arange(B, device=device), t]
        alpha4 = alpha.view(-1, 1, 1, 1).to(dtype=z0.dtype)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            s0 = model.encode(z0)
            s1 = model.encode(z1)
            st = model.encode(zt)
            s_lerp = (1.0 - alpha4) * s0 + alpha4 * s1
            z_hat = model.decode(s_lerp)
            z_recon = model.decode(st)

            lin_loss = _loss(s_lerp, st, args.loss_type)
            recon_loss = _loss(z_recon, zt, args.loss_type)
            interp_loss = _loss(z_hat, zt, args.loss_type)
            if args.id_weight > 0.0:
                id_loss = _loss(st, zt, args.loss_type)
            else:
                id_loss = torch.tensor(0.0, device=device)
            if args.iso_weight > 0.0:
                iso_loss = _iso_loss(st, mode=args.iso_mode)
            else:
                iso_loss = torch.tensor(0.0, device=device)
            loss = (
                float(args.lin_weight) * lin_loss
                + float(args.recon_weight) * recon_loss
                + float(args.interp_weight) * interp_loss
                + float(args.id_weight) * id_loss
                + float(args.iso_weight) * iso_loss
            )

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
            writer.add_scalar("train/lin_loss", lin_loss.item(), step)
            writer.add_scalar("train/recon_loss", recon_loss.item(), step)
            writer.add_scalar("train/interp_loss", interp_loss.item(), step)
            writer.add_scalar("train/id_loss", id_loss.item(), step)
            writer.add_scalar("train/iso_loss", iso_loss.item(), step)
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
            )
            writer.add_scalar("val/lerp_l1", metrics["val_lerp_l1"], step)
            writer.add_scalar("val/straight_l1", metrics["val_straight_l1"], step)
            writer.add_scalar("val/recon_l1", metrics["val_recon_l1"], step)
            writer.add_scalar("val/lin_l1", metrics["val_lin_l1"], step)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "latent_straightener_wansynth",
                "T": args.T,
                "in_channels": C0,
                "arch": str(args.arch),
                "hidden_channels": int(args.hidden_channels),
                "blocks": int(args.blocks),
                "kernel_size": int(args.kernel_size),
                "patch_size": int(args.patch_size),
                "d_model": int(args.d_model),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "d_ff": int(args.d_ff),
                "dropout": float(args.dropout),
                "use_residual": bool(args.use_residual),
                "min_gap": args.min_gap,
                "lin_weight": args.lin_weight,
                "recon_weight": args.recon_weight,
                "interp_weight": args.interp_weight,
                "id_weight": args.id_weight,
                "loss_type": args.loss_type,
                "iso_weight": args.iso_weight,
                "iso_mode": args.iso_mode,
                "model_dtype": str(model_dtype).replace("torch.", ""),
            }
            save_checkpoint(ckpt_path, model, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "latent_straightener_wansynth",
        "T": args.T,
        "in_channels": C0,
        "arch": str(args.arch),
        "hidden_channels": int(args.hidden_channels),
        "blocks": int(args.blocks),
        "kernel_size": int(args.kernel_size),
        "patch_size": int(args.patch_size),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "d_ff": int(args.d_ff),
        "dropout": float(args.dropout),
        "use_residual": bool(args.use_residual),
        "min_gap": args.min_gap,
        "lin_weight": args.lin_weight,
        "recon_weight": args.recon_weight,
        "interp_weight": args.interp_weight,
        "id_weight": args.id_weight,
        "loss_type": args.loss_type,
        "iso_weight": args.iso_weight,
        "iso_mode": args.iso_mode,
        "model_dtype": str(model_dtype).replace("torch.", ""),
    }
    save_checkpoint(final_path, model, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
