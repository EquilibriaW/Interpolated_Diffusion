import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch
from src.corruptions.video_keyframes import interpolate_video_from_indices
from src.models.video_interpolator import TinyTemporalInterpolator
from src.utils.video_tokens import patchify_latents
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.seed import set_seed
from src.utils.logging import create_writer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--w_anchor", type=float, default=0.3)
    p.add_argument("--w_missing", type=float, default=1.0)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/video_interp_wansynth")
    p.add_argument("--log_dir", type=str, default="runs/video_interp_wansynth")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=200)
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Wan synth interpolator training.")
    autocast_dtype = get_autocast_dtype()

    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shuffle=True,
        shardshuffle=True,
    )
    it = iter(loader)

    # Prime one batch to infer dims.
    batch0 = next(it)
    latents0 = batch0["latents"].to(device)
    tokens0, _ = patchify_latents(latents0, args.patch_size)
    B0, T0, N0, D0 = tokens0.shape
    if T0 != args.T:
        raise ValueError(f"T mismatch: batch={T0} args={args.T}")
    data_dim = D0

    model = TinyTemporalInterpolator(
        data_dim=data_dim, kernel_size=args.kernel_size, n_layers=args.n_layers, dropout=args.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt, ema=None, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 1)

    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    for step in pbar:
        if step == start_step:
            batch = batch0
        else:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
        latents = batch["latents"].to(device)
        tokens, _ = patchify_latents(latents, args.patch_size)
        B, T, N, D = tokens.shape

        z0 = tokens.permute(0, 2, 1, 3).reshape(B * N, T, D)
        idx, mask = sample_fixed_k_indices_uniform_batch(
            z0.shape[0], T, args.K, generator=gen, device=device, ensure_endpoints=True
        )
        vals = z0.gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D))
        z_interp = interpolate_video_from_indices(idx, vals, T, mode="linear")

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            z_pred = model(z_interp)
            z_pred = z_pred.scatter(1, idx.unsqueeze(-1).expand_as(vals), vals)
            diff = (z_pred - z0).pow(2).sum(-1)
            w = torch.where(
                mask,
                torch.full_like(mask.float(), float(args.w_anchor)),
                torch.full_like(mask.float(), float(args.w_missing)),
            )
            loss = (diff * w).sum() / (w.sum() * D + 1e-8)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "T": args.T,
                "data_dim": data_dim,
                "patch_size": args.patch_size,
                "kernel_size": args.kernel_size,
                "n_layers": args.n_layers,
            }
            save_checkpoint(ckpt_path, model, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "T": args.T,
        "data_dim": data_dim,
        "patch_size": args.patch_size,
        "kernel_size": args.kernel_size,
        "n_layers": args.n_layers,
    }
    save_checkpoint(final_path, model, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
