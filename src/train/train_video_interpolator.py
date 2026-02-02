import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.toy_video import MovingShapesVideoDataset
from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch
from src.corruptions.video_keyframes import interpolate_video_from_indices
from src.models.video_interpolator import TinyTemporalInterpolator
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.seed import set_seed
from src.utils.logging import create_writer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--latent_size", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--w_anchor", type=float, default=0.3)
    p.add_argument("--w_missing", type=float, default=1.0)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/video_interp")
    p.add_argument("--log_dir", type=str, default="runs/video_interp")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--save_preview_every", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for toy interpolator training.")
    print(f"[toy interp model] device={device}")
    print(f"[toy interp model] cuda={torch.cuda.get_device_name(0)}")
    autocast_dtype = get_autocast_dtype()

    dataset = MovingShapesVideoDataset(
        T=args.T, H=args.H, n_samples=args.num_samples, seed=args.seed, latent_size=args.latent_size
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)

    data_dim = dataset.data_dim
    model = TinyTemporalInterpolator(data_dim=data_dim, kernel_size=args.kernel_size, n_layers=args.n_layers, dropout=args.dropout).to(device)
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
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        z0 = batch["x"].to(device)
        B, T, D = z0.shape

        idx, mask = sample_fixed_k_indices_uniform_batch(B, T, args.K, generator=gen, device=device, ensure_endpoints=True)
        vals = z0.gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D))
        z_interp = interpolate_video_from_indices(idx, vals, T, mode="linear")

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            z_pred = model(z_interp)
            z_pred = z_pred.scatter(1, idx.unsqueeze(-1).expand_as(vals), vals)
            diff = (z_pred - z0).pow(2).sum(-1)
            w = torch.where(mask, torch.full_like(mask.float(), float(args.w_anchor)), torch.full_like(mask.float(), float(args.w_missing)))
            loss = (diff * w).sum() / (w.sum() * D + 1e-8)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), step)

        if args.save_preview_every > 0 and step % args.save_preview_every == 0:
            try:
                import imageio.v2 as imageio
            except Exception:
                imageio = None
            if imageio is not None:
                from src.data.toy_video import decode_latents

                frames_dir = os.path.join(args.log_dir, "preview")
                os.makedirs(frames_dir, exist_ok=True)
                gt = decode_latents(z0[0])
                interp = decode_latents(z_interp[0])
                pred = decode_latents(z_pred[0])
                for t in range(gt.shape[0]):
                    tiles = [interp[t], pred[t], gt[t]]
                    frame = torch.cat(tiles, dim=2).detach().cpu().clamp(0, 1)
                    frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    imageio.imwrite(os.path.join(frames_dir, f"step_{step:07d}_t{t:03d}.png"), frame)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "T": args.T,
                "latent_size": args.latent_size,
                "data_dim": data_dim,
                "kernel_size": args.kernel_size,
                "n_layers": args.n_layers,
            }
            save_checkpoint(ckpt_path, model, opt, step, ema=None, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "T": args.T,
        "latent_size": args.latent_size,
        "data_dim": data_dim,
        "kernel_size": args.kernel_size,
        "n_layers": args.n_layers,
    }
    save_checkpoint(final_path, model, opt, args.steps, ema=None, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
