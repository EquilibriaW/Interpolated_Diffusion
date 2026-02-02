import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.toy_video import MovingShapesVideoDataset
from src.corruptions.keyframes import build_nested_masks_batch
from src.corruptions.video_keyframes import build_video_interp_adjacent_batch, build_video_interp_level_batch
from src.models.denoiser_interp_levels import InterpLevelDenoiser
from src.models.video_interpolator import TinyTemporalInterpolator
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype
from src.utils.ema import EMA
from src.utils.seed import set_seed
from src.utils.logging import create_writer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--latent_size", type=int, default=16)
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--stage2_mode", type=str, default="adj", choices=["adj", "x0"])
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--anchor_conf", type=int, default=1)
    p.add_argument("--anchor_conf_teacher", type=float, default=0.95)
    p.add_argument("--anchor_conf_student", type=float, default=0.5)
    p.add_argument("--anchor_conf_endpoints", type=float, default=1.0)
    p.add_argument("--anchor_conf_missing", type=float, default=0.0)
    p.add_argument("--anchor_conf_anneal", type=int, default=1)
    p.add_argument("--anchor_conf_anneal_mode", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--w_anchor", type=float, default=0.3)
    p.add_argument("--w_missing", type=float, default=1.0)
    p.add_argument("--corrupt_mode", type=str, default="dist", choices=["none", "dist", "gauss"])
    p.add_argument("--corrupt_sigma", type=float, default=0.02)
    p.add_argument("--corrupt_anchor_frac", type=float, default=0.25)
    p.add_argument("--student_replace_prob", type=float, default=0.5)
    p.add_argument("--student_noise_std", type=float, default=0.02)
    p.add_argument("--video_interp_mode", type=str, default="linear", choices=["linear", "smooth", "learned"])
    p.add_argument("--video_interp_ckpt", type=str, default="")
    p.add_argument("--video_interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/interp_levels_toy_video")
    p.add_argument("--log_dir", type=str, default="runs/interp_levels_toy_video")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")
    return p


def _anneal_conf(conf: torch.Tensor, s_idx: torch.Tensor, levels: int, mode: str) -> torch.Tensor:
    if conf is None or mode == "none" or levels <= 0:
        return conf
    frac = s_idx.float() / float(levels)
    if mode == "linear":
        lam = 1.0 - frac
    elif mode == "cosine":
        lam = 0.5 * (1.0 + torch.cos(torch.pi * frac))
    else:
        lam = torch.zeros_like(frac)
    lam = lam.view(-1, 1)
    return conf + (1.0 - conf) * lam


def _sample_level_indices(B: int, levels: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for toy interp-level training.")
    print(f"[toy interp] device={device}")
    print(f"[toy interp] cuda={torch.cuda.get_device_name(0)}")
    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    dataset = MovingShapesVideoDataset(
        T=args.T, n_samples=args.num_samples, seed=args.seed, latent_size=args.latent_size
    )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)

    data_dim = dataset.data_dim
    video_interp_model = None
    smooth_kernel = None
    if args.video_interp_mode == "smooth":
        smooth_kernel = torch.tensor([float(x) for x in args.video_interp_smooth_kernel.split(",")], dtype=torch.float32)
    elif args.video_interp_mode == "learned":
        if not args.video_interp_ckpt:
            raise ValueError("--video_interp_ckpt is required for video_interp_mode=learned")
        video_interp_model = TinyTemporalInterpolator(data_dim=data_dim)
        payload = torch.load(args.video_interp_ckpt, map_location="cpu")
        state = payload.get("model", payload)
        video_interp_model.load_state_dict(state)
        video_interp_model.to(device)
        video_interp_model.eval()

    mask_channels = (2 if args.stage2_mode == "adj" else 1) + (1 if args.anchor_conf else 0)
    model = InterpLevelDenoiser(
        data_dim=data_dim,
        use_sdf=False,
        max_levels=args.levels,
        use_start_goal=False,
        mask_channels=mask_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 23)

    model.train()
    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        z0 = batch["x"].to(device)

        masks_levels, idx_levels = build_nested_masks_batch(
            z0.shape[0],
            args.T,
            args.K_min,
            args.levels,
            generator=gen,
            device=device,
            k_schedule=args.k_schedule,
            k_geom_gamma=args.k_geom_gamma,
        )
        s_idx = _sample_level_indices(z0.shape[0], args.levels, gen, device)

        if args.stage2_mode == "adj":
            z_s, z_prev, mask_s, mask_prev, s_idx, _, _, conf_s, conf_prev = build_video_interp_adjacent_batch(
                z0,
                args.K_min,
                args.levels,
                gen,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma=args.corrupt_sigma,
                anchor_noise_frac=args.corrupt_anchor_frac,
                student_replace_prob=args.student_replace_prob,
                student_noise_std=args.student_noise_std,
                conf_anchor=args.anchor_conf_teacher,
                conf_student=args.anchor_conf_student,
                conf_endpoints=args.anchor_conf_endpoints,
                conf_missing=args.anchor_conf_missing,
                clamp_endpoints=False,
                interp_mode=args.video_interp_mode,
                interp_model=video_interp_model,
                smooth_kernel=smooth_kernel,
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
                prev_idx = torch.clamp(s_idx - 1, min=0)
                conf_prev = _anneal_conf(conf_prev, prev_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
            else:
                mask_in = torch.stack([mask_s, mask_prev], dim=-1)
            target = z_prev - z_s
            weight_mask = conf_prev if args.anchor_conf else mask_prev
        else:
            z_s, mask_s, s_idx, _, _, conf_s = build_video_interp_level_batch(
                z0,
                args.K_min,
                args.levels,
                gen,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma=args.corrupt_sigma,
                anchor_noise_frac=args.corrupt_anchor_frac,
                student_replace_prob=args.student_replace_prob,
                student_noise_std=args.student_noise_std,
                conf_anchor=args.anchor_conf_teacher,
                conf_student=args.anchor_conf_student,
                conf_endpoints=args.anchor_conf_endpoints,
                conf_missing=args.anchor_conf_missing,
                clamp_endpoints=False,
                interp_mode=args.video_interp_mode,
                interp_model=video_interp_model,
                smooth_kernel=smooth_kernel,
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), conf_s], dim=-1)
            else:
                mask_in = mask_s
            target = z0 - z_s
            weight_mask = conf_s if args.anchor_conf else mask_s

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            delta_hat = model(z_s, s_idx, mask_in, {})
            diff = (delta_hat - target).pow(2).sum(dim=-1)
            if args.anchor_conf:
                w = float(args.w_missing) + (float(args.w_anchor) - float(args.w_missing)) * weight_mask
            else:
                w = torch.where(
                    weight_mask,
                    torch.tensor(args.w_anchor, device=device),
                    torch.tensor(args.w_missing, device=device),
                )
            loss = (diff * w).sum() / (w.sum() * z0.shape[-1] + 1e-8)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model.parameters())

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item(), step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "interp_levels_toy_video",
                "T": args.T,
                "K_min": args.K_min,
                "levels": args.levels,
                "data_dim": data_dim,
                "stage2_mode": args.stage2_mode,
                "mask_channels": mask_channels,
                "k_schedule": args.k_schedule,
                "k_geom_gamma": args.k_geom_gamma,
                "anchor_conf": bool(args.anchor_conf),
                "anchor_conf_teacher": float(args.anchor_conf_teacher),
                "anchor_conf_student": float(args.anchor_conf_student),
                "anchor_conf_endpoints": float(args.anchor_conf_endpoints),
                "anchor_conf_missing": float(args.anchor_conf_missing),
                "anchor_conf_anneal": bool(args.anchor_conf_anneal),
                "anchor_conf_anneal_mode": args.anchor_conf_anneal_mode,
                "corrupt_mode": args.corrupt_mode,
                "corrupt_sigma": float(args.corrupt_sigma),
                "corrupt_anchor_frac": float(args.corrupt_anchor_frac),
                "student_replace_prob": float(args.student_replace_prob),
                "student_noise_std": float(args.student_noise_std),
                "video_interp_mode": args.video_interp_mode,
                "video_interp_ckpt": args.video_interp_ckpt,
                "video_interp_smooth_kernel": args.video_interp_smooth_kernel,
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "interp_levels_toy_video",
        "T": args.T,
        "K_min": args.K_min,
        "levels": args.levels,
        "data_dim": data_dim,
        "stage2_mode": args.stage2_mode,
        "mask_channels": mask_channels,
        "k_schedule": args.k_schedule,
        "k_geom_gamma": args.k_geom_gamma,
        "anchor_conf": bool(args.anchor_conf),
        "anchor_conf_teacher": float(args.anchor_conf_teacher),
        "anchor_conf_student": float(args.anchor_conf_student),
        "anchor_conf_endpoints": float(args.anchor_conf_endpoints),
        "anchor_conf_missing": float(args.anchor_conf_missing),
        "anchor_conf_anneal": bool(args.anchor_conf_anneal),
        "anchor_conf_anneal_mode": args.anchor_conf_anneal_mode,
        "corrupt_mode": args.corrupt_mode,
        "corrupt_sigma": float(args.corrupt_sigma),
        "corrupt_anchor_frac": float(args.corrupt_anchor_frac),
        "student_replace_prob": float(args.student_replace_prob),
        "student_noise_std": float(args.student_noise_std),
        "video_interp_mode": args.video_interp_mode,
        "video_interp_ckpt": args.video_interp_ckpt,
        "video_interp_smooth_kernel": args.video_interp_smooth_kernel,
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
