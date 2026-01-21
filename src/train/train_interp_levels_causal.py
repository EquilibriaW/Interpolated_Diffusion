import argparse
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import build_nested_masks_batch, interpolate_from_indices
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_interp_levels_causal import InterpLevelCausalDenoiser
from src.models.denoiser_keypoints import KeypointDenoiser
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype, get_device
from src.utils.ema import EMA
from src.utils.logging import create_writer
from src.utils.normalize import logit_pos, sigmoid_pos
from src.utils.seed import get_seed_from_env, set_seed


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K_min", type=int, default=8)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--recompute_vel", type=int, default=1)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default="d4rl", choices=["particle", "synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="runs/interp_levels_causal")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/interp_levels_causal")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_checkpoint", type=int, default=0)
    p.add_argument("--w_anchor", type=float, default=0.1)
    p.add_argument("--w_missing", type=float, default=1.0)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--allow_tf32", type=int, default=1)
    p.add_argument("--enable_flash_sdp", type=int, default=1)
    p.add_argument("--bootstrap_stage1_ckpt", type=str, default=None)
    p.add_argument("--bootstrap_use_ema", type=int, default=1)
    p.add_argument("--bootstrap_ddim_steps", type=int, default=5)
    p.add_argument("--bootstrap_prob_start", type=float, default=0.0)
    p.add_argument("--bootstrap_prob_end", type=float, default=0.3)
    p.add_argument("--bootstrap_warmup_steps", type=int, default=5000)
    p.add_argument("--bootstrap_prob_cap", type=float, default=0.5)
    p.add_argument("--bootstrap_mode", type=str, default="batch", choices=["batch", "per_example"])
    return p


def _build_known_mask_values(idx: torch.Tensor, cond: dict, D: int, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K = idx.shape
    known_mask = torch.zeros((B, K, D), device=idx.device, dtype=torch.bool)
    known_values = torch.zeros((B, K, D), device=idx.device, dtype=torch.float32)
    if "start_goal" in cond and D >= 2:
        start = cond["start_goal"][:, :2]
        goal = cond["start_goal"][:, 2:]
        start_pos = start.unsqueeze(1).expand(B, K, 2)
        goal_pos = goal.unsqueeze(1).expand(B, K, 2)
        mask_start = (idx == 0).unsqueeze(-1)
        mask_goal = (idx == T - 1).unsqueeze(-1)
        known_mask[:, :, :2] = mask_start | mask_goal
        known_values[:, :, :2] = torch.where(mask_start, start_pos, known_values[:, :, :2])
        known_values[:, :, :2] = torch.where(mask_goal, goal_pos, known_values[:, :, :2])
    return known_mask, known_values


def _sample_keypoints_ddim(
    model,
    schedule,
    idx: torch.Tensor,
    known_mask: torch.Tensor,
    known_values: torch.Tensor,
    cond: dict,
    steps: int,
    T: int,
):
    device = idx.device
    B, K = idx.shape
    D = known_values.shape[-1]
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps)
    z = torch.randn((B, K, D), device=device)
    z = torch.where(known_mask, known_values, z)
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(z, t, idx, known_mask, cond, T)
        z = ddim_step(z, eps, t, t_prev, schedule, eta=0.0)
        z = torch.where(known_mask, known_values, z)
    return z


def build_interp_level_batch(
    x0: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    recompute_velocity: bool = False,
    x0_override: Optional[torch.Tensor] = None,
    masks_levels: Optional[torch.Tensor] = None,
    idx_levels: Optional[List[torch.Tensor]] = None,
):
    B, T, D = x0.shape
    device = x0.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    x_s = torch.zeros_like(x0)
    mask_s = torch.zeros((B, T), dtype=torch.bool, device=device)
    s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)
    source = x0_override if x0_override is not None else x0
    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        vals = source[sel].gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D))
        xs = interpolate_from_indices(idx, vals, T, recompute_velocity=recompute_velocity)
        x_s[sel] = xs
        mask_s[sel] = masks_levels[sel, s]
    return x_s, mask_s, s_idx, masks_levels, idx_levels


def main():
    args = build_argparser().parse_args()
    if args.dataset != "d4rl":
        raise ValueError("Particle/synthetic datasets are disabled; use --dataset d4rl with Maze2D envs.")
    seed = args.seed if args.seed is not None else get_seed_from_env()
    set_seed(seed, deterministic=bool(args.deterministic))

    device = get_device(args.device)
    if device.type == "cuda" and not bool(args.deterministic):
        torch.backends.cudnn.benchmark = True
        if bool(args.allow_tf32):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        try:
            torch.backends.cuda.enable_flash_sdp(bool(args.enable_flash_sdp))
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass
    autocast_dtype = get_autocast_dtype()
    use_fp16 = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    dataset_name = "synthetic" if args.dataset == "synthetic" else args.dataset
    if dataset_name == "d4rl":
        dataset = D4RLMazeDataset(
            env_id=args.env_id,
            num_samples=args.num_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            seed=seed,
            flip_y=bool(args.d4rl_flip_y),
        )
    else:
        dataset = ParticleMazeDataset(
            num_samples=args.num_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            cache_dir=args.cache_dir,
        )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    it = iter(loader)

    data_dim = 4 if args.with_velocity else 2
    model = InterpLevelCausalDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        max_levels=args.levels,
        use_checkpoint=bool(args.use_checkpoint),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    bootstrap_model = None
    bootstrap_schedule = None
    bootstrap_logit = False
    bootstrap_logit_eps = 1e-5
    if args.bootstrap_stage1_ckpt is not None:
        bootstrap_model = KeypointDenoiser(
            data_dim=data_dim,
            use_sdf=bool(args.use_sdf),
        ).to(device)
        if args.bootstrap_use_ema:
            ema_boot = EMA(bootstrap_model.parameters())
        else:
            ema_boot = None
        _, payload = load_checkpoint(
            args.bootstrap_stage1_ckpt,
            bootstrap_model,
            optimizer=None,
            ema=ema_boot,
            map_location=device,
            return_payload=True,
        )
        if ema_boot is not None:
            ema_boot.copy_to(bootstrap_model.parameters())
        bootstrap_model.eval()
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta.get("stage") != "keypoints":
            raise ValueError("bootstrap_stage1_ckpt does not appear to be a keypoints checkpoint")
        n_train = meta.get("N_train")
        schedule_name = meta.get("schedule")
        if n_train is None or schedule_name is None:
            raise ValueError("Keypoint checkpoint missing meta for N_train/schedule")
        if meta.get("logit_space") is not None:
            bootstrap_logit = bool(meta.get("logit_space"))
        if meta.get("logit_eps") is not None:
            bootstrap_logit_eps = float(meta.get("logit_eps"))
        betas_boot = make_beta_schedule(schedule_name, n_train).to(device)
        bootstrap_schedule = make_alpha_bars(betas_boot)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 29)

    model.train()
    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x0 = batch["x"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}

        masks_levels, idx_levels = build_nested_masks_batch(
            x0.shape[0], args.T, args.K_min, args.levels, generator=gen, device=device
        )

        x0_used = x0
        if bootstrap_model is not None:
            if args.bootstrap_warmup_steps <= 0:
                p_boot = args.bootstrap_prob_end
            else:
                frac = min(1.0, float(step) / float(args.bootstrap_warmup_steps))
                p_boot = args.bootstrap_prob_start + frac * (args.bootstrap_prob_end - args.bootstrap_prob_start)
            p_boot = min(p_boot, args.bootstrap_prob_cap)
            if args.bootstrap_mode == "batch":
                use_boot = torch.rand((), generator=gen, device=device) < p_boot
                use_mask = torch.full((x0.shape[0],), bool(use_boot), device=device, dtype=torch.bool)
            else:
                use_mask = torch.rand((x0.shape[0],), generator=gen, device=device) < p_boot
            if torch.any(use_mask):
                idx_s = idx_levels[args.levels]
                known_mask, known_values = _build_known_mask_values(idx_s, cond, data_dim, args.T)
                if bootstrap_logit:
                    known_values = logit_pos(known_values, eps=bootstrap_logit_eps)
                with torch.no_grad():
                    z_hat = _sample_keypoints_ddim(
                        bootstrap_model,
                        bootstrap_schedule,
                        idx_s,
                        known_mask,
                        known_values,
                        cond,
                        args.bootstrap_ddim_steps,
                        args.T,
                    )
                    if bootstrap_logit:
                        z_hat = sigmoid_pos(z_hat)
                x0_aug = x0.clone()
                if use_mask.all():
                    x0_aug[:, :, :2].scatter_(
                        1, idx_s.unsqueeze(-1).expand(-1, idx_s.shape[1], 2), z_hat[:, :, :2]
                    )
                    x0_used = x0_aug
                else:
                    sel_idx = idx_s[use_mask]
                    if sel_idx.numel() > 0:
                        x0_sel = x0_aug[use_mask]
                        z_sel = z_hat[use_mask]
                        x0_sel[:, :, :2].scatter_(
                            1, sel_idx.unsqueeze(-1).expand(-1, sel_idx.shape[1], 2), z_sel[:, :, :2]
                        )
                        x0_aug[use_mask] = x0_sel
                        x0_used = torch.where(use_mask.view(-1, 1, 1), x0_aug, x0)

        x_s, mask_s, s_idx, _, _ = build_interp_level_batch(
            x0, args.K_min, args.levels, gen, recompute_velocity=bool(args.recompute_vel),
            x0_override=x0_used, masks_levels=masks_levels, idx_levels=idx_levels
        )
        target = x0 - x_s

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            delta_hat = model(x_s, s_idx, mask_s, cond)
            diff = (delta_hat - target) ** 2
            diff = diff.sum(dim=-1)
            w = torch.where(mask_s, torch.tensor(args.w_anchor, device=device), torch.tensor(args.w_missing, device=device))
            loss = (diff * w).sum() / (w.sum() * x0.shape[-1] + 1e-8)
            loss = loss / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model.parameters())

        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f}")
            writer.add_scalar("train/loss", loss.item() * args.grad_accum, step)

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            save_checkpoint(ckpt_path, model, optimizer, step, ema)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    save_checkpoint(final_path, model, optimizer, args.steps, ema)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
