import argparse
import sys
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import build_nested_masks_batch, interpolate_from_indices
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset, PreparedTrajectoryDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_interp_levels import InterpLevelDenoiser
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
    p.add_argument("--stage2_mode", type=str, default="adj", choices=["x0", "adj"])
    p.add_argument("--level_sampling", type=str, default="high", choices=["uniform", "high"])
    p.add_argument("--level_high_prob", type=float, default=0.5)
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
    p.add_argument("--dataset", type=str, default="d4rl", choices=["particle", "synthetic", "d4rl", "d4rl_prepared"])
    p.add_argument("--prepared_path", type=str, default=None)
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--d4rl_flip_y", type=int, default=0)
    p.add_argument("--max_collision_rate", type=float, default=0.0)
    p.add_argument("--max_resample_tries", type=int, default=50)
    p.add_argument("--min_goal_dist", type=float, default=None)
    p.add_argument("--min_path_len", type=float, default=None)
    p.add_argument("--min_tortuosity", type=float, default=None)
    p.add_argument("--min_turns", type=int, default=None)
    p.add_argument("--turn_angle_deg", type=float, default=30.0)
    p.add_argument("--window_mode", type=str, default="end", choices=["end", "random", "episode"])
    p.add_argument("--goal_mode", type=str, default="window_end", choices=["env", "window_end"])
    p.add_argument("--episode_split_mod", type=int, default=None)
    p.add_argument("--episode_split_val", type=int, default=0)
    p.add_argument("--use_start_goal", type=int, default=1, help="Deprecated. Use --clamp_endpoints/--cond_start_goal.")
    p.add_argument("--clamp_endpoints", type=int, default=1)
    p.add_argument("--cond_start_goal", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="runs/interp_levels")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/interp_levels")
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
    p.add_argument("--bootstrap_ddim_schedule", type=str, default="quadratic", choices=["linear", "quadratic", "sqrt"])
    p.add_argument("--bootstrap_prob_start", type=float, default=0.0)
    p.add_argument("--bootstrap_prob_end", type=float, default=0.3)
    p.add_argument("--bootstrap_warmup_steps", type=int, default=5000)
    p.add_argument("--bootstrap_prob_cap", type=float, default=0.5)
    p.add_argument("--bootstrap_mode", type=str, default="batch", choices=["batch", "per_example"])
    p.add_argument("--bootstrap_replace_prob", type=float, default=0.5)
    p.add_argument("--k_schedule", type=str, default="doubling", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--anchor_conf", type=int, default=1)
    p.add_argument("--anchor_conf_teacher", type=float, default=0.95)
    p.add_argument("--anchor_conf_student", type=float, default=0.5)
    p.add_argument("--anchor_conf_endpoints", type=float, default=1.0)
    p.add_argument("--anchor_conf_missing", type=float, default=0.0)
    p.add_argument("--anchor_conf_anneal", type=int, default=1)
    p.add_argument("--anchor_conf_anneal_mode", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--corrupt_mode", type=str, default="dist", choices=["none", "dist"])
    p.add_argument("--corrupt_sigma_max", type=float, default=0.08)
    p.add_argument("--corrupt_sigma_min", type=float, default=0.012)
    p.add_argument("--corrupt_sigma_pow", type=float, default=0.75)
    p.add_argument("--corrupt_anchor_frac", type=float, default=0.25)
    p.add_argument("--corrupt_index_jitter_max", type=int, default=0)
    p.add_argument("--corrupt_index_jitter_prob", type=float, default=0.0)
    p.add_argument("--corrupt_index_jitter_pow", type=float, default=1.0)
    p.add_argument("--debug_corrupt_stats", type=int, default=0)
    p.add_argument("--debug_corrupt_every", type=int, default=500)
    return p


def _build_known_mask_values(
    idx: torch.Tensor, cond: dict, D: int, T: int, clamp_endpoints: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K = idx.shape
    known_mask = torch.zeros((B, K, D), device=idx.device, dtype=torch.bool)
    known_values = torch.zeros((B, K, D), device=idx.device, dtype=torch.float32)
    if clamp_endpoints:
        if "start_goal" not in cond:
            raise ValueError("clamp_endpoints=True but start_goal missing from cond")
        if D < 2:
            return known_mask, known_values
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
    schedule_name: str = "linear",
):
    device = idx.device
    B, K = idx.shape
    D = known_values.shape[-1]
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps, schedule=schedule_name)
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
    s_idx: Optional[torch.Tensor] = None,
    corrupt_mode: str = "none",
    corrupt_sigma_max: float = 0.0,
    corrupt_sigma_min: float = 0.0,
    corrupt_sigma_pow: float = 1.0,
    corrupt_anchor_frac: float = 0.0,
    corrupt_index_jitter_max: int = 0,
    corrupt_index_jitter_prob: float = 0.0,
    corrupt_index_jitter_pow: float = 1.0,
    clamp_endpoints: bool = True,
):
    B, T, D = x0.shape
    device = x0.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    x_s = torch.zeros_like(x0)
    mask_s = torch.zeros((B, T), dtype=torch.bool, device=device)
    if s_idx is None:
        s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)
    source = x0_override if x0_override is not None else x0
    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        vals = source[sel].gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D))
        if corrupt_mode != "none":
            K_s = idx.shape[1]
            sigma = _compute_sigma_for_level(K_s, K_min, corrupt_sigma_max, corrupt_sigma_min, corrupt_sigma_pow)
            anchor_sigma = sigma * float(corrupt_anchor_frac)
            jitter = _compute_jitter_for_level(
                K_s, K_min, corrupt_index_jitter_max, corrupt_index_jitter_pow
            )
            xs = _corrupt_from_anchors(
                source[sel],
                idx,
                T,
                generator,
                sigma,
                anchor_sigma,
                jitter,
                corrupt_index_jitter_prob,
                clamp_endpoints,
                recompute_velocity,
            )
        else:
            xs = interpolate_from_indices(idx, vals, T, recompute_velocity=recompute_velocity)
        x_s[sel] = xs
        mask_s[sel] = masks_levels[sel, s]
    return x_s, mask_s, s_idx, masks_levels, idx_levels


def build_interp_adjacent_batch(
    x0: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    recompute_velocity: bool = False,
    x0_override: Optional[torch.Tensor] = None,
    masks_levels: Optional[torch.Tensor] = None,
    idx_levels: Optional[List[torch.Tensor]] = None,
    s_idx: Optional[torch.Tensor] = None,
    corrupt_mode: str = "none",
    corrupt_sigma_max: float = 0.0,
    corrupt_sigma_min: float = 0.0,
    corrupt_sigma_pow: float = 1.0,
    corrupt_anchor_frac: float = 0.0,
    corrupt_index_jitter_max: int = 0,
    corrupt_index_jitter_prob: float = 0.0,
    corrupt_index_jitter_pow: float = 1.0,
    clamp_endpoints: bool = True,
):
    B, T, D = x0.shape
    device = x0.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    x_s = torch.zeros_like(x0)
    x_prev = torch.zeros_like(x0)
    mask_s = torch.zeros((B, T), dtype=torch.bool, device=device)
    mask_prev = torch.zeros((B, T), dtype=torch.bool, device=device)
    if s_idx is None:
        s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)
    source = x0_override if x0_override is not None else x0
    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        idx_prev = idx_levels[s - 1][sel]
        vals = source[sel].gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D))
        vals_prev = source[sel].gather(1, idx_prev.unsqueeze(-1).expand(-1, idx_prev.shape[1], D))
        if corrupt_mode != "none":
            K_s = idx.shape[1]
            K_prev = idx_prev.shape[1]
            sigma_s = _compute_sigma_for_level(K_s, K_min, corrupt_sigma_max, corrupt_sigma_min, corrupt_sigma_pow)
            sigma_prev = _compute_sigma_for_level(K_prev, K_min, corrupt_sigma_max, corrupt_sigma_min, corrupt_sigma_pow)
            anchor_sigma_s = sigma_s * float(corrupt_anchor_frac)
            anchor_sigma_prev = sigma_prev * float(corrupt_anchor_frac)
            jitter_s = _compute_jitter_for_level(
                K_s, K_min, corrupt_index_jitter_max, corrupt_index_jitter_pow
            )
            jitter_prev = _compute_jitter_for_level(
                K_prev, K_min, corrupt_index_jitter_max, corrupt_index_jitter_pow
            )
            x_s[sel] = _corrupt_from_anchors(
                source[sel],
                idx,
                T,
                generator,
                sigma_s,
                anchor_sigma_s,
                jitter_s,
                corrupt_index_jitter_prob,
                clamp_endpoints,
                recompute_velocity,
            )
            x_prev[sel] = _corrupt_from_anchors(
                source[sel],
                idx_prev,
                T,
                generator,
                sigma_prev,
                anchor_sigma_prev,
                jitter_prev,
                corrupt_index_jitter_prob,
                clamp_endpoints,
                recompute_velocity,
            )
        else:
            x_s[sel] = interpolate_from_indices(idx, vals, T, recompute_velocity=recompute_velocity)
            x_prev[sel] = interpolate_from_indices(idx_prev, vals_prev, T, recompute_velocity=recompute_velocity)
        mask_s[sel] = masks_levels[sel, s]
        mask_prev[sel] = masks_levels[sel, s - 1]
    return x_s, x_prev, mask_s, mask_prev, s_idx, masks_levels, idx_levels


def _compute_sigma_for_level(
    K_s: int,
    K_min: int,
    sigma_max: float,
    sigma_min: float,
    sigma_pow: float,
) -> float:
    if sigma_max <= 0.0:
        return 0.0
    K_s = max(1, int(K_s))
    K_min = max(1, int(K_min))
    ratio = float(K_min) / float(K_s)
    sigma = float(sigma_max) * (ratio ** float(sigma_pow))
    sigma = min(float(sigma_max), sigma)
    sigma = max(float(sigma_min), sigma)
    return sigma


def _compute_jitter_for_level(K_s: int, K_min: int, jitter_max: int, jitter_pow: float) -> int:
    if jitter_max <= 0:
        return 0
    K_s = max(1, int(K_s))
    K_min = max(1, int(K_min))
    ratio = float(K_min) / float(K_s)
    jitter = int(round(float(jitter_max) * (ratio ** float(jitter_pow))))
    jitter = max(0, min(int(jitter_max), jitter))
    return jitter


def _distance_alpha(idx: torch.Tensor, T: int) -> torch.Tensor:
    B, K = idx.shape
    device = idx.device
    t_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    seg = torch.searchsorted(idx, t_grid, right=True) - 1
    seg = seg.clamp(0, K - 2)
    left_idx = idx.gather(1, seg)
    right_idx = idx.gather(1, seg + 1)
    gap = (right_idx - left_idx).clamp(min=1)
    dist = torch.minimum(t_grid - left_idx, right_idx - t_grid)
    alpha = (2.0 * dist.float() / gap.float()).clamp(0.0, 1.0)
    return alpha.unsqueeze(-1)


def _corrupt_from_anchors(
    source: torch.Tensor,
    idx: torch.Tensor,
    T: int,
    generator: torch.Generator,
    sigma: float,
    anchor_sigma: float,
    index_jitter: int,
    index_jitter_prob: float,
    clamp_endpoints: bool,
    recompute_velocity: bool,
) -> torch.Tensor:
    B, _, D = source.shape
    idx_j = idx
    if index_jitter > 0 and index_jitter_prob > 0.0:
        jitter = torch.randint(
            0,
            2 * index_jitter + 1,
            (B, idx.shape[1]),
            generator=generator,
            device=idx.device,
        ) - int(index_jitter)
        use = torch.rand((B, idx.shape[1]), generator=generator, device=idx.device) < float(index_jitter_prob)
        if clamp_endpoints:
            use = use & ~(idx == 0) & ~(idx == (T - 1))
        idx_j = torch.where(use, idx + jitter, idx).clamp(0, T - 1)
    vals = source.gather(1, idx_j.unsqueeze(-1).expand(-1, idx.shape[1], D)).clone()
    if anchor_sigma > 0.0:
        noise_vals = torch.zeros_like(vals)
        noise_vals[:, :, :2] = torch.randn(
            (B, idx.shape[1], 2), generator=generator, device=source.device, dtype=source.dtype
        ) * float(anchor_sigma)
        if clamp_endpoints:
            mask_end = (idx == 0) | (idx == T - 1)
            noise_vals[mask_end] = 0.0
        vals[:, :, :2] = vals[:, :, :2] + noise_vals[:, :, :2]
    x = interpolate_from_indices(idx, vals, T, recompute_velocity=False)
    if sigma > 0.0:
        alpha = _distance_alpha(idx, T)
        noise = torch.zeros_like(x)
        noise[:, :, :2] = torch.randn(
            (B, T, 2), generator=generator, device=source.device, dtype=source.dtype
        ) * float(sigma)
        x[:, :, :2] = x[:, :, :2] + noise[:, :, :2] * alpha
    if recompute_velocity and D == 4:
        pos = x[:, :, :2]
        v = torch.zeros_like(pos)
        dt = 1.0 / float(T)
        v[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / dt
        v[:, -1] = 0.0
        x = torch.cat([pos, v], dim=-1)
    return x


def _log_corrupt_stats(
    writer,
    step: int,
    s_idx: torch.Tensor,
    levels: int,
    x0: torch.Tensor,
    x_s: torch.Tensor,
    x_prev: Optional[torch.Tensor] = None,
):
    lines = []
    with torch.no_grad():
        for s in range(1, levels + 1):
            sel = s_idx == s
            if not torch.any(sel):
                continue
            xs = x_s[sel][..., :2]
            x0s = x0[sel][..., :2]
            d_x0 = (x0s - xs).pow(2).sum(-1).sqrt().mean()
            if x_prev is not None:
                xps = x_prev[sel][..., :2]
                d_adj = (xps - xs).pow(2).sum(-1).sqrt().mean()
                lines.append(f"s{s}: ||x0-xs||={d_x0.item():.4f} ||xprev-xs||={d_adj.item():.4f}")
                if writer is not None:
                    writer.add_scalar(f"debug/corrupt_x0_s{s}", d_x0.item(), step)
                    writer.add_scalar(f"debug/corrupt_adj_s{s}", d_adj.item(), step)
            else:
                lines.append(f"s{s}: ||x0-xs||={d_x0.item():.4f}")
                if writer is not None:
                    writer.add_scalar(f"debug/corrupt_x0_s{s}", d_x0.item(), step)
    if lines:
        print(f"[debug] corrupt_stats step={step} | " + " | ".join(lines))


def _build_anchor_conf(
    mask_s: torch.Tensor,
    student_mask: Optional[torch.Tensor],
    conf_teacher: float,
    conf_student: float,
    conf_endpoints: float,
    conf_missing: float,
    clamp_endpoints: bool,
) -> torch.Tensor:
    conf = torch.full_like(mask_s.float(), float(conf_missing))
    conf = torch.where(mask_s, torch.full_like(conf, float(conf_teacher)), conf)
    if student_mask is not None:
        conf = torch.where(student_mask & mask_s, torch.full_like(conf, float(conf_student)), conf)
    if clamp_endpoints:
        conf[:, 0] = float(conf_endpoints)
        conf[:, -1] = float(conf_endpoints)
    return conf


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

def _sample_level_indices(
    B: int,
    levels: int,
    generator: torch.Generator,
    device: torch.device,
    mode: str,
    high_prob: float,
) -> torch.Tensor:
    if mode == "uniform" or levels <= 1:
        return torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)
    high_prob = float(max(0.0, min(1.0, high_prob)))
    draw = torch.rand((B,), generator=generator, device=device)
    s_idx = torch.empty((B,), device=device, dtype=torch.long)
    high = draw < high_prob
    if torch.any(high):
        s_idx[high] = levels
    if torch.any(~high):
        s_idx[~high] = torch.randint(1, levels + 1, (int((~high).sum().item()),), generator=generator, device=device, dtype=torch.long)
    return s_idx


def main():
    args = build_argparser().parse_args()
    if "--clamp_endpoints" not in sys.argv and "--cond_start_goal" not in sys.argv:
        args.clamp_endpoints = int(bool(args.use_start_goal))
        args.cond_start_goal = int(bool(args.use_start_goal))
    else:
        if "--clamp_endpoints" not in sys.argv:
            args.clamp_endpoints = int(bool(args.use_start_goal))
        if "--cond_start_goal" not in sys.argv:
            args.cond_start_goal = int(bool(args.use_start_goal))
    args.use_start_goal = int(bool(args.cond_start_goal))
    if args.dataset not in {"d4rl", "d4rl_prepared"}:
        raise ValueError("Particle/synthetic datasets are disabled; use --dataset d4rl or d4rl_prepared.")
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
    if dataset_name == "d4rl_prepared":
        if args.prepared_path is None:
            raise ValueError("--prepared_path is required for dataset d4rl_prepared")
        dataset = PreparedTrajectoryDataset(args.prepared_path, use_sdf=bool(args.use_sdf))
    elif dataset_name == "d4rl":
        dataset = D4RLMazeDataset(
            env_id=args.env_id,
            num_samples=args.num_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            seed=seed,
            flip_y=bool(args.d4rl_flip_y),
            max_collision_rate=args.max_collision_rate,
            max_resample_tries=args.max_resample_tries,
            min_goal_dist=args.min_goal_dist,
            min_path_len=args.min_path_len,
            min_tortuosity=args.min_tortuosity,
        min_turns=args.min_turns,
        turn_angle_deg=args.turn_angle_deg,
        window_mode=args.window_mode,
        goal_mode=args.goal_mode,
        episode_split_mod=args.episode_split_mod,
        episode_split_val=args.episode_split_val,
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
    mask_channels = (2 if args.stage2_mode == "adj" else 1) + (1 if args.anchor_conf else 0)
    model = InterpLevelDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        max_levels=args.levels,
        use_checkpoint=bool(args.use_checkpoint),
        use_start_goal=bool(args.cond_start_goal),
        mask_channels=mask_channels,
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
        payload_meta = {}
        try:
            payload_meta = torch.load(args.bootstrap_stage1_ckpt, map_location="cpu")
        except Exception:
            payload_meta = {}
        meta = payload_meta.get("meta", {}) if isinstance(payload_meta, dict) else {}
        if meta.get("cond_start_goal") is not None:
            use_start_goal_kp = bool(meta.get("cond_start_goal"))
        else:
            use_start_goal_kp = bool(meta.get("use_start_goal", args.use_start_goal))
        if meta.get("clamp_endpoints") is not None:
            clamp_endpoints_kp = bool(meta.get("clamp_endpoints"))
        else:
            clamp_endpoints_kp = bool(args.clamp_endpoints)
        bootstrap_model = KeypointDenoiser(
            data_dim=data_dim,
            use_sdf=bool(args.use_sdf),
            use_start_goal=use_start_goal_kp,
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
        meta = payload.get("meta", {}) if isinstance(payload, dict) else meta
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
    gen.manual_seed(seed + 23)

    model.train()
    pbar = tqdm(range(start_step, args.steps), dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    use_mask = torch.zeros((args.batch,), device=device, dtype=torch.bool)
    student_mask = torch.zeros((args.batch, args.T), device=device, dtype=torch.bool)
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x0 = batch["x"].to(device)
        cond = {k: v.to(device) for k, v in batch["cond"].items()}

        masks_levels, idx_levels = build_nested_masks_batch(
            x0.shape[0],
            args.T,
            args.K_min,
            args.levels,
            generator=gen,
            device=device,
            k_schedule=args.k_schedule,
            k_geom_gamma=args.k_geom_gamma,
        )

        x0_used = x0
        use_mask = torch.zeros((x0.shape[0],), device=device, dtype=torch.bool)
        student_mask = torch.zeros((x0.shape[0], x0.shape[1]), device=device, dtype=torch.bool)
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
                known_mask, known_values = _build_known_mask_values(
                    idx_s, cond, data_dim, args.T, bool(clamp_endpoints_kp)
                )
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
                        schedule_name=args.bootstrap_ddim_schedule,
                    )
                    if bootstrap_logit:
                        z_hat = sigmoid_pos(z_hat)
                x0_aug = x0.clone()
                replace_mask = torch.rand((x0.shape[0], idx_s.shape[1]), generator=gen, device=device) < float(
                    args.bootstrap_replace_prob
                )
                replace_mask = replace_mask & use_mask.view(-1, 1)
                replace_mask = replace_mask & ~(idx_s == 0) & ~(idx_s == (args.T - 1))
                student_mask.scatter_(1, idx_s, replace_mask)
                vals = x0_aug[:, :, :2].gather(1, idx_s.unsqueeze(-1).expand(-1, idx_s.shape[1], 2))
                vals = torch.where(replace_mask.unsqueeze(-1), z_hat[:, :, :2], vals)
                x0_aug[:, :, :2].scatter_(1, idx_s.unsqueeze(-1).expand(-1, idx_s.shape[1], 2), vals)
                if torch.any(replace_mask):
                    x0_used = torch.where(use_mask.view(-1, 1, 1), x0_aug, x0)

        s_idx = _sample_level_indices(
            x0.shape[0],
            args.levels,
            gen,
            device,
            args.level_sampling,
            args.level_high_prob,
        )
        if args.stage2_mode == "adj":
            x_s, x_prev, mask_s, mask_prev, s_idx, _, _ = build_interp_adjacent_batch(
                x0,
                args.K_min,
                args.levels,
                gen,
                recompute_velocity=bool(args.recompute_vel),
                x0_override=x0_used,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma_max=args.corrupt_sigma_max,
                corrupt_sigma_min=args.corrupt_sigma_min,
                corrupt_sigma_pow=args.corrupt_sigma_pow,
                corrupt_anchor_frac=args.corrupt_anchor_frac,
                corrupt_index_jitter_max=args.corrupt_index_jitter_max,
                corrupt_index_jitter_prob=args.corrupt_index_jitter_prob,
                corrupt_index_jitter_pow=args.corrupt_index_jitter_pow,
                clamp_endpoints=bool(args.clamp_endpoints),
            )
            if args.debug_corrupt_stats and (step % args.debug_corrupt_every == 0):
                _log_corrupt_stats(writer, step, s_idx, args.levels, x0, x_s, x_prev)
            conf_s = _build_anchor_conf(
                mask_s,
                student_mask if bootstrap_model is not None else None,
                args.anchor_conf_teacher,
                args.anchor_conf_student,
                args.anchor_conf_endpoints,
                args.anchor_conf_missing,
                bool(args.clamp_endpoints),
            )
            conf_prev = _build_anchor_conf(
                mask_prev,
                student_mask if bootstrap_model is not None else None,
                args.anchor_conf_teacher,
                args.anchor_conf_student,
                args.anchor_conf_endpoints,
                args.anchor_conf_missing,
                bool(args.clamp_endpoints),
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
                prev_idx = torch.clamp(s_idx - 1, min=0)
                conf_prev = _anneal_conf(conf_prev, prev_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
            else:
                mask_in = torch.stack([mask_s, mask_prev], dim=-1)
            target = x_prev - x_s
            weight_mask = conf_prev if args.anchor_conf else mask_prev
        else:
            x_s, mask_s, s_idx, _, _ = build_interp_level_batch(
                x0,
                args.K_min,
                args.levels,
                gen,
                recompute_velocity=bool(args.recompute_vel),
                x0_override=x0_used,
                masks_levels=masks_levels,
                idx_levels=idx_levels,
                s_idx=s_idx,
                corrupt_mode=args.corrupt_mode,
                corrupt_sigma_max=args.corrupt_sigma_max,
                corrupt_sigma_min=args.corrupt_sigma_min,
                corrupt_sigma_pow=args.corrupt_sigma_pow,
                corrupt_anchor_frac=args.corrupt_anchor_frac,
                corrupt_index_jitter_max=args.corrupt_index_jitter_max,
                corrupt_index_jitter_prob=args.corrupt_index_jitter_prob,
                corrupt_index_jitter_pow=args.corrupt_index_jitter_pow,
                clamp_endpoints=bool(args.clamp_endpoints),
            )
            if args.debug_corrupt_stats and (step % args.debug_corrupt_every == 0):
                _log_corrupt_stats(writer, step, s_idx, args.levels, x0, x_s, None)
            conf_s = _build_anchor_conf(
                mask_s,
                student_mask if bootstrap_model is not None else None,
                args.anchor_conf_teacher,
                args.anchor_conf_student,
                args.anchor_conf_endpoints,
                args.anchor_conf_missing,
                bool(args.clamp_endpoints),
            )
            if args.anchor_conf_anneal:
                conf_s = _anneal_conf(conf_s, s_idx, args.levels, args.anchor_conf_anneal_mode)
            if args.anchor_conf:
                mask_in = torch.stack([mask_s.float(), conf_s], dim=-1)
            else:
                mask_in = mask_s
            target = x0 - x_s
            weight_mask = conf_s if args.anchor_conf else mask_s

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            delta_hat = model(x_s, s_idx, mask_in, cond)
            diff = (delta_hat - target) ** 2
            diff = diff.sum(dim=-1)
            if args.anchor_conf:
                w = float(args.w_missing) + (float(args.w_anchor) - float(args.w_missing)) * weight_mask
            else:
                w = torch.where(
                    weight_mask,
                    torch.tensor(args.w_anchor, device=device),
                    torch.tensor(args.w_missing, device=device),
                )
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
            meta = {
                "stage": "interp_levels",
                "T": args.T,
                "K_min": args.K_min,
                "levels": args.levels,
                "data_dim": data_dim,
                "use_sdf": bool(args.use_sdf),
                "with_velocity": bool(args.with_velocity),
                "dataset": args.dataset,
                "env_id": args.env_id,
                "d4rl_flip_y": bool(args.d4rl_flip_y),
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
                "bootstrap_replace_prob": float(args.bootstrap_replace_prob),
                "corrupt_mode": args.corrupt_mode,
                "corrupt_sigma_max": float(args.corrupt_sigma_max),
                "corrupt_sigma_min": float(args.corrupt_sigma_min),
                "corrupt_sigma_pow": float(args.corrupt_sigma_pow),
                "corrupt_anchor_frac": float(args.corrupt_anchor_frac),
                "corrupt_index_jitter_max": int(args.corrupt_index_jitter_max),
                "corrupt_index_jitter_prob": float(args.corrupt_index_jitter_prob),
                "corrupt_index_jitter_pow": float(args.corrupt_index_jitter_pow),
                "use_start_goal": bool(args.cond_start_goal),
                "clamp_endpoints": bool(args.clamp_endpoints),
                "cond_start_goal": bool(args.cond_start_goal),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "interp_levels",
        "T": args.T,
        "K_min": args.K_min,
        "levels": args.levels,
        "data_dim": data_dim,
        "use_sdf": bool(args.use_sdf),
        "with_velocity": bool(args.with_velocity),
        "dataset": args.dataset,
        "env_id": args.env_id,
        "d4rl_flip_y": bool(args.d4rl_flip_y),
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
        "bootstrap_replace_prob": float(args.bootstrap_replace_prob),
        "corrupt_mode": args.corrupt_mode,
        "corrupt_sigma_max": float(args.corrupt_sigma_max),
        "corrupt_sigma_min": float(args.corrupt_sigma_min),
        "corrupt_sigma_pow": float(args.corrupt_sigma_pow),
        "corrupt_anchor_frac": float(args.corrupt_anchor_frac),
        "corrupt_index_jitter_max": int(args.corrupt_index_jitter_max),
        "corrupt_index_jitter_prob": float(args.corrupt_index_jitter_prob),
        "corrupt_index_jitter_pow": float(args.corrupt_index_jitter_pow),
        "use_start_goal": bool(args.cond_start_goal),
        "clamp_endpoints": bool(args.clamp_endpoints),
        "cond_start_goal": bool(args.cond_start_goal),
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
