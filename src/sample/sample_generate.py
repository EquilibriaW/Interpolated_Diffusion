import argparse
import sys
import csv
import json
import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import (
    interpolate_from_indices,
    sample_fixed_k_indices_batch,
    sample_fixed_k_indices_uniform_batch,
    build_nested_masks_from_base,
    build_nested_masks_from_logits,
    build_nested_masks_from_level_logits,
    _compute_k_schedule,
)
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset, PreparedTrajectoryDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.eval.metrics import collision_rate, goal_distance, success
from src.eval.visualize import plot_maze2d_geom_walls, plot_maze2d_trajectories, plot_trajectories
from src.models.denoiser_interp_levels import InterpLevelDenoiser
from src.models.denoiser_keypoints import KeypointDenoiser
from src.models.keypoint_selector import KeypointSelector, select_topk_indices
from src.models.segment_cost import SegmentCostPredictor
from src.selection.epiplexity_dp import build_segment_features_from_idx
from src.utils.clamp import apply_clamp, apply_soft_clamp
from src.utils.device import get_device
from src.utils.normalize import logit_pos, sigmoid_pos


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_keypoints", type=str, default="checkpoints/keypoints/ckpt_final.pt")
    p.add_argument("--ckpt_interp", type=str, default="checkpoints/interp_levels/ckpt_final.pt")
    p.add_argument("--out_dir", type=str, default="runs/gen")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--K_min", type=int, default=None)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--stage2_mode", type=str, default="adj", choices=["x0", "adj"])
    p.add_argument("--N_train", type=int, default=None)
    p.add_argument("--schedule", type=str, default=None, choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ddim_schedule", type=str, default="quadratic", choices=["linear", "quadratic", "sqrt"])
    p.add_argument("--k_schedule", type=str, default="doubling", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--anchor_conf", type=int, default=1)
    p.add_argument("--anchor_conf_teacher", type=float, default=0.95)
    p.add_argument("--anchor_conf_student", type=float, default=0.5)
    p.add_argument("--anchor_conf_endpoints", type=float, default=1.0)
    p.add_argument("--anchor_conf_missing", type=float, default=0.0)
    p.add_argument("--anchor_conf_anneal", type=int, default=1)
    p.add_argument("--anchor_conf_anneal_mode", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--soft_anchor_clamp", type=int, default=1)
    p.add_argument("--soft_clamp_schedule", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--soft_clamp_max", type=float, default=1.0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--recompute_vel", type=int, default=1)
    p.add_argument("--logit_space", type=int, default=1)
    p.add_argument("--logit_eps", type=float, default=1e-5)
    p.add_argument("--pos_clip", type=int, default=0)
    p.add_argument("--pos_clip_min", type=float, default=0.0)
    p.add_argument("--pos_clip_max", type=float, default=1.0)
    p.add_argument("--s2_sample_noise_mode", type=str, default="none", choices=["none", "constant", "level"])
    p.add_argument("--s2_sample_noise_sigma", type=float, default=0.0)
    p.add_argument("--s2_sample_noise_scale", type=float, default=1.0)
    p.add_argument("--s2_corrupt_sigma_max", type=float, default=None)
    p.add_argument("--s2_corrupt_sigma_min", type=float, default=None)
    p.add_argument("--s2_corrupt_sigma_pow", type=float, default=None)
    p.add_argument("--use_kp_feat", type=int, default=0)
    p.add_argument("--kp_feat_dim", type=int, default=0)
    p.add_argument("--dphi_ckpt", type=str, default=None)
    p.add_argument("--dphi_use_ema", type=int, default=1)
    p.add_argument("--kp_d_model", type=int, default=None)
    p.add_argument("--kp_n_layers", type=int, default=None)
    p.add_argument("--kp_n_heads", type=int, default=None)
    p.add_argument("--kp_d_ff", type=int, default=None)
    p.add_argument("--kp_d_cond", type=int, default=None)
    p.add_argument("--kp_maze_channels", type=str, default=None)
    p.add_argument("--selector_ckpt", type=str, default=None)
    p.add_argument("--selector_use_ema", type=int, default=1)
    p.add_argument("--s2_d_model", type=int, default=None)
    p.add_argument("--s2_n_layers", type=int, default=None)
    p.add_argument("--s2_n_heads", type=int, default=None)
    p.add_argument("--s2_d_ff", type=int, default=None)
    p.add_argument("--s2_d_cond", type=int, default=None)
    p.add_argument("--s2_maze_channels", type=str, default=None)
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default="d4rl", choices=["particle", "synthetic", "d4rl", "d4rl_prepared"])
    p.add_argument("--prepared_path", type=str, default=None)
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
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
    p.add_argument("--cond_start_goal_kp", type=int, default=None)
    p.add_argument("--cond_start_goal_s2", type=int, default=None)
    p.add_argument("--override_meta", type=int, default=0)
    p.add_argument("--override_interp_meta", type=int, default=0)
    p.add_argument("--clamp_policy", type=str, default="endpoints", choices=["none", "endpoints", "all_anchors"])
    p.add_argument("--clamp_dims", type=str, default="pos", choices=["pos", "all"])
    p.add_argument("--oracle_keypoints", type=int, default=0)
    p.add_argument("--plot_gt", type=int, default=1)
    p.add_argument("--save_debug", type=int, default=1)
    p.add_argument("--save_diffusion_frames", type=int, default=0)
    p.add_argument("--frames_stride", type=int, default=1)
    p.add_argument("--frames_include_stage2", type=int, default=1)
    p.add_argument("--export_video", type=str, default="none", choices=["none", "mp4", "gif"])
    p.add_argument("--video_fps", type=int, default=8)
    p.add_argument("--save_npz", type=int, default=1)
    p.add_argument("--save_steps_npz", type=int, default=0)
    p.add_argument("--skip_stage2", type=int, default=0)
    p.add_argument("--compare_oracle", type=int, default=0)
    p.add_argument("--plot_keypoints", type=int, default=1)
    p.add_argument("--plot_points", type=int, default=0)
    p.add_argument("--force_single", type=int, default=0)
    p.add_argument(
        "--kp_index_mode",
        type=str,
        default="uniform",
        choices=["random", "uniform", "uniform_jitter", "selector"],
    )
    p.add_argument("--kp_jitter", type=float, default=0.0)
    p.add_argument("--sample_random", type=int, default=1)
    p.add_argument("--sample_seed", type=int, default=1234)
    p.add_argument("--stage1_cache", type=str, default="")
    p.add_argument(
        "--stage1_cache_mode",
        type=str,
        default="none",
        choices=["none", "save", "load", "auto"],
    )
    p.add_argument("--sample_unique", type=int, default=1)
    return p


def _compute_sigma_for_level(
    K_s: int,
    K_min: int,
    sigma_max: float,
    sigma_min: float,
    sigma_pow: float,
) -> float:
    if sigma_max <= 0.0:
        return 0.0
    if K_s <= K_min:
        return float(sigma_max)
    ratio = float(K_min) / float(K_s)
    sigma = float(sigma_max) * (ratio ** float(sigma_pow))
    sigma = min(float(sigma_max), sigma)
    sigma = max(float(sigma_min), sigma)
    return sigma


def _apply_s2_noise(x: torch.Tensor, mask: Optional[torch.Tensor], sigma: float, scale: float) -> torch.Tensor:
    if sigma <= 0.0 or scale <= 0.0:
        return x
    noise = torch.randn_like(x[:, :, :2]) * float(sigma) * float(scale)
    if mask is not None:
        miss = ~mask.bool()
        noise = noise * miss.unsqueeze(-1)
    x[:, :, :2] = x[:, :, :2] + noise
    return x


def get_cond_from_sample(sample: dict) -> dict:
    return sample["cond"]


def _ensure_mujoco_env():
    if "MUJOCO_PY_MUJOCO_PATH" not in os.environ:
        for candidate in ("/workspace/mujoco210", os.path.expanduser("~/.mujoco/mujoco210")):
            if os.path.isdir(candidate):
                os.environ["MUJOCO_PY_MUJOCO_PATH"] = candidate
                break
    os.environ.setdefault("MUJOCO_GL", "egl")
    mujoco_path = os.environ.get("MUJOCO_PY_MUJOCO_PATH", "")
    if mujoco_path:
        bin_path = os.path.join(mujoco_path, "bin")
        ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
        if bin_path not in ld_paths:
            ld_paths.append(bin_path)
            os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)

def _export_video(frames_dir: str, fmt: str, fps: int):
    if fmt == "none":
        return
    frames = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    frames.sort()
    if not frames:
        return
    out_path = os.path.join(frames_dir, f"video.{fmt}")
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
            for fname in frames:
                writer.append_data(imageio.imread(os.path.join(frames_dir, fname)))
        return
    except Exception:
        print("imageio unavailable or failed to write video; trying ffmpeg if possible.")
    if fmt == "mp4":
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("ffmpeg not found and imageio unavailable; skipping video export.")
            return
        pattern = os.path.join(frames_dir, "step_%03d.png")
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern, "-pix_fmt", "yuv420p", out_path]
        subprocess.run(cmd, check=False)
    else:
        print("Video export skipped (gif requires imageio).")




def _normalize_walls(
    walls: list,
    pos_low: torch.Tensor,
    pos_scale: torch.Tensor,
    flip_y: bool,
) -> Optional[list]:
    if not walls or pos_low is None or pos_scale is None:
        return None
    low = pos_low.detach().cpu().numpy()
    scale = pos_scale.detach().cpu().numpy()
    out = []
    for poly in walls:
        arr = np.asarray(poly, dtype=np.float32)
        arr = (arr - low[None, :2]) / scale[None, :2]
        if flip_y:
            arr[:, 1] = 1.0 - arr[:, 1]
        out.append(arr)
    return out


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


def _kp_feat_from_idx(
    idx: torch.Tensor,
    T: int,
    kp_feat_dim: int,
    left_diff: torch.Tensor | None = None,
    right_diff: torch.Tensor | None = None,
) -> torch.Tensor:
    B, K = idx.shape
    feat = torch.zeros((B, K, kp_feat_dim), device=idx.device, dtype=torch.float32)
    if kp_feat_dim <= 0:
        return feat
    denom = float(max(1, T - 1))
    t_norm = idx.float() / denom
    feat[:, :, 0] = 0.0
    feat[:, :, 1] = 0.0
    if K > 1:
        gaps = (idx[:, 1:] - idx[:, :-1]).float() / denom
        feat[:, 1:, 0] = gaps
        feat[:, :-1, 1] = gaps
    if kp_feat_dim >= 3:
        feat[:, :, 2] = t_norm
    if kp_feat_dim >= 5 and left_diff is not None and right_diff is not None:
        feat[:, :, 3] = left_diff
        feat[:, :, 4] = right_diff
    return feat


def _parse_int_list(spec: str, fallback: str) -> tuple[int, ...]:
    if spec is None:
        spec = fallback
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise ValueError("empty int list")
    return tuple(int(p) for p in parts)


def _build_anchor_conf(
    mask_s: torch.Tensor,
    student_mask: Optional[torch.Tensor],
    use_student: bool,
    conf_teacher: float,
    conf_student: float,
    conf_endpoints: float,
    conf_missing: float,
    clamp_endpoints: bool,
) -> torch.Tensor:
    conf = torch.full_like(mask_s.float(), float(conf_missing))
    conf = torch.where(mask_s, torch.full_like(conf, float(conf_teacher)), conf)
    if student_mask is not None and use_student:
        conf = torch.where(student_mask & mask_s, torch.full_like(conf, float(conf_student)), conf)
    if clamp_endpoints:
        conf[:, 0] = float(conf_endpoints)
        conf[:, -1] = float(conf_endpoints)
    return conf


def _soft_clamp_lambda(s: int, levels: int, schedule: str, max_val: float) -> float:
    if levels <= 0:
        return float(max_val)
    frac = float(s) / float(levels)
    if schedule == "linear":
        return float(max_val) * frac
    if schedule == "cosine":
        return float(max_val) * 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))
    return float(max_val)


def _anneal_conf(conf: torch.Tensor, s: int, levels: int, mode: str) -> torch.Tensor:
    if conf is None or mode == "none" or levels <= 0:
        return conf
    frac = float(s) / float(levels)
    if mode == "linear":
        lam = 1.0 - frac
    elif mode == "cosine":
        lam = 0.5 * (1.0 + math.cos(math.pi * frac))
    else:
        lam = 0.0
    return conf + (1.0 - conf) * float(lam)


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
    return_intermediates: bool = False,
    pos_clip: bool = False,
    pos_clip_min: float = 0.0,
    pos_clip_max: float = 1.0,
):
    device = idx.device
    B, K = idx.shape
    D = known_values.shape[-1]
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps, schedule=schedule_name)
    def _clip_pos(z_in: torch.Tensor) -> torch.Tensor:
        if not pos_clip:
            return z_in
        z_in[..., :2] = z_in[..., :2].clamp(min=pos_clip_min, max=pos_clip_max)
        return z_in

    z = torch.randn((B, K, D), device=device)
    z = torch.where(known_mask, known_values, z)
    z = _clip_pos(z)
    intermediates = [z.detach().clone()] if return_intermediates else None
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(z, t, idx, known_mask, cond, T)
        z = ddim_step(z, eps, t, t_prev, schedule, eta=0.0)
        z = torch.where(known_mask, known_values, z)
        z = _clip_pos(z)
        if return_intermediates:
            intermediates.append(z.detach().clone())
    if return_intermediates:
        return z, intermediates
    return z


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
    os.makedirs(args.out_dir, exist_ok=True)
    _ensure_mujoco_env()
    if args.dataset not in {"d4rl", "d4rl_prepared"}:
        raise ValueError("Particle/synthetic datasets are disabled; use --dataset d4rl or d4rl_prepared.")
    if args.ckpt_interp == "":
        args.skip_stage2 = 1

    # Load keypoint payload early to configure model dims before instantiation.
    payload_kp = {}
    if os.path.exists(args.ckpt_keypoints):
        try:
            payload_meta = torch.load(args.ckpt_keypoints, map_location="cpu")
            if isinstance(payload_meta, dict):
                payload_kp = payload_meta
        except Exception:
            payload_kp = {}
    meta = payload_kp.get("meta", {}) if isinstance(payload_kp, dict) else {}
    def _normalize_dataset_name(name: Optional[str]) -> Optional[str]:
        if name == "d4rl_prepared":
            return "d4rl"
        return name

    if meta.get("stage") == "keypoints" and not args.override_meta:
        args.T = meta.get("T", args.T)
        args.N_train = meta.get("N_train", args.N_train)
        args.schedule = meta.get("schedule", args.schedule)
        if args.K_min is None:
            args.K_min = meta.get("K", args.K_min)
        if meta.get("logit_space") is not None:
            args.logit_space = int(bool(meta.get("logit_space")))
        if meta.get("logit_eps") is not None:
            args.logit_eps = float(meta.get("logit_eps"))
        if meta.get("use_sdf") is not None:
            args.use_sdf = int(bool(meta.get("use_sdf")))
        if meta.get("with_velocity") is not None:
            args.with_velocity = int(bool(meta.get("with_velocity")))
        if meta.get("logit_space") is not None:
            args.logit_space = int(bool(meta.get("logit_space")))
        if meta.get("logit_eps") is not None:
            args.logit_eps = float(meta.get("logit_eps"))
        if meta.get("clamp_endpoints") is not None:
            args.clamp_endpoints = int(bool(meta.get("clamp_endpoints")))
        elif meta.get("use_start_goal") is not None:
            args.clamp_endpoints = int(bool(meta.get("use_start_goal")))
        if meta.get("cond_start_goal") is not None:
            args.cond_start_goal = int(bool(meta.get("cond_start_goal")))
        elif meta.get("use_start_goal") is not None:
            args.cond_start_goal = int(bool(meta.get("use_start_goal")))
        args.use_start_goal = int(bool(args.cond_start_goal))
        if "--use_kp_feat" not in sys.argv and meta.get("use_kp_feat") is not None:
            args.use_kp_feat = int(bool(meta.get("use_kp_feat")))
        if "--kp_feat_dim" not in sys.argv and meta.get("kp_feat_dim") is not None:
            args.kp_feat_dim = int(meta.get("kp_feat_dim"))
        if meta.get("window_mode") is not None:
            args.window_mode = str(meta.get("window_mode"))
        if meta.get("goal_mode") is not None:
            args.goal_mode = str(meta.get("goal_mode"))
        if meta.get("dataset") is not None:
            if _normalize_dataset_name(args.dataset) != _normalize_dataset_name(meta.get("dataset")):
                raise ValueError(
                    f"Keypoint checkpoint dataset mismatch: ckpt={meta.get('dataset')} args={args.dataset}. "
                    "Use --override_meta 1 to force."
                )
            args.dataset = meta.get("dataset")
        if meta.get("env_id") is not None:
            if args.env_id != meta.get("env_id"):
                raise ValueError(
                    f"Keypoint checkpoint env_id mismatch: ckpt={meta.get('env_id')} args={args.env_id}. "
                    "Use --override_meta 1 to force."
                )
            args.env_id = meta.get("env_id")
        if meta.get("d4rl_flip_y") is not None:
            args.d4rl_flip_y = int(bool(meta.get("d4rl_flip_y")))
        if "--kp_d_model" not in sys.argv and meta.get("kp_d_model") is not None:
            args.kp_d_model = int(meta.get("kp_d_model"))
        if "--kp_n_layers" not in sys.argv and meta.get("kp_n_layers") is not None:
            args.kp_n_layers = int(meta.get("kp_n_layers"))
        if "--kp_n_heads" not in sys.argv and meta.get("kp_n_heads") is not None:
            args.kp_n_heads = int(meta.get("kp_n_heads"))
        if "--kp_d_ff" not in sys.argv and meta.get("kp_d_ff") is not None:
            args.kp_d_ff = int(meta.get("kp_d_ff"))
        if "--kp_d_cond" not in sys.argv and meta.get("kp_d_cond") is not None:
            args.kp_d_cond = int(meta.get("kp_d_cond"))
        if "--kp_maze_channels" not in sys.argv and meta.get("kp_maze_channels") is not None:
            args.kp_maze_channels = str(meta.get("kp_maze_channels"))

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2
    cond_start_goal_kp = (
        bool(args.cond_start_goal_kp) if args.cond_start_goal_kp is not None else bool(args.cond_start_goal)
    )

    dphi_model = None
    dphi_seg_feat_dim = None
    if args.dphi_ckpt:
        payload = torch.load(args.dphi_ckpt, map_location="cpu")
        meta_dphi = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta_dphi.get("stage") != "segment_cost":
            raise ValueError("dphi_ckpt does not appear to be a segment_cost checkpoint")
        if meta_dphi.get("T") is not None and int(meta_dphi.get("T")) != int(args.T):
            raise ValueError(f"dphi_ckpt T mismatch: ckpt={meta_dphi.get('T')} args={args.T}")
        if meta_dphi.get("use_sdf") is not None and bool(meta_dphi.get("use_sdf")) != bool(args.use_sdf):
            raise ValueError("dphi_ckpt use_sdf mismatch")
        if meta_dphi.get("cond_start_goal") is not None and bool(meta_dphi.get("cond_start_goal")) != bool(args.cond_start_goal):
            raise ValueError("dphi_ckpt cond_start_goal mismatch")
        if not bool(args.use_kp_feat) or int(args.kp_feat_dim) < 5:
            raise ValueError("dphi_ckpt requires use_kp_feat=1 and kp_feat_dim>=5")
        maze_channels = meta_dphi.get("maze_channels", "32,64")
        dphi_model = SegmentCostPredictor(
            d_cond=int(meta_dphi.get("d_cond", 128)),
            seg_feat_dim=int(meta_dphi.get("seg_feat_dim", 3)),
            hidden_dim=int(meta_dphi.get("hidden_dim", 256)),
            n_layers=int(meta_dphi.get("n_layers", 3)),
            dropout=float(meta_dphi.get("dropout", 0.0)),
            use_sdf=bool(args.use_sdf),
            use_start_goal=bool(args.cond_start_goal),
            maze_channels=_parse_int_list(maze_channels, "32,64"),
        ).to(device)
        dphi_state = payload.get("model", payload)
        dphi_model.load_state_dict(dphi_state)
        if bool(args.dphi_use_ema) and isinstance(payload, dict) and "ema" in payload:
            from src.utils.ema import EMA

            ema_dphi = EMA(dphi_model.parameters())
            ema_dphi.load_state_dict(payload["ema"])
            ema_dphi.copy_to(dphi_model.parameters())
        dphi_model.eval()
        dphi_seg_feat_dim = int(dphi_model.seg_feat_dim)
    elif bool(args.use_kp_feat) and int(args.kp_feat_dim) > 3:
        raise ValueError("kp_feat_dim>3 requires --dphi_ckpt for predicted difficulties")

    selector_model = None
    selector_use_level = False
    selector_level_mode = "k_norm"
    if args.kp_index_mode == "selector":
        if args.selector_ckpt is None:
            raise ValueError("kp_index_mode=selector requires --selector_ckpt")
        payload = torch.load(args.selector_ckpt, map_location="cpu")
        meta_sel = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta_sel.get("stage") != "selector":
            raise ValueError("selector_ckpt does not appear to be a selector checkpoint")
        if meta_sel.get("T") is not None and int(meta_sel.get("T")) != int(args.T):
            raise ValueError(f"selector_ckpt T mismatch: ckpt={meta_sel.get('T')} args={args.T}")
        if meta_sel.get("K") is not None and int(meta_sel.get("K")) != int(args.K_min):
            raise ValueError(f"selector_ckpt K mismatch: ckpt={meta_sel.get('K')} args={args.K_min}")
        if meta_sel.get("use_sdf") is not None and bool(meta_sel.get("use_sdf")) != bool(args.use_sdf):
            raise ValueError("selector_ckpt use_sdf mismatch")
        if meta_sel.get("cond_start_goal") is not None and bool(meta_sel.get("cond_start_goal")) != bool(args.cond_start_goal):
            raise ValueError("selector_ckpt cond_start_goal mismatch")
        selector_use_level = bool(meta_sel.get("use_level", False))
        selector_level_mode = str(meta_sel.get("level_mode", "k_norm"))
        maze_channels = meta_sel.get("maze_channels", "32,64")
        selector_model = KeypointSelector(
            T=int(meta_sel.get("T", args.T)),
            d_model=int(meta_sel.get("d_model", 256)),
            n_heads=int(meta_sel.get("n_heads", 8)),
            d_ff=int(meta_sel.get("d_ff", 512)),
            n_layers=int(meta_sel.get("n_layers", 2)),
            pos_dim=int(meta_sel.get("pos_dim", 64)),
            dropout=float(meta_sel.get("dropout", 0.0)),
            use_sdf=bool(args.use_sdf),
            use_start_goal=bool(args.cond_start_goal),
            use_sg_map=bool(meta_sel.get("use_sg_map", True)),
            use_sg_token=bool(meta_sel.get("use_sg_token", True)),
            use_goal_dist_token=bool(meta_sel.get("use_goal_dist_token", False)),
            use_cond_bias=bool(meta_sel.get("use_cond_bias", False)),
            cond_bias_mode=str(meta_sel.get("cond_bias_mode", "memory")),
            use_level=selector_use_level,
            level_mode=selector_level_mode,
            sg_map_sigma=float(meta_sel.get("sg_map_sigma", 1.5)),
            maze_channels=_parse_int_list(maze_channels, "32,64"),
        ).to(device)
        selector_state = payload.get("model", payload)
        selector_model.load_state_dict(selector_state)
        if bool(args.selector_use_ema) and isinstance(payload, dict) and "ema" in payload:
            from src.utils.ema import EMA

            ema_sel = EMA(selector_model.parameters())
            ema_sel.load_state_dict(payload["ema"])
            ema_sel.copy_to(selector_model.parameters())
        selector_model.eval()

    kp_model = KeypointDenoiser(
        d_model=int(args.kp_d_model) if args.kp_d_model is not None else 256,
        n_layers=int(args.kp_n_layers) if args.kp_n_layers is not None else 8,
        n_heads=int(args.kp_n_heads) if args.kp_n_heads is not None else 8,
        d_ff=int(args.kp_d_ff) if args.kp_d_ff is not None else 1024,
        d_cond=int(args.kp_d_cond) if args.kp_d_cond is not None else 128,
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        use_start_goal=cond_start_goal_kp,
        kp_feat_dim=int(args.kp_feat_dim) if bool(args.use_kp_feat) else 0,
        maze_channels=_parse_int_list(args.kp_maze_channels, "32,64"),
    ).to(device)
    if "model" not in payload_kp:
        raise FileNotFoundError(f"Checkpoint not found or invalid: {args.ckpt_keypoints}")

    kp_model.load_state_dict(payload_kp["model"])
    interp_model = None
    payload_interp = {}
    interp_meta = {}
    if not args.skip_stage2:
        if os.path.exists(args.ckpt_interp):
            payload_interp = torch.load(args.ckpt_interp, map_location="cpu")
            if isinstance(payload_interp, dict):
                interp_meta = payload_interp.get("meta", {}) if isinstance(payload_interp, dict) else {}
        cond_start_goal_s2 = (
            bool(args.cond_start_goal_s2) if args.cond_start_goal_s2 is not None else bool(args.cond_start_goal)
        )
        if interp_meta and args.cond_start_goal_s2 is None:
            if interp_meta.get("cond_start_goal") is not None:
                cond_start_goal_s2 = bool(interp_meta.get("cond_start_goal"))
            elif interp_meta.get("use_start_goal") is not None:
                cond_start_goal_s2 = bool(interp_meta.get("use_start_goal"))
        if interp_meta:
            if "--s2_d_model" not in sys.argv and interp_meta.get("s2_d_model") is not None:
                args.s2_d_model = int(interp_meta.get("s2_d_model"))
            if "--s2_n_layers" not in sys.argv and interp_meta.get("s2_n_layers") is not None:
                args.s2_n_layers = int(interp_meta.get("s2_n_layers"))
            if "--s2_n_heads" not in sys.argv and interp_meta.get("s2_n_heads") is not None:
                args.s2_n_heads = int(interp_meta.get("s2_n_heads"))
            if "--s2_d_ff" not in sys.argv and interp_meta.get("s2_d_ff") is not None:
                args.s2_d_ff = int(interp_meta.get("s2_d_ff"))
            if "--s2_d_cond" not in sys.argv and interp_meta.get("s2_d_cond") is not None:
                args.s2_d_cond = int(interp_meta.get("s2_d_cond"))
            if "--s2_maze_channels" not in sys.argv and interp_meta.get("s2_maze_channels") is not None:
                args.s2_maze_channels = str(interp_meta.get("s2_maze_channels"))

        s2_corrupt_sigma_max = float(
            args.s2_corrupt_sigma_max
            if args.s2_corrupt_sigma_max is not None
            else interp_meta.get("corrupt_sigma_max", 0.0)
        )
        s2_corrupt_sigma_min = float(
            args.s2_corrupt_sigma_min
            if args.s2_corrupt_sigma_min is not None
            else interp_meta.get("corrupt_sigma_min", 0.0)
        )
        s2_corrupt_sigma_pow = float(
            args.s2_corrupt_sigma_pow
            if args.s2_corrupt_sigma_pow is not None
            else interp_meta.get("corrupt_sigma_pow", 1.0)
        )

        mask_channels = (2 if args.stage2_mode == "adj" else 1) + (1 if args.anchor_conf else 0)
        interp_model = InterpLevelDenoiser(
            d_model=int(args.s2_d_model) if args.s2_d_model is not None else 256,
            n_layers=int(args.s2_n_layers) if args.s2_n_layers is not None else 8,
            n_heads=int(args.s2_n_heads) if args.s2_n_heads is not None else 8,
            d_ff=int(args.s2_d_ff) if args.s2_d_ff is not None else 1024,
            d_cond=int(args.s2_d_cond) if args.s2_d_cond is not None else 128,
            data_dim=data_dim,
            use_sdf=bool(args.use_sdf),
            max_levels=args.levels,
            use_start_goal=cond_start_goal_s2,
            mask_channels=mask_channels,
            maze_channels=_parse_int_list(args.s2_maze_channels, "32,64"),
        ).to(device)
        if "model" not in payload_interp:
            raise FileNotFoundError(f"Checkpoint not found or invalid: {args.ckpt_interp}")
        if interp_meta and not bool(args.override_interp_meta):
            if interp_meta.get("cond_start_goal") is not None:
                interp_cond = bool(interp_meta.get("cond_start_goal"))
            elif interp_meta.get("use_start_goal") is not None:
                interp_cond = bool(interp_meta.get("use_start_goal"))
            else:
                interp_cond = None
            if interp_meta.get("clamp_endpoints") is not None:
                interp_clamp = bool(interp_meta.get("clamp_endpoints"))
            elif interp_meta.get("use_start_goal") is not None:
                interp_clamp = bool(interp_meta.get("use_start_goal"))
            else:
                interp_clamp = None
            if interp_meta.get("anchor_conf") is not None:
                interp_anchor_conf = bool(interp_meta.get("anchor_conf"))
                if interp_anchor_conf != bool(args.anchor_conf):
                    raise ValueError(
                        f"Stage2 checkpoint anchor_conf mismatch: ckpt={interp_anchor_conf} args={bool(args.anchor_conf)}. "
                        "Use --override_interp_meta 1 to force."
                    )
            if interp_cond is not None and interp_cond != bool(cond_start_goal_s2):
                raise ValueError(
                    f"Stage2 checkpoint cond_start_goal mismatch: ckpt={interp_cond} args={bool(cond_start_goal_s2)}. "
                    "Use --override_interp_meta 1 to force."
                )
            if interp_clamp is not None and interp_clamp != bool(args.clamp_endpoints):
                raise ValueError(
                    f"Stage2 checkpoint clamp_endpoints mismatch: ckpt={interp_clamp} args={bool(args.clamp_endpoints)}. "
                    "Use --override_interp_meta 1 to force."
                )
        interp_model.load_state_dict(payload_interp["model"])

    ema_warned = False
    if args.use_ema:
        from src.utils.ema import EMA

        if isinstance(payload_kp, dict) and "ema" in payload_kp:
            ema_kp = EMA(kp_model.parameters())
            ema_kp.load_state_dict(payload_kp["ema"])
            ema_kp.copy_to(kp_model.parameters())
        else:
            if not ema_warned:
                print("Checkpoint has no EMA; using raw model weights.")
                ema_warned = True
        if not args.skip_stage2 and interp_model is not None:
            if isinstance(payload_interp, dict) and "ema" in payload_interp:
                ema_interp = EMA(interp_model.parameters())
                ema_interp.load_state_dict(payload_interp["ema"])
                ema_interp.copy_to(interp_model.parameters())
            else:
                if not ema_warned:
                    print("Checkpoint has no EMA; using raw model weights.")
                    ema_warned = True
    kp_model.eval()
    if interp_model is not None:
        interp_model.eval()

    meta = payload_kp.get("meta", {}) if isinstance(payload_kp, dict) else {}
    if meta.get("stage") == "keypoints" and not args.override_meta:
        args.T = meta.get("T", args.T)
        args.N_train = meta.get("N_train", args.N_train)
        args.schedule = meta.get("schedule", args.schedule)
        if args.K_min is None:
            args.K_min = meta.get("K", args.K_min)
    if args.T is None or args.N_train is None or args.schedule is None:
        raise ValueError("Missing T/N_train/schedule. Provide args or use a keypoint checkpoint with meta.")

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    dataset_name = "synthetic" if args.dataset == "synthetic" else args.dataset
    if dataset_name == "d4rl_prepared":
        if args.prepared_path is None:
            raise ValueError("--prepared_path is required for dataset d4rl_prepared")
        dataset = PreparedTrajectoryDataset(args.prepared_path, use_sdf=bool(args.use_sdf))
    elif dataset_name == "d4rl":
        dataset = D4RLMazeDataset(
            env_id=args.env_id,
            num_samples=args.n_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            seed=1234,
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
            num_samples=args.n_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
        )
    sampler = None
    if bool(args.sample_random):
        from torch.utils.data import RandomSampler

        gen_cpu = torch.Generator()
        gen_cpu.manual_seed(int(args.sample_seed))
        replacement = not bool(args.sample_unique)
        sampler = RandomSampler(dataset, replacement=replacement, num_samples=args.n_samples, generator=gen_cpu)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler)

    maze_map = None
    maze_scale = None
    mj_walls = None
    mj_walls_norm = None
    pos_low = None
    pos_scale = None
    flip_y = False
    if dataset_name in {"d4rl", "d4rl_prepared"}:
        maze_map = getattr(dataset, "maze_map", None)
        maze_scale = getattr(dataset, "maze_size_scaling", None)
        mj_walls = getattr(dataset, "mj_walls", None)
        pos_low = getattr(dataset, "pos_low", None)
        pos_scale = getattr(dataset, "pos_scale", None)
        flip_y = bool(getattr(dataset, "flip_y", False))
        if dataset_name == "d4rl_prepared" and args.prepared_path:
            meta_dir = os.path.dirname(args.prepared_path)
            meta_candidates = [
                os.path.join(meta_dir, "meta.json"),
                os.path.splitext(args.prepared_path)[0] + "_meta.json",
            ]
            for meta_path in meta_candidates:
                if not os.path.exists(meta_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_prepared = json.load(f)
                    if meta_prepared is not None and "d4rl_flip_y" in meta_prepared:
                        flip_y = bool(meta_prepared.get("d4rl_flip_y"))
                        break
                except Exception:
                    continue
        if mj_walls and pos_low is not None and pos_scale is not None:
            mj_walls_norm = _normalize_walls(mj_walls, pos_low, pos_scale, flip_y)
        if maze_scale is None and maze_map is not None and pos_scale is not None:
            maze_arr = np.array(maze_map)
            if maze_arr.ndim == 2:
                h, w = maze_arr.shape
                if w > 0 and h > 0:
                    maze_scale = float(min(pos_scale[0].item() / w, pos_scale[1].item() / h))

    def _denorm_xy(xy: np.ndarray) -> np.ndarray:
        if pos_low is None or pos_scale is None:
            return xy
        out = xy.copy()
        if flip_y:
            out[:, 1] = 1.0 - out[:, 1]
        out = out * pos_scale[:2].cpu().numpy() + pos_low[:2].cpu().numpy()
        return out

    def _plot_compare_panels(
        occ: np.ndarray,
        panels: list,
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
        if n == 1:
            axes = [axes]
        h, w = occ.shape
        scale_w = max(w - 1, 1)
        scale_h = max(h - 1, 1)
        for ax, panel in zip(axes, panels):
            traj = panel["traj"]
            label = panel["label"]
            coll = panel.get("coll", float("nan"))
            kp = panel.get("kp", None)
            sg = panel.get("sg", None)
            footer = panel.get("footer", None)
            origin = "upper" if flip_y else "lower"
            ax.imshow(occ, cmap="gray_r", origin=origin)
            xy = traj[:, :2]
            if bool(args.plot_points):
                base_color = "0.6" if (kp is not None and kp.size > 0) else "tab:blue"
                ax.scatter(xy[:, 0] * scale_w, xy[:, 1] * scale_h, color=base_color, s=6)
            else:
                ax.plot(xy[:, 0] * scale_w, xy[:, 1] * scale_h, color="tab:blue", linewidth=2.0)
            if sg is not None and sg.shape[0] >= 4:
                ax.scatter([sg[0] * scale_w], [sg[1] * scale_h], s=25, color="tab:green")
                ax.scatter([sg[2] * scale_w], [sg[3] * scale_h], s=40, color="tab:red", marker="x")
            if kp is not None and kp.size > 0:
                ax.scatter(kp[:, 0] * scale_w, kp[:, 1] * scale_h, s=20, color="tab:orange")
            ax.set_title(f"{label} | coll={coll:.3f}", fontsize=9)
            ax.set_xlim([0, scale_w])
            if flip_y:
                ax.set_ylim([scale_h, 0])
            else:
                ax.set_ylim([0, scale_h])
            ax.set_xticks([])
            ax.set_yticks([])
            if footer:
                ax.text(0.5, -0.08, footer, transform=ax.transAxes, ha="center", fontsize=8)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.savefig(out_path)
        plt.close(fig)

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    idx_global = 0

    cache = None
    cache_mode = args.stage1_cache_mode
    cache_path = args.stage1_cache
    if cache_mode != "none" and cache_path:
        if cache_mode in {"load", "auto"} and os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu")
            if "idx" not in cache or "z_pred" not in cache or "x_pred" not in cache:
                raise ValueError("stage1_cache missing required keys: idx, z_pred, x_pred")
        elif cache_mode == "load":
            raise ValueError(f"stage1_cache_mode=load but file not found: {cache_path}")
    cache_write = None
    if cache is None and cache_mode in {"save", "auto"} and cache_path:
        cache_write = {"idx": [], "z_pred": [], "x_pred": []}

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    interp_list = []
    refined_list = []
    gt_list = []
    keypoints_list = []
    idx_list = []
    mask_list = []
    start_goal_list = []
    difficulty_list = []
    steps_list = []
    occ_list = []
    sdf_list = []
    occ_np = None
    sdf_np = None
    run_config = {
        "args": vars(args),
        "kp_meta": payload_kp.get("meta", {}) if isinstance(payload_kp, dict) else {},
        "interp_meta": payload_interp.get("meta", {}) if isinstance(payload_interp, dict) else {},
    }
    with open(metrics_path, "w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(
            [
                "sample_id",
                "collision_interp",
                "collision_refined",
                "goal_dist_interp",
                "goal_dist_refined",
                "success_interp",
                "success_refined",
                "collision_oracle_interp",
                "collision_pred_interp",
                "collision_oracle_refined",
                "goal_dist_oracle_refined",
                "success_oracle_refined",
            ]
        )
        with torch.no_grad():
            for batch in tqdm(loader, dynamic_ncols=True):
                cond = {k: v.to(device) for k, v in get_cond_from_sample(batch).items()}
                x0_gt = batch.get("x", None)
                if x0_gt is not None:
                    x0_gt = x0_gt.to(device)
                B = cond["start_goal"].shape[0]
                selector_logits = None
                use_cache = cache is not None
                if use_cache:
                    if idx_global + B > cache["idx"].shape[0]:
                        raise ValueError("stage1_cache has fewer samples than requested")
                    idx = cache["idx"][idx_global : idx_global + B].to(device)
                    z_pred = cache["z_pred"][idx_global : idx_global + B].to(device)
                    x_pred = cache["x_pred"][idx_global : idx_global + B].to(device)
                    if bool(args.clamp_endpoints) and "start_goal" in cond:
                        sg = cond["start_goal"]
                        start = sg[:, :2]
                        goal = sg[:, 2:]
                        err_start = (x_pred[:, 0, :2] - start).abs().max().item()
                        err_goal = (x_pred[:, -1, :2] - goal).abs().max().item()
                        if err_start > 1e-4 or err_goal > 1e-4:
                            raise ValueError(
                                "stage1_cache endpoints mismatch with current start_goal. "
                                "Clear cache or set STAGE1_CACHE_MODE=none."
                            )
                    masks = torch.zeros((B, args.T), device=device, dtype=torch.bool)
                    masks.scatter_(1, idx, True)
                    z_steps = None
                else:
                    if args.kp_index_mode == "random":
                        idx, masks = sample_fixed_k_indices_batch(
                            B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True
                        )
                    elif args.kp_index_mode in {"uniform", "uniform_jitter"}:
                        jitter = args.kp_jitter if args.kp_index_mode == "uniform_jitter" else 0.0
                        idx, masks = sample_fixed_k_indices_uniform_batch(
                            B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True, jitter=jitter
                        )
                    else:
                        if selector_model is None:
                            raise ValueError("kp_index_mode=selector but selector model not loaded")
                        with torch.no_grad():
                            if selector_use_level:
                                if selector_level_mode == "s_norm":
                                    level_val = float(args.levels) / float(max(1, args.levels))
                                else:
                                    level_val = float(args.K_min) / float(max(1, args.T - 1))
                                cond_sel = dict(cond)
                                cond_sel["level"] = torch.full((B, 1), level_val, device=device)
                                logits = selector_model(cond_sel)
                            else:
                                logits = selector_model(cond)
                        idx = select_topk_indices(logits, args.K_min)
                        masks = torch.zeros((B, args.T), device=device, dtype=torch.bool)
                        masks.scatter_(1, idx, True)
                    selector_logits = logits if args.kp_index_mode == "selector" else None

                    if bool(args.use_kp_feat) and int(args.kp_feat_dim) > 0:
                        left_diff = None
                        right_diff = None
                        if dphi_model is not None:
                            seg_feat_sel = build_segment_features_from_idx(idx, args.T, dphi_seg_feat_dim)
                            seg_cost = dphi_model(cond, seg_feat_sel)
                            K = idx.shape[1]
                            left_diff = torch.zeros((B, K), device=device, dtype=torch.float32)
                            right_diff = torch.zeros((B, K), device=device, dtype=torch.float32)
                            left_diff[:, 1:] = seg_cost
                            right_diff[:, :-1] = seg_cost
                        cond["kp_feat"] = _kp_feat_from_idx(
                            idx, args.T, int(args.kp_feat_dim), left_diff, right_diff
                        )

                    # Predicted keypoints (Stage 1).
                    known_mask, known_values = _build_known_mask_values(
                        idx, cond, data_dim, args.T, bool(args.clamp_endpoints)
                    )
                    if args.logit_space:
                        known_values = logit_pos(known_values, eps=args.logit_eps)
                    if args.save_diffusion_frames:
                        z_pred, z_steps = _sample_keypoints_ddim(
                            kp_model,
                            schedule,
                            idx,
                            known_mask,
                            known_values,
                            cond,
                            args.ddim_steps,
                            args.T,
                            schedule_name=args.ddim_schedule,
                            return_intermediates=True,
                            pos_clip=bool(args.pos_clip) and not bool(args.logit_space),
                            pos_clip_min=args.pos_clip_min,
                            pos_clip_max=args.pos_clip_max,
                        )
                    else:
                        z_pred = _sample_keypoints_ddim(
                            kp_model,
                            schedule,
                            idx,
                            known_mask,
                            known_values,
                            cond,
                            args.ddim_steps,
                            args.T,
                            schedule_name=args.ddim_schedule,
                            pos_clip=bool(args.pos_clip) and not bool(args.logit_space),
                            pos_clip_min=args.pos_clip_min,
                            pos_clip_max=args.pos_clip_max,
                        )
                        z_steps = None
                    if args.logit_space:
                        z_pred = sigmoid_pos(z_pred)
                        if z_steps is not None:
                            z_steps = [sigmoid_pos(z_step) for z_step in z_steps]

                    x_pred = interpolate_from_indices(
                        idx, z_pred, args.T, recompute_velocity=bool(args.recompute_vel)
                    )
                    if cache_write is not None:
                        cache_write["idx"].append(idx.detach().cpu())
                        cache_write["z_pred"].append(z_pred.detach().cpu())
                        cache_write["x_pred"].append(x_pred.detach().cpu())
                conf_pred = None
                if args.anchor_conf:
                    conf_pred = _build_anchor_conf(
                        masks,
                        masks,
                        True,
                        args.anchor_conf_teacher,
                        args.anchor_conf_student,
                        args.anchor_conf_endpoints,
                        args.anchor_conf_missing,
                        bool(args.clamp_endpoints),
                    )

                # Oracle keypoints interpolation.
                z_oracle = None
                x_oracle = None
                if args.oracle_keypoints or args.compare_oracle:
                    if x0_gt is None:
                        raise ValueError("oracle_keypoints/compare_oracle requires batch['x'] to be present.")
                    idx_exp = idx.unsqueeze(-1).expand(-1, -1, x0_gt.shape[-1])
                    z_oracle = x0_gt.gather(1, idx_exp)
                    x_oracle = interpolate_from_indices(idx, z_oracle, args.T, recompute_velocity=bool(args.recompute_vel))

                stage2_steps = None
                x_hat_oracle = None
                if args.skip_stage2 or interp_model is None:
                    x_hat = x_pred
                elif args.stage2_mode == "adj":
                    if args.kp_index_mode == "selector":
                        if selector_model is None:
                            raise ValueError("kp_index_mode=selector but selector model not loaded")
                        if selector_use_level:
                            k_list = _compute_k_schedule(
                                args.T, args.K_min, args.levels, schedule=args.k_schedule, geom_gamma=args.k_geom_gamma
                            )
                            logits_levels = []
                            for s in range(args.levels + 1):
                                if selector_level_mode == "s_norm":
                                    level_val = float(s) / float(max(1, args.levels))
                                else:
                                    level_val = float(k_list[s]) / float(max(1, args.T - 1))
                                cond_sel = dict(cond)
                                cond_sel["level"] = torch.full((B, 1), level_val, device=device)
                                with torch.no_grad():
                                    logits_s = selector_model(cond_sel)
                                logits_levels.append(logits_s)
                            logits_levels = torch.stack(logits_levels, dim=1)
                            masks_levels, _ = build_nested_masks_from_level_logits(
                                logits_levels,
                                args.K_min,
                                args.levels,
                                k_schedule=args.k_schedule,
                                k_geom_gamma=args.k_geom_gamma,
                            )
                        else:
                            if selector_logits is None:
                                with torch.no_grad():
                                    selector_logits = selector_model(cond)
                            masks_levels, _ = build_nested_masks_from_logits(
                                selector_logits,
                                args.K_min,
                                args.levels,
                                k_schedule=args.k_schedule,
                                k_geom_gamma=args.k_geom_gamma,
                            )
                    else:
                        masks_levels, _ = build_nested_masks_from_base(
                            idx,
                            args.T,
                            args.levels,
                            generator=gen,
                            device=device,
                            k_schedule=args.k_schedule,
                            k_geom_gamma=args.k_geom_gamma,
                        )
                    sigma_levels = None
                    if args.s2_sample_noise_mode == "level":
                        k_list = _compute_k_schedule(
                            args.T, args.K_min, args.levels, schedule=args.k_schedule, geom_gamma=args.k_geom_gamma
                        )
                        sigma_levels = [
                            _compute_sigma_for_level(
                                int(k_list[s]),
                                args.K_min,
                                s2_corrupt_sigma_max,
                                s2_corrupt_sigma_min,
                                s2_corrupt_sigma_pow,
                            )
                            for s in range(args.levels + 1)
                        ]
                    x_curr = x_pred
                    if args.save_diffusion_frames:
                        stage2_steps = []
                    for s in range(args.levels, 0, -1):
                        mask_s = masks_levels[:, s]
                        mask_prev = masks_levels[:, s - 1]
                        if args.anchor_conf:
                            conf_s = _build_anchor_conf(
                                mask_s,
                                None,
                                False,
                                args.anchor_conf_teacher,
                                args.anchor_conf_student,
                                args.anchor_conf_endpoints,
                                args.anchor_conf_missing,
                                bool(args.clamp_endpoints),
                            )
                            conf_s = _anneal_conf(conf_s, s, args.levels, args.anchor_conf_anneal_mode)
                            mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
                        else:
                            conf_s = None
                            mask_in = torch.stack([mask_s, mask_prev], dim=-1)
                        s_level = torch.full((B,), s, device=device, dtype=torch.long)
                        delta_hat = interp_model(x_curr, s_level, mask_in, cond)
                        x_curr = x_curr + delta_hat
                        if args.s2_sample_noise_mode != "none":
                            if args.s2_sample_noise_mode == "constant":
                                sigma = float(args.s2_sample_noise_sigma)
                            else:
                                sigma = float(sigma_levels[s]) if sigma_levels is not None else 0.0
                            x_curr = _apply_s2_noise(x_curr, mask_s, sigma, args.s2_sample_noise_scale)
                        if args.soft_anchor_clamp and conf_s is not None:
                            lam = _soft_clamp_lambda(s, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                            x_curr = apply_soft_clamp(x_curr, x_pred, conf_s, lam, args.clamp_dims)
                        if args.clamp_policy == "all_anchors":
                            clamp_mask = mask_s
                        elif args.clamp_policy == "endpoints":
                            clamp_mask = torch.zeros_like(mask_s)
                            clamp_mask[:, 0] = True
                            clamp_mask[:, -1] = True
                        else:
                            clamp_mask = None
                        if clamp_mask is not None:
                            x_curr = apply_clamp(x_curr, x_pred, clamp_mask, args.clamp_dims)
                        if args.pos_clip:
                            x_curr[:, :, :2] = x_curr[:, :, :2].clamp(min=args.pos_clip_min, max=args.pos_clip_max)
                        if stage2_steps is not None:
                            stage2_steps.append(x_curr.detach().clone())
                    x_hat = x_curr
                    if x_oracle is not None:
                        x_curr = x_oracle
                        for s in range(args.levels, 0, -1):
                            mask_s = masks_levels[:, s]
                            mask_prev = masks_levels[:, s - 1]
                            if args.anchor_conf:
                                conf_s = _build_anchor_conf(
                                    mask_s,
                                    None,
                                    False,
                                    args.anchor_conf_teacher,
                                    args.anchor_conf_student,
                                    args.anchor_conf_endpoints,
                                    args.anchor_conf_missing,
                                    bool(args.clamp_endpoints),
                                )
                                conf_s = _anneal_conf(conf_s, s, args.levels, args.anchor_conf_anneal_mode)
                                mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
                            else:
                                conf_s = None
                                mask_in = torch.stack([mask_s, mask_prev], dim=-1)
                            s_level = torch.full((B,), s, device=device, dtype=torch.long)
                            delta_hat = interp_model(x_curr, s_level, mask_in, cond)
                            x_curr = x_curr + delta_hat
                            if args.s2_sample_noise_mode != "none":
                                if args.s2_sample_noise_mode == "constant":
                                    sigma = float(args.s2_sample_noise_sigma)
                                else:
                                    sigma = float(sigma_levels[s]) if sigma_levels is not None else 0.0
                                x_curr = _apply_s2_noise(x_curr, mask_s, sigma, args.s2_sample_noise_scale)
                            if args.soft_anchor_clamp and conf_s is not None:
                                lam = _soft_clamp_lambda(s, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                                x_curr = apply_soft_clamp(x_curr, x_oracle, conf_s, lam, args.clamp_dims)
                            if args.clamp_policy == "all_anchors":
                                clamp_mask = mask_s
                            elif args.clamp_policy == "endpoints":
                                clamp_mask = torch.zeros_like(mask_s)
                                clamp_mask[:, 0] = True
                                clamp_mask[:, -1] = True
                            else:
                                clamp_mask = None
                            if clamp_mask is not None:
                                x_curr = apply_clamp(x_curr, x_oracle, clamp_mask, args.clamp_dims)
                            if args.pos_clip:
                                x_curr[:, :, :2] = x_curr[:, :, :2].clamp(min=args.pos_clip_min, max=args.pos_clip_max)
                            x_hat_oracle = x_curr
                else:
                    s_level = torch.full((B,), args.levels, device=device, dtype=torch.long)
                    if args.anchor_conf and conf_pred is not None:
                        conf_s = _anneal_conf(conf_pred, args.levels, args.levels, args.anchor_conf_anneal_mode)
                        mask_in = torch.stack([masks.float(), conf_s], dim=-1)
                    else:
                        mask_in = masks
                    delta_hat = interp_model(x_pred, s_level, mask_in, cond)
                    x_hat = x_pred + delta_hat
                    if args.s2_sample_noise_mode != "none":
                        if args.s2_sample_noise_mode == "constant":
                            sigma = float(args.s2_sample_noise_sigma)
                        else:
                            sigma = _compute_sigma_for_level(
                                int(args.K_min),
                                args.K_min,
                                s2_corrupt_sigma_max,
                                s2_corrupt_sigma_min,
                                s2_corrupt_sigma_pow,
                            )
                        x_hat = _apply_s2_noise(x_hat, masks, sigma, args.s2_sample_noise_scale)
                    if args.soft_anchor_clamp and conf_pred is not None:
                        lam = _soft_clamp_lambda(args.levels, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                        x_hat = apply_soft_clamp(x_hat, x_pred, conf_pred, lam, args.clamp_dims)
                    if args.clamp_policy == "all_anchors":
                        clamp_mask = masks
                    elif args.clamp_policy == "endpoints":
                        clamp_mask = torch.zeros_like(masks)
                        clamp_mask[:, 0] = True
                        clamp_mask[:, -1] = True
                    else:
                        clamp_mask = None
                    if clamp_mask is not None:
                        x_hat = apply_clamp(x_hat, x_pred, clamp_mask, args.clamp_dims)
                    if x_oracle is not None:
                        if args.anchor_conf:
                            conf_oracle = _build_anchor_conf(
                                masks,
                                None,
                                False,
                                args.anchor_conf_teacher,
                                args.anchor_conf_student,
                                args.anchor_conf_endpoints,
                                args.anchor_conf_missing,
                                bool(args.clamp_endpoints),
                            )
                            conf_oracle = _anneal_conf(conf_oracle, args.levels, args.levels, args.anchor_conf_anneal_mode)
                            mask_in = torch.stack([masks.float(), conf_oracle], dim=-1)
                        else:
                            conf_oracle = None
                            mask_in = masks
                        delta_hat = interp_model(x_oracle, s_level, mask_in, cond)
                        x_hat_oracle = x_oracle + delta_hat
                        if args.s2_sample_noise_mode != "none":
                            if args.s2_sample_noise_mode == "constant":
                                sigma = float(args.s2_sample_noise_sigma)
                            else:
                                sigma = _compute_sigma_for_level(
                                    int(args.K_min),
                                    args.K_min,
                                    s2_corrupt_sigma_max,
                                    s2_corrupt_sigma_min,
                                    s2_corrupt_sigma_pow,
                                )
                            x_hat_oracle = _apply_s2_noise(x_hat_oracle, masks, sigma, args.s2_sample_noise_scale)
                        if args.soft_anchor_clamp and conf_oracle is not None:
                            lam = _soft_clamp_lambda(args.levels, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                            x_hat_oracle = apply_soft_clamp(x_hat_oracle, x_oracle, conf_oracle, lam, args.clamp_dims)
                        if clamp_mask is not None:
                            x_hat_oracle = apply_clamp(x_hat_oracle, x_oracle, clamp_mask, args.clamp_dims)

                for b in range(B):
                    occ_t = cond["occ"][b, 0]
                    interp_t = x_pred[b]
                    refined_t = x_hat[b]
                    if args.clamp_endpoints and "start_goal" in cond:
                        goal_t = cond["start_goal"][b, 2:]
                    elif x0_gt is not None:
                        goal_t = x0_gt[b, -1, :2]
                    else:
                        goal_t = None

                    if goal_t is None:
                        goal_dist_interp = float("nan")
                        goal_dist_refined = float("nan")
                        success_interp = float("nan")
                        success_refined = float("nan")
                    else:
                        goal_dist_interp = goal_distance(goal_t, interp_t)
                        goal_dist_refined = goal_distance(goal_t, refined_t)
                        success_interp = success(goal_t, interp_t, occ_t.shape[-2], occ_t.shape[-1])
                        success_refined = success(goal_t, refined_t, occ_t.shape[-2], occ_t.shape[-1])

                    coll_oracle = float("nan")
                    coll_pred = collision_rate(occ_t, interp_t)
                    if x_oracle is not None:
                        coll_oracle = collision_rate(occ_t, x_oracle[b])
                    coll_oracle_refined = float("nan")
                    goal_dist_oracle_refined = float("nan")
                    success_oracle_refined = float("nan")
                    if x_hat_oracle is not None:
                        coll_oracle_refined = collision_rate(occ_t, x_hat_oracle[b])
                        if goal_t is not None:
                            goal_dist_oracle_refined = goal_distance(goal_t, x_hat_oracle[b])
                            success_oracle_refined = success(
                                goal_t, x_hat_oracle[b], occ_t.shape[-2], occ_t.shape[-1]
                            )

                    writer.writerow(
                        [
                            idx_global,
                            collision_rate(occ_t, interp_t),
                            collision_rate(occ_t, refined_t),
                            goal_dist_interp,
                            goal_dist_refined,
                            success_interp,
                            success_refined,
                            coll_oracle,
                            coll_pred,
                            coll_oracle_refined,
                            goal_dist_oracle_refined,
                            success_oracle_refined,
                        ]
                    )

                    if args.save_npz:
                        occ_list.append(occ_t.detach().cpu().numpy())
                        if "sdf" in cond:
                            sdf_t = cond["sdf"][b]
                            if sdf_t.dim() == 3:
                                sdf_t = sdf_t[0]
                            sdf_list.append(sdf_t.detach().cpu().numpy())
                        if occ_np is None:
                            occ_np = occ_t.detach().cpu().numpy()
                        if sdf_np is None and "sdf" in cond:
                            sdf_np = cond["sdf"][b, 0].detach().cpu().numpy()
                        interp_list.append(interp_t.detach().cpu().numpy())
                        refined_list.append(refined_t.detach().cpu().numpy())
                        if x0_gt is not None:
                            gt_list.append(x0_gt[b].detach().cpu().numpy())
                        keypoints_list.append(z_pred[b].detach().cpu().numpy())
                        idx_list.append(idx[b].detach().cpu().numpy())
                        mask_list.append(masks[b].detach().cpu().numpy())
                        if "start_goal" in cond:
                            start_goal_list.append(cond["start_goal"][b].detach().cpu().numpy())
                        if "difficulty" in batch:
                            difficulty_list.append(int(batch["difficulty"][b].item()))

                    if args.save_debug:
                        out_path = os.path.join(args.out_dir, f"sample_{idx_global:04d}.png")
                        interp_np = interp_t.detach().cpu().numpy()
                        refined_np = refined_t.detach().cpu().numpy()
                        if args.compare_oracle and x_oracle is not None:
                            occ = occ_t.detach().cpu().numpy()
                            sg = cond["start_goal"][b].detach().cpu().numpy() if "start_goal" in cond else None
                            panels = []
                            panels.append(
                                {
                                    "traj": x_oracle[b].detach().cpu().numpy()[:, :2],
                                    "label": "oracle interp",
                                    "coll": coll_oracle,
                                    "kp": z_oracle[b].detach().cpu().numpy()[:, :2]
                                    if (args.plot_keypoints and z_oracle is not None)
                                    else None,
                                    "sg": sg,
                                    "footer": "phase1 (oracle)",
                                }
                            )
                            panels.append(
                                {
                                    "traj": interp_np[:, :2],
                                    "label": "pred interp",
                                    "coll": coll_pred,
                                    "kp": z_pred[b].detach().cpu().numpy()[:, :2] if args.plot_keypoints else None,
                                    "sg": sg,
                                    "footer": "phase1 (pred)",
                                }
                            )
                            if not args.skip_stage2:
                                panels.append(
                                    {
                                        "traj": refined_np[:, :2],
                                        "label": "stage2 (pred)",
                                        "coll": collision_rate(occ_t, refined_t),
                                        "kp": None,
                                        "sg": sg,
                                        "footer": "phase2 (pred)",
                                    }
                                )
                                if x_hat_oracle is not None:
                                    panels.append(
                                        {
                                            "traj": x_hat_oracle[b].detach().cpu().numpy()[:, :2],
                                            "label": "stage2 (oracle)",
                                            "coll": coll_oracle_refined,
                                            "kp": None,
                                            "sg": sg,
                                            "footer": "phase2 (oracle)",
                                        }
                                    )
                            _plot_compare_panels(occ, panels, out_path)
                        else:
                            trajs = [interp_np[:, :2], refined_np[:, :2]]
                            labels = ["interp", "refined"]
                            if args.force_single:
                                trajs = [interp_np[:, :2]]
                                labels = ["interp"]
                            if args.plot_gt and x0_gt is not None:
                                gt_np = x0_gt[b].detach().cpu().numpy()
                                trajs.append(gt_np[:, :2])
                                labels.append("gt")
                            if dataset_name in {"d4rl", "d4rl_prepared"}:
                                occ = occ_t.detach().cpu().numpy()
                                kps = None
                                if args.plot_keypoints:
                                    kps = [z_pred[b].detach().cpu().numpy()[:, :2]] + [None] * (len(trajs) - 1)
                                plot_trajectories(
                                    occ,
                                    trajs,
                                    labels,
                                    out_path=out_path,
                                    plot_points=bool(args.plot_points),
                                    keypoints=kps,
                                    flip_y=flip_y,
                                )
                            elif mj_walls_norm is not None:
                                norm_trajs = [interp_np[:, :2], refined_np[:, :2]]
                                norm_labels = ["interp", "refined"]
                                if args.force_single:
                                    norm_trajs = [interp_np[:, :2]]
                                    norm_labels = ["interp"]
                                if args.plot_gt and x0_gt is not None:
                                    norm_trajs.append(gt_np[:, :2])
                                    norm_labels.append("gt")
                                plot_maze2d_geom_walls(
                                    mj_walls_norm,
                                    norm_trajs,
                                    norm_labels,
                                    out_path=out_path,
                                    bounds=((0.0, 1.0), (0.0, 1.0)),
                                    plot_points=bool(args.plot_points),
                                )
                            elif maze_map is not None and maze_scale is not None:
                                plot_trajs = [_denorm_xy(t) for t in trajs]
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    plot_trajs,
                                    labels,
                                    out_path=out_path,
                                    plot_points=bool(args.plot_points),
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                plot_trajectories(
                                    occ,
                                    trajs,
                                    labels,
                                    out_path=out_path,
                                    plot_points=bool(args.plot_points),
                                    flip_y=flip_y,
                                )

                    if args.save_debug and args.save_diffusion_frames and z_steps is not None:
                        frames_dir = os.path.join(args.out_dir, "diffusion_steps", f"sample_{idx_global:04d}")
                        os.makedirs(frames_dir, exist_ok=True)
                        step_indices = list(range(0, len(z_steps), max(1, args.frames_stride)))
                        for si, step_idx in enumerate(step_indices):
                            z_step = z_steps[step_idx][b : b + 1]
                            x_step = interpolate_from_indices(
                                idx[b : b + 1], z_step, args.T, recompute_velocity=bool(args.recompute_vel)
                            )
                            frame_path = os.path.join(frames_dir, f"step_{si:03d}.png")
                            step_np = x_step[0].detach().cpu().numpy()
                            if dataset_name in {"d4rl", "d4rl_prepared"}:
                                occ = occ_t.detach().cpu().numpy()
                                footer = f"phase1 step {si}/{len(step_indices)-1}"
                                plot_trajectories(
                                    occ,
                                    [step_np[:, :2]],
                                    [f"phase1"],
                                    out_path=frame_path,
                                    footer=footer,
                                    plot_points=bool(args.plot_points),
                                    keypoints=[z_step[0].detach().cpu().numpy()[:, :2]] if args.plot_keypoints else None,
                                    flip_y=flip_y,
                                )
                            elif mj_walls_norm is not None:
                                plot_maze2d_geom_walls(
                                    mj_walls_norm,
                                    [step_np[:, :2]],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    bounds=((0.0, 1.0), (0.0, 1.0)),
                                    plot_points=bool(args.plot_points),
                                    keypoints=[z_step[0].detach().cpu().numpy()[:, :2]] if args.plot_keypoints else None,
                                )
                            elif maze_map is not None and maze_scale is not None:
                                step_plot = _denorm_xy(step_np[:, :2])
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    [step_plot],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    plot_points=bool(args.plot_points),
                                    keypoints=[_denorm_xy(z_step[0].detach().cpu().numpy()[:, :2])]
                                    if args.plot_keypoints
                                    else None,
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                plot_trajectories(
                                    occ,
                                    [step_np[:, :2]],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    plot_points=bool(args.plot_points),
                                    keypoints=[z_step[0].detach().cpu().numpy()[:, :2]] if args.plot_keypoints else None,
                                    flip_y=flip_y,
                                )
                        if args.frames_include_stage2 and not args.skip_stage2:
                            base = len(step_indices)

                            def _render_phase2(traj_np: np.ndarray, frame_path: str, footer: str):
                                if dataset_name in {"d4rl", "d4rl_prepared"}:
                                    occ = occ_t.detach().cpu().numpy()
                                    plot_trajectories(
                                        occ,
                                        [traj_np[:, :2]],
                                        ["phase2"],
                                        out_path=frame_path,
                                        footer=footer,
                                        plot_points=bool(args.plot_points),
                                        flip_y=flip_y,
                                    )
                                elif mj_walls_norm is not None:
                                    plot_maze2d_geom_walls(
                                        mj_walls_norm,
                                        [traj_np[:, :2]],
                                        ["phase2"],
                                        out_path=frame_path,
                                        bounds=((0.0, 1.0), (0.0, 1.0)),
                                        footer=footer,
                                        plot_points=bool(args.plot_points),
                                    )
                                elif maze_map is not None and maze_scale is not None:
                                    traj_plot = _denorm_xy(traj_np[:, :2])
                                    plot_maze2d_trajectories(
                                        maze_map,
                                        maze_scale,
                                        [traj_plot],
                                        ["phase2"],
                                        out_path=frame_path,
                                        footer=footer,
                                        plot_points=bool(args.plot_points),
                                    )
                                else:
                                    occ = occ_t.detach().cpu().numpy()
                                    plot_trajectories(
                                        occ,
                                        [traj_np[:, :2]],
                                        ["phase2"],
                                        out_path=frame_path,
                                        footer=footer,
                                        plot_points=bool(args.plot_points),
                                        flip_y=flip_y,
                                    )

                            if stage2_steps is not None and len(stage2_steps) > 0:
                                for si2, x_step2 in enumerate(stage2_steps):
                                    frame_path = os.path.join(frames_dir, f"step_{base + si2:03d}.png")
                                    step_np2 = x_step2[b].detach().cpu().numpy()
                                    footer = f"phase2 step {si2+1}/{len(stage2_steps)}"
                                    _render_phase2(step_np2, frame_path, footer)
                                final_path = os.path.join(frames_dir, "stage2.png")
                                final_np = stage2_steps[-1][b].detach().cpu().numpy()
                                _render_phase2(
                                    final_np,
                                    final_path,
                                    f"phase2 step {len(stage2_steps)}/{len(stage2_steps)}",
                                )
                            else:
                                stage2_step = base
                                final_step_path = os.path.join(frames_dir, f"step_{stage2_step:03d}.png")
                                final_path = os.path.join(frames_dir, "stage2.png")
                                refined_np = refined_t.detach().cpu().numpy()
                                _render_phase2(refined_np, final_step_path, "phase2 step 1/1")
                                if final_path != final_step_path:
                                    _render_phase2(refined_np, final_path, "phase2 step 1/1")
                        _export_video(frames_dir, args.export_video, args.video_fps)
                    idx_global += 1

                if args.save_steps_npz and z_steps is not None:
                    # Store interpolated trajectories for each diffusion step.
                    for b in range(B):
                        step_trajs = []
                        for z_step in z_steps:
                            x_step = interpolate_from_indices(
                                idx[b : b + 1], z_step[b : b + 1], args.T, recompute_velocity=bool(args.recompute_vel)
                            )
                            step_trajs.append(x_step[0].detach().cpu().numpy())
                        steps_list.append(np.stack(step_trajs, axis=0))

    if cache_write is not None:
        cache_out = {
            "idx": torch.cat(cache_write["idx"], dim=0) if cache_write["idx"] else torch.empty((0,), dtype=torch.long),
            "z_pred": torch.cat(cache_write["z_pred"], dim=0) if cache_write["z_pred"] else torch.empty((0,)),
            "x_pred": torch.cat(cache_write["x_pred"], dim=0) if cache_write["x_pred"] else torch.empty((0,)),
            "args": vars(args),
        }
        torch.save(cache_out, cache_path)

    if args.save_npz:
        out_npz = os.path.join(args.out_dir, "samples.npz")
        save_kwargs = {
            "interp": np.stack(interp_list, axis=0) if interp_list else np.zeros((0,)),
            "refined": np.stack(refined_list, axis=0) if refined_list else np.zeros((0,)),
            "keypoints": np.stack(keypoints_list, axis=0) if keypoints_list else np.zeros((0,)),
            "idx": np.stack(idx_list, axis=0) if idx_list else np.zeros((0,), dtype=np.int64),
            "mask": np.stack(mask_list, axis=0) if mask_list else np.zeros((0,)),
            "start_goal": np.stack(start_goal_list, axis=0) if start_goal_list else np.zeros((0,)),
        }
        if gt_list:
            save_kwargs["gt"] = np.stack(gt_list, axis=0)
        if difficulty_list:
            save_kwargs["difficulty"] = np.asarray(difficulty_list, dtype=np.int64)
        if occ_list:
            save_kwargs["occ"] = np.stack(occ_list, axis=0)
        elif occ_np is not None:
            save_kwargs["occ"] = occ_np
        if sdf_list:
            save_kwargs["sdf"] = np.stack(sdf_list, axis=0)
        elif sdf_np is not None:
            save_kwargs["sdf"] = sdf_np
        if args.save_steps_npz and steps_list:
            save_kwargs["interp_steps"] = np.stack(steps_list, axis=0)
        np.savez_compressed(out_npz, **save_kwargs)

        cfg_path = os.path.join(args.out_dir, "run_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
