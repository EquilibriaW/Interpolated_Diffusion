import argparse
import csv
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import (
    interpolate_from_indices,
    sample_fixed_k_indices_batch,
    sample_fixed_k_indices_uniform_batch,
)
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset, PreparedTrajectoryDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.eval.metrics import collision_rate, goal_distance, success
from src.eval.visualize import plot_maze2d_geom_walls, plot_maze2d_trajectories, plot_trajectories
from src.models.denoiser_interp_levels import InterpLevelDenoiser
from src.models.denoiser_keypoints import KeypointDenoiser
from src.utils.clamp import apply_clamp
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
    p.add_argument("--N_train", type=int, default=None)
    p.add_argument("--schedule", type=str, default=None, choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--recompute_vel", type=int, default=1)
    p.add_argument("--logit_space", type=int, default=0)
    p.add_argument("--logit_eps", type=float, default=1e-5)
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
    p.add_argument("--use_start_goal", type=int, default=1)
    p.add_argument("--override_meta", type=int, default=0)
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
    p.add_argument("--force_single", type=int, default=0)
    p.add_argument("--kp_index_mode", type=str, default="uniform", choices=["random", "uniform", "uniform_jitter"])
    p.add_argument("--kp_jitter", type=float, default=0.0)
    return p


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

        with imageio.get_writer(out_path, fps=fps) as writer:
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
    idx: torch.Tensor, cond: dict, D: int, T: int, use_start_goal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K = idx.shape
    known_mask = torch.zeros((B, K, D), device=idx.device, dtype=torch.bool)
    known_values = torch.zeros((B, K, D), device=idx.device, dtype=torch.float32)
    if use_start_goal and "start_goal" in cond and D >= 2:
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
    return_intermediates: bool = False,
):
    device = idx.device
    B, K = idx.shape
    D = known_values.shape[-1]
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps)
    z = torch.randn((B, K, D), device=device)
    z = torch.where(known_mask, known_values, z)
    intermediates = [z.detach().clone()] if return_intermediates else None
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(z, t, idx, known_mask, cond, T)
        z = ddim_step(z, eps, t, t_prev, schedule, eta=0.0)
        z = torch.where(known_mask, known_values, z)
        if return_intermediates:
            intermediates.append(z.detach().clone())
    if return_intermediates:
        return z, intermediates
    return z


def main():
    args = build_argparser().parse_args()
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
        if meta.get("use_start_goal") is not None:
            args.use_start_goal = int(bool(meta.get("use_start_goal")))
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

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2

    kp_model = KeypointDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        use_start_goal=bool(args.use_start_goal),
    ).to(device)
    if "model" not in payload_kp:
        raise FileNotFoundError(f"Checkpoint not found or invalid: {args.ckpt_keypoints}")

    kp_model.load_state_dict(payload_kp["model"])
    interp_model = None
    payload_interp = {}
    if not args.skip_stage2:
        interp_model = InterpLevelDenoiser(
            data_dim=data_dim,
            use_sdf=bool(args.use_sdf),
            max_levels=args.levels,
            use_start_goal=bool(args.use_start_goal),
        ).to(device)
        payload_interp = torch.load(args.ckpt_interp, map_location="cpu") if os.path.exists(args.ckpt_interp) else {}
        if "model" not in payload_interp:
            raise FileNotFoundError(f"Checkpoint not found or invalid: {args.ckpt_interp}")
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
        )
    else:
        dataset = ParticleMazeDataset(
            num_samples=args.n_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
        )
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

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

    def _gridify_xy(xy: np.ndarray, h: int, w: int) -> np.ndarray:
        x = xy[:, 0]
        y = xy[:, 1]
        j = np.floor(x * w).astype(np.int64)
        i = np.floor(y * h).astype(np.int64)
        j = np.clip(j, 0, w - 1)
        i = np.clip(i, 0, h - 1)
        out = np.stack([j / float(w), i / float(h)], axis=-1)
        return out

    def _plot_side_by_side(
        occ: np.ndarray,
        left_traj: np.ndarray,
        right_traj: np.ndarray,
        left_label: str,
        right_label: str,
        left_collision: float,
        right_collision: float,
        left_kp: Optional[np.ndarray],
        right_kp: Optional[np.ndarray],
        left_sg: Optional[np.ndarray],
        right_sg: Optional[np.ndarray],
        out_path: str,
    ):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        h, w = occ.shape
        for ax, traj, label, coll, kp, sg in [
            (axes[0], left_traj, left_label, left_collision, left_kp, left_sg),
            (axes[1], right_traj, right_label, right_collision, right_kp, right_sg),
        ]:
            ax.imshow(occ, cmap="gray_r", origin="upper")
            xy = traj[:, :2]
            ax.plot(xy[:, 0] * w, xy[:, 1] * h, color="tab:blue", linewidth=2.0)
            if sg is not None and sg.shape[0] >= 4:
                ax.scatter([sg[0] * w], [sg[1] * h], s=25, color="tab:green")
                ax.scatter([sg[2] * w], [sg[3] * h], s=40, color="tab:red", marker="x")
            if kp is not None and kp.size > 0:
                ax.scatter(kp[:, 0] * w, kp[:, 1] * h, s=20, color="tab:orange")
            ax.set_title(f"{label} | coll={coll:.3f}", fontsize=9)
            ax.set_xlim([0, w])
            ax.set_ylim([h, 0])
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    idx_global = 0

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
            ]
        )
        with torch.no_grad():
            for batch in tqdm(loader, dynamic_ncols=True):
                cond = {k: v.to(device) for k, v in get_cond_from_sample(batch).items()}
                x0_gt = batch.get("x", None)
                if x0_gt is not None:
                    x0_gt = x0_gt.to(device)
                B = cond["start_goal"].shape[0]
                if args.kp_index_mode == "random":
                    idx, masks = sample_fixed_k_indices_batch(
                        B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True
                    )
                else:
                    jitter = args.kp_jitter if args.kp_index_mode == "uniform_jitter" else 0.0
                    idx, masks = sample_fixed_k_indices_uniform_batch(
                        B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True, jitter=jitter
                    )

                # Predicted keypoints (Stage 1).
                known_mask, known_values = _build_known_mask_values(idx, cond, data_dim, args.T, bool(args.use_start_goal))
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
                        return_intermediates=True,
                    )
                else:
                    z_pred = _sample_keypoints_ddim(
                        kp_model, schedule, idx, known_mask, known_values, cond, args.ddim_steps, args.T
                    )
                    z_steps = None
                if args.logit_space:
                    z_pred = sigmoid_pos(z_pred)
                    if z_steps is not None:
                        z_steps = [sigmoid_pos(z_step) for z_step in z_steps]

                x_pred = interpolate_from_indices(idx, z_pred, args.T, recompute_velocity=bool(args.recompute_vel))

                # Oracle keypoints interpolation.
                z_oracle = None
                x_oracle = None
                if args.oracle_keypoints or args.compare_oracle:
                    if x0_gt is None:
                        raise ValueError("oracle_keypoints/compare_oracle requires batch['x'] to be present.")
                    idx_exp = idx.unsqueeze(-1).expand(-1, -1, x0_gt.shape[-1])
                    z_oracle = x0_gt.gather(1, idx_exp)
                    x_oracle = interpolate_from_indices(idx, z_oracle, args.T, recompute_velocity=bool(args.recompute_vel))

                if args.skip_stage2 or interp_model is None:
                    x_hat = x_pred
                else:
                    s_level = torch.full((B,), args.levels, device=device, dtype=torch.long)
                    delta_hat = interp_model(x_pred, s_level, masks, cond)
                    x_hat = x_pred + delta_hat
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

                for b in range(B):
                    occ_t = cond["occ"][b, 0]
                    interp_t = x_pred[b]
                    refined_t = x_hat[b]
                    if args.use_start_goal:
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
                        ]
                    )

                    if args.save_npz:
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
                            h, w = occ.shape
                            left_traj = _gridify_xy(x_oracle[b].detach().cpu().numpy()[:, :2], h, w)
                            right_traj = _gridify_xy(interp_np[:, :2], h, w)
                            left_kp = (
                                z_oracle[b].detach().cpu().numpy()[:, :2]
                                if (args.plot_keypoints and z_oracle is not None)
                                else None
                            )
                            right_kp = z_pred[b].detach().cpu().numpy()[:, :2] if args.plot_keypoints else None
                            left_kp = _gridify_xy(left_kp, h, w) if left_kp is not None else None
                            right_kp = _gridify_xy(right_kp, h, w) if right_kp is not None else None
                            sg = cond["start_goal"][b].detach().cpu().numpy() if "start_goal" in cond else None
                            if sg is not None:
                                sg_xy = sg.reshape(2, 2)
                                sg_xy = _gridify_xy(sg_xy, h, w)
                                sg = sg_xy.reshape(4,)
                            _plot_side_by_side(
                                occ,
                                left_traj,
                                right_traj,
                                "oracle",
                                "pred",
                                coll_oracle,
                                coll_pred,
                                left_kp,
                                right_kp,
                                sg,
                                sg,
                                out_path,
                            )
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
                                h, w = occ.shape
                                trajs_grid = [_gridify_xy(t, h, w) for t in trajs]
                                plot_trajectories(occ, trajs_grid, labels, out_path=out_path)
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
                                )
                            elif maze_map is not None and maze_scale is not None:
                                plot_trajs = [_denorm_xy(t) for t in trajs]
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    plot_trajs,
                                    labels,
                                    out_path=out_path,
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                plot_trajectories(occ, trajs, labels, out_path=out_path)

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
                                h, w = occ.shape
                                step_grid = _gridify_xy(step_np[:, :2], h, w)
                                plot_trajectories(occ, [step_grid], [f"step {step_idx}"], out_path=frame_path)
                            elif mj_walls_norm is not None:
                                plot_maze2d_geom_walls(
                                    mj_walls_norm,
                                    [step_np[:, :2]],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    bounds=((0.0, 1.0), (0.0, 1.0)),
                                )
                            elif maze_map is not None and maze_scale is not None:
                                step_plot = _denorm_xy(step_np[:, :2])
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    [step_plot],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                plot_trajectories(occ, [step_np[:, :2]], [f"step {step_idx}"], out_path=frame_path)
                        if args.frames_include_stage2 and not args.skip_stage2:
                            final_path = os.path.join(frames_dir, "stage2.png")
                            if dataset_name in {"d4rl", "d4rl_prepared"}:
                                occ = occ_t.detach().cpu().numpy()
                                h, w = occ.shape
                                refined_np = refined_t.detach().cpu().numpy()
                                refined_grid = _gridify_xy(refined_np[:, :2], h, w)
                                plot_trajectories(occ, [refined_grid], ["stage2"], out_path=final_path)
                            elif mj_walls_norm is not None:
                                plot_maze2d_geom_walls(
                                    mj_walls_norm,
                                    [refined_np[:, :2]],
                                    ["stage2"],
                                    out_path=final_path,
                                    bounds=((0.0, 1.0), (0.0, 1.0)),
                                )
                            elif maze_map is not None and maze_scale is not None:
                                refined_plot = _denorm_xy(refined_np[:, :2])
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    [refined_plot],
                                    ["stage2"],
                                    out_path=final_path,
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                refined_np = refined_t.detach().cpu().numpy()
                                plot_trajectories(occ, [refined_np[:, :2]], ["stage2"], out_path=final_path)
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
        if occ_np is not None:
            save_kwargs["occ"] = occ_np
        if sdf_np is not None:
            save_kwargs["sdf"] = sdf_np
        if args.save_steps_npz and steps_list:
            save_kwargs["interp_steps"] = np.stack(steps_list, axis=0)
        np.savez_compressed(out_npz, **save_kwargs)

        cfg_path = os.path.join(args.out_dir, "run_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
