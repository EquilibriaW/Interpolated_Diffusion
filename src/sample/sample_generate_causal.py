import argparse
import os
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm

from src.corruptions.keyframes import interpolate_from_indices, sample_fixed_k_indices_batch
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset, PreparedTrajectoryDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.eval.visualize import plot_maze2d_geom_walls, plot_maze2d_trajectories, plot_trajectories
from src.models.denoiser_interp_levels_causal import InterpLevelCausalDenoiser
from src.models.denoiser_keypoints import KeypointDenoiser
from src.utils.clamp import apply_clamp
from src.utils.device import get_device
from src.utils.normalize import logit_pos, sigmoid_pos


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_keypoints", type=str, default="checkpoints/keypoints/ckpt_final.pt")
    p.add_argument("--ckpt_interp", type=str, default="checkpoints/interp_levels_causal/ckpt_final.pt")
    p.add_argument("--out_dir", type=str, default="runs/gen_causal")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--chunk", type=int, default=16)
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
    p.add_argument("--save_chunk_frames", type=int, default=0)
    p.add_argument("--frames_stride", type=int, default=1)
    p.add_argument("--export_video", type=str, default="none", choices=["none", "mp4", "gif"])
    p.add_argument("--video_fps", type=int, default=8)
    return p


def _heuristic_right(left: torch.Tensor, goal: torch.Tensor, L: int, remaining: int) -> torch.Tensor:
    frac = min(1.0, float(L) / max(1, remaining))
    return left + frac * (goal - left)


def _unnormalize_pos(x: torch.Tensor, pos_low: torch.Tensor, pos_scale: torch.Tensor, flip_y: bool) -> torch.Tensor:
    pos = x[..., :2].clone()
    if flip_y:
        pos[..., 1] = 1.0 - pos[..., 1]
    return pos * pos_scale[:2] + pos_low[:2]


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
        pattern = os.path.join(frames_dir, "frame_%03d.png")
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern, "-pix_fmt", "yuv420p", out_path]
        subprocess.run(cmd, check=False)
    else:
        print("Video export skipped (gif requires imageio).")


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


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _ensure_mujoco_env()
    if args.dataset not in {"d4rl", "d4rl_prepared"}:
        raise ValueError("Particle/synthetic datasets are disabled; use --dataset d4rl or d4rl_prepared.")

    meta = {}
    if os.path.exists(args.ckpt_keypoints):
        try:
            payload_meta = torch.load(args.ckpt_keypoints, map_location="cpu")
            if isinstance(payload_meta, dict):
                meta = payload_meta.get("meta", {}) or {}
        except Exception:
            meta = {}
    def _normalize_dataset_name(name):
        return "d4rl" if name == "d4rl_prepared" else name

    if meta.get("stage") == "keypoints" and not args.override_meta:
        args.T = meta.get("T", args.T)
        args.N_train = meta.get("N_train", args.N_train)
        args.schedule = meta.get("schedule", args.schedule)
        if args.K_min is None:
            args.K_min = meta.get("K", args.K_min)
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
    interp_model = InterpLevelCausalDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        max_levels=args.levels,
        use_start_goal=bool(args.use_start_goal),
    ).to(device)

    payload_kp = torch.load(args.ckpt_keypoints, map_location="cpu") if os.path.exists(args.ckpt_keypoints) else {}
    payload_interp = torch.load(args.ckpt_interp, map_location="cpu") if os.path.exists(args.ckpt_interp) else {}
    if "model" not in payload_kp:
        raise FileNotFoundError(f"Checkpoint not found or invalid: {args.ckpt_keypoints}")
    if "model" not in payload_interp:
        raise FileNotFoundError(f"Checkpoint not found or invalid: {args.ckpt_interp}")
    kp_model.load_state_dict(payload_kp["model"])
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
        if isinstance(payload_interp, dict) and "ema" in payload_interp:
            ema_interp = EMA(interp_model.parameters())
            ema_interp.load_state_dict(payload_interp["ema"])
            ema_interp.copy_to(interp_model.parameters())
        else:
            if not ema_warned:
                print("Checkpoint has no EMA; using raw model weights.")
                ema_warned = True
    kp_model.eval()
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

    maze_map = None
    maze_scale = None
    mj_walls = None
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
        if maze_scale is None and maze_map is not None and pos_scale is not None:
            maze_arr = np.array(maze_map)
            if maze_arr.ndim == 2:
                h, w = maze_arr.shape
                if w > 0 and h > 0:
                    maze_scale = float(min(pos_scale[0].item() / w, pos_scale[1].item() / h))

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)

    for i in tqdm(range(args.n_samples), dynamic_ncols=True):
        sample = dataset[i]
        cond = {k: v.unsqueeze(0).to(device) for k, v in sample["cond"].items()}
        start_goal = cond["start_goal"][0]
        start = start_goal[:2]
        goal = start_goal[2:]

        x_gen = torch.zeros((args.T, data_dim), device=device)
        x_gen[0, :2] = start
        if data_dim > 2:
            x_gen[0, 2:] = 0.0

        cur = 1
        frames_dir = None
        frame_idx = 0
        if args.save_chunk_frames:
            frames_dir = os.path.join(args.out_dir, "chunk_frames", f"sample_{i:04d}")
            os.makedirs(frames_dir, exist_ok=True)
        while cur < args.T:
            end = min(args.T - 1, cur + args.chunk - 1)
            L = end - cur + 1
            remaining = args.T - cur
            left = x_gen[cur - 1, :2]
            right = goal if end == args.T - 1 else _heuristic_right(left, goal, L, remaining)
            local_T = L + 1

            idx_local, mask_local = sample_fixed_k_indices_batch(
                1, local_T, min(args.K_min, local_T), generator=gen, device=device, ensure_endpoints=True
            )
            mask_local = mask_local[0]
            known_mask = torch.zeros((1, idx_local.shape[1], data_dim), device=device, dtype=torch.bool)
            known_values = torch.zeros((1, idx_local.shape[1], data_dim), device=device)
            if args.use_start_goal:
                known_mask[:, :, :2] = (idx_local == 0).unsqueeze(-1) | (idx_local == local_T - 1).unsqueeze(-1)
                known_values[:, :, :2] = torch.where(
                    (idx_local == 0).unsqueeze(-1), left.view(1, 1, 2), known_values[:, :, :2]
                )
                known_values[:, :, :2] = torch.where(
                    (idx_local == local_T - 1).unsqueeze(-1), right.view(1, 1, 2), known_values[:, :, :2]
                )
            if args.logit_space:
                known_values = logit_pos(known_values, eps=args.logit_eps)

            cond_chunk = {k: v for k, v in cond.items()}
            cond_chunk["start_goal"] = torch.cat([left, right], dim=0).view(1, 4)

            z_hat = _sample_keypoints_ddim(
                kp_model, schedule, idx_local, known_mask, known_values, cond_chunk, args.ddim_steps, local_T
            )
            if args.logit_space:
                z_hat = sigmoid_pos(z_hat)

            x_s = interpolate_from_indices(idx_local, z_hat, local_T, recompute_velocity=bool(args.recompute_vel))

            # Build a sequence with full prefix context (0..cur-2) and current chunk (cur-1..end).
            full_len = end + 1
            x_full = torch.zeros((1, full_len, data_dim), device=device)
            mask_full = torch.zeros((1, full_len), dtype=torch.bool, device=device)
            if cur > 1:
                x_full[0, : cur - 1] = x_gen[: cur - 1]
                mask_full[0, : cur - 1] = True
            x_full[0, cur - 1 : full_len] = x_s[0]
            mask_full[0, cur - 1 : full_len] = mask_local

            s_level = torch.full((1,), args.levels, device=device, dtype=torch.long)
            delta_hat = interp_model(x_full, s_level, mask_full, cond_chunk)
            x_hat = x_full + delta_hat
            if args.clamp_policy == "all_anchors":
                clamp_mask = mask_full
            elif args.clamp_policy == "endpoints":
                clamp_mask = torch.zeros_like(mask_full)
                clamp_mask[:, cur - 1] = True
                clamp_mask[:, full_len - 1] = True
            else:
                clamp_mask = None
            if clamp_mask is not None:
                x_hat = apply_clamp(x_hat, x_full, clamp_mask, args.clamp_dims)

            x_gen[cur : end + 1, :2] = x_hat[0, cur : end + 1, :2]
            if data_dim > 2 and args.recompute_vel:
                x_gen[cur : end + 1, 2:] = x_hat[0, cur : end + 1, 2:]
            cur = end + 1
            if args.save_chunk_frames and frames_dir is not None and frame_idx % max(1, args.frames_stride) == 0:
                traj = x_gen.clone()
                if cur < args.T:
                    traj[cur:, :] = float("nan")
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
                if dataset_name in {"d4rl", "d4rl_prepared"} and pos_low is not None and pos_scale is not None:
                    traj_world = _unnormalize_pos(traj, pos_low.to(device), pos_scale.to(device), flip_y)
                    bounds = (
                        (float(pos_low[0].item()), float((pos_low[0] + pos_scale[0]).item())),
                        (float(pos_low[1].item()), float((pos_low[1] + pos_scale[1]).item())),
                    )
                    if mj_walls:
                        plot_maze2d_geom_walls(
                            mj_walls,
                            [traj_world.detach().cpu().numpy()],
                            [f"chunk {frame_idx}"],
                            out_path=frame_path,
                            bounds=bounds,
                        )
                    elif maze_map is not None and maze_scale is not None:
                        plot_maze2d_trajectories(
                            maze_map,
                            maze_scale,
                            [traj_world.detach().cpu().numpy()],
                            [f"chunk {frame_idx}"],
                            out_path=frame_path,
                            bounds=bounds,
                        )
                    else:
                        occ = cond["occ"][0, 0].detach().cpu().numpy()
                        plot_trajectories(occ, [traj[:, :2].detach().cpu().numpy()], [f"chunk {frame_idx}"], out_path=frame_path)
                else:
                    occ = cond["occ"][0, 0].detach().cpu().numpy()
                    plot_trajectories(occ, [traj[:, :2].detach().cpu().numpy()], [f"chunk {frame_idx}"], out_path=frame_path)
                frame_idx += 1

        if data_dim > 2 and args.recompute_vel:
            pos = x_gen[:, :2]
            v = torch.zeros_like(pos)
            dt = 1.0 / float(args.T)
            v[:-1] = (pos[1:] - pos[:-1]) / dt
            v[-1] = 0.0
            x_gen = torch.cat([pos, v], dim=-1)

        out_path = os.path.join(args.out_dir, f"sample_{i:04d}.png")
        if dataset_name in {"d4rl", "d4rl_prepared"} and pos_low is not None and pos_scale is not None:
            pred_world = _unnormalize_pos(x_gen, pos_low.to(device), pos_scale.to(device), flip_y)
            bounds = (
                (float(pos_low[0].item()), float((pos_low[0] + pos_scale[0]).item())),
                (float(pos_low[1].item()), float((pos_low[1] + pos_scale[1]).item())),
            )
            if mj_walls:
                plot_maze2d_geom_walls(
                    mj_walls,
                    [pred_world.detach().cpu().numpy()],
                    ["pred"],
                    out_path=out_path,
                    bounds=bounds,
                )
            elif maze_map is not None and maze_scale is not None:
                plot_maze2d_trajectories(
                    maze_map,
                    maze_scale,
                    [pred_world.detach().cpu().numpy()],
                    ["pred"],
                    out_path=out_path,
                    bounds=bounds,
                )
            else:
                occ = cond["occ"][0, 0].detach().cpu().numpy()
                pred = x_gen.detach().cpu().numpy()
                plot_trajectories(occ, [pred[:, :2]], ["pred"], out_path=out_path)
        else:
            occ = cond["occ"][0, 0].detach().cpu().numpy()
            pred = x_gen.detach().cpu().numpy()
            plot_trajectories(occ, [pred[:, :2]], ["pred"], out_path=out_path)
        if args.save_chunk_frames and frames_dir is not None:
            _export_video(frames_dir, args.export_video, args.video_fps)


if __name__ == "__main__":
    main()
