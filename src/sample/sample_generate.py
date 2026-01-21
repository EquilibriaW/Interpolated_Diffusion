import argparse
import csv
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import interpolate_from_indices, sample_fixed_k_indices_batch
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
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
    p.add_argument("--dataset", type=str, default="d4rl", choices=["particle", "synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--d4rl_flip_y", type=int, default=1)
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


def _unnormalize_pos(x: torch.Tensor, pos_low: torch.Tensor, pos_scale: torch.Tensor, flip_y: bool) -> torch.Tensor:
    pos = x[..., :2].clone()
    if flip_y:
        pos[..., 1] = 1.0 - pos[..., 1]
    return pos * pos_scale[:2] + pos_low[:2]


def _bounds_from_walls(walls: list) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = np.concatenate([np.asarray(w) for w in walls], axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return (float(xmin), float(xmax)), (float(ymin), float(ymax))


def _unnormalize_pos_with_bounds(
    x: torch.Tensor,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    flip_y: bool,
) -> torch.Tensor:
    pos = x[..., :2].clone()
    if flip_y:
        pos[..., 1] = 1.0 - pos[..., 1]
    (xmin, xmax), (ymin, ymax) = bounds
    scale = torch.tensor([xmax - xmin, ymax - ymin], device=pos.device, dtype=pos.dtype)
    low = torch.tensor([xmin, ymin], device=pos.device, dtype=pos.dtype)
    return pos * scale + low


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
    if args.dataset != "d4rl":
        raise ValueError("Particle/synthetic datasets are disabled; use --dataset d4rl with Maze2D envs.")

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
        if meta.get("dataset") is not None:
            if args.dataset != meta.get("dataset"):
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

    kp_model = KeypointDenoiser(data_dim=data_dim, use_sdf=bool(args.use_sdf)).to(device)
    interp_model = InterpLevelDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        max_levels=args.levels,
    ).to(device)

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
    if dataset_name == "d4rl":
        dataset = D4RLMazeDataset(
            env_id=args.env_id,
            num_samples=args.n_samples,
            T=args.T,
            with_velocity=bool(args.with_velocity),
            use_sdf=bool(args.use_sdf),
            seed=1234,
            flip_y=bool(args.d4rl_flip_y),
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
    walls_bounds = None
    pos_low = None
    pos_scale = None
    flip_y = False
    if dataset_name == "d4rl":
        maze_map = getattr(dataset, "maze_map", None)
        maze_scale = getattr(dataset, "maze_size_scaling", None)
        mj_walls = getattr(dataset, "mj_walls", None)
        pos_low = getattr(dataset, "pos_low", None)
        pos_scale = getattr(dataset, "pos_scale", None)
        flip_y = bool(getattr(dataset, "flip_y", False))
        if mj_walls:
            walls_bounds = _bounds_from_walls(mj_walls)
        if maze_scale is None and maze_map is not None and pos_scale is not None:
            maze_arr = np.array(maze_map)
            if maze_arr.ndim == 2:
                h, w = maze_arr.shape
                if w > 0 and h > 0:
                    maze_scale = float(min(pos_scale[0].item() / w, pos_scale[1].item() / h))

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    idx_global = 0

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
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
            ]
        )
        with torch.no_grad():
            for batch in tqdm(loader, dynamic_ncols=True):
                cond = {k: v.to(device) for k, v in get_cond_from_sample(batch).items()}
                x0_gt = batch.get("x", None)
                if x0_gt is not None:
                    x0_gt = x0_gt.to(device)
                B = cond["start_goal"].shape[0]
                idx, masks = sample_fixed_k_indices_batch(
                    B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True
                )
                if args.oracle_keypoints:
                    if x0_gt is None:
                        raise ValueError("oracle_keypoints requires batch['x'] to be present.")
                    idx_exp = idx.unsqueeze(-1).expand(-1, -1, x0_gt.shape[-1])
                    z_hat = x0_gt.gather(1, idx_exp)
                    z_steps = None
                else:
                    known_mask, known_values = _build_known_mask_values(idx, cond, data_dim, args.T)
                    if args.logit_space:
                        known_values = logit_pos(known_values, eps=args.logit_eps)
                    if args.save_diffusion_frames:
                        z_hat, z_steps = _sample_keypoints_ddim(
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
                        z_hat = _sample_keypoints_ddim(
                            kp_model, schedule, idx, known_mask, known_values, cond, args.ddim_steps, args.T
                        )
                        z_steps = None
                    if args.logit_space:
                        z_hat = sigmoid_pos(z_hat)
                        if z_steps is not None:
                            z_steps = [sigmoid_pos(z_step) for z_step in z_steps]

                x_s = interpolate_from_indices(idx, z_hat, args.T, recompute_velocity=bool(args.recompute_vel))

                s_level = torch.full((B,), args.levels, device=device, dtype=torch.long)
                delta_hat = interp_model(x_s, s_level, masks, cond)
                x_hat = x_s + delta_hat
                if args.clamp_policy == "all_anchors":
                    clamp_mask = masks
                elif args.clamp_policy == "endpoints":
                    clamp_mask = torch.zeros_like(masks)
                    clamp_mask[:, 0] = True
                    clamp_mask[:, -1] = True
                else:
                    clamp_mask = None
                if clamp_mask is not None:
                    x_hat = apply_clamp(x_hat, x_s, clamp_mask, args.clamp_dims)

                for b in range(B):
                    occ_t = cond["occ"][b, 0]
                    goal_t = cond["start_goal"][b, 2:]
                    interp_t = x_s[b]
                    refined_t = x_hat[b]
                    writer.writerow(
                        [
                            idx_global,
                            collision_rate(occ_t, interp_t),
                            collision_rate(occ_t, refined_t),
                            goal_distance(goal_t, interp_t),
                            goal_distance(goal_t, refined_t),
                            success(goal_t, interp_t, occ_t.shape[-2], occ_t.shape[-1]),
                            success(goal_t, refined_t, occ_t.shape[-2], occ_t.shape[-1]),
                        ]
                    )

                    if args.save_debug:
                        out_path = os.path.join(args.out_dir, f"sample_{idx_global:04d}.png")
                        interp_np = interp_t.detach().cpu().numpy()
                        refined_np = refined_t.detach().cpu().numpy()
                        trajs = [interp_np[:, :2], refined_np[:, :2]]
                        labels = ["interp", "refined"]
                        if args.plot_gt and x0_gt is not None:
                            gt_np = x0_gt[b].detach().cpu().numpy()
                            trajs.append(gt_np[:, :2])
                            labels.append("gt")
                        if dataset_name == "d4rl":
                            if mj_walls and walls_bounds is not None:
                                interp_world = _unnormalize_pos_with_bounds(interp_t, walls_bounds, flip_y)
                                refined_world = _unnormalize_pos_with_bounds(refined_t, walls_bounds, flip_y)
                                world_trajs = [interp_world.detach().cpu().numpy(), refined_world.detach().cpu().numpy()]
                                world_labels = ["interp", "refined"]
                                if args.plot_gt and x0_gt is not None:
                                    gt_world = _unnormalize_pos_with_bounds(x0_gt[b], walls_bounds, flip_y)
                                    world_trajs.append(gt_world.detach().cpu().numpy())
                                    world_labels.append("gt")
                                plot_maze2d_geom_walls(
                                    mj_walls, world_trajs, world_labels, out_path=out_path, bounds=walls_bounds
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                plot_trajectories(occ, trajs, labels, out_path=out_path)
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
                            if dataset_name == "d4rl" and mj_walls and walls_bounds is not None:
                                step_world = _unnormalize_pos_with_bounds(x_step[0], walls_bounds, flip_y)
                                plot_maze2d_geom_walls(
                                    mj_walls,
                                    [step_world.detach().cpu().numpy()],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    bounds=walls_bounds,
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                plot_trajectories(occ, [step_np[:, :2]], [f"step {step_idx}"], out_path=frame_path)
                        if args.frames_include_stage2:
                            final_path = os.path.join(frames_dir, "stage2.png")
                            if dataset_name == "d4rl" and mj_walls and walls_bounds is not None:
                                refined_world = _unnormalize_pos_with_bounds(refined_t, walls_bounds, flip_y)
                                plot_maze2d_geom_walls(
                                    mj_walls,
                                    [refined_world.detach().cpu().numpy()],
                                    ["stage2"],
                                    out_path=final_path,
                                    bounds=walls_bounds,
                                )
                            else:
                                occ = occ_t.detach().cpu().numpy()
                                refined_np = refined_t.detach().cpu().numpy()
                                plot_trajectories(occ, [refined_np[:, :2]], ["stage2"], out_path=final_path)
                        _export_video(frames_dir, args.export_video, args.video_fps)
                    idx_global += 1


if __name__ == "__main__":
    main()
