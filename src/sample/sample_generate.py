import argparse
import os
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import interpolate_from_indices, sample_fixed_k_indices_batch
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.eval.visualize import plot_maze2d_geom_walls, plot_maze2d_trajectories, plot_trajectories
from src.models.denoiser_interp_levels import InterpLevelDenoiser
from src.models.denoiser_keypoints import KeypointDenoiser
from src.utils.checkpoint import load_checkpoint
from src.utils.clamp import apply_clamp
from src.utils.device import get_device


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
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", type=str, default="particle", choices=["particle", "synthetic", "d4rl"])
    p.add_argument("--env_id", type=str, default="maze2d-medium-v1")
    p.add_argument("--d4rl_flip_y", type=int, default=1)
    p.add_argument("--override_meta", type=int, default=0)
    p.add_argument("--clamp_policy", type=str, default="endpoints", choices=["none", "endpoints", "all_anchors"])
    p.add_argument("--clamp_dims", type=str, default="pos", choices=["pos", "all"])
    p.add_argument("--save_diffusion_frames", type=int, default=0)
    p.add_argument("--frames_stride", type=int, default=1)
    p.add_argument("--frames_include_stage2", type=int, default=1)
    p.add_argument("--export_video", type=str, default="none", choices=["none", "mp4", "gif"])
    p.add_argument("--video_fps", type=int, default=8)
    return p


def get_cond_from_sample(sample: dict) -> dict:
    return sample["cond"]


def _unnormalize_pos(x: torch.Tensor, pos_low: torch.Tensor, pos_scale: torch.Tensor, flip_y: bool) -> torch.Tensor:
    pos = x[..., :2].clone()
    if flip_y:
        pos[..., 1] = 1.0 - pos[..., 1]
    return pos * pos_scale[:2] + pos_low[:2]


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

    # Load keypoint meta early to configure model dims before instantiation.
    meta = {}
    if os.path.exists(args.ckpt_keypoints):
        try:
            payload_meta = torch.load(args.ckpt_keypoints, map_location="cpu")
            if isinstance(payload_meta, dict):
                meta = payload_meta.get("meta", {}) or {}
        except Exception:
            meta = {}
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

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2

    kp_model = KeypointDenoiser(data_dim=data_dim, use_sdf=bool(args.use_sdf)).to(device)
    interp_model = InterpLevelDenoiser(
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        max_levels=args.levels,
    ).to(device)

    if args.use_ema:
        from src.utils.ema import EMA

        ema_kp = EMA(kp_model.parameters())
        ema_interp = EMA(interp_model.parameters())
    else:
        ema_kp = None
        ema_interp = None

    _, payload_kp = load_checkpoint(args.ckpt_keypoints, kp_model, optimizer=None, ema=ema_kp, map_location=device, return_payload=True)
    load_checkpoint(args.ckpt_interp, interp_model, optimizer=None, ema=ema_interp, map_location=device)
    if ema_kp is not None:
        ema_kp.copy_to(kp_model.parameters())
    if ema_interp is not None:
        ema_interp.copy_to(interp_model.parameters())
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
        if maze_scale is None and maze_map is not None and pos_scale is not None:
            maze_arr = np.array(maze_map)
            if maze_arr.ndim == 2:
                h, w = maze_arr.shape
                if w > 0 and h > 0:
                    maze_scale = float(min(pos_scale[0].item() / w, pos_scale[1].item() / h))

    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    idx_global = 0

    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True):
            cond = {k: v.to(device) for k, v in get_cond_from_sample(batch).items()}
            B = cond["start_goal"].shape[0]
            idx, masks = sample_fixed_k_indices_batch(
                B, args.T, args.K_min, generator=gen, device=device, ensure_endpoints=True
            )
            known_mask, known_values = _build_known_mask_values(idx, cond, data_dim, args.T)

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
                z_hat = _sample_keypoints_ddim(kp_model, schedule, idx, known_mask, known_values, cond, args.ddim_steps, args.T)
                z_steps = None

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
                out_path = os.path.join(args.out_dir, f"sample_{idx_global:04d}.png")
                if dataset_name == "d4rl" and pos_low is not None and pos_scale is not None:
                    pred_world = _unnormalize_pos(x_hat[b], pos_low.to(device), pos_scale.to(device), flip_y)
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
                        occ = cond["occ"][b, 0].detach().cpu().numpy()
                        pred = x_hat[b].detach().cpu().numpy()
                        plot_trajectories(occ, [pred[:, :2]], ["pred"], out_path=out_path)
                else:
                    occ = cond["occ"][b, 0].detach().cpu().numpy()
                    pred = x_hat[b].detach().cpu().numpy()
                    plot_trajectories(occ, [pred[:, :2]], ["pred"], out_path=out_path)
                idx_global += 1
                if args.save_diffusion_frames and z_steps is not None:
                    frames_dir = os.path.join(args.out_dir, "diffusion_steps", f"sample_{idx_global - 1:04d}")
                    os.makedirs(frames_dir, exist_ok=True)
                    step_indices = list(range(0, len(z_steps), max(1, args.frames_stride)))
                    for si, step_idx in enumerate(step_indices):
                        z_step = z_steps[step_idx][b : b + 1]
                        x_step = interpolate_from_indices(idx[b : b + 1], z_step, args.T, recompute_velocity=bool(args.recompute_vel))
                        frame_path = os.path.join(frames_dir, f"step_{si:03d}.png")
                        if dataset_name == "d4rl" and pos_low is not None and pos_scale is not None:
                            step_world = _unnormalize_pos(x_step[0], pos_low.to(device), pos_scale.to(device), flip_y)
                            bounds = (
                                (float(pos_low[0].item()), float((pos_low[0] + pos_scale[0]).item())),
                                (float(pos_low[1].item()), float((pos_low[1] + pos_scale[1]).item())),
                            )
                            if mj_walls:
                                plot_maze2d_geom_walls(
                                    mj_walls,
                                    [step_world.detach().cpu().numpy()],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    bounds=bounds,
                                )
                            elif maze_map is not None and maze_scale is not None:
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    [step_world.detach().cpu().numpy()],
                                    [f"step {step_idx}"],
                                    out_path=frame_path,
                                    bounds=bounds,
                                )
                            else:
                                occ = cond["occ"][b, 0].detach().cpu().numpy()
                                step_np = x_step[0].detach().cpu().numpy()
                                plot_trajectories(occ, [step_np[:, :2]], [f"step {step_idx}"], out_path=frame_path)
                        else:
                            occ = cond["occ"][b, 0].detach().cpu().numpy()
                            step_np = x_step[0].detach().cpu().numpy()
                            plot_trajectories(occ, [step_np[:, :2]], [f"step {step_idx}"], out_path=frame_path)
                    if args.frames_include_stage2:
                        final_path = os.path.join(frames_dir, "stage2.png")
                        if dataset_name == "d4rl" and pos_low is not None and pos_scale is not None:
                            if mj_walls:
                                plot_maze2d_geom_walls(
                                    mj_walls,
                                    [pred_world.detach().cpu().numpy()],
                                    ["stage2"],
                                    out_path=final_path,
                                    bounds=bounds,
                                )
                            elif maze_map is not None and maze_scale is not None:
                                plot_maze2d_trajectories(
                                    maze_map,
                                    maze_scale,
                                    [pred_world.detach().cpu().numpy()],
                                    ["stage2"],
                                    out_path=final_path,
                                    bounds=bounds,
                                )
                            else:
                                plot_trajectories(occ, [pred[:, :2]], ["stage2"], out_path=final_path)
                        else:
                            plot_trajectories(occ, [pred[:, :2]], ["stage2"], out_path=final_path)
                    _export_video(frames_dir, args.export_video, args.video_fps)


if __name__ == "__main__":
    main()
