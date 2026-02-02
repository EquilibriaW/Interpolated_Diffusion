import argparse
import csv
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import build_nested_masks_from_base, sample_fixed_k_indices_uniform_batch
from src.corruptions.video_keyframes import interpolate_video_from_indices
from src.data.toy_video import MovingShapesVideoDataset, decode_latents
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_interp_levels import InterpLevelDenoiser
from src.models.denoiser_keypoints import KeypointDenoiser
from src.models.video_interpolator import TinyTemporalInterpolator
from src.utils.clamp import apply_clamp, apply_soft_clamp
from src.utils.device import get_device
from src.utils.normalize import logit_pos, sigmoid_pos


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_keypoints", type=str, default="")
    p.add_argument("--ckpt_interp", type=str, default="")
    p.add_argument("--out_dir", type=str, default="runs/gen_toy_video")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--stage2_mode", type=str, default="adj", choices=["x0", "adj"])
    p.add_argument("--N_train", type=int, default=None)
    p.add_argument("--schedule", type=str, default=None, choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ddim_schedule", type=str, default="quadratic", choices=["linear", "quadratic", "sqrt"])
    p.add_argument("--k_schedule", type=str, default="geom", choices=["doubling", "linear", "geom"])
    p.add_argument("--k_geom_gamma", type=float, default=None)
    p.add_argument("--anchor_conf", type=int, default=1)
    p.add_argument("--anchor_conf_teacher", type=float, default=0.95)
    p.add_argument("--anchor_conf_student", type=float, default=0.5)
    p.add_argument("--anchor_conf_missing", type=float, default=0.0)
    p.add_argument("--anchor_conf_anneal", type=int, default=1)
    p.add_argument("--anchor_conf_anneal_mode", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--soft_anchor_clamp", type=int, default=1)
    p.add_argument("--soft_clamp_schedule", type=str, default="linear", choices=["linear", "cosine", "none"])
    p.add_argument("--soft_clamp_max", type=float, default=0.3)
    p.add_argument("--clamp_policy", type=str, default="none", choices=["none", "all_anchors"])
    p.add_argument("--clamp_dims", type=str, default="all", choices=["all"])
    p.add_argument("--logit_space", type=int, default=0)
    p.add_argument("--logit_eps", type=float, default=1e-5)
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--override_meta", type=int, default=0)
    p.add_argument("--compare_oracle", type=int, default=1)
    p.add_argument("--skip_stage2", type=int, default=0)
    p.add_argument("--interp_mode", type=str, default="linear", choices=["linear", "smooth", "learned"])
    p.add_argument("--interp_ckpt", type=str, default="")
    p.add_argument("--interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
    p.add_argument("--latent_size", type=int, default=16)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clamp_endpoints", type=int, default=1)
    p.add_argument("--save_frames", type=int, default=1)
    p.add_argument("--frames_stride", type=int, default=1)
    p.add_argument("--export_video", type=str, default="mp4")
    p.add_argument("--video_fps", type=int, default=8)
    p.add_argument("--label_frames", type=int, default=1)
    p.add_argument("--save_diffusion_frames", type=int, default=0)
    p.add_argument("--diffusion_frame_t", type=int, default=-1)
    p.add_argument("--plot_gt", type=int, default=1)
    return p


def _soft_clamp_lambda(level: int, levels: int, schedule: str, lam_max: float) -> float:
    if schedule == "none" or levels <= 0:
        return 0.0
    frac = float(level) / float(levels)
    if schedule == "linear":
        lam = 1.0 - frac
    elif schedule == "cosine":
        lam = 0.5 * (1.0 + torch.cos(torch.tensor(frac * torch.pi)).item())
    else:
        lam = 0.0
    return float(lam) * float(lam_max)


def _anneal_conf(conf: torch.Tensor, level: int, levels: int, mode: str) -> torch.Tensor:
    if conf is None or mode == "none" or levels <= 0:
        return conf
    frac = float(level) / float(levels)
    if mode == "linear":
        lam = 1.0 - frac
    elif mode == "cosine":
        lam = 0.5 * (1.0 + torch.cos(torch.tensor(frac * torch.pi)).item())
    else:
        lam = 0.0
    return conf + (1.0 - conf) * float(lam)


def _build_anchor_conf(mask: torch.Tensor, conf_anchor: float, conf_missing: float) -> torch.Tensor:
    conf = torch.full_like(mask.float(), float(conf_missing))
    return conf + (mask.float() * (float(conf_anchor) - float(conf_missing)))


def _sample_keypoints_ddim(
    model,
    schedule,
    idx: torch.Tensor,
    known_mask: torch.Tensor,
    known_values: torch.Tensor,
    steps: int,
    T: int,
    schedule_name: str = "linear",
    return_intermediates: bool = False,
):
    device = idx.device
    B, K = idx.shape
    D = known_values.shape[-1]
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps, schedule=schedule_name)
    z = torch.randn((B, K, D), device=device)
    z = torch.where(known_mask, known_values, z)
    intermediates = []
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(z, t, idx, known_mask, {}, T)
        z = ddim_step(z, eps, t, t_prev, schedule, eta=0.0)
        z = torch.where(known_mask, known_values, z)
        if return_intermediates:
            intermediates.append(z.detach().clone())
    return (z, intermediates) if return_intermediates else z


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_video(frames: torch.Tensor, path: str, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    if imageio is None:
        return
    frames = frames.detach().cpu().clamp(0, 1)
    frames = (frames * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _stack_panels(frames_list: list[torch.Tensor]) -> torch.Tensor:
    frames = [f.detach().cpu() for f in frames_list]
    return torch.cat(frames, dim=-1)


def _label_panel_frame(frame: torch.Tensor, labels: list[str], t_idx: int) -> torch.Tensor:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    frame = frame.detach().cpu().clamp(0, 1)
    img = (frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[1], frame.shape[2]
    panel_w = w // len(labels)
    for i, label in enumerate(labels):
        text = f"{label} | t={t_idx:02d}"
        x = i * panel_w + 4
        y = 4
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(None)
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for toy video sampling.")
    _ensure_dir(args.out_dir)
    torch.manual_seed(args.seed)

    # Interpolator for visualization/corruption (optional learned).
    interp_model = None
    smooth_kernel = None
    if args.interp_mode == "smooth":
        smooth_kernel = torch.tensor([float(x) for x in args.interp_smooth_kernel.split(",")], dtype=torch.float32)
    elif args.interp_mode == "learned":
        if not args.interp_ckpt:
            raise ValueError("--interp_ckpt is required for interp_mode=learned")
        interp_model = TinyTemporalInterpolator(data_dim=3 * args.latent_size * args.latent_size)
        payload = torch.load(args.interp_ckpt, map_location="cpu")
        state = payload.get("model", payload)
        interp_model.load_state_dict(state)
        interp_model.to(device)
        interp_model.eval()

    # Stage 1 (keypoints) model (read meta before dataset).
    kp_model = None
    schedule = None
    if args.ckpt_keypoints:
        meta = {}
        payload = torch.load(args.ckpt_keypoints, map_location="cpu")
        if isinstance(payload, dict):
            meta = payload.get("meta", {}) or {}
        if meta.get("stage") == "keypoints" and not args.override_meta:
            if meta.get("T") is not None:
                args.T = int(meta.get("T"))
            if meta.get("K") is not None:
                args.K_min = int(meta.get("K"))
            if meta.get("N_train") is not None:
                args.N_train = int(meta.get("N_train"))
            if meta.get("schedule") is not None:
                args.schedule = str(meta.get("schedule"))
            if meta.get("logit_space") is not None:
                args.logit_space = int(bool(meta.get("logit_space")))
            if meta.get("logit_eps") is not None:
                args.logit_eps = float(meta.get("logit_eps"))
            if meta.get("clamp_endpoints") is not None:
                args.clamp_endpoints = int(bool(meta.get("clamp_endpoints")))
        data_dim = int(meta.get("data_dim", 3 * args.latent_size * args.latent_size))
        if args.N_train is None or args.schedule is None:
            raise ValueError("Keypoint meta missing N_train/schedule; pass --N_train/--schedule.")
        kp_model = KeypointDenoiser(data_dim=data_dim, use_sdf=False, use_start_goal=False).to(device)
        state = payload.get("model", payload)
        kp_model.load_state_dict(state)
        if args.use_ema and isinstance(payload, dict) and "ema" in payload:
            from src.utils.ema import EMA

            ema = EMA(kp_model.parameters())
            ema.load_state_dict(payload["ema"])
            ema.copy_to(kp_model.parameters())
        elif args.use_ema:
            print("Warning: checkpoint has no EMA; using raw keypoint weights.")
        kp_model.eval()
        betas = make_beta_schedule(args.schedule, args.N_train).to(device)
        schedule = make_alpha_bars(betas)
    else:
        data_dim = 3 * args.latent_size * args.latent_size

    # Stage 2 model (read meta before dataset).
    interp_level_model = None
    if args.ckpt_interp and not args.skip_stage2:
        payload = torch.load(args.ckpt_interp, map_location="cpu")
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta.get("stage") == "interp_levels_toy_video" and not args.override_meta:
            if meta.get("T") is not None:
                args.T = int(meta.get("T"))
            if meta.get("K_min") is not None:
                args.K_min = int(meta.get("K_min"))
            if meta.get("levels") is not None:
                args.levels = int(meta.get("levels"))
            if meta.get("stage2_mode") is not None:
                args.stage2_mode = str(meta.get("stage2_mode"))
        mask_channels = int(meta.get("mask_channels", 2 if args.stage2_mode == "adj" else 1))
        expected_no_conf = 2 if args.stage2_mode == "adj" else 1
        if mask_channels == expected_no_conf:
            args.anchor_conf = 0
        elif mask_channels == expected_no_conf + 1:
            args.anchor_conf = 1
        elif mask_channels != expected_no_conf:
            raise ValueError(f"Unexpected mask_channels={mask_channels} for stage2_mode={args.stage2_mode}")
        interp_level_model = InterpLevelDenoiser(
            data_dim=data_dim,
            use_sdf=False,
            max_levels=args.levels,
            use_start_goal=False,
            mask_channels=mask_channels,
        ).to(device)
        interp_level_model.load_state_dict(payload["model"])
        interp_level_model.eval()
    else:
        if not args.skip_stage2:
            raise ValueError("Stage2 disabled: ckpt_interp is empty or missing.")
        args.skip_stage2 = 1

    dataset = MovingShapesVideoDataset(
        T=args.T or 16,
        H=args.H,
        n_samples=args.num_samples,
        seed=args.seed,
        latent_size=args.latent_size,
    )
    if dataset.data_dim != data_dim:
        raise ValueError(f"Data dim mismatch: dataset {dataset.data_dim} vs model {data_dim}")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, drop_last=False)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 7)

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(
            [
                "sample_id",
                "mse_pred_interp",
                "mse_pred_refined",
                "mse_pred_delta",
                "mse_oracle_interp",
                "mse_oracle_refined",
                "mse_oracle_delta",
            ]
        )
        idx_global = 0
        for batch in tqdm(loader, dynamic_ncols=True):
            if idx_global >= args.n_samples:
                break
            z0 = batch["x"].to(device)
            B = z0.shape[0]
            if idx_global + B > args.n_samples:
                z0 = z0[: args.n_samples - idx_global]
                B = z0.shape[0]
            T = z0.shape[1]

            idx, masks = sample_fixed_k_indices_uniform_batch(
                B, T, args.K_min, generator=gen, device=device, ensure_endpoints=True
            )
            z_oracle = z0.gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], z0.shape[-1]))
            x_oracle = interpolate_video_from_indices(
                idx, z_oracle, T, mode=args.interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model
            )

            z_pred = None
            z_steps = None
            x_pred = x_oracle
            if kp_model is not None:
                known_mask = torch.zeros((B, idx.shape[1], z0.shape[-1]), device=device, dtype=torch.bool)
                known_values = torch.zeros_like(known_mask, dtype=z0.dtype).float()
                if args.clamp_endpoints:
                    start_goal = batch.get("cond", {}).get("start_goal", None)
                    if start_goal is None:
                        raise ValueError("Toy video batch missing cond['start_goal']")
                    start_goal = start_goal.to(device)
                    start = start_goal[:, : z0.shape[-1]]
                    goal = start_goal[:, z0.shape[-1] :]
                    mask_start = (idx == 0).unsqueeze(-1)
                    mask_goal = (idx == (T - 1)).unsqueeze(-1)
                    start_vals = start.unsqueeze(1).expand(-1, idx.shape[1], -1)
                    goal_vals = goal.unsqueeze(1).expand(-1, idx.shape[1], -1)
                    known_mask = mask_start | mask_goal
                    known_values = torch.where(mask_start, start_vals, known_values)
                    known_values = torch.where(mask_goal, goal_vals, known_values)
                if args.logit_space:
                    known_values = logit_pos(known_values, eps=args.logit_eps)
                if args.save_diffusion_frames:
                    z_pred, z_steps = _sample_keypoints_ddim(
                        kp_model,
                        schedule,
                        idx,
                        known_mask,
                        known_values,
                        args.ddim_steps,
                        T,
                        args.ddim_schedule,
                        return_intermediates=True,
                    )
                else:
                    z_pred = _sample_keypoints_ddim(
                        kp_model, schedule, idx, known_mask, known_values, args.ddim_steps, T, args.ddim_schedule
                    )
                if args.logit_space:
                    z_pred = sigmoid_pos(z_pred)
                x_pred = interpolate_video_from_indices(
                    idx, z_pred, T, mode=args.interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model
                )

            x_hat = x_pred
            x_hat_oracle = None
            stage2_steps_pred = None
            stage2_steps_oracle = None
            if not args.skip_stage2 and interp_level_model is not None:
                if args.stage2_mode == "adj":
                    masks_levels, _ = build_nested_masks_from_base(
                        idx,
                        T,
                        args.levels,
                        generator=gen,
                        device=device,
                        k_schedule=args.k_schedule,
                        k_geom_gamma=args.k_geom_gamma,
                    )
                    x_curr = x_pred
                    if args.save_diffusion_frames:
                        stage2_steps_pred = []
                    for s in range(args.levels, 0, -1):
                        mask_s = masks_levels[:, s]
                        mask_prev = masks_levels[:, s - 1]
                        if args.anchor_conf:
                            conf_s = _build_anchor_conf(mask_s, args.anchor_conf_student, args.anchor_conf_missing)
                            conf_s = _anneal_conf(conf_s, s, args.levels, args.anchor_conf_anneal_mode)
                            mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
                        else:
                            conf_s = None
                            mask_in = torch.stack([mask_s, mask_prev], dim=-1)
                        s_level = torch.full((B,), s, device=device, dtype=torch.long)
                        delta_hat = interp_level_model(x_curr, s_level, mask_in, {})
                        x_curr = x_curr + delta_hat
                        if args.soft_anchor_clamp and conf_s is not None:
                            lam = _soft_clamp_lambda(s, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                            x_curr = apply_soft_clamp(x_curr, x_pred, conf_s, lam, args.clamp_dims)
                        if args.clamp_policy == "all_anchors":
                            x_curr = apply_clamp(x_curr, x_pred, masks, args.clamp_dims)
                        if stage2_steps_pred is not None:
                            stage2_steps_pred.append(x_curr.detach().clone())
                    x_hat = x_curr

                    if args.compare_oracle:
                        x_curr = x_oracle
                        if args.save_diffusion_frames:
                            stage2_steps_oracle = []
                        for s in range(args.levels, 0, -1):
                            mask_s = masks_levels[:, s]
                            mask_prev = masks_levels[:, s - 1]
                            if args.anchor_conf:
                                conf_s = _build_anchor_conf(mask_s, args.anchor_conf_teacher, args.anchor_conf_missing)
                                conf_s = _anneal_conf(conf_s, s, args.levels, args.anchor_conf_anneal_mode)
                                mask_in = torch.stack([mask_s.float(), mask_prev.float(), conf_s], dim=-1)
                            else:
                                conf_s = None
                                mask_in = torch.stack([mask_s, mask_prev], dim=-1)
                            s_level = torch.full((B,), s, device=device, dtype=torch.long)
                            delta_hat = interp_level_model(x_curr, s_level, mask_in, {})
                            x_curr = x_curr + delta_hat
                            if args.soft_anchor_clamp and conf_s is not None:
                                lam = _soft_clamp_lambda(s, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                                x_curr = apply_soft_clamp(x_curr, x_oracle, conf_s, lam, args.clamp_dims)
                            if args.clamp_policy == "all_anchors":
                                x_curr = apply_clamp(x_curr, x_oracle, masks, args.clamp_dims)
                            if stage2_steps_oracle is not None:
                                stage2_steps_oracle.append(x_curr.detach().clone())
                        x_hat_oracle = x_curr
                else:
                    s_level = torch.full((B,), args.levels, device=device, dtype=torch.long)
                    if args.anchor_conf:
                        conf_s = _build_anchor_conf(masks, args.anchor_conf_student, args.anchor_conf_missing)
                        conf_s = _anneal_conf(conf_s, args.levels, args.levels, args.anchor_conf_anneal_mode)
                        mask_in = torch.stack([masks.float(), conf_s], dim=-1)
                    else:
                        conf_s = None
                        mask_in = masks
                    delta_hat = interp_level_model(x_pred, s_level, mask_in, {})
                    x_hat = x_pred + delta_hat
                    if args.soft_anchor_clamp and conf_s is not None:
                        lam = _soft_clamp_lambda(args.levels, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                        x_hat = apply_soft_clamp(x_hat, x_pred, conf_s, lam, args.clamp_dims)
                    if args.clamp_policy == "all_anchors":
                        x_hat = apply_clamp(x_hat, x_pred, masks, args.clamp_dims)
                    if args.compare_oracle:
                        if args.anchor_conf:
                            conf_s = _build_anchor_conf(masks, args.anchor_conf_teacher, args.anchor_conf_missing)
                            conf_s = _anneal_conf(conf_s, args.levels, args.levels, args.anchor_conf_anneal_mode)
                            mask_in = torch.stack([masks.float(), conf_s], dim=-1)
                        else:
                            conf_s = None
                            mask_in = masks
                        delta_hat = interp_level_model(x_oracle, s_level, mask_in, {})
                        x_hat_oracle = x_oracle + delta_hat
                        if args.soft_anchor_clamp and conf_s is not None:
                            lam = _soft_clamp_lambda(args.levels, args.levels, args.soft_clamp_schedule, args.soft_clamp_max)
                            x_hat_oracle = apply_soft_clamp(x_hat_oracle, x_oracle, conf_s, lam, args.clamp_dims)
                        if args.clamp_policy == "all_anchors":
                            x_hat_oracle = apply_clamp(x_hat_oracle, x_oracle, masks, args.clamp_dims)

            for b in range(B):
                mse_pred_interp = torch.mean((x_pred[b] - z0[b]) ** 2).item()
                mse_pred_refined = torch.mean((x_hat[b] - z0[b]) ** 2).item()
                mse_pred_delta = torch.mean((x_hat[b] - x_pred[b]) ** 2).item()
                mse_oracle_interp = torch.mean((x_oracle[b] - z0[b]) ** 2).item()
                mse_oracle_refined = float("nan")
                mse_oracle_delta = float("nan")
                if x_hat_oracle is not None:
                    mse_oracle_refined = torch.mean((x_hat_oracle[b] - z0[b]) ** 2).item()
                    mse_oracle_delta = torch.mean((x_hat_oracle[b] - x_oracle[b]) ** 2).item()

                writer.writerow(
                    [
                        idx_global,
                        mse_pred_interp,
                        mse_pred_refined,
                        mse_pred_delta,
                        mse_oracle_interp,
                        mse_oracle_refined,
                        mse_oracle_delta,
                    ]
                )

                if args.save_frames or args.export_video:
                    sample_dir = os.path.join(args.out_dir, f"sample_{idx_global:04d}")
                    _ensure_dir(sample_dir)
                    gt_frames = decode_latents(z0[b])
                    interp_pred_frames = decode_latents(x_pred[b])
                    panels = []
                    labels = []
                    if args.plot_gt:
                        panels.append(gt_frames)
                        labels.append("gt")
                    if args.compare_oracle:
                        panels.append(decode_latents(x_oracle[b]))
                        labels.append("oracle interp")
                    panels.append(interp_pred_frames)
                    labels.append("pred interp")
                    if not args.skip_stage2:
                        panels.append(decode_latents(x_hat[b]))
                        labels.append("stage2 pred")
                        if x_hat_oracle is not None:
                            panels.append(decode_latents(x_hat_oracle[b]))
                            labels.append("stage2 oracle")
                    stacked = _stack_panels(panels)

                    if args.save_frames:
                        stride = max(1, args.frames_stride)
                        for t in range(0, stacked.shape[0], stride):
                            frame = stacked[t]
                            if args.label_frames:
                                frame = _label_panel_frame(frame, labels, t)
                            frame = frame.detach().cpu().clamp(0, 1)
                            frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                            out_path = os.path.join(sample_dir, f"frame_{t:03d}.png")
                            try:
                                import imageio.v2 as imageio

                                imageio.imwrite(out_path, frame)
                            except Exception:
                                pass

                    if args.export_video:
                        video_path = os.path.join(sample_dir, f"video.{args.export_video}")
                        if args.label_frames:
                            labeled = []
                            for t in range(stacked.shape[0]):
                                labeled.append(_label_panel_frame(stacked[t], labels, t))
                            labeled = torch.stack(labeled, dim=0)
                            _save_video(labeled, video_path, args.video_fps)
                        else:
                            _save_video(stacked, video_path, args.video_fps)

                if args.save_diffusion_frames:
                    sample_dir = os.path.join(args.out_dir, f"sample_{idx_global:04d}")
                    _ensure_dir(sample_dir)
                    t_vis = args.diffusion_frame_t
                    if t_vis < 0:
                        t_vis = T // 2
                    # Phase 1 diffusion steps (predicted keypoints)
                    if z_steps is not None:
                        for si, z_step in enumerate(z_steps):
                            x_step = interpolate_video_from_indices(
                                idx[b : b + 1],
                                z_step[b : b + 1],
                                T,
                                mode=args.interp_mode,
                                smooth_kernel=smooth_kernel,
                                interp_model=interp_model,
                            )
                            frame = decode_latents(x_step)[t_vis]
                            frame = _label_panel_frame(frame, [f"phase1 step {si+1}/{len(z_steps)}"], t_vis)
                            frame = frame.clamp(0, 1)
                            frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                            out_path = os.path.join(sample_dir, f"phase1_step_{si:03d}.png")
                            try:
                                import imageio.v2 as imageio

                                imageio.imwrite(out_path, frame)
                            except Exception:
                                pass
                    # Phase 2 diffusion steps (adjacent refinement)
                    if stage2_steps_pred is not None:
                        for si, x_step in enumerate(stage2_steps_pred):
                            frame = decode_latents(x_step[b])[t_vis]
                            frame = _label_panel_frame(frame, [f"phase2 step {si+1}/{len(stage2_steps_pred)}"], t_vis)
                            frame = frame.clamp(0, 1)
                            frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                            out_path = os.path.join(sample_dir, f"phase2_step_{si:03d}.png")
                            try:
                                import imageio.v2 as imageio

                                imageio.imwrite(out_path, frame)
                            except Exception:
                                pass

                idx_global += 1
            if idx_global >= args.n_samples:
                break


if __name__ == "__main__":
    main()
