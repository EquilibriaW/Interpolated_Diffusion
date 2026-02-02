import argparse
import sys
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import sample_fixed_k_indices_batch, sample_fixed_k_indices_uniform_batch
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset, PreparedTrajectoryDataset
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_keypoints import KeypointDenoiser
from src.models.segment_cost import SegmentCostPredictor
from src.selection.epiplexity_dp import build_segment_features, build_segment_precompute
from src.utils.device import get_device
from src.utils.normalize import logit_pos, sigmoid_pos


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/keypoints")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--N_train", type=int, default=None)
    p.add_argument("--schedule", type=str, default=None, choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ddim_schedule", type=str, default="quadratic", choices=["linear", "quadratic", "sqrt"])
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--logit_space", type=int, default=1)
    p.add_argument("--logit_eps", type=float, default=1e-5)
    p.add_argument("--use_ema", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--override_meta", type=int, default=0)
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
    p.add_argument("--kp_index_mode", type=str, default="uniform", choices=["random", "uniform", "uniform_jitter"])
    p.add_argument("--kp_jitter", type=float, default=0.0)
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


def _sample_ddim(
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

    meta = {}
    if os.path.exists(args.ckpt):
        try:
            payload_meta = torch.load(args.ckpt, map_location="cpu")
            if isinstance(payload_meta, dict):
                meta = payload_meta.get("meta", {}) or {}
        except Exception:
            meta = {}
    def _normalize_dataset_name(name):
        return "d4rl" if name == "d4rl_prepared" else name

    if meta.get("stage") == "keypoints" and not args.override_meta:
        args.T = meta.get("T", args.T)
        args.K = meta.get("K", args.K)
        args.N_train = meta.get("N_train", args.N_train)
        args.schedule = meta.get("schedule", args.schedule)
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

    device = get_device(args.device)
    data_dim = 4 if args.with_velocity else 2

    dphi_model = None
    seg_feat = None
    seg_id = None
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
        precomp = build_segment_precompute(args.T, 1, device)
        seg_feat = build_segment_features(args.T, precomp.seg_i, precomp.seg_j).to(device)
        seg_id = precomp.seg_id
    elif bool(args.use_kp_feat) and int(args.kp_feat_dim) > 3:
        raise ValueError("kp_feat_dim>3 requires --dphi_ckpt for predicted difficulties")

    model = KeypointDenoiser(
        d_model=int(args.kp_d_model) if args.kp_d_model is not None else 256,
        n_layers=int(args.kp_n_layers) if args.kp_n_layers is not None else 8,
        n_heads=int(args.kp_n_heads) if args.kp_n_heads is not None else 8,
        d_ff=int(args.kp_d_ff) if args.kp_d_ff is not None else 1024,
        d_cond=int(args.kp_d_cond) if args.kp_d_cond is not None else 128,
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        use_start_goal=bool(args.cond_start_goal),
        kp_feat_dim=int(args.kp_feat_dim) if bool(args.use_kp_feat) else 0,
        maze_channels=_parse_int_list(args.kp_maze_channels, "32,64"),
    ).to(device)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    payload = torch.load(args.ckpt, map_location="cpu")
    if "model" not in payload:
        raise KeyError(f"Checkpoint missing model weights: {args.ckpt}")
    model.load_state_dict(payload["model"])
    if args.use_ema:
        from src.utils.ema import EMA

        if "ema" in payload:
            ema = EMA(model.parameters())
            ema.load_state_dict(payload["ema"])
            ema.copy_to(model.parameters())
        else:
            print("Checkpoint has no EMA; using raw model weights.")
    model.eval()

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if meta.get("stage") == "keypoints" and not args.override_meta:
        args.T = meta.get("T", args.T)
        args.K = meta.get("K", args.K)
        args.N_train = meta.get("N_train", args.N_train)
        args.schedule = meta.get("schedule", args.schedule)
    if args.T is None or args.K is None or args.N_train is None or args.schedule is None:
        raise ValueError("Missing T/K/N_train/schedule. Provide args or use a checkpoint with meta.")

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
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    all_samples = []
    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True):
            cond = {k: v.to(device) for k, v in batch["cond"].items()}
            B = cond["start_goal"].shape[0]
            if args.kp_index_mode == "random":
                idx, _ = sample_fixed_k_indices_batch(B, args.T, args.K, device=device, ensure_endpoints=True)
            else:
                jitter = args.kp_jitter if args.kp_index_mode == "uniform_jitter" else 0.0
                idx, _ = sample_fixed_k_indices_uniform_batch(
                    B, args.T, args.K, device=device, ensure_endpoints=True, jitter=jitter
                )
            if bool(args.use_kp_feat) and int(args.kp_feat_dim) > 0:
                left_diff = None
                right_diff = None
                if dphi_model is not None:
                    seg_pred = dphi_model(cond, seg_feat)
                    seg_ids = seg_id[idx[:, :-1], idx[:, 1:]]
                    if torch.any(seg_ids < 0):
                        raise ValueError("Invalid segment id lookup for kp_feat")
                    seg_cost = seg_pred.gather(1, seg_ids)
                    K = idx.shape[1]
                    left_diff = torch.zeros((B, K), device=device, dtype=torch.float32)
                    right_diff = torch.zeros((B, K), device=device, dtype=torch.float32)
                    left_diff[:, 1:] = seg_cost
                    right_diff[:, :-1] = seg_cost
                cond["kp_feat"] = _kp_feat_from_idx(
                    idx, args.T, int(args.kp_feat_dim), left_diff, right_diff
                )
            known_mask, known_values = _build_known_mask_values(
                idx, cond, data_dim, args.T, bool(args.clamp_endpoints)
            )
            if args.logit_space:
                known_values = logit_pos(known_values, eps=args.logit_eps)
            z_hat = _sample_ddim(
                model,
                schedule,
                idx,
                known_mask,
                known_values,
                cond,
                args.ddim_steps,
                args.T,
                schedule_name=args.ddim_schedule,
            )
            if args.logit_space:
                z_hat = sigmoid_pos(z_hat)
            all_samples.append({"idx": idx.detach().cpu(), "z": z_hat.detach().cpu()})

    torch.save(all_samples, os.path.join(args.out_dir, "keypoints.pt"))
    print(f"saved {len(all_samples)} batches to {os.path.join(args.out_dir, 'keypoints.pt')}")


if __name__ == "__main__":
    main()
