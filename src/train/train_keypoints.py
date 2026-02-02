import argparse
import sys
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.keyframes import sample_fixed_k_indices_batch, sample_fixed_k_indices_uniform_batch
from src.data.dataset import D4RLMazeDataset, ParticleMazeDataset, PreparedTrajectoryDataset
from src.diffusion.ddpm import q_sample
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.denoiser_keypoints import KeypointDenoiser
from src.models.keypoint_selector import KeypointSelector, select_topk_indices
from src.models.segment_cost import SegmentCostPredictor
from src.selection.epiplexity_dp import build_segment_features_from_idx
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.device import get_autocast_dtype, get_device
from src.utils.ema import EMA
from src.utils.logging import create_writer
from src.utils.normalize import logit_pos
from src.utils.run_config import write_run_config
from src.utils.seed import get_seed_from_env, set_seed


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--kp_d_model", type=int, default=384)
    p.add_argument("--kp_n_layers", type=int, default=12)
    p.add_argument("--kp_n_heads", type=int, default=12)
    p.add_argument("--kp_d_ff", type=int, default=1536)
    p.add_argument("--kp_d_cond", type=int, default=128)
    p.add_argument("--kp_maze_channels", type=str, default="32,64,128,128")
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_sdf", type=int, default=0)
    p.add_argument("--with_velocity", type=int, default=0)
    p.add_argument("--logit_space", type=int, default=1)
    p.add_argument("--logit_eps", type=float, default=1e-5)
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
    p.add_argument("--log_dir", type=str, default="runs/keypoints")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/keypoints")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ema", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_checkpoint", type=int, default=0)
    p.add_argument("--deterministic", type=int, default=1)
    p.add_argument("--allow_tf32", type=int, default=1)
    p.add_argument("--enable_flash_sdp", type=int, default=1)
    p.add_argument("--idx_source", type=str, default="random", choices=["random", "uniform", "dp_precomputed", "selector"])
    p.add_argument("--idx_policy_mix", type=str, default="dp:0.7,uniform:0.2,random:0.1")
    p.add_argument("--use_kp_feat", type=int, default=1)
    p.add_argument("--kp_feat_dim", type=int, default=3)
    p.add_argument("--dphi_ckpt", type=str, default=None)
    p.add_argument("--dphi_use_ema", type=int, default=1)
    p.add_argument("--selector_ckpt", type=str, default=None)
    p.add_argument("--selector_use_ema", type=int, default=1)
    return p


def _gather_keypoints(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    idx_exp = idx.unsqueeze(-1).expand(B, idx.shape[1], D)
    return torch.gather(x, dim=1, index=idx_exp)


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


def _build_keypoint_batch(
    x0: torch.Tensor,
    K: int,
    cond: dict,
    generator: torch.Generator,
    logit_space: bool,
    logit_eps: float,
    clamp_endpoints: bool,
    idx_override: Optional[torch.Tensor] = None,
):
    B, T, D = x0.shape
    if idx_override is None:
        idx, _ = sample_fixed_k_indices_batch(B, T, K, generator=generator, device=x0.device, ensure_endpoints=True)
    else:
        idx = idx_override
    z0 = _gather_keypoints(x0, idx)
    known_mask, known_values = _build_known_mask_values(idx, cond, D, T, clamp_endpoints)
    if logit_space:
        z0 = logit_pos(z0, eps=logit_eps)
        known_values = logit_pos(known_values, eps=logit_eps)
    return z0, idx, known_mask, known_values


def _parse_idx_policy_mix(spec: str):
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    modes = []
    weights = []
    for part in parts:
        if ":" not in part:
            continue
        name, val = part.split(":", 1)
        name = name.strip()
        try:
            weight = float(val.strip())
        except ValueError:
            continue
        if name not in {"dp", "uniform", "random", "selector"}:
            continue
        modes.append(name)
        weights.append(weight)
    if not modes:
        return None
    total = sum(weights)
    if total <= 0:
        return None
    weights = [w / total for w in weights]
    return modes, torch.tensor(weights, dtype=torch.float32)


def _parse_int_list(spec: str) -> tuple[int, ...]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise ValueError("empty int list")
    return tuple(int(p) for p in parts)


def _kp_feat_from_idx(
    idx: torch.Tensor,
    T: int,
    kp_feat_dim: int,
    left_diff: Optional[torch.Tensor] = None,
    right_diff: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, K = idx.shape
    feat = torch.zeros((B, K, kp_feat_dim), device=idx.device, dtype=torch.float32)
    if kp_feat_dim <= 0:
        return feat
    denom = float(max(1, T - 1))
    if kp_feat_dim >= 3:
        feat[:, :, 2] = idx.float() / denom
    if K > 1:
        gaps = (idx[:, 1:] - idx[:, :-1]).float() / denom
        feat[:, 1:, 0] = gaps
        feat[:, :-1, 1] = gaps
    if kp_feat_dim >= 5 and left_diff is not None and right_diff is not None:
        feat[:, :, 3] = left_diff
        feat[:, :, 4] = right_diff
    return feat


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
    model = KeypointDenoiser(
        d_model=int(args.kp_d_model),
        n_layers=int(args.kp_n_layers),
        n_heads=int(args.kp_n_heads),
        d_ff=int(args.kp_d_ff),
        d_cond=int(args.kp_d_cond),
        data_dim=data_dim,
        use_sdf=bool(args.use_sdf),
        use_start_goal=bool(args.cond_start_goal),
        use_checkpoint=bool(args.use_checkpoint),
        kp_feat_dim=int(args.kp_feat_dim) if bool(args.use_kp_feat) else 0,
        maze_channels=_parse_int_list(args.kp_maze_channels),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model.parameters(), decay=args.ema_decay) if args.ema else None

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = create_writer(args.log_dir)
    write_run_config(
        args.log_dir,
        args,
        writer=writer,
        prepared_path=args.prepared_path,
        extra={"stage": "keypoints"},
    )

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, ema, map_location=device)

    dphi_model = None
    seg_feat = None
    seg_id = None
    if args.dphi_ckpt:
        payload = torch.load(args.dphi_ckpt, map_location="cpu")
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta.get("stage") != "segment_cost":
            raise ValueError("dphi_ckpt does not appear to be a segment_cost checkpoint")
        if meta.get("T") is not None and int(meta.get("T")) != int(args.T):
            raise ValueError(f"dphi_ckpt T mismatch: ckpt={meta.get('T')} args={args.T}")
        if meta.get("use_sdf") is not None and bool(meta.get("use_sdf")) != bool(args.use_sdf):
            raise ValueError("dphi_ckpt use_sdf mismatch")
        if meta.get("cond_start_goal") is not None and bool(meta.get("cond_start_goal")) != bool(args.cond_start_goal):
            raise ValueError("dphi_ckpt cond_start_goal mismatch")
        if not bool(args.use_kp_feat) or int(args.kp_feat_dim) < 5:
            raise ValueError("dphi_ckpt requires use_kp_feat=1 and kp_feat_dim>=5")
        maze_channels = meta.get("maze_channels", "32,64")
        dphi_model = SegmentCostPredictor(
            d_cond=int(meta.get("d_cond", 128)),
            seg_feat_dim=int(meta.get("seg_feat_dim", 3)),
            hidden_dim=int(meta.get("hidden_dim", 256)),
            n_layers=int(meta.get("n_layers", 3)),
            dropout=float(meta.get("dropout", 0.0)),
            use_sdf=bool(args.use_sdf),
            use_start_goal=bool(args.cond_start_goal),
            maze_channels=_parse_int_list(maze_channels),
        ).to(device)
        dphi_state = payload.get("model", payload)
        dphi_model.load_state_dict(dphi_state)
        if bool(args.dphi_use_ema) and isinstance(payload, dict) and "ema" in payload:
            ema_dphi = EMA(dphi_model.parameters())
            ema_dphi.load_state_dict(payload["ema"])
            ema_dphi.copy_to(dphi_model.parameters())
        dphi_model.eval()
        dphi_seg_feat_dim = int(dphi_model.seg_feat_dim)

    selector_model = None
    selector_use_level = False
    selector_level_mode = "k_norm"
    needs_selector = False
    idx_policy_mix = _parse_idx_policy_mix(args.idx_policy_mix)
    if idx_policy_mix is not None and "selector" in idx_policy_mix[0]:
        needs_selector = True
    if args.idx_source == "selector":
        needs_selector = True
    if needs_selector:
        if args.selector_ckpt is None:
            raise ValueError("selector indices requested but --selector_ckpt not provided")
        payload = torch.load(args.selector_ckpt, map_location="cpu")
        meta_sel = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta_sel.get("stage") != "selector":
            raise ValueError("selector_ckpt does not appear to be a selector checkpoint")
        if meta_sel.get("T") is not None and int(meta_sel.get("T")) != int(args.T):
            raise ValueError(f"selector_ckpt T mismatch: ckpt={meta_sel.get('T')} args={args.T}")
        if meta_sel.get("K") is not None and int(meta_sel.get("K")) != int(args.K):
            raise ValueError(f"selector_ckpt K mismatch: ckpt={meta_sel.get('K')} args={args.K}")
        if meta_sel.get("use_sdf") is not None and bool(meta_sel.get("use_sdf")) != bool(args.use_sdf):
            raise ValueError("selector_ckpt use_sdf mismatch")
        if meta_sel.get("cond_start_goal") is not None and bool(meta_sel.get("cond_start_goal")) != bool(
            args.cond_start_goal
        ):
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
            maze_channels=_parse_int_list(maze_channels),
        ).to(device)
        selector_state = payload.get("model", payload)
        selector_model.load_state_dict(selector_state)
        if bool(args.selector_use_ema) and isinstance(payload, dict) and "ema" in payload:
            ema_sel = EMA(selector_model.parameters())
            ema_sel.load_state_dict(payload["ema"])
            ema_sel.copy_to(selector_model.parameters())
        selector_model.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 11)

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

        B = x0.shape[0]
        idx_override = None
        if idx_policy_mix is not None:
            modes, probs = idx_policy_mix
            probs = probs.to(device)
            choice = torch.multinomial(probs, B, replacement=True)
            idx_buf = torch.empty((B, args.K), device=device, dtype=torch.long)
            mode_to_idx = {}
            if "random" in modes:
                idx_r, _ = sample_fixed_k_indices_batch(
                    B, args.T, args.K, generator=gen, device=device, ensure_endpoints=True
                )
                mode_to_idx["random"] = idx_r
            if "uniform" in modes:
                idx_u, _ = sample_fixed_k_indices_uniform_batch(
                    B, args.T, args.K, generator=gen, device=device, ensure_endpoints=True
                )
                mode_to_idx["uniform"] = idx_u
            if "dp" in modes:
                if "kp_idx" not in cond:
                    raise ValueError("idx_policy_mix includes dp but batch missing cond['kp_idx']")
                idx_dp = cond["kp_idx"].to(device)
                if idx_dp.shape[1] != args.K:
                    raise ValueError("kp_idx K mismatch with args.K")
                idx_dp = idx_dp.clone()
                idx_dp[:, 0] = 0
                idx_dp[:, -1] = args.T - 1
                idx_dp = torch.sort(idx_dp, dim=1).values
                mode_to_idx["dp"] = idx_dp
            if "selector" in modes:
                if selector_model is None:
                    raise ValueError("idx_policy_mix includes selector but selector model not loaded")
                with torch.no_grad():
                    if selector_use_level:
                        if selector_level_mode == "s_norm":
                            level_val = float(args.K) / float(max(1, args.K))
                        else:
                            level_val = float(args.K) / float(max(1, args.T - 1))
                        cond_sel = dict(cond)
                        cond_sel["level"] = torch.full((B, 1), level_val, device=device)
                        logits = selector_model(cond_sel)
                    else:
                        logits = selector_model(cond)
                mode_to_idx["selector"] = select_topk_indices(logits, args.K)
            for m_i, m in enumerate(modes):
                mask = choice == m_i
                if mask.any():
                    idx_buf[mask] = mode_to_idx[m][mask]
            idx_override = idx_buf
        else:
            if args.idx_source == "dp_precomputed":
                if "kp_idx" not in cond:
                    raise ValueError("idx_source=dp_precomputed but batch missing cond['kp_idx']")
                idx_dp = cond["kp_idx"].to(device)
                if idx_dp.shape[1] != args.K:
                    raise ValueError("kp_idx K mismatch with args.K")
                idx_dp = idx_dp.clone()
                idx_dp[:, 0] = 0
                idx_dp[:, -1] = args.T - 1
                idx_dp = torch.sort(idx_dp, dim=1).values
                idx_override = idx_dp
            elif args.idx_source == "uniform":
                idx_u, _ = sample_fixed_k_indices_uniform_batch(
                    B, args.T, args.K, generator=gen, device=device, ensure_endpoints=True
                )
                idx_override = idx_u
            elif args.idx_source == "selector":
                if selector_model is None:
                    raise ValueError("idx_source=selector but selector model not loaded")
                with torch.no_grad():
                    if selector_use_level:
                        if selector_level_mode == "s_norm":
                            level_val = float(args.K) / float(max(1, args.K))
                        else:
                            level_val = float(args.K) / float(max(1, args.T - 1))
                        cond_sel = dict(cond)
                        cond_sel["level"] = torch.full((B, 1), level_val, device=device)
                        logits = selector_model(cond_sel)
                    else:
                        logits = selector_model(cond)
                idx_override = select_topk_indices(logits, args.K)
            else:
                idx_override = None

        z0, idx, known_mask, known_values = _build_keypoint_batch(
            x0,
            args.K,
            cond,
            gen,
            bool(args.logit_space),
            args.logit_eps,
            bool(args.clamp_endpoints),
            idx_override=idx_override,
        )
        if bool(args.use_kp_feat):
            left_diff = None
            right_diff = None
            if dphi_model is not None:
                seg_feat_sel = build_segment_features_from_idx(idx, args.T, dphi_seg_feat_dim)
                with torch.no_grad():
                    seg_cost = dphi_model(cond, seg_feat_sel)
                left_diff = torch.zeros((B, args.K), device=device, dtype=torch.float32)
                right_diff = torch.zeros((B, args.K), device=device, dtype=torch.float32)
                left_diff[:, 1:] = seg_cost
                right_diff[:, :-1] = seg_cost
            cond["kp_feat"] = _kp_feat_from_idx(idx, args.T, int(args.kp_feat_dim), left_diff, right_diff)

        t = torch.randint(0, args.N_train, (x0.shape[0],), device=device, dtype=torch.long)
        z_t, eps = q_sample(z0, t, schedule)
        z_t = torch.where(known_mask, known_values, z_t)
        eps = eps * (~known_mask)

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            eps_hat = model(z_t, t, idx, known_mask, cond, args.T)
            diff = (eps_hat - eps) ** 2
            valid = (~known_mask).float()
            loss = (diff * valid).sum() / (valid.sum() + 1e-8)
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
                "stage": "keypoints",
                "T": args.T,
                "K": args.K,
                "data_dim": data_dim,
                "N_train": args.N_train,
                "schedule": args.schedule,
                "use_sdf": bool(args.use_sdf),
                "with_velocity": bool(args.with_velocity),
                "dataset": args.dataset,
                "env_id": args.env_id,
                "d4rl_flip_y": bool(args.d4rl_flip_y),
                "logit_space": bool(args.logit_space),
                "logit_eps": float(args.logit_eps),
                "use_start_goal": bool(args.cond_start_goal),
                "clamp_endpoints": bool(args.clamp_endpoints),
                "cond_start_goal": bool(args.cond_start_goal),
                "window_mode": args.window_mode,
                "goal_mode": args.goal_mode,
                "min_tortuosity": args.min_tortuosity,
                "min_turns": args.min_turns,
                "turn_angle_deg": args.turn_angle_deg,
                "use_kp_feat": bool(args.use_kp_feat),
                "kp_feat_dim": int(args.kp_feat_dim) if bool(args.use_kp_feat) else 0,
                "idx_source": args.idx_source,
                "idx_policy_mix": args.idx_policy_mix,
                "kp_d_model": int(args.kp_d_model),
                "kp_n_layers": int(args.kp_n_layers),
                "kp_n_heads": int(args.kp_n_heads),
                "kp_d_ff": int(args.kp_d_ff),
                "kp_d_cond": int(args.kp_d_cond),
                "kp_maze_channels": args.kp_maze_channels,
                "dphi_ckpt": args.dphi_ckpt,
                "dphi_use_ema": bool(args.dphi_use_ema),
                "selector_ckpt": args.selector_ckpt,
                "selector_use_ema": bool(args.selector_use_ema),
            }
            save_checkpoint(ckpt_path, model, optimizer, step, ema, meta=meta)

    final_path = os.path.join(args.ckpt_dir, "ckpt_final.pt")
    meta = {
        "stage": "keypoints",
        "T": args.T,
        "K": args.K,
        "data_dim": data_dim,
        "N_train": args.N_train,
        "schedule": args.schedule,
        "use_sdf": bool(args.use_sdf),
        "with_velocity": bool(args.with_velocity),
        "dataset": args.dataset,
        "env_id": args.env_id,
        "d4rl_flip_y": bool(args.d4rl_flip_y),
        "logit_space": bool(args.logit_space),
        "logit_eps": float(args.logit_eps),
        "use_start_goal": bool(args.cond_start_goal),
        "clamp_endpoints": bool(args.clamp_endpoints),
        "cond_start_goal": bool(args.cond_start_goal),
        "window_mode": args.window_mode,
        "goal_mode": args.goal_mode,
        "min_tortuosity": args.min_tortuosity,
        "min_turns": args.min_turns,
        "turn_angle_deg": args.turn_angle_deg,
        "use_kp_feat": bool(args.use_kp_feat),
        "kp_feat_dim": int(args.kp_feat_dim) if bool(args.use_kp_feat) else 0,
        "idx_source": args.idx_source,
        "idx_policy_mix": args.idx_policy_mix,
        "kp_d_model": int(args.kp_d_model),
        "kp_n_layers": int(args.kp_n_layers),
        "kp_n_heads": int(args.kp_n_heads),
        "kp_d_ff": int(args.kp_d_ff),
        "kp_d_cond": int(args.kp_d_cond),
        "kp_maze_channels": args.kp_maze_channels,
        "dphi_ckpt": args.dphi_ckpt,
        "dphi_use_ema": bool(args.dphi_use_ema),
        "selector_ckpt": args.selector_ckpt,
        "selector_use_ema": bool(args.selector_use_ema),
    }
    save_checkpoint(final_path, model, optimizer, args.steps, ema, meta=meta)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
