import argparse
import json
import os

import torch
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.corruptions.keyframes import sample_fixed_k_indices_uniform_batch
from src.diffusion.ddpm import _timesteps, ddim_step
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.video_token_denoisers import VideoTokenKeypointDenoiser
from src.models.encoders import TextConditionEncoder
from src.models.wan_backbone import load_wan_transformer, resolve_dtype
from src.corruptions.video_keyframes import interpolate_video_from_indices
from src.utils.video_tokens import unpatchify_tokens
from src.utils.video_tokens import patchify_latents
from src.utils.device import get_autocast_dtype


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for anchor shards")
    p.add_argument("--phase1_ckpt", type=str, required=True, help="Phase-1 keypoint checkpoint")
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ddim_schedule", type=str, default="quadratic", choices=["linear", "quadratic", "sqrt"])
    p.add_argument("--idx_source", type=str, default="uniform", choices=["uniform", "npz"])
    p.add_argument("--idx_npz", type=str, default="")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_cond", type=int, default=256)
    p.add_argument("--use_wan", type=int, default=0)
    p.add_argument("--wan_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--wan_subfolder", type=str, default="transformer")
    p.add_argument("--wan_dtype", type=str, default="")
    p.add_argument("--wan_attn", type=str, default="default", choices=["default", "sla", "sagesla"])
    p.add_argument("--sla_topk", type=float, default=0.1)
    p.add_argument("--video_interp_mode", type=str, default="smooth", choices=["linear", "smooth", "flow", "sinkhorn"])
    p.add_argument("--video_interp_smooth_kernel", type=str, default="0.25,0.5,0.25")
    p.add_argument("--flow_interp_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_win", type=int, default=5)
    p.add_argument("--sinkhorn_angles", type=str, default="-10,-5,0,5,10")
    p.add_argument("--sinkhorn_shift", type=int, default=4)
    p.add_argument("--sinkhorn_global_mode", type=str, default="phasecorr", choices=["se2", "phasecorr", "none"])
    p.add_argument("--sinkhorn_warp_space", type=str, default="s", choices=["z", "s"])
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--sinkhorn_tau", type=float, default=0.05)
    p.add_argument("--sinkhorn_dustbin", type=float, default=-2.0)
    p.add_argument("--sinkhorn_d_match", type=int, default=0)
    p.add_argument("--sinkhorn_straightener_ckpt", type=str, default="")
    p.add_argument("--sinkhorn_straightener_dtype", type=str, default="")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--shuffle_buffer", type=int, default=0)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--max_batches", type=int, default=0, help="Stop after this many batches (0 = no limit)")
    p.add_argument("--anchor_dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    return p


def _sample_keypoints_ddim_tokens(
    model: torch.nn.Module,
    schedule: dict,
    idx: torch.Tensor,
    cond: dict,
    steps: int,
    T: int,
    spatial_shape: tuple[int, int],
    tokens_shape: tuple[int, int, int, int],
    schedule_name: str = "quadratic",
) -> torch.Tensor:
    device = idx.device
    B, K = idx.shape
    _, _, N, D = tokens_shape
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps, schedule=schedule_name)
    model_dtype = getattr(model, "dtype", next(model.parameters()).dtype)
    z = torch.randn((B, K, N, D), device=device, dtype=model_dtype)
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        eps = model(z, t, idx, cond, T, spatial_shape)
        z = ddim_step(z, eps, t, t_prev, schedule, eta=0.0)
    return z


def _sample_keypoints_ddim_wan(
    model: torch.nn.Module,
    schedule: dict,
    idx: torch.Tensor,
    tokens: torch.Tensor,
    text_embed: torch.Tensor,
    steps: int,
    T: int,
    spatial_shape: tuple[int, int],
    patch_size: int,
    schedule_name: str = "quadratic",
    interp_mode: str = "smooth",
    smooth_kernel: torch.Tensor | None = None,
    flow_warper: object | None = None,
    sinkhorn_warper: object | None = None,
) -> torch.Tensor:
    device = idx.device
    B, K = idx.shape
    B2, T2, N, D = tokens.shape
    if B2 != B or T2 != T:
        raise ValueError("tokens shape mismatch for wan sampling")
    n_train = schedule["alpha_bar"].shape[0]
    times = _timesteps(n_train, steps, schedule=schedule_name)
    model_dtype = getattr(model, "dtype", next(model.parameters()).dtype)
    z = torch.randn((B, K, N, D), device=device, dtype=model_dtype)
    for i in range(len(times) - 1):
        t = torch.full((B,), int(times[i]), device=device, dtype=torch.long)
        t_prev = torch.full((B,), int(times[i + 1]), device=device, dtype=torch.long)
        if interp_mode in ("flow", "sinkhorn"):
            warper = flow_warper if interp_mode == "flow" else sinkhorn_warper
            if warper is None:
                raise ValueError(f"{interp_mode} interpolator requested but warper is None")
            z_seq = torch.zeros((B, T, N, D), device=device, dtype=z.dtype)
            z_seq.scatter_(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), z)
            latents_seq = unpatchify_tokens(z_seq, patch_size, spatial_shape)
            try:
                flow_dtype = next(warper.parameters()).dtype
            except StopIteration:
                flow_dtype = latents_seq.dtype
            latents_seq = latents_seq.to(dtype=flow_dtype)
            latents_interp, _ = warper.interpolate(latents_seq, idx)
            latents_t = latents_interp
        else:
            idx_rep = idx.repeat_interleave(N, dim=0)
            vals_rep = z.permute(0, 2, 1, 3).reshape(B * N, K, D)
            z_interp_flat = interpolate_video_from_indices(
                idx_rep, vals_rep, T, mode=interp_mode, smooth_kernel=smooth_kernel
            )
            z_interp = z_interp_flat.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
            z_interp = z_interp.scatter(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D), z)
            latents_t = unpatchify_tokens(z_interp, patch_size, spatial_shape)
        latents_t = latents_t.permute(0, 2, 1, 3, 4)
        latents_t = latents_t.to(dtype=model_dtype)
        text_embed = text_embed.to(dtype=model_dtype)
        with torch.no_grad():
            pred_latents = model(latents_t, t, text_embed).sample
        pred_latents = pred_latents.permute(0, 2, 1, 3, 4)
        pred_tokens, _ = patchify_latents(pred_latents, patch_size)
        pred_key = pred_tokens.gather(1, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D))
        z = ddim_step(z, pred_key, t, t_prev, schedule, eta=0.0)
    return z


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for anchor precompute.")

    if args.num_workers > 0:
        import warnings

        warnings.warn(
            "Anchor precompute with num_workers>0 can reorder samples and break key alignment. "
            "Use num_workers=0 for deterministic ordering.",
            stacklevel=2,
        )

    os.makedirs(args.out_dir, exist_ok=True)

    idx_all = None
    if args.idx_source == "npz":
        if not args.idx_npz:
            raise ValueError("--idx_npz is required when idx_source=npz")
        payload = torch.load(args.idx_npz, map_location="cpu") if args.idx_npz.endswith(".pt") else None
        if payload is not None:
            idx_all = payload.get("kp_idx")
        if idx_all is None:
            import numpy as np

            npz = np.load(args.idx_npz)
            if "kp_idx" not in npz:
                raise ValueError("idx_npz missing kp_idx")
            idx_all = npz["kp_idx"]
        idx_all = torch.as_tensor(idx_all, dtype=torch.long)

    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=max(1, args.shuffle_buffer),
        shuffle=False,
        shardshuffle=False,
        return_keys=True,
    )

    betas = make_beta_schedule(args.schedule, args.N_train).to(device)
    schedule = make_alpha_bars(betas)

    # Load checkpoint meta if present.
    payload = torch.load(args.phase1_ckpt, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

    model = None
    cond_encoder = None
    shard_writer = None
    flow_warper = None
    sinkhorn_warper = None
    in_channels = int(meta.get("in_channels", 16))
    if args.video_interp_mode == "flow":
        if not args.flow_interp_ckpt:
            raise ValueError("--flow_interp_ckpt is required for video_interp_mode=flow")
        from src.models.latent_flow_interpolator import load_latent_flow_interpolator
        flow_dtype = resolve_dtype(args.wan_dtype) or get_autocast_dtype()
        flow_warper, _ = load_latent_flow_interpolator(args.flow_interp_ckpt, device=device, dtype=flow_dtype)
    elif args.video_interp_mode == "sinkhorn":
        from src.models.sinkhorn_warp import SinkhornWarpInterpolator
        from src.models.latent_straightener import load_latent_straightener

        straightener = None
        if args.sinkhorn_straightener_ckpt:
            s_dtype = resolve_dtype(args.sinkhorn_straightener_dtype) or get_autocast_dtype()
            straightener, _ = load_latent_straightener(args.sinkhorn_straightener_ckpt, device=device, dtype=s_dtype)
        angles = [float(x) for x in args.sinkhorn_angles.split(",") if x.strip()]
        sinkhorn_warper = SinkhornWarpInterpolator(
            in_channels=in_channels,
            patch_size=args.patch_size,
            win_size=args.sinkhorn_win,
            global_mode=args.sinkhorn_global_mode,
            angles_deg=angles,
            shift_range=args.sinkhorn_shift,
            sinkhorn_iters=args.sinkhorn_iters,
            sinkhorn_tau=args.sinkhorn_tau,
            dustbin_logit=args.sinkhorn_dustbin,
            d_match=args.sinkhorn_d_match,
            straightener=straightener,
            warp_space=args.sinkhorn_warp_space,
        ).to(device=device, dtype=get_autocast_dtype())

    anchor_dtype = torch.float16 if args.anchor_dtype == "float16" else torch.float32
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 19)

    try:
        import webdataset as wds

        shard_writer = wds.ShardWriter(os.path.join(args.out_dir, "anchor-%06d.tar"), maxcount=args.samples_per_shard)
        for step, batch in enumerate(tqdm(loader, desc="precompute anchors")):
            if args.max_batches and step >= int(args.max_batches):
                break
            latents = batch["latents"].to(device)
            text_embed = batch.get("text_embed")
            if text_embed is None:
                raise RuntimeError("text_embed missing from Wan synth dataset")
            text_embed = text_embed.to(device)
            keys = batch.get("key")
            if keys is None:
                raise RuntimeError("dataset did not provide keys; set return_keys=True")

            tokens, spatial_shape = patchify_latents(latents, args.patch_size)
            B, T, N, D = tokens.shape
            if T != args.T:
                raise ValueError(f"T mismatch: batch={T} args={args.T}")

            if model is None:
                d_model = int(meta.get("d_model", args.d_model))
                d_ff = int(meta.get("d_ff", args.d_ff))
                n_layers = int(meta.get("n_layers", args.n_layers))
                n_heads = int(meta.get("n_heads", args.n_heads))
                d_cond = int(meta.get("d_cond", args.d_cond))
                text_dim = int(meta.get("text_dim", text_embed.shape[-1]))
                if args.use_wan:
                    wan_dtype = resolve_dtype(args.wan_dtype)
                    model = load_wan_transformer(
                        args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device
                    )
                else:
                    cond_encoder = TextConditionEncoder(text_dim=text_dim, d_cond=d_cond).to(device)
                    model = VideoTokenKeypointDenoiser(
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        d_cond=d_cond,
                        use_sdf=False,
                        use_start_goal=False,
                        data_dim=D,
                        cond_encoder=cond_encoder,
                    ).to(device)

                lora_rank = int(meta.get("lora_rank", 0))
                if lora_rank > 0:
                    from src.models.lora import LoRAConfig, inject_lora

                    config = LoRAConfig(
                        rank=lora_rank,
                        alpha=float(meta.get("lora_alpha", 1.0)),
                        dropout=float(meta.get("lora_dropout", 0.0)),
                        targets=tuple(
                            [t.strip() for t in str(meta.get("lora_targets", "attn,ffn")).split(",") if t.strip()]
                        ),
                    )
                    inject_lora(model, config)

                state = payload.get("model", payload) if isinstance(payload, dict) else payload
                has_sla_state = isinstance(state, dict) and any("processor.sla." in k for k in state.keys())

                if args.use_wan and args.wan_attn != "default":
                    from src.models.wan_sla import apply_wan_sla

                    wan_dtype = resolve_dtype(args.wan_dtype)
                    use_bf16 = wan_dtype == torch.bfloat16 if wan_dtype is not None else True
                    apply_wan_sla(
                        model,
                        topk=float(args.sla_topk),
                        attention_type=str(args.wan_attn),
                        use_bf16=use_bf16,
                    )

                missing, unexpected = model.load_state_dict(state, strict=False)
                if has_sla_state:
                    if missing or unexpected:
                        raise RuntimeError(
                            f"Checkpoint mismatch after SLA load. Missing={missing}, Unexpected={unexpected}"
                        )
                else:
                    missing = [k for k in missing if "processor.sla." not in k]
                    unexpected = [k for k in unexpected if "processor.sla." not in k]
                    if missing or unexpected:
                        raise RuntimeError(
                            f"Checkpoint mismatch after load. Missing={missing}, Unexpected={unexpected}"
                        )
                model.eval()

            if args.idx_source == "uniform":
                idx, _ = sample_fixed_k_indices_uniform_batch(
                    B, T, args.K, generator=gen, device=device, ensure_endpoints=False
                )
            else:
                if idx_all is None:
                    raise RuntimeError("idx_source=npz requires idx_all")
                start = step * B
                end = start + B
                if end > idx_all.shape[0]:
                    raise RuntimeError("idx_npz exhausted before dataset")
                idx = idx_all[start:end].to(device)

            if args.use_wan:
                smooth_kernel = None
                if args.video_interp_mode == "smooth":
                    smooth_kernel = torch.tensor(
                        [float(x) for x in args.video_interp_smooth_kernel.split(",")], dtype=torch.float32, device=device
                    )
                anchors = _sample_keypoints_ddim_wan(
                    model,
                    schedule,
                    idx,
                    tokens,
                    text_embed,
                    args.ddim_steps,
                    T,
                    spatial_shape,
                    args.patch_size,
                    schedule_name=args.ddim_schedule,
                    interp_mode=args.video_interp_mode,
                    smooth_kernel=smooth_kernel,
                    flow_warper=flow_warper,
                    sinkhorn_warper=sinkhorn_warper,
                )
            else:
                cond = {"text_embed": text_embed}
                with torch.no_grad():
                    anchors = _sample_keypoints_ddim_tokens(
                        model,
                        schedule,
                        idx,
                        cond,
                        args.ddim_steps,
                        T,
                        spatial_shape,
                        tokens.shape,
                        schedule_name=args.ddim_schedule,
                    )

            anchors = anchors.to(dtype=anchor_dtype).cpu()
            idx_cpu = idx.cpu()

            for b in range(B):
                shard_writer.write(
                    {
                        "__key__": str(keys[b]),
                        "anchor.pth": anchors[b],
                        "anchor_idx.pth": idx_cpu[b],
                    }
                )
    finally:
        if shard_writer is not None:
            shard_writer.close()

    meta_out = {
        "source_pattern": args.data_pattern,
        "phase1_ckpt": args.phase1_ckpt,
        "T": args.T,
        "K": args.K,
        "patch_size": args.patch_size,
        "N_train": args.N_train,
        "schedule": args.schedule,
        "ddim_steps": args.ddim_steps,
        "ddim_schedule": args.ddim_schedule,
        "idx_source": args.idx_source,
        "use_wan": bool(args.use_wan),
        "wan_repo": args.wan_repo,
        "wan_subfolder": args.wan_subfolder,
        "wan_dtype": args.wan_dtype,
        "wan_attn": args.wan_attn,
        "sla_topk": float(args.sla_topk),
        "video_interp_mode": args.video_interp_mode,
        "flow_interp_ckpt": args.flow_interp_ckpt,
        "sinkhorn_win": args.sinkhorn_win,
        "sinkhorn_angles": args.sinkhorn_angles,
        "sinkhorn_shift": args.sinkhorn_shift,
        "sinkhorn_global_mode": args.sinkhorn_global_mode,
        "sinkhorn_warp_space": args.sinkhorn_warp_space,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tau": args.sinkhorn_tau,
        "sinkhorn_dustbin": args.sinkhorn_dustbin,
        "sinkhorn_d_match": args.sinkhorn_d_match,
        "sinkhorn_straightener_ckpt": args.sinkhorn_straightener_ckpt,
        "sinkhorn_straightener_dtype": args.sinkhorn_straightener_dtype,
        "anchor_dtype": args.anchor_dtype,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)


if __name__ == "__main__":
    main()
