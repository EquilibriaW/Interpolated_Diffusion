import argparse
import time

import psutil
import torch

from src.data.wan_synth import create_wan_synth_dataloader
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule
from src.models.wan_backbone import load_wan_transformer, resolve_dtype
from src.utils.device import get_autocast_dtype
from src.utils.logging import create_writer
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shuffle_buffer", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--T", type=int, default=21)
    p.add_argument("--N_train", type=int, default=1000)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])

    p.add_argument("--wan_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--wan_subfolder", type=str, default="transformer")
    p.add_argument("--wan_dtype", type=str, default="bf16")
    p.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Optional checkpoint path from our training scripts. If set, load weights from this ckpt.",
    )

    p.add_argument("--sla_topk", type=float, default=0.07)
    p.add_argument("--sla_type", type=str, default="sagesla", choices=["sla", "sagesla"])

    p.add_argument("--log_dir", type=str, default="runs/eval_wan_sla_gap_wansynth")
    p.add_argument("--log_every", type=int, default=10)
    return p


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = create_writer(args.log_dir)
    proc = psutil.Process()

    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        shuffle_buffer=int(args.shuffle_buffer),
        shuffle=False,
        shardshuffle=False,
        pin_memory=True,
    )
    it = iter(loader)

    betas = make_beta_schedule(str(args.schedule), int(args.N_train)).to(device)
    schedule = make_alpha_bars(betas)
    sqrt_ab = schedule["sqrt_alpha_bar"].to(dtype=torch.float32)
    sqrt_1mab = schedule["sqrt_one_minus_alpha_bar"].to(dtype=torch.float32)

    wan_dtype = resolve_dtype(args.wan_dtype)
    if wan_dtype is None:
        raise ValueError("--wan_dtype must be set (recommend bf16)")

    # Two models with identical weights: dense is "teacher", SLA is approximation.
    model_dense = load_wan_transformer(args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device)
    model_sla = load_wan_transformer(args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device)
    from src.models.wan_sla import apply_wan_sla

    replaced = apply_wan_sla(model_sla, topk=float(args.sla_topk), attention_type=str(args.sla_type), use_bf16=True)
    writer.add_scalar("eval/sla_layers", float(replaced), 0)

    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu")
        sd = payload.get("model", {})
        meta = payload.get("meta", {}) or {}

        # If training attached a per-frame conditioning projector, attach it before loading.
        if any(k.startswith("frame_cond_proj.") for k in sd.keys()):
            from src.models.wan_frame_cond import FrameCondProjector

            feat_dim = int(meta.get("wan_frame_cond_feat_dim", 5))
            text_dim = int(meta.get("text_dim", 4096))
            hidden_dim = int(meta.get("wan_frame_cond_hidden", 256))
            n_layers = int(meta.get("wan_frame_cond_layers", 2))
            dropout = float(meta.get("wan_frame_cond_dropout", 0.0))

            proj_dtype = getattr(model_dense, "dtype", None) or wan_dtype
            model_dense.frame_cond_proj = FrameCondProjector(
                feat_dim=feat_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
            ).to(device=device, dtype=proj_dtype)
            model_sla.frame_cond_proj = FrameCondProjector(
                feat_dim=feat_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
            ).to(device=device, dtype=proj_dtype)

        missing_dense, unexpected_dense = model_dense.load_state_dict(sd, strict=False)
        missing_sla, unexpected_sla = model_sla.load_state_dict(sd, strict=False)
        writer.add_text(
            "eval/ckpt_load",
            f"ckpt={args.ckpt}\n"
            f"missing_dense={len(missing_dense)} unexpected_dense={len(unexpected_dense)}\n"
            f"missing_sla={len(missing_sla)} unexpected_sla={len(unexpected_sla)}",
            0,
        )

    model_dense.eval()
    model_sla.eval()
    model_dtype = getattr(model_dense, "dtype", None)
    autocast_dtype = resolve_dtype(args.wan_dtype) or get_autocast_dtype()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 123)

    ema_gap = 0.0
    ema_dense = 0.0
    ema_sla = 0.0
    ema_beta = 0.98
    start = time.perf_counter()

    for step in range(int(args.max_batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        latents = batch["latents"]
        text_embed = batch.get("text_embed")
        if text_embed is None:
            raise RuntimeError("text_embed missing from Wan synth dataset")

        if model_dtype is not None:
            latents = latents.to(device, dtype=model_dtype, non_blocking=True)
            text_embed = text_embed.to(device, dtype=model_dtype, non_blocking=True)
        else:
            latents = latents.to(device, non_blocking=True)
            text_embed = text_embed.to(device, non_blocking=True)

        if latents.shape[1] != int(args.T):
            raise ValueError(f"T mismatch: batch T={latents.shape[1]} args.T={args.T}")

        B, T, C, H, W = latents.shape
        t = torch.randint(0, int(args.N_train), (B,), generator=gen, device=device, dtype=torch.long)
        eps = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=gen)
        zt = sqrt_ab[t].view(B, 1, 1, 1, 1) * latents + sqrt_1mab[t].view(B, 1, 1, 1, 1) * eps

        latents_in = zt.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            pred_dense = model_dense(latents_in, t, text_embed).sample
            pred_sla = model_sla(latents_in, t, text_embed).sample
        pred_dense = pred_dense.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
        pred_sla = pred_sla.permute(0, 2, 1, 3, 4)

        mse_dense = (pred_dense - eps).float().pow(2).mean()
        mse_sla = (pred_sla - eps).float().pow(2).mean()
        mse_gap = (pred_sla - pred_dense).float().pow(2).mean()

        if step == 0:
            ema_dense = float(mse_dense.item())
            ema_sla = float(mse_sla.item())
            ema_gap = float(mse_gap.item())
        else:
            ema_dense = ema_beta * ema_dense + (1.0 - ema_beta) * float(mse_dense.item())
            ema_sla = ema_beta * ema_sla + (1.0 - ema_beta) * float(mse_sla.item())
            ema_gap = ema_beta * ema_gap + (1.0 - ema_beta) * float(mse_gap.item())

        if step % int(args.log_every) == 0:
            dt = time.perf_counter() - start
            sps = float((step + 1) * B) / max(dt, 1e-8)
            rss_gb = proc.memory_info().rss / (1024**3)
            writer.add_scalar("eval/mse_dense_eps", float(mse_dense.item()), step)
            writer.add_scalar("eval/mse_sla_eps", float(mse_sla.item()), step)
            writer.add_scalar("eval/mse_sla_vs_dense", float(mse_gap.item()), step)
            writer.add_scalar("eval/mse_dense_eps_ema", float(ema_dense), step)
            writer.add_scalar("eval/mse_sla_eps_ema", float(ema_sla), step)
            writer.add_scalar("eval/mse_sla_vs_dense_ema", float(ema_gap), step)
            writer.add_scalar("eval/mse_sla_over_dense", float(mse_sla.item() / max(mse_dense.item(), 1e-12)), step)
            writer.add_scalar("eval/t_mean", float(t.float().mean().item()), step)
            writer.add_scalar("eval/samples_per_sec", sps, step)
            writer.add_scalar("eval/cpu_rss_gb", float(rss_gb), step)
            if torch.cuda.is_available():
                writer.add_scalar("eval/max_mem_gb", float(torch.cuda.max_memory_allocated() / (1024**3)), step)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
