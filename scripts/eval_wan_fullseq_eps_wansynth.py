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
    p.add_argument("--wan_dtype", type=str, default="")
    p.add_argument("--wan_attn", type=str, default="default", choices=["default", "sla", "sagesla"])
    p.add_argument("--sla_topk", type=float, default=0.07)

    p.add_argument("--log_dir", type=str, default="runs/eval_wan_fullseq_eps_wansynth")
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
        drop_last=True,
    )
    it = iter(loader)

    schedule = make_beta_schedule(str(args.schedule), int(args.N_train))
    alpha_bars = make_alpha_bars(schedule)
    sqrt_ab = torch.tensor(alpha_bars, device=device, dtype=torch.float32).sqrt()
    sqrt_1mab = torch.tensor(1.0 - alpha_bars, device=device, dtype=torch.float32).sqrt()

    wan_dtype = resolve_dtype(args.wan_dtype)
    model = load_wan_transformer(args.wan_repo, subfolder=args.wan_subfolder, torch_dtype=wan_dtype, device=device)
    if args.wan_attn != "default":
        from src.models.wan_sla import apply_wan_sla

        use_bf16 = wan_dtype == torch.bfloat16 if wan_dtype is not None else True
        replaced = apply_wan_sla(model, topk=float(args.sla_topk), attention_type=str(args.wan_attn), use_bf16=use_bf16)
        writer.add_scalar("eval/sla_layers", float(replaced), 0)

    model.eval()
    model_dtype = getattr(model, "dtype", None)
    autocast_dtype = resolve_dtype(args.wan_dtype) or get_autocast_dtype()

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 123)

    ema_mse = 0.0
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
        eps = torch.randn_like(latents, generator=gen)
        zt = sqrt_ab[t].view(B, 1, 1, 1, 1) * latents + sqrt_1mab[t].view(B, 1, 1, 1, 1) * eps

        latents_in = zt.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            pred = model(latents_in, t, text_embed).sample
        pred = pred.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]

        mse = (pred - eps).float().pow(2).mean()
        if step == 0:
            ema_mse = float(mse.item())
        else:
            ema_mse = ema_beta * ema_mse + (1.0 - ema_beta) * float(mse.item())

        if step % int(args.log_every) == 0:
            dt = time.perf_counter() - start
            sps = float((step + 1) * B) / max(dt, 1e-8)
            rss_gb = proc.memory_info().rss / (1024**3)
            writer.add_scalar("eval/mse_eps", float(mse.item()), step)
            writer.add_scalar("eval/mse_eps_ema", float(ema_mse), step)
            writer.add_scalar("eval/t_mean", float(t.float().mean().item()), step)
            writer.add_scalar("eval/samples_per_sec", sps, step)
            writer.add_scalar("eval/cpu_rss_gb", float(rss_gb), step)
            if torch.cuda.is_available():
                writer.add_scalar("eval/max_mem_gb", float(torch.cuda.max_memory_allocated() / (1024**3)), step)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()

