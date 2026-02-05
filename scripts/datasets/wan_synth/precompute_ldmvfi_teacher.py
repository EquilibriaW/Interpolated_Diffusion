import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.wan_backbone import resolve_dtype
from src.teachers.ldmvfi_teacher import LDMVFITEACHER


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True, help="Shard pattern for Wan2.1 synthetic dataset")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for teacher shards")
    p.add_argument("--ldmvfi_root", type=str, default="tmp_repos/LDMVFI")
    p.add_argument("--ldmvfi_config", type=str, default="tmp_repos/LDMVFI/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml")
    p.add_argument("--ldmvfi_ckpt", type=str, required=True)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=1.0)
    p.add_argument("--teacher_dtype", type=str, default="fp32")
    p.add_argument("--teacher_autocast", type=str, default="")
    p.add_argument("--vae_repo", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--vae_subfolder", type=str, default="vae")
    p.add_argument("--vae_dtype", type=str, default="bf16")
    p.add_argument("--vae_tiling", type=int, default=0)
    p.add_argument("--vae_slicing", type=int, default=0)
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--min_gap", type=int, default=2)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--shuffle_buffer", type=int, default=0)
    p.add_argument("--samples_per_shard", type=int, default=500)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--max_hw", type=int, default=0, help="If set, downscale inputs so max(H,W) <= max_hw")
    p.add_argument("--store_dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    return p


def _round_down(x: int, mult: int) -> int:
    return max(mult, (x // mult) * mult)


def _sample_even_mid_triplets(B: int, T: int, min_gap: int, gen: torch.Generator, device: torch.device):
    t0 = torch.empty((B,), dtype=torch.long, device=device)
    t1 = torch.empty((B,), dtype=torch.long, device=device)
    t = torch.empty((B,), dtype=torch.long, device=device)
    if min_gap % 2 == 1:
        min_gap = min_gap + 1
    max_gap = max(1, T - 1)
    even_gaps = [g for g in range(min_gap, max_gap + 1) if g % 2 == 0]
    if not even_gaps:
        raise ValueError(f"No valid even gaps for T={T} and min_gap={min_gap}")
    gaps = torch.tensor(even_gaps, device=device, dtype=torch.long)
    for i in range(B):
        gap = int(gaps[torch.randint(0, len(gaps), (1,), generator=gen, device=device)].item())
        lo = int(torch.randint(0, T - gap, (1,), generator=gen, device=device).item())
        hi = lo + gap
        t0[i] = lo
        t1[i] = hi
        t[i] = lo + gap // 2
    return t0, t1, t


def _decode_wan_latents(vae, z_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # z_norm: [B,C,H,W] normalized
    z = z_norm.unsqueeze(2)
    z = z * std + mean
    rgb = vae.decode(z, return_dict=False)[0]
    rgb = rgb.squeeze(2)
    return torch.clamp(rgb, min=-1.0, max=1.0)


def _encode_wan_latents(vae, rgb: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # rgb: [B,3,H,W] in [-1,1]
    rgb = rgb.unsqueeze(2)
    posterior = vae.encode(rgb).latent_dist
    z_raw = posterior.mode()
    z_norm = (z_raw - mean) / std
    return z_norm.squeeze(2)


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for LDMVFI teacher precompute.")

    os.makedirs(args.out_dir, exist_ok=True)

    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=max(1, args.shuffle_buffer),
        shuffle=False,
        shardshuffle=False,
        return_keys=True,
    )

    from diffusers import AutoencoderKLWan

    vae_dtype = resolve_dtype(args.vae_dtype) or torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.vae_repo, subfolder=args.vae_subfolder, torch_dtype=vae_dtype)
    vae.to(device)
    vae.eval()
    if args.vae_tiling:
        vae.enable_tiling()
    if args.vae_slicing:
        vae.enable_slicing()

    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae_dtype).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=vae_dtype).view(1, vae.config.z_dim, 1, 1, 1)

    teacher_dtype = resolve_dtype(args.teacher_dtype) or torch.float32
    autocast_dtype = resolve_dtype(args.teacher_autocast)
    teacher = LDMVFITEACHER(
        repo_root=args.ldmvfi_root,
        config_path=args.ldmvfi_config,
        ckpt_path=args.ldmvfi_ckpt,
        device=device,
        dtype=teacher_dtype,
        use_ddim=True,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        autocast_dtype=autocast_dtype,
    )

    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 17)

    try:
        import webdataset as wds

        shard_writer = wds.ShardWriter(os.path.join(args.out_dir, "teacher-%06d.tar"), maxcount=args.samples_per_shard)
        for step, batch in enumerate(tqdm(loader, desc="precompute teacher")):
            if args.max_batches and step >= int(args.max_batches):
                break

            latents = batch["latents"].to(device, dtype=vae_dtype, non_blocking=True)
            text_embed = batch.get("text_embed")
            text = batch.get("text")
            keys = batch.get("key")
            if keys is None:
                raise RuntimeError("dataset did not provide keys; set return_keys=True")
            if latents.dim() != 5:
                raise ValueError("latents must be [B,T,C,H,W]")
            B, T, C, H, W = latents.shape
            if T != args.T:
                raise ValueError(f"T mismatch: batch={T} args={args.T}")

            t0, t1, t = _sample_even_mid_triplets(B, T, args.min_gap, gen, device)
            z0 = latents[torch.arange(B, device=device), t0]
            z1 = latents[torch.arange(B, device=device), t1]

            with torch.no_grad():
                x0 = _decode_wan_latents(vae, z0, mean, std)
                x1 = _decode_wan_latents(vae, z1, mean, std)

            if args.max_hw and max(H, W) > int(args.max_hw):
                scale = float(args.max_hw) / float(max(H, W))
                new_h = _round_down(int(round(H * scale)), 8)
                new_w = _round_down(int(round(W * scale)), 8)
                x0_small = F.interpolate(x0, size=(new_h, new_w), mode="bilinear", align_corners=False)
                x1_small = F.interpolate(x1, size=(new_h, new_w), mode="bilinear", align_corners=False)
                x_t = teacher.interpolate(x0_small, x1_small)
                x_t = F.interpolate(x_t, size=(H, W), mode="bilinear", align_corners=False)
            else:
                x_t = teacher.interpolate(x0, x1)
            x_t = torch.clamp(x_t, min=-1.0, max=1.0).to(dtype=vae_dtype)

            with torch.no_grad():
                zt = _encode_wan_latents(vae, x_t, mean, std)

            zt = zt.to(dtype=store_dtype, device="cpu")
            latents_cpu = latents.to(dtype=store_dtype, device="cpu")
            if text_embed is not None:
                text_embed = text_embed.to(device="cpu")
            t_info = torch.stack([t0, t1, t], dim=1).to(device="cpu")

            for i in range(B):
                key = keys[i]
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                # Clone to avoid saving the entire batch storage for each sample.
                zt_i = zt[i].contiguous().clone()
                t_info_i = t_info[i].contiguous().clone()
                lat_i = latents_cpu[i].contiguous().clone()
                sample = {
                    "__key__": str(key),
                    "latent.pth": lat_i,
                    "teacher.pth": zt_i,
                    "teacher_idx.pth": t_info_i,
                }
                if text_embed is not None:
                    sample["embed.pth"] = text_embed[i].contiguous().clone()
                if text is not None:
                    text_i = text[i]
                    if isinstance(text_i, bytes):
                        text_i = text_i.decode("utf-8")
                    sample["prompt.txt"] = str(text_i)
                shard_writer.write(sample)
    finally:
        if "shard_writer" in locals() and shard_writer is not None:
            shard_writer.close()


if __name__ == "__main__":
    main()
