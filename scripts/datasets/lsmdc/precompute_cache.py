from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import torch
from tqdm import tqdm

from src.data.lsmdc import LSMDCVideoDataset
from src.models.clip_text import CLIPTextEncoder
from src.models.frame_vae import FrameAutoencoderKL
from src.utils.device import get_autocast_dtype
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/LSMDC")
    p.add_argument("--video_dir", type=str, default="data/LSMDC/videos")
    p.add_argument("--out_dir", type=str, default="data/lsmdc_cache")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=256)
    p.add_argument("--clip_seconds", type=float, default=5.0)
    p.add_argument("--clip_strategy", type=str, default="center", choices=["center", "random"])
    p.add_argument("--use_padded", type=int, default=1)
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--shard_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--vae_scale", type=float, default=0.18215)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return p


def _dtype_from_arg(arg: str) -> torch.dtype:
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    return torch.float32


def _save_shard(out_dir: str, split: str, shard_id: int, latents, text_embed, text, meta) -> str:
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    shard_path = os.path.join(out_dir, split, f"shard_{shard_id:05d}.pt")
    payload = {
        "latents": latents,
        "text_embed": text_embed,
        "text": text,
        "meta": meta,
    }
    torch.save(payload, shard_path)
    return shard_path


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for caching")

    autocast_dtype = _dtype_from_arg(args.dtype)

    dataset = LSMDCVideoDataset(
        data_dir=args.data_dir,
        video_dir=args.video_dir,
        split=args.split,
        T=args.T,
        frame_size=args.frame_size,
        clip_seconds=args.clip_seconds,
        clip_strategy=args.clip_strategy,
        use_padded=bool(args.use_padded),
        max_items=args.max_items or None,
    )
    text_encoder = CLIPTextEncoder(model_name=args.text_model, device=device, dtype=autocast_dtype)
    vae = FrameAutoencoderKL(
        model_name=args.vae_model,
        device=device,
        dtype=autocast_dtype,
        scale=args.vae_scale,
        use_mean=True,
        freeze=True,
    )

    latents_buf = []
    text_buf = []
    text_embed_buf = []
    meta_buf = []
    shard_paths = []

    total = 0
    failed = 0
    shard_id = 0
    start_time = time.time()
    target = len(dataset)
    if args.max_items and args.max_items > 0:
        target = min(target, args.max_items)
    pbar = tqdm(total=target, dynamic_ncols=True)

    batch_frames: List[torch.Tensor] = []
    batch_texts: List[str] = []
    batch_meta: List[object] = []

    for idx in range(target):
        try:
            item = dataset[idx]
        except Exception:
            failed += 1
            continue
        batch_frames.append(item["frames"])
        batch_texts.append(str(item.get("text", "")))
        batch_meta.append(item.get("meta"))
        pbar.update(1)

        if len(batch_frames) < args.batch:
            continue

        frames = torch.stack(batch_frames, dim=0).to(device)
        texts = list(batch_texts)

        with torch.no_grad():
            latents = vae.encode(frames)
            text_embed = text_encoder(texts)

        latents = latents.detach().cpu().to(torch.float16)
        text_embed = text_embed.detach().cpu().to(torch.float16)

        for i in range(latents.shape[0]):
            latents_buf.append(latents[i])
            text_embed_buf.append(text_embed[i])
            text_buf.append(texts[i])
            meta_buf.append(batch_meta[i])
            total += 1

            if len(latents_buf) >= args.shard_size:
                shard_path = _save_shard(
                    args.out_dir,
                    args.split,
                    shard_id,
                    torch.stack(latents_buf, dim=0),
                    torch.stack(text_embed_buf, dim=0),
                    list(text_buf),
                    list(meta_buf),
                )
                shard_paths.append({"path": shard_path, "count": len(latents_buf)})
                shard_id += 1
                latents_buf.clear()
                text_embed_buf.clear()
                text_buf.clear()
                meta_buf.clear()

        batch_frames.clear()
        batch_texts.clear()
        batch_meta.clear()

        elapsed = time.time() - start_time
        if total > 0:
            pbar.set_description(f"cached {total} clips, {elapsed/total:.2f}s/clip")

    if len(batch_frames) > 0:
        frames = torch.stack(batch_frames, dim=0).to(device)
        texts = list(batch_texts)
        with torch.no_grad():
            latents = vae.encode(frames)
            text_embed = text_encoder(texts)
        latents = latents.detach().cpu().to(torch.float16)
        text_embed = text_embed.detach().cpu().to(torch.float16)
        for i in range(latents.shape[0]):
            latents_buf.append(latents[i])
            text_embed_buf.append(text_embed[i])
            text_buf.append(texts[i])
            meta_buf.append(batch_meta[i])
            total += 1

    if len(latents_buf) > 0:
        shard_path = _save_shard(
            args.out_dir,
            args.split,
            shard_id,
            torch.stack(latents_buf, dim=0),
            torch.stack(text_embed_buf, dim=0),
            list(text_buf),
            list(meta_buf),
        )
        shard_paths.append({"path": shard_path, "count": len(latents_buf)})

    index = {
        "total": total,
        "shards": shard_paths,
        "config": {
            "T": args.T,
            "frame_size": args.frame_size,
            "clip_seconds": args.clip_seconds,
            "clip_strategy": args.clip_strategy,
            "use_padded": bool(args.use_padded),
            "text_model": args.text_model,
            "vae_model": args.vae_model,
            "vae_scale": args.vae_scale,
        },
        "failed": failed,
    }
    os.makedirs(os.path.join(args.out_dir, args.split), exist_ok=True)
    index_path = os.path.join(args.out_dir, args.split, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
