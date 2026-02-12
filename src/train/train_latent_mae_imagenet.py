from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.frame_vae import FrameAutoencoderKL
from src.models.latent_mae import LatentMAE
from src.optim import build_optimizer
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import create_writer
from src.utils.seed import set_seed


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return use_ddp, rank, local_rank, world_size, device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="imagefolder", choices=["imagefolder", "fake"])
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--fake_num_samples", type=int, default=100000)

    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_weight_decay", type=float, default=0.01)
    p.add_argument("--muon_adam_beta1", type=float, default=0.9)
    p.add_argument("--muon_adam_beta2", type=float, default=0.95)
    p.add_argument("--muon_adam_eps", type=float, default=1e-10)

    p.add_argument("--base_width", type=int, default=256)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--mask_patch", type=int, default=2)

    p.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--vae_scale", type=float, default=0.18215)
    p.add_argument("--vae_use_mean", type=int, default=1)

    p.add_argument("--amp_bf16", type=int, default=1)

    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/latent_mae_imagenet")
    p.add_argument("--log_dir", type=str, default="runs/latent_mae_imagenet")
    p.add_argument("--resume", type=str, default="")
    return p


def _build_dataset(args: argparse.Namespace) -> Dataset:
    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as exc:  # pragma: no cover
        raise ImportError("torchvision is required for latent MAE training") from exc

    if args.dataset == "imagefolder":
        if not args.data_root:
            raise ValueError("--data_root is required for dataset=imagefolder")
        tfm = T.Compose(
            [
                T.RandomResizedCrop(int(args.image_size), scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        ds = torchvision.datasets.ImageFolder(args.data_root, transform=tfm)
        return ds

    tfm = torchvision.transforms.ToTensor()
    ds = torchvision.datasets.FakeData(
        size=int(args.fake_num_samples),
        image_size=(3, int(args.image_size), int(args.image_size)),
        num_classes=int(args.num_classes),
        transform=tfm,
        random_offset=int(args.seed),
    )
    return ds


def _sample_patch_mask(
    batch_size: int,
    h: int,
    w: int,
    *,
    patch: int,
    ratio: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if patch <= 0:
        raise ValueError("patch must be > 0")
    if (h % patch) != 0 or (w % patch) != 0:
        raise ValueError("latent size must be divisible by mask patch")
    gh = h // patch
    gw = w // patch
    m = torch.rand((batch_size, gh, gw), generator=generator, device=device) < float(ratio)
    m = m.repeat_interleave(patch, dim=1).repeat_interleave(patch, dim=2)  # [B,H,W]
    return m.unsqueeze(1)  # [B,1,H,W]


def main() -> None:
    args = build_parser().parse_args()
    use_ddp, rank, local_rank, world_size, device = _init_distributed()
    seed_rank = int(args.seed) + (rank * 100003)
    set_seed(seed_rank, deterministic=False)

    if device.type != "cuda":
        raise RuntimeError("CUDA is required for latent MAE training")
    is_main = rank == 0
    if is_main:
        print(f"[latent-mae] device={device} cuda={torch.cuda.get_device_name(local_rank if use_ddp else 0)}")
        print(f"[latent-mae] ddp={int(use_ddp)} world_size={world_size}")

    ds = _build_dataset(args)
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed_rank,
            drop_last=True,
        )
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            sampler=sampler,
            num_workers=int(args.num_workers),
            pin_memory=True,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            pin_memory=True,
            drop_last=True,
        )
    it = iter(loader)

    model = LatentMAE(in_channels=4, base_width=int(args.base_width)).to(device)
    optimizer, opt_info = build_optimizer(
        model=model,
        optimizer_name=str(args.optimizer),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        muon_lr=float(args.muon_lr),
        muon_momentum=float(args.muon_momentum),
        muon_weight_decay=float(args.muon_weight_decay),
        muon_adam_betas=(float(args.muon_adam_beta1), float(args.muon_adam_beta2)),
        muon_adam_eps=float(args.muon_adam_eps),
    )
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    model_raw = model.module if isinstance(model, DDP) else model

    vae = FrameAutoencoderKL(
        model_name=str(args.vae_model),
        device=device,
        dtype=torch.float16,
        scale=float(args.vae_scale),
        use_mean=bool(args.vae_use_mean),
        freeze=True,
    )

    if is_main:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        writer = create_writer(args.log_dir)
    else:
        writer = _NullWriter()
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model_raw, optimizer, ema=None, map_location=device)
    if use_ddp:
        start_t = torch.tensor([int(start_step)], device=device, dtype=torch.long)
        dist.broadcast(start_t, src=0)
        start_step = int(start_t.item())

    amp_bf16 = bool(args.amp_bf16)
    grad_accum = max(1, int(args.grad_accum))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_rank + 17)
    model.train()
    if is_main:
        eff_b = int(args.batch_size) * int(world_size) * int(grad_accum)
        print(
            f"[latent-mae] optimizer={opt_info.name} n_muon={opt_info.n_muon} n_adam={opt_info.n_adam} "
            f"local_batch={int(args.batch_size)} grad_accum={grad_accum} effective_global_batch={eff_b}"
        )

    pbar = tqdm(range(start_step, int(args.steps)), dynamic_ncols=True, disable=not is_main)
    for step in pbar:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        losses_micro = []
        last_latents = None
        last_mask = None
        for _ in range(grad_accum):
            try:
                x0, _ = next(it)
            except StopIteration:
                if use_ddp:
                    sampler.set_epoch(step + 1 + rank)
                it = iter(loader)
                x0, _ = next(it)
            x0 = x0.to(device, non_blocking=True).clamp(0.0, 1.0)

            with torch.no_grad():
                latents = vae.encode(x0).to(dtype=torch.float32)
            b, c, h, w = latents.shape
            m = _sample_patch_mask(
                batch_size=b,
                h=h,
                w=w,
                patch=int(args.mask_patch),
                ratio=float(args.mask_ratio),
                generator=gen,
                device=device,
            )
            x_masked = latents * (~m).to(dtype=latents.dtype)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_bf16):
                recon = model(x_masked)
                mse = (recon - latents).pow(2)
                m_f = m.to(dtype=mse.dtype)
                loss = (mse * m_f).sum() / (m_f.sum() * float(c) + 1e-8)
            (loss / float(grad_accum)).backward()
            losses_micro.append(float(loss.detach().item()))
            last_latents = latents
            last_mask = m

        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        loss_value = float(sum(losses_micro) / max(1, len(losses_micro)))
        step_time = time.time() - t0
        if is_main and (step % int(args.log_every) == 0):
            writer.add_scalar("train/loss", loss_value, step)
            writer.add_scalar("train/step_time_sec", float(step_time), step)
            writer.add_scalar(
                "train/samples_per_sec",
                float((int(args.batch_size) * grad_accum) / max(step_time, 1e-8)),
                step,
            )
            writer.add_scalar("train/mask_ratio_target", float(args.mask_ratio), step)
            if last_mask is not None:
                writer.add_scalar("train/mask_ratio_actual", float(last_mask.float().mean().item()), step)
            if last_latents is not None:
                writer.add_scalar("train/latent_std", float(last_latents.float().std().item()), step)
            pbar.set_description(f"loss={loss_value:.5f} step={step_time:.3f}s")

        if is_main and step > 0 and (step % int(args.save_every) == 0):
            ckpt = os.path.join(args.ckpt_dir, f"ckpt_{step:07d}.pt")
            meta = {
                "stage": "latent_mae_imagenet",
                "dataset": str(args.dataset),
                "data_root": str(args.data_root),
                "image_size": int(args.image_size),
                "batch_size_local": int(args.batch_size),
                "world_size": int(world_size),
                "grad_accum": int(grad_accum),
                "optimizer": str(args.optimizer),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "muon_lr": float(args.muon_lr),
                "muon_momentum": float(args.muon_momentum),
                "muon_weight_decay": float(args.muon_weight_decay),
                "base_width": int(args.base_width),
                "mask_ratio": float(args.mask_ratio),
                "mask_patch": int(args.mask_patch),
                "vae_model": str(args.vae_model),
                "vae_scale": float(args.vae_scale),
                "vae_use_mean": bool(args.vae_use_mean),
            }
            save_checkpoint(ckpt, model_raw, optimizer, step, ema=None, meta=meta)

    if is_main:
        final = os.path.join(args.ckpt_dir, "ckpt_final.pt")
        meta = {
            "stage": "latent_mae_imagenet",
            "dataset": str(args.dataset),
            "data_root": str(args.data_root),
            "image_size": int(args.image_size),
            "batch_size_local": int(args.batch_size),
            "world_size": int(world_size),
            "grad_accum": int(grad_accum),
            "optimizer": str(args.optimizer),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "muon_lr": float(args.muon_lr),
            "muon_momentum": float(args.muon_momentum),
            "muon_weight_decay": float(args.muon_weight_decay),
            "base_width": int(args.base_width),
            "mask_ratio": float(args.mask_ratio),
            "mask_patch": int(args.mask_patch),
            "vae_model": str(args.vae_model),
            "vae_scale": float(args.vae_scale),
            "vae_use_mean": bool(args.vae_use_mean),
        }
        save_checkpoint(final, model_raw, optimizer, int(args.steps), ema=None, meta=meta)
    writer.flush()
    writer.close()
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

