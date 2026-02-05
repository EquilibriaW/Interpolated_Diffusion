from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.latent_flow_interpolator import LatentResidualRefiner


class LatentLerpResidualInterpolator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        residual_channels: Optional[int] = None,
        residual_blocks: int = 2,
        gap_cond: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.residual_blocks = int(residual_blocks)
        self.gap_cond = bool(gap_cond)
        cond_channels = 1 if self.gap_cond else 0
        if residual_channels is None:
            residual_channels = base_channels
        # Inputs: z_base, z0, z1, t (+ gap)
        in_ch = self.in_channels * 3 + 1 + cond_channels
        out_ch = self.in_channels + 1  # residual + uncertainty
        self.residual = LatentResidualRefiner(
            in_channels=in_ch,
            hidden_channels=int(residual_channels),
            out_channels=out_ch,
            n_blocks=self.residual_blocks,
        )

    def _expand_cond(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.view(-1, 1, 1, 1)
        elif cond.dim() == 2:
            cond = cond.view(cond.shape[0], cond.shape[1], 1, 1)
        return cond.to(dtype=z0.dtype, device=z0.device).expand(-1, -1, z0.shape[-2], z0.shape[-1])

    def interpolate_pair(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        alpha: torch.Tensor,
        gap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha.dim() == 1:
            alpha = alpha.view(-1, 1, 1, 1)
        alpha = alpha.to(dtype=z0.dtype, device=z0.device).clamp(0.0, 1.0)
        z_base = (1.0 - alpha) * z0 + alpha * z1
        t_chan = alpha.expand(-1, 1, z_base.shape[-2], z_base.shape[-1])
        if self.gap_cond:
            if gap is None:
                raise ValueError("gap must be provided when gap_cond is enabled")
            gap_chan = self._expand_cond(z_base, gap)
            res_in = torch.cat([z_base, z0, z1, t_chan, gap_chan], dim=1)
        else:
            res_in = torch.cat([z_base, z0, z1, t_chan], dim=1)
        out = self.residual(res_in)
        res = out[:, : self.in_channels]
        unc = torch.sigmoid(out[:, self.in_channels : self.in_channels + 1])
        # Endpoint-locked residual.
        scale = alpha * (1.0 - alpha)
        z_hat = z_base + scale * res
        return z_hat, unc

    @torch.no_grad()
    def interpolate(self, latents: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # latents: [B,T,C,H,W], idx: [B,K]
        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T, _, _, _ = latents.shape
        if idx.dim() != 2 or idx.shape[0] != B:
            raise ValueError("idx must be [B,K]")
        out = torch.zeros_like(latents)
        conf = torch.zeros((B, T, latents.shape[-2], latents.shape[-1]), device=latents.device, dtype=latents.dtype)
        for b in range(B):
            idx_b = idx[b].clone()
            idx_b, _ = torch.sort(idx_b)
            anchors = latents[b, idx_b]  # [K,C,H,W]
            out[b, idx_b] = anchors
            conf[b, idx_b] = 1.0
            for k in range(idx_b.shape[0] - 1):
                t0 = int(idx_b[k].item())
                t1 = int(idx_b[k + 1].item())
                if t1 <= t0:
                    continue
                gap = t1 - t0
                if gap <= 1:
                    continue
                z0 = anchors[k : k + 1]
                z1 = anchors[k + 1 : k + 2]
                steps = gap - 1
                alpha = torch.linspace(1, steps, steps, device=latents.device, dtype=latents.dtype) / float(gap)
                alpha = alpha.view(-1, 1, 1, 1)
                z0_rep = z0.expand(steps, -1, -1, -1)
                z1_rep = z1.expand(steps, -1, -1, -1)
                gap_val = torch.tensor([gap / max(T - 1, 1)], device=latents.device, dtype=latents.dtype)
                gap_rep = gap_val.expand(steps)
                z_t, unc = self.interpolate_pair(z0_rep, z1_rep, alpha, gap=gap_rep if self.gap_cond else None)
                out[b, t0 + 1 : t1] = z_t
                conf[b, t0 + 1 : t1] = (1.0 - unc[:, 0]).clamp(0.0, 1.0)

            first = int(idx_b[0].item())
            last = int(idx_b[-1].item())
            if first > 0:
                out[b, :first] = anchors[0].unsqueeze(0).expand(first, -1, -1, -1)
                conf[b, :first] = conf[b, first : first + 1]
            if last < T - 1:
                out[b, last + 1 :] = anchors[-1].unsqueeze(0).expand(T - last - 1, -1, -1, -1)
                conf[b, last + 1 :] = conf[b, last : last + 1]
        return out, conf


def load_latent_lerp_interpolator(
    ckpt_path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> Tuple[LatentLerpResidualInterpolator, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    in_channels = int(meta.get("in_channels", 4))
    base_channels = int(meta.get("base_channels", 32))
    residual_blocks = int(meta.get("residual_blocks", 2))
    residual_channels = int(meta.get("residual_channels", base_channels))
    gap_cond = bool(meta.get("gap_cond", False))
    if dtype is None:
        meta_dtype = meta.get("model_dtype")
        if meta_dtype:
            name = str(meta_dtype).lower()
            if name in {"bf16", "bfloat16"}:
                dtype = torch.bfloat16
            elif name in {"fp16", "float16"}:
                dtype = torch.float16
            elif name in {"fp32", "float32"}:
                dtype = torch.float32
    model = LatentLerpResidualInterpolator(
        in_channels=in_channels,
        base_channels=base_channels,
        residual_channels=residual_channels,
        residual_blocks=residual_blocks,
        gap_cond=gap_cond,
    )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, meta


__all__ = ["LatentLerpResidualInterpolator", "load_latent_lerp_interpolator"]
