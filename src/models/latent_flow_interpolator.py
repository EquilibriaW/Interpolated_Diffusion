from __future__ import annotations

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + x)


class LatentResidualRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_blocks: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(hidden_channels) for _ in range(max(0, n_blocks))])
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out_proj(h)


def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    # flow: [B,2,H,W] in pixels
    bsz, _, h, w = flow.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=-1).to(dtype=flow.dtype)
    base = base.unsqueeze(0).expand(bsz, -1, -1, -1)
    grid = base + flow.permute(0, 2, 3, 1)
    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1).to(dtype=flow.dtype)


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    grid = _flow_to_grid(flow)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _cost_volume(
    z0: torch.Tensor,
    z1: torch.Tensor,
    radius: int = 2,
    downscale: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive for cost volume")
    if downscale < 1:
        raise ValueError("downscale must be >= 1")
    if downscale > 1:
        z0s = F.avg_pool2d(z0, downscale, stride=downscale)
        z1s = F.avg_pool2d(z1, downscale, stride=downscale)
    else:
        z0s = z0
        z1s = z1
    if normalize:
        z0s = F.normalize(z0s, dim=1, eps=1e-6)
        z1s = F.normalize(z1s, dim=1, eps=1e-6)

    B, C, H, W = z0s.shape
    pad = int(radius)
    z1p = F.pad(z1s, (pad, pad, pad, pad), mode="replicate")
    vols = []
    for dy in range(-pad, pad + 1):
        y0 = dy + pad
        y1 = y0 + H
        for dx in range(-pad, pad + 1):
            x0 = dx + pad
            x1 = x0 + W
            z1_shift = z1p[:, :, y0:y1, x0:x1]
            corr = (z0s * z1_shift).sum(dim=1, keepdim=True)
            vols.append(corr)
    cv = torch.cat(vols, dim=1)
    cv = cv / math.sqrt(max(1.0, float(C)))
    if downscale > 1:
        cv = F.interpolate(cv, size=z0.shape[-2:], mode="bilinear", align_corners=False)
    return cv


class LatentFlowPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        max_flow: float = 20.0,
        cond_channels: int = 0,
        time_mask: bool = False,
        cost_volume: bool = False,
        cv_radius: int = 2,
        cv_downscale: int = 2,
        cv_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.max_flow = float(max_flow)
        self.cond_channels = int(cond_channels)
        self.time_mask = bool(time_mask)
        self.cost_volume = bool(cost_volume)
        self.cv_radius = int(cv_radius)
        self.cv_downscale = int(cv_downscale)
        self.cv_norm = bool(cv_norm)
        cv_channels = (2 * self.cv_radius + 1) ** 2 if self.cost_volume else 0
        self.cv_channels = int(cv_channels)
        self.enc1 = _ConvBlock(in_channels * 2 + self.cond_channels + self.cv_channels, base_channels, stride=1)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = _ConvBlock(base_channels * 2, base_channels * 2, stride=1)
        self.dec1 = _ConvBlock(base_channels * 3, base_channels, stride=1)
        out_ch = 7 if self.time_mask else 6
        self.out = nn.Conv2d(base_channels, out_ch, kernel_size=3, padding=1)

    def forward(
        self, z0: torch.Tensor, z1: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z0/z1: [B,C,H,W]
        if self.cost_volume:
            cv = _cost_volume(z0, z1, radius=self.cv_radius, downscale=self.cv_downscale, normalize=self.cv_norm)
        else:
            cv = None

        if self.cond_channels > 0:
            if cond is None:
                raise ValueError("cond is required when cond_channels > 0")
            if cond.dim() == 2:
                cond = cond.view(cond.shape[0], cond.shape[1], 1, 1)
            if cond.dim() == 4 and cond.shape[-2:] != z0.shape[-2:]:
                cond = cond.expand(-1, -1, z0.shape[-2], z0.shape[-1])
            if cv is None:
                x = torch.cat([z0, z1, cond], dim=1)
            else:
                x = torch.cat([z0, z1, cond, cv], dim=1)
        else:
            if cv is None:
                x = torch.cat([z0, z1], dim=1)
            else:
                x = torch.cat([z0, z1, cv], dim=1)
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h3_up = F.interpolate(h3, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h3_up, h1], dim=1)
        h = self.dec1(h)
        out = self.out(h)
        flow01 = torch.tanh(out[:, 0:2]) * self.max_flow
        flow10 = torch.tanh(out[:, 2:4]) * self.max_flow
        if self.time_mask:
            mask_a = out[:, 4:5]
            mask_b = out[:, 5:6]
            uncertainty = torch.sigmoid(out[:, 6:7])
        else:
            mask_a = torch.sigmoid(out[:, 4:5])
            mask_b = torch.zeros_like(mask_a)
            uncertainty = torch.sigmoid(out[:, 5:6])
        return flow01, flow10, mask_a, mask_b, uncertainty


class LatentFlowInterpolator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        max_flow: float = 20.0,
        residual_channels: Optional[int] = None,
        residual_blocks: int = 2,
        time_mask: bool = False,
        gap_cond: bool = False,
        cost_volume: bool = False,
        cv_radius: int = 2,
        cv_downscale: int = 2,
        cv_norm: bool = True,
    ) -> None:
        super().__init__()
        self.time_mask = bool(time_mask)
        self.gap_cond = bool(gap_cond)
        cond_channels = 1 if self.gap_cond else 0
        self.cond_channels = int(cond_channels)
        self.cost_volume = bool(cost_volume)
        self.cv_radius = int(cv_radius)
        self.cv_downscale = int(cv_downscale)
        self.cv_norm = bool(cv_norm)
        self.net = LatentFlowPredictor(
            in_channels,
            base_channels=base_channels,
            max_flow=max_flow,
            cond_channels=cond_channels,
            time_mask=self.time_mask,
            cost_volume=self.cost_volume,
            cv_radius=self.cv_radius,
            cv_downscale=self.cv_downscale,
            cv_norm=self.cv_norm,
        )
        if residual_channels is None:
            residual_channels = base_channels
        self.residual_blocks = int(residual_blocks)
        if self.residual_blocks > 0:
            self.residual = LatentResidualRefiner(
                in_channels=in_channels * 3 + 1 + cond_channels,
                hidden_channels=residual_channels,
                out_channels=in_channels,
                n_blocks=self.residual_blocks,
            )
        else:
            self.residual = None

    def forward(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(z0, z1)

    def _expand_cond(self, z0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.view(-1, 1, 1, 1)
        elif cond.dim() == 2:
            cond = cond.view(cond.shape[0], cond.shape[1], 1, 1)
        return cond.to(dtype=z0.dtype, device=z0.device).expand(-1, -1, z0.shape[-2], z0.shape[-1])

    def predict_flow(
        self, z0: torch.Tensor, z1: torch.Tensor, gap: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.gap_cond:
            if gap is None:
                raise ValueError("gap must be provided when gap_cond is enabled")
            cond = self._expand_cond(z0, gap)
            return self.net(z0, z1, cond)
        return self.net(z0, z1)

    def blend_from_flow(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        alpha: torch.Tensor,
        flow01: torch.Tensor,
        flow10: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if alpha.dim() == 1:
            alpha = alpha.view(-1, 1, 1, 1)
        alpha = alpha.to(dtype=z0.dtype, device=z0.device).clamp(0.0, 1.0)
        if self.time_mask:
            if mask_b is None:
                raise ValueError("mask_b must be provided when time_mask is enabled")
            mask = torch.sigmoid(mask_a + mask_b * (2.0 * alpha - 1.0))
        else:
            mask = mask_a
        z0_w = _warp(z0, -alpha * flow01)
        z1_w = _warp(z1, -(1.0 - alpha) * flow10)
        mask = mask.to(dtype=z0.dtype)
        z_t = mask * z0_w + (1.0 - mask) * z1_w
        if self.residual is not None:
            t_chan = alpha.expand(-1, 1, z_t.shape[-2], z_t.shape[-1])
            if self.gap_cond:
                if gap is None:
                    raise ValueError("gap must be provided when gap_cond is enabled")
                gap_chan = self._expand_cond(z_t, gap)
                res_in = torch.cat([z_t, z0, z1, t_chan, gap_chan], dim=1)
            else:
                res_in = torch.cat([z_t, z0, z1, t_chan], dim=1)
            z_t = z_t + self.residual(res_in)
        return z_t

    def interpolate_pair(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        alpha: torch.Tensor,
        gap: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flow01, flow10, mask_a, mask_b, uncertainty = self.predict_flow(z0, z1, gap=gap)
        z_t = self.blend_from_flow(z0, z1, alpha, flow01, flow10, mask_a, mask_b, gap=gap)
        return z_t, uncertainty

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
                gap_val = torch.tensor([gap / max(T - 1, 1)], device=latents.device, dtype=latents.dtype)
                flow01, flow10, mask_a, mask_b, uncertainty = self.predict_flow(z0, z1, gap=gap_val)
                steps = gap - 1
                alpha = torch.linspace(1, steps, steps, device=latents.device, dtype=latents.dtype) / float(gap)
                alpha = alpha.view(-1, 1, 1, 1)
                z0_rep = z0.expand(steps, -1, -1, -1)
                z1_rep = z1.expand(steps, -1, -1, -1)
                flow01_rep = flow01.expand(steps, -1, -1, -1) * alpha
                flow10_rep = flow10.expand(steps, -1, -1, -1) * (1.0 - alpha)
                mask_a_rep = mask_a.expand(steps, -1, -1, -1)
                mask_b_rep = mask_b.expand(steps, -1, -1, -1)
                gap_rep = gap_val.expand(steps)
                z_t = self.blend_from_flow(
                    z0_rep, z1_rep, alpha, flow01_rep, flow10_rep, mask_a_rep, mask_b_rep, gap=gap_rep
                )
                out[b, t0 + 1 : t1] = z_t
                conf[b, t0 + 1 : t1] = (1.0 - uncertainty[0, 0]).clamp(0.0, 1.0)

            first = int(idx_b[0].item())
            last = int(idx_b[-1].item())
            if first > 0:
                out[b, :first] = anchors[0].unsqueeze(0).expand(first, -1, -1, -1)
                conf[b, :first] = conf[b, first : first + 1]
            if last < T - 1:
                out[b, last + 1 :] = anchors[-1].unsqueeze(0).expand(T - last - 1, -1, -1, -1)
                conf[b, last + 1 :] = conf[b, last : last + 1]

        return out, conf


def load_latent_flow_interpolator(
    ckpt_path: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
) -> Tuple[LatentFlowInterpolator, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    in_channels = int(meta.get("in_channels", 4))
    base_channels = int(meta.get("base_channels", 32))
    max_flow = float(meta.get("max_flow", 20.0))
    residual_blocks = int(meta.get("residual_blocks", 0))
    residual_channels = int(meta.get("residual_channels", base_channels))
    time_mask = bool(meta.get("time_mask", False))
    gap_cond = bool(meta.get("gap_cond", False))
    cost_volume = bool(meta.get("cost_volume", False))
    cv_radius = int(meta.get("cv_radius", 2))
    cv_downscale = int(meta.get("cv_downscale", 2))
    cv_norm = bool(meta.get("cv_norm", True))
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
    model = LatentFlowInterpolator(
        in_channels,
        base_channels=base_channels,
        max_flow=max_flow,
        residual_channels=residual_channels,
        residual_blocks=residual_blocks,
        time_mask=time_mask,
        gap_cond=gap_cond,
        cost_volume=cost_volume,
        cv_radius=cv_radius,
        cv_downscale=cv_downscale,
        cv_norm=cv_norm,
    )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=residual_blocks > 0)
    if device is not None or dtype is not None:
        model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, meta


__all__ = ["LatentFlowInterpolator", "LatentFlowPredictor", "load_latent_flow_interpolator"]
