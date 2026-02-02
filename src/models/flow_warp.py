from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAFTFlowEstimator(nn.Module):
    def __init__(
        self,
        variant: str = "large",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        from torchvision.models.optical_flow import (  # type: ignore
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if variant == "large":
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=True)
        elif variant == "small":
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=True)
        else:
            raise ValueError(f"unknown raft variant {variant}")

        self.model = model
        self.preprocess = weights.transforms()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # img1/img2: [B,3,H,W] in [0,1]
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        img1, img2 = self.preprocess(img1, img2)
        flows = self.model(img1, img2)
        return flows[-1]


def _resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    b, c, h, w = flow.shape
    new_h, new_w = size
    scale_y = float(new_h) / float(h)
    scale_x = float(new_w) / float(w)
    flow = F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=False)
    flow[:, 0] = flow[:, 0] * scale_x
    flow[:, 1] = flow[:, 1] * scale_y
    return flow


def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    # flow: [B,2,H,W] in pixels
    b, _, h, w = flow.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=-1).to(dtype=flow.dtype)
    base = base.unsqueeze(0).expand(b, -1, -1, -1)
    flow_t = flow.permute(0, 2, 3, 1)
    grid = base + flow_t
    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1).to(dtype=flow.dtype)


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W], flow: [B,2,H,W]
    grid = _flow_to_grid(flow).to(dtype=x.dtype)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _flow_confidence(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, sigma: float) -> torch.Tensor:
    # forward-backward consistency at time of flow_fwd
    flow_bwd_warp = _warp(flow_bwd, flow_fwd)
    fb = flow_fwd + flow_bwd_warp
    err = torch.norm(fb, dim=1, keepdim=True)
    conf = torch.exp(-(err * err) / max(float(sigma) ** 2, 1e-6))
    return conf


class FlowWarpInterpolator:
    def __init__(
        self,
        flow_estimator: RAFTFlowEstimator,
        vae: nn.Module,
        frame_size: int,
        latent_downsample: int = 8,
        conf_sigma: float = 1.0,
    ) -> None:
        self.flow_estimator = flow_estimator
        self.vae = vae
        self.frame_size = int(frame_size)
        self.latent_downsample = int(latent_downsample)
        self.conf_sigma = float(conf_sigma)

    @torch.no_grad()
    def interpolate(self, latents: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # latents: [B,T,C,Hl,Wl], idx: [B,K]
        if latents.dim() != 5:
            raise ValueError("latents must have shape [B,T,C,H,W]")
        bsz, T, _, H_l, W_l = latents.shape
        out = torch.zeros_like(latents)
        conf = torch.zeros((bsz, T, H_l, W_l), device=latents.device, dtype=latents.dtype)

        for b in range(bsz):
            idx_b = idx[b].clone()
            idx_b, _ = torch.sort(idx_b)
            anchors = latents[b, idx_b]  # [K,C,Hl,Wl]
            rgb = self.vae.decode(anchors)  # [K,3,H,W]
            rgb = rgb.to(self.flow_estimator.device, dtype=torch.float32)

            # default fill with nearest anchor
            out[b] = anchors[0].unsqueeze(0).expand(T, -1, -1, -1)
            conf[b] = 1.0

            for k in range(idx_b.shape[0] - 1):
                t0 = int(idx_b[k].item())
                t1 = int(idx_b[k + 1].item())
                if t1 <= t0:
                    continue
                rgb0 = rgb[k : k + 1]
                rgb1 = rgb[k + 1 : k + 2]
                flow01 = self.flow_estimator(rgb0, rgb1)
                flow10 = self.flow_estimator(rgb1, rgb0)
                conf0 = _flow_confidence(flow01, flow10, self.conf_sigma)
                conf1 = _flow_confidence(flow10, flow01, self.conf_sigma)

                flow01_l = _resize_flow(flow01, (H_l, W_l))
                flow10_l = _resize_flow(flow10, (H_l, W_l))
                conf0_l = F.interpolate(conf0, size=(H_l, W_l), mode="bilinear", align_corners=False).to(
                    dtype=latents.dtype
                )
                conf1_l = F.interpolate(conf1, size=(H_l, W_l), mode="bilinear", align_corners=False).to(
                    dtype=latents.dtype
                )

                z0 = anchors[k : k + 1]
                z1 = anchors[k + 1 : k + 2]
                out[b, t0] = z0
                out[b, t1] = z1
                conf[b, t0] = conf0_l[0]
                conf[b, t1] = conf1_l[0]

                for t in range(t0 + 1, t1):
                    alpha = float(t - t0) / float(t1 - t0)
                    flow0 = -alpha * flow01_l
                    flow1 = -(1.0 - alpha) * flow10_l
                    z0_w = _warp(z0, flow0.to(dtype=latents.dtype))
                    z1_w = _warp(z1, flow1.to(dtype=latents.dtype))
                    w0 = (1.0 - alpha) * conf0_l
                    w1 = alpha * conf1_l
                    denom = w0 + w1 + 1e-6
                    out[b, t] = (w0 * z0_w + w1 * z1_w) / denom
                    conf[b, t] = (1.0 - alpha) * conf0_l[0] + alpha * conf1_l[0]

            # fill outside endpoints if needed
            first = int(idx_b[0].item())
            last = int(idx_b[-1].item())
            if first > 0:
                out[b, :first] = anchors[0].unsqueeze(0).expand(first, -1, -1, -1)
                conf[b, :first] = conf[b, first : first + 1]
            if last < T - 1:
                out[b, last + 1 :] = anchors[-1].unsqueeze(0).expand(T - last - 1, -1, -1, -1)
                conf[b, last + 1 :] = conf[b, last : last + 1]

        return out, conf


__all__ = ["RAFTFlowEstimator", "FlowWarpInterpolator"]
