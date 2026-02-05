from __future__ import annotations

from typing import Iterable, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.video_tokens import patchify_latents


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
    if flow.dtype != x.dtype:
        flow = flow.to(dtype=x.dtype)
    grid = _flow_to_grid(flow)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _sinkhorn_log(logits: torch.Tensor, iters: int) -> torch.Tensor:
    logp = logits
    for _ in range(int(iters)):
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
        logp = logp - torch.logsumexp(logp, dim=0, keepdim=True)
    return logp


class SinkhornWarpInterpolator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 4,
        win_size: int = 5,
        angles_deg: Optional[Iterable[float]] = None,
        shift_range: int = 4,
        sinkhorn_iters: int = 20,
        sinkhorn_tau: float = 0.05,
        dustbin_logit: float = -2.0,
        d_match: int = 0,
        straightener: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.win_size = int(win_size)
        if angles_deg is None:
            angles_deg = (-10.0, -5.0, 0.0, 5.0, 10.0)
        angles = [float(a) * math.pi / 180.0 for a in angles_deg]
        self.register_buffer("angles", torch.tensor(angles, dtype=torch.float32), persistent=False)
        self.shift_range = int(shift_range)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tau = float(sinkhorn_tau)
        self.dustbin_logit = float(dustbin_logit)
        self.d_match = int(d_match)
        self.straightener = straightener
        if self.straightener is not None:
            self.straightener.eval()
            for p in self.straightener.parameters():
                p.requires_grad_(False)

    def _token_features(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # z: [B,C,H,W]
        if self.straightener is not None:
            with torch.no_grad():
                z = self.straightener.encode(z)
        tokens, (hp, wp) = patchify_latents(z.unsqueeze(1), self.patch_size)
        tokens = tokens[:, 0]  # [B,N,D]
        tokens = tokens.float()
        B, N, D = tokens.shape
        if self.d_match > 0 and self.d_match < D:
            if D % self.d_match != 0:
                raise ValueError(f"d_match {self.d_match} must divide token dim {D}")
            tokens = tokens.view(B, N, self.d_match, D // self.d_match).mean(dim=-1)
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        feats = tokens.view(B, hp, wp, -1)
        return feats, hp, wp

    def _apply_se2(self, feats: torch.Tensor, theta: float, dx: int, dy: int) -> torch.Tensor:
        # feats: [B,Hp,Wp,D]
        B, Hp, Wp, D = feats.shape
        feat_map = feats.permute(0, 3, 1, 2)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        tx = 0.0 if Wp <= 1 else 2.0 * float(dx) / float(Wp - 1)
        ty = 0.0 if Hp <= 1 else 2.0 * float(dy) / float(Hp - 1)
        mat = torch.tensor(
            [[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], device=feat_map.device, dtype=feat_map.dtype
        )
        mat = mat.unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(mat, size=feat_map.shape, align_corners=True)
        warped = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped.permute(0, 2, 3, 1)

    def _estimate_global_se2(self, f0: torch.Tensor, f1: torch.Tensor) -> Tuple[float, int, int]:
        # f0/f1: [1,Hp,Wp,D]
        best_score = None
        best_theta = 0.0
        best_dx = 0
        best_dy = 0
        for theta in self.angles.tolist():
            for dy in range(-self.shift_range, self.shift_range + 1):
                for dx in range(-self.shift_range, self.shift_range + 1):
                    f1_aligned = self._apply_se2(f1, theta, dx, dy)
                    score = (f0 * f1_aligned).mean()
                    score_val = float(score.item())
                    if best_score is None or score_val > best_score:
                        best_score = score_val
                        best_theta = float(theta)
                        best_dx = int(dx)
                        best_dy = int(dy)
        return best_theta, best_dx, best_dy

    def _local_sinkhorn_delta(self, f0: torch.Tensor, f1: torch.Tensor, hp: int, wp: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # f0/f1: [1,Hp,Wp,D]
        delta = torch.zeros((hp, wp, 2), device=f0.device, dtype=torch.float32)
        conf = torch.zeros((hp, wp), device=f0.device, dtype=torch.float32)
        win = self.win_size
        for y0 in range(0, hp, win):
            for x0 in range(0, wp, win):
                h = min(win, hp - y0)
                w = min(win, wp - x0)
                if h <= 0 or w <= 0:
                    continue
                x = f0[0, y0 : y0 + h, x0 : x0 + w].reshape(h * w, -1)
                y = f1[0, y0 : y0 + h, x0 : x0 + w].reshape(h * w, -1)
                n = x.shape[0]
                if n == 0:
                    continue
                logits = (x @ y.T) / math.sqrt(max(1.0, float(x.shape[1])))
                logits = logits / max(self.sinkhorn_tau, 1e-6)
                logp = torch.full((n + 1, n + 1), self.dustbin_logit, device=logits.device, dtype=logits.dtype)
                logp[:n, :n] = logits
                logp = _sinkhorn_log(logp, self.sinkhorn_iters)
                p = torch.exp(logp)
                p_xy = p[:n, :n]
                mass = p_xy.sum(dim=1, keepdim=True).clamp_min(1e-8)
                yy, xx = torch.meshgrid(
                    torch.arange(h, device=logits.device),
                    torch.arange(w, device=logits.device),
                    indexing="ij",
                )
                coords = torch.stack([xx, yy], dim=-1).view(n, 2).float()
                q = (p_xy @ coords) / mass
                delta_block = (q - coords).view(h, w, 2)
                conf_block = (1.0 - p[:n, n]).view(h, w)
                delta[y0 : y0 + h, x0 : x0 + w] = delta_block
                conf[y0 : y0 + h, x0 : x0 + w] = conf_block
        return delta, conf

    def _compose_flow(
        self,
        delta: torch.Tensor,
        theta: float,
        dx: int,
        dy: int,
        hp: int,
        wp: int,
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(hp, device=delta.device),
            torch.arange(wp, device=delta.device),
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1).float()
        center = torch.tensor([(wp - 1) / 2.0, (hp - 1) / 2.0], device=delta.device, dtype=coords.dtype)
        coords_centered = coords - center
        v = coords_centered + delta
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        qx = cos_t * v[..., 0] - sin_t * v[..., 1]
        qy = sin_t * v[..., 0] + cos_t * v[..., 1]
        q = torch.stack([qx, qy], dim=-1) + center + torch.tensor([dx, dy], device=delta.device, dtype=coords.dtype)
        flow = q - coords
        return flow

    def _compute_flow(self, z0: torch.Tensor, z1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f0, hp, wp = self._token_features(z0)
        f1, _, _ = self._token_features(z1)
        theta, dx, dy = self._estimate_global_se2(f0, f1)
        f1_aligned = self._apply_se2(f1, theta, dx, dy)
        delta, conf = self._local_sinkhorn_delta(f0, f1_aligned, hp, wp)
        flow_tok = self._compose_flow(delta, theta, dx, dy, hp, wp)
        return flow_tok, conf

    @torch.no_grad()
    def interpolate(self, latents: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # latents: [B,T,C,H,W], idx: [B,K]
        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T, _, H, W = latents.shape
        if idx.dim() != 2 or idx.shape[0] != B:
            raise ValueError("idx must be [B,K]")
        out = torch.zeros_like(latents)
        conf = torch.zeros((B, T, H, W), device=latents.device, dtype=latents.dtype)
        for b in range(B):
            idx_b = idx[b].clone()
            idx_b, _ = torch.sort(idx_b)
            anchors = latents[b, idx_b]
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
                flow01_tok, conf_tok = self._compute_flow(z0, z1)
                flow10_tok, _ = self._compute_flow(z1, z0)
                flow01_tok = flow01_tok.permute(2, 0, 1).unsqueeze(0)  # [1,2,Hp,Wp]
                flow10_tok = flow10_tok.permute(2, 0, 1).unsqueeze(0)
                flow01 = F.interpolate(flow01_tok, size=(H, W), mode="bilinear", align_corners=True) * float(
                    self.patch_size
                )
                flow10 = F.interpolate(flow10_tok, size=(H, W), mode="bilinear", align_corners=True) * float(
                    self.patch_size
                )
                conf_lat = F.interpolate(conf_tok.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True)
                conf_lat = conf_lat.clamp(0.0, 1.0)

                steps = gap - 1
                alpha = torch.linspace(1, steps, steps, device=latents.device, dtype=latents.dtype) / float(gap)
                alpha = alpha.view(-1, 1, 1, 1)
                z0_rep = z0.expand(steps, -1, -1, -1)
                z1_rep = z1.expand(steps, -1, -1, -1)
                flow01_rep = flow01.expand(steps, -1, -1, -1) * alpha
                flow10_rep = flow10.expand(steps, -1, -1, -1) * (1.0 - alpha)
                conf_rep = conf_lat.expand(steps, -1, -1, -1)

                z0_w = _warp(z0_rep, -flow01_rep)
                z1_w = _warp(z1_rep, -flow10_rep)
                w0 = (1.0 - alpha) * conf_rep
                w1 = alpha * conf_rep
                denom = w0 + w1 + 1e-6
                z_t = (w0 * z0_w + w1 * z1_w) / denom
                out[b, t0 + 1 : t1] = z_t
                conf[b, t0 + 1 : t1] = conf_lat[0, 0]

            first = int(idx_b[0].item())
            last = int(idx_b[-1].item())
            if first > 0:
                out[b, :first] = anchors[0].unsqueeze(0).expand(first, -1, -1, -1)
                conf[b, :first] = conf[b, first : first + 1]
            if last < T - 1:
                out[b, last + 1 :] = anchors[-1].unsqueeze(0).expand(T - last - 1, -1, -1, -1)
                conf[b, last + 1 :] = conf[b, last : last + 1]
        return out, conf


__all__ = ["SinkhornWarpInterpolator"]
