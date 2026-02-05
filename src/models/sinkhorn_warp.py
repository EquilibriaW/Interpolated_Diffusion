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


def _sinkhorn_log_batch(logits: torch.Tensor, iters: int) -> torch.Tensor:
    # logits: [B, N, N]
    logp = logits
    for _ in range(int(iters)):
        logp = logp - torch.logsumexp(logp, dim=2, keepdim=True)
        logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
    return logp


class SinkhornWarpInterpolator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 4,
        win_size: int = 5,
        angles_deg: Optional[Iterable[float]] = None,
        shift_range: int = 4,
        se2_chunk: int = 64,
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
        self.se2_chunk = int(se2_chunk)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tau = float(sinkhorn_tau)
        self.dustbin_logit = float(dustbin_logit)
        self.d_match = int(d_match)
        self.straightener = straightener
        self._se2_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
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

    def _get_se2_candidates(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._se2_cache.get(key)
        if cached is not None:
            return cached
        angles = self.angles.to(device=device, dtype=dtype)
        shifts = torch.arange(-self.shift_range, self.shift_range + 1, device=device, dtype=angles.dtype)
        theta_grid, dy_grid, dx_grid = torch.meshgrid(angles, shifts, shifts, indexing="ij")
        theta = theta_grid.reshape(-1)
        dx = dx_grid.reshape(-1).to(torch.long)
        dy = dy_grid.reshape(-1).to(torch.long)
        self._se2_cache[key] = (theta, dx, dy)
        return theta, dx, dy

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

    def _apply_se2_per_sample(
        self, feats: torch.Tensor, theta: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
    ) -> torch.Tensor:
        # feats: [B,Hp,Wp,D], theta/dx/dy: [B]
        B, Hp, Wp, _ = feats.shape
        feat_map = feats.permute(0, 3, 1, 2)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        tx = torch.zeros_like(theta) if Wp <= 1 else 2.0 * dx.to(theta.dtype) / float(Wp - 1)
        ty = torch.zeros_like(theta) if Hp <= 1 else 2.0 * dy.to(theta.dtype) / float(Hp - 1)
        mat = torch.zeros((B, 2, 3), device=feat_map.device, dtype=feat_map.dtype)
        mat[:, 0, 0] = cos_t
        mat[:, 0, 1] = -sin_t
        mat[:, 0, 2] = tx
        mat[:, 1, 0] = sin_t
        mat[:, 1, 1] = cos_t
        mat[:, 1, 2] = ty
        grid = F.affine_grid(mat, size=feat_map.shape, align_corners=True)
        warped = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped.permute(0, 2, 3, 1)

    def _estimate_global_se2_batch(self, f0: torch.Tensor, f1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # f0/f1: [B,Hp,Wp,D]
        device = f0.device
        B, Hp, Wp, _ = f0.shape
        f0_map = f0.permute(0, 3, 1, 2)
        f1_map = f1.permute(0, 3, 1, 2)
        # Use a low-channel score map for global SE(2) to keep memory bounded.
        f0_score = f0_map.mean(dim=1, keepdim=True)
        f1_score = f1_map.mean(dim=1, keepdim=True)

        theta_all, dx_all, dy_all = self._get_se2_candidates(device=device, dtype=f0.dtype)
        num_cand = int(theta_all.numel())
        best_score = torch.full((B,), -float("inf"), device=device, dtype=f0.dtype)
        best_idx = torch.zeros((B,), device=device, dtype=torch.long)

        chunk = max(1, int(self.se2_chunk))
        for start in range(0, num_cand, chunk):
            end = min(num_cand, start + chunk)
            theta = theta_all[start:end]
            dx = dx_all[start:end]
            dy = dy_all[start:end]
            csz = int(theta.numel())

            theta_bc = theta.view(1, csz).expand(B, csz)
            dx_bc = dx.view(1, csz).expand(B, csz)
            dy_bc = dy.view(1, csz).expand(B, csz)

            cos_t = torch.cos(theta_bc)
            sin_t = torch.sin(theta_bc)
            tx = torch.zeros_like(theta_bc) if Wp <= 1 else 2.0 * dx_bc.to(theta_bc.dtype) / float(Wp - 1)
            ty = torch.zeros_like(theta_bc) if Hp <= 1 else 2.0 * dy_bc.to(theta_bc.dtype) / float(Hp - 1)

            mat = torch.zeros((B, csz, 2, 3), device=device, dtype=f0.dtype)
            mat[:, :, 0, 0] = cos_t
            mat[:, :, 0, 1] = -sin_t
            mat[:, :, 0, 2] = tx
            mat[:, :, 1, 0] = sin_t
            mat[:, :, 1, 1] = cos_t
            mat[:, :, 1, 2] = ty

            mat = mat.view(B * csz, 2, 3)
            f1_rep = f1_score.unsqueeze(1).expand(B, csz, -1, -1, -1).reshape(B * csz, 1, Hp, Wp)
            grid = F.affine_grid(mat, size=f1_rep.shape, align_corners=True)
            f1_aligned = F.grid_sample(
                f1_rep, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(B, csz, 1, Hp, Wp)

            score = (f1_aligned * f0_score.unsqueeze(1)).mean(dim=(2, 3, 4))
            score_chunk, idx_chunk = score.max(dim=1)
            better = score_chunk > best_score
            best_score = torch.where(better, score_chunk, best_score)
            best_idx = torch.where(better, idx_chunk + start, best_idx)

        best_theta = theta_all[best_idx]
        best_dx = dx_all[best_idx]
        best_dy = dy_all[best_idx]
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

    def _local_sinkhorn_delta_batch(
        self, f0: torch.Tensor, f1: torch.Tensor, hp: int, wp: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # f0/f1: [B,Hp,Wp,D]
        B = f0.shape[0]
        delta = torch.zeros((B, hp, wp, 2), device=f0.device, dtype=torch.float32)
        conf = torch.zeros((B, hp, wp), device=f0.device, dtype=torch.float32)
        win = self.win_size

        def process_block(y0: int, x0: int, h: int, w: int) -> None:
            if h <= 0 or w <= 0:
                return
            x = f0[:, y0 : y0 + h, x0 : x0 + w].reshape(B, h * w, -1)
            y = f1[:, y0 : y0 + h, x0 : x0 + w].reshape(B, h * w, -1)
            n = x.shape[1]
            if n == 0:
                return
            logits = torch.bmm(x, y.transpose(1, 2)) / math.sqrt(max(1.0, float(x.shape[2])))
            logits = logits / max(self.sinkhorn_tau, 1e-6)
            logp = torch.full(
                (B, n + 1, n + 1),
                self.dustbin_logit,
                device=logits.device,
                dtype=logits.dtype,
            )
            logp[:, :n, :n] = logits
            logp = _sinkhorn_log_batch(logp, self.sinkhorn_iters)
            p = torch.exp(logp)
            p_xy = p[:, :n, :n]
            mass = p_xy.sum(dim=2, keepdim=True).clamp_min(1e-8)
            yy, xx = torch.meshgrid(
                torch.arange(h, device=logits.device),
                torch.arange(w, device=logits.device),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=-1).view(n, 2).float()
            coords = coords.unsqueeze(0).expand(B, -1, -1)
            q = torch.bmm(p_xy, coords) / mass
            delta_block = (q - coords).view(B, h, w, 2)
            conf_block = (1.0 - p[:, :n, n]).view(B, h, w)
            delta[:, y0 : y0 + h, x0 : x0 + w] = delta_block
            conf[:, y0 : y0 + h, x0 : x0 + w] = conf_block

        nH = hp // win
        nW = wp // win

        if nH > 0 and nW > 0:
            h_main = nH * win
            w_main = nW * win
            f0_main = f0[:, :h_main, :w_main]
            f1_main = f1[:, :h_main, :w_main]
            D = f0_main.shape[-1]

            f0_blk = (
                f0_main.view(B, nH, win, nW, win, D)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(B * nH * nW, win * win, D)
            )
            f1_blk = (
                f1_main.view(B, nH, win, nW, win, D)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(B * nH * nW, win * win, D)
            )

            logits = torch.bmm(f0_blk, f1_blk.transpose(1, 2)) / math.sqrt(max(1.0, float(D)))
            logits = logits / max(self.sinkhorn_tau, 1e-6)
            logp = torch.full(
                (B * nH * nW, win * win + 1, win * win + 1),
                self.dustbin_logit,
                device=logits.device,
                dtype=logits.dtype,
            )
            logp[:, : win * win, : win * win] = logits
            logp = _sinkhorn_log_batch(logp, self.sinkhorn_iters)
            p = torch.exp(logp)
            p_xy = p[:, : win * win, : win * win]
            mass = p_xy.sum(dim=2, keepdim=True).clamp_min(1e-8)
            yy, xx = torch.meshgrid(
                torch.arange(win, device=logits.device),
                torch.arange(win, device=logits.device),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=-1).view(win * win, 2).float()
            coords = coords.unsqueeze(0).expand(B * nH * nW, -1, -1)
            q = torch.bmm(p_xy, coords) / mass
            delta_blk = (q - coords).view(B, nH, nW, win, win, 2)
            conf_blk = (1.0 - p[:, : win * win, win * win]).view(B, nH, nW, win, win)

            delta_blk = (
                delta_blk.permute(0, 1, 3, 2, 4, 5)
                .reshape(B, h_main, w_main, 2)
            )
            conf_blk = conf_blk.permute(0, 1, 3, 2, 4).reshape(B, h_main, w_main)
            delta[:, :h_main, :w_main] = delta_blk
            conf[:, :h_main, :w_main] = conf_blk

        rem_h = hp - nH * win
        rem_w = wp - nW * win
        if rem_w > 0 and nH > 0:
            x0 = nW * win
            for y0 in range(0, nH * win, win):
                process_block(y0, x0, win, rem_w)
        if rem_h > 0 and nW > 0:
            y0 = nH * win
            for x0 in range(0, nW * win, win):
                process_block(y0, x0, rem_h, win)
        if rem_h > 0 and rem_w > 0:
            process_block(nH * win, nW * win, rem_h, rem_w)

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

    def _compose_flow_batch(
        self,
        delta: torch.Tensor,
        theta: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
        hp: int,
        wp: int,
    ) -> torch.Tensor:
        # delta: [B,Hp,Wp,2], theta/dx/dy: [B]
        B = delta.shape[0]
        y, x = torch.meshgrid(
            torch.arange(hp, device=delta.device),
            torch.arange(wp, device=delta.device),
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1).float()
        center = torch.tensor([(wp - 1) / 2.0, (hp - 1) / 2.0], device=delta.device, dtype=coords.dtype)
        coords_centered = coords - center
        v = coords_centered.unsqueeze(0) + delta
        cos_t = torch.cos(theta).view(B, 1, 1)
        sin_t = torch.sin(theta).view(B, 1, 1)
        qx = cos_t * v[..., 0] - sin_t * v[..., 1]
        qy = sin_t * v[..., 0] + cos_t * v[..., 1]
        q = torch.stack([qx, qy], dim=-1)
        trans = torch.stack([dx.to(q.dtype), dy.to(q.dtype)], dim=-1).view(B, 1, 1, 2)
        q = q + center + trans
        flow = q - coords
        return flow

    def _compute_flow_from_feats_batch(self, f0: torch.Tensor, f1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # f0/f1: [B,Hp,Wp,D]
        hp, wp = f0.shape[1], f0.shape[2]
        theta, dx, dy = self._estimate_global_se2_batch(f0, f1)
        f1_aligned = self._apply_se2_per_sample(f1, theta, dx, dy)
        delta, conf = self._local_sinkhorn_delta_batch(f0, f1_aligned, hp, wp)
        flow_tok = self._compose_flow_batch(delta, theta, dx, dy, hp, wp)
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
        # precompute token features for all frames
        latents_flat = latents.reshape(B * T, latents.shape[2], H, W)
        feats, hp, wp = self._token_features(latents_flat)
        feats = feats.view(B, T, hp, wp, -1)

        pair_meta = []
        idx_sorted = []
        for b in range(B):
            idx_b = idx[b].clone()
            idx_b, _ = torch.sort(idx_b)
            idx_sorted.append(idx_b)
            anchors = latents[b, idx_b]
            out[b, idx_b] = anchors
            conf[b, idx_b] = 1.0

            first = int(idx_b[0].item())
            last = int(idx_b[-1].item())
            if first > 0:
                out[b, :first] = anchors[0].unsqueeze(0).expand(first, -1, -1, -1)
                conf[b, :first] = conf[b, first : first + 1]
            if last < T - 1:
                out[b, last + 1 :] = anchors[-1].unsqueeze(0).expand(T - last - 1, -1, -1, -1)
                conf[b, last + 1 :] = conf[b, last : last + 1]

        idx_sorted = torch.stack(idx_sorted, dim=0)
        t0 = idx_sorted[:, :-1]
        t1 = idx_sorted[:, 1:]
        gap = t1 - t0
        pair_mask = gap > 1
        if torch.any(pair_mask):
            pair_b, pair_k = torch.nonzero(pair_mask, as_tuple=True)
            pair_t0 = t0[pair_b, pair_k]
            pair_t1 = t1[pair_b, pair_k]
        else:
            pair_b = pair_k = pair_t0 = pair_t1 = None

        if pair_b is None or pair_b.numel() == 0:
            return out, conf

        f0 = feats[pair_b, pair_t0]
        f1 = feats[pair_b, pair_t1]

        flow01_tok, conf01_tok = self._compute_flow_from_feats_batch(f0, f1)
        flow10_tok, conf10_tok = self._compute_flow_from_feats_batch(f1, f0)

        flow01_tok = flow01_tok.permute(0, 3, 1, 2)  # [P,2,Hp,Wp]
        flow10_tok = flow10_tok.permute(0, 3, 1, 2)
        flow01 = F.interpolate(flow01_tok, size=(H, W), mode="bilinear", align_corners=True) * float(self.patch_size)
        flow10 = F.interpolate(flow10_tok, size=(H, W), mode="bilinear", align_corners=True) * float(self.patch_size)
        conf01_lat = F.interpolate(conf01_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True)
        conf10_lat = F.interpolate(conf10_tok.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True)
        conf01_lat = conf01_lat.clamp(0.0, 1.0)
        conf10_lat = conf10_lat.clamp(0.0, 1.0)

        for p_idx in range(pair_b.numel()):
            b = int(pair_b[p_idx].item())
            t0 = int(pair_t0[p_idx].item())
            t1 = int(pair_t1[p_idx].item())
            z0 = latents[b : b + 1, t0]
            z1 = latents[b : b + 1, t1]
            flow01_p = flow01[p_idx : p_idx + 1]
            flow10_p = flow10[p_idx : p_idx + 1]
            conf01_p = conf01_lat[p_idx : p_idx + 1]
            conf10_p = conf10_lat[p_idx : p_idx + 1]
            conf_mix = torch.minimum(conf01_p, conf10_p)

            gap = t1 - t0
            steps = gap - 1
            alpha = torch.linspace(1, steps, steps, device=latents.device, dtype=latents.dtype) / float(gap)
            alpha = alpha.view(-1, 1, 1, 1)
            z0_rep = z0.expand(steps, -1, -1, -1)
            z1_rep = z1.expand(steps, -1, -1, -1)
            flow01_rep = flow01_p.expand(steps, -1, -1, -1) * alpha
            flow10_rep = flow10_p.expand(steps, -1, -1, -1) * (1.0 - alpha)
            conf01_rep = conf01_p.expand(steps, -1, -1, -1)
            conf10_rep = conf10_p.expand(steps, -1, -1, -1)

            z0_w = _warp(z0_rep, -flow01_rep)
            z1_w = _warp(z1_rep, -flow10_rep)
            w0 = (1.0 - alpha) * conf01_rep
            w1 = alpha * conf10_rep
            denom = w0 + w1 + 1e-6
            z_t = (w0 * z0_w + w1 * z1_w) / denom
            out[b, t0 + 1 : t1] = z_t
            conf[b, t0 + 1 : t1] = conf_mix[0, 0]
        return out, conf


__all__ = ["SinkhornWarpInterpolator"]
