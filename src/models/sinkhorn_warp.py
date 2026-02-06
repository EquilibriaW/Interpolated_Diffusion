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
        win_stride: int = 0,
        global_mode: str = "se2",
        phasecorr_mode: str = "mean",
        phasecorr_level: str = "token",
        angles_deg: Optional[Iterable[float]] = None,
        shift_range: int = 4,
        se2_chunk: int = 64,
        sinkhorn_iters: int = 20,
        sinkhorn_tau: float = 0.05,
        dustbin_logit: float = -2.0,
        spatial_gamma: float = 0.0,
        spatial_radius: int = 0,
        fb_sigma: float = 0.0,
        d_match: int = 0,
        proj_mode: str = "groupmean",
        learn_tau: bool = False,
        learn_dustbin: bool = False,
        tau_min: float = 1e-3,
        straightener: Optional[nn.Module] = None,
        warp_space: str = "z",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.win_size = int(win_size)
        self.win_stride = int(win_stride) if int(win_stride) > 0 else int(win_size)
        if self.win_stride <= 0:
            raise ValueError("win_stride must be >= 1")
        self.global_mode = str(global_mode)
        if self.global_mode not in ("se2", "phasecorr", "none"):
            raise ValueError(f"global_mode must be one of se2/phasecorr/none, got {self.global_mode}")
        self.phasecorr_mode = str(phasecorr_mode)
        if self.phasecorr_mode not in ("mean", "multi"):
            raise ValueError(f"phasecorr_mode must be 'mean' or 'multi', got {self.phasecorr_mode}")
        self.phasecorr_level = str(phasecorr_level)
        if self.phasecorr_level not in ("token", "latent"):
            raise ValueError(f"phasecorr_level must be 'token' or 'latent', got {self.phasecorr_level}")
        if angles_deg is None:
            angles_deg = (-10.0, -5.0, 0.0, 5.0, 10.0)
        angles = [float(a) * math.pi / 180.0 for a in angles_deg]
        # Keep a python copy to avoid device sync (tolist()) during phase correlation search.
        self._angles_list = list(angles)
        self.register_buffer("angles", torch.tensor(angles, dtype=torch.float32), persistent=False)
        self.shift_range = int(shift_range)
        self.se2_chunk = int(se2_chunk)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.learn_tau = bool(learn_tau)
        self.learn_dustbin = bool(learn_dustbin)
        self.tau_min = float(tau_min)
        if self.tau_min < 0.0:
            raise ValueError("tau_min must be >= 0")
        # Parameterize tau via softplus to keep it positive and avoid tau->0 instabilities.
        if self.learn_tau:
            init_tau = float(sinkhorn_tau)
            init_tau = max(init_tau - self.tau_min, 1e-6)
            init_raw = math.log(math.expm1(init_tau))  # inverse softplus
            self.tau_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        else:
            self.register_buffer(
                "tau_const", torch.tensor(float(sinkhorn_tau), dtype=torch.float32), persistent=False
            )
        if self.learn_dustbin:
            self.dustbin_param = nn.Parameter(torch.tensor(float(dustbin_logit), dtype=torch.float32))
        else:
            self.register_buffer(
                "dustbin_const", torch.tensor(float(dustbin_logit), dtype=torch.float32), persistent=False
            )
        self.spatial_gamma = float(spatial_gamma)
        self.spatial_radius = int(spatial_radius)
        self.fb_sigma = float(fb_sigma)
        self.d_match = int(d_match)
        self.proj_mode = str(proj_mode)
        if self.proj_mode not in ("none", "groupmean", "linear"):
            raise ValueError(f"proj_mode must be one of none/groupmean/linear, got {self.proj_mode}")
        token_dim = int(self.in_channels) * int(self.patch_size) * int(self.patch_size)
        self.token_dim = int(token_dim)
        self.token_proj: nn.Module | None = None
        if self.proj_mode == "linear":
            if self.d_match <= 0:
                raise ValueError("proj_mode='linear' requires d_match > 0")
            self.token_proj = nn.Linear(self.token_dim, self.d_match, bias=False)
        self.straightener = straightener
        self.warp_space = str(warp_space)
        if self.warp_space not in ("z", "s"):
            raise ValueError(f"warp_space must be 'z' or 's', got {self.warp_space}")
        if self.warp_space == "s" and self.straightener is None:
            raise ValueError("warp_space='s' requires a straightener")
        self._se2_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._dist_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}
        if self.straightener is not None:
            self.straightener.eval()
            for p in self.straightener.parameters():
                p.requires_grad_(False)

    def _tau(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.learn_tau:
            return (F.softplus(self.tau_raw) + self.tau_min).to(device=device, dtype=dtype)
        return self.tau_const.to(device=device, dtype=dtype)

    def _dustbin(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.learn_dustbin:
            return self.dustbin_param.to(device=device, dtype=dtype)
        return self.dustbin_const.to(device=device, dtype=dtype)

    def _spatial_dist2(self, h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Returns [n,n] squared distances between token coords within an h x w window.
        # Cache in float32 and cast at use sites.
        key = (device, int(h), int(w))
        cached = self._dist_cache.get(key)
        if cached is None:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=-1).reshape(h * w, 2).float()
            diff = coords[:, None, :] - coords[None, :, :]
            dist2 = (diff * diff).sum(dim=-1)  # [n,n]
            self._dist_cache[key] = dist2
            cached = dist2
        return cached.to(dtype=dtype)

    def _token_features(self, z: torch.Tensor, *, assume_straightened: bool = False) -> Tuple[torch.Tensor, int, int]:
        # z: [B,C,H,W]
        if self.straightener is not None and not assume_straightened:
            with torch.no_grad():
                z = self.straightener.encode(z)
        tokens, (hp, wp) = patchify_latents(z.unsqueeze(1), self.patch_size)
        tokens = tokens[:, 0]  # [B,N,D]
        tokens = tokens.float()
        B, N, D = tokens.shape
        if self.proj_mode == "linear":
            if self.token_proj is None:
                raise RuntimeError("token_proj missing for proj_mode='linear'")
            if D != self.token_dim:
                raise ValueError(f"token dim mismatch: expected {self.token_dim}, got {D}")
            tokens = self.token_proj(tokens)
        elif self.proj_mode == "groupmean":
            if self.d_match > 0 and self.d_match < D:
                if D % self.d_match != 0:
                    raise ValueError(f"d_match {self.d_match} must divide token dim {D}")
                tokens = tokens.view(B, N, self.d_match, D // self.d_match).mean(dim=-1)
        elif self.proj_mode == "none":
            pass
        else:
            raise RuntimeError(f"unknown proj_mode {self.proj_mode}")
        tokens = F.normalize(tokens, dim=-1, eps=1e-6)
        feats = tokens.view(B, hp, wp, -1)
        return feats, hp, wp

    def token_features(self, z: torch.Tensor, *, assume_straightened: bool = False) -> Tuple[torch.Tensor, int, int]:
        return self._token_features(z, assume_straightened=assume_straightened)

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

    def _apply_se2_map_per_sample(
        self, x: torch.Tensor, theta: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor
    ) -> torch.Tensor:
        # x: [B,C,H,W], theta/dx/dy: [B] where dx/dy are in pixel units for the HxW map.
        if x.dim() != 4:
            raise ValueError("_apply_se2_map_per_sample expects [B,C,H,W]")
        B, _, H, W = x.shape
        x = x.float()
        theta = theta.to(device=x.device, dtype=torch.float32)
        dx = dx.to(device=x.device, dtype=torch.float32)
        dy = dy.to(device=x.device, dtype=torch.float32)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        tx = torch.zeros_like(theta) if W <= 1 else 2.0 * dx / float(W - 1)
        ty = torch.zeros_like(theta) if H <= 1 else 2.0 * dy / float(H - 1)
        mat = torch.zeros((B, 2, 3), device=x.device, dtype=torch.float32)
        mat[:, 0, 0] = cos_t
        mat[:, 0, 1] = -sin_t
        mat[:, 0, 2] = tx
        mat[:, 1, 0] = sin_t
        mat[:, 1, 1] = cos_t
        mat[:, 1, 2] = ty
        grid = F.affine_grid(mat, size=x.shape, align_corners=True)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

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

    def _phasecorr_shift_batch(
        self, f0: torch.Tensor, f1: torch.Tensor, *, return_peak: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # f0/f1: [B,Hp,Wp], returns dx/dy in token units (float)
        B, Hp, Wp = f0.shape
        f0 = f0 - f0.mean(dim=(1, 2), keepdim=True)
        f1 = f1 - f1.mean(dim=(1, 2), keepdim=True)
        F0 = torch.fft.rfft2(f0)
        F1 = torch.fft.rfft2(f1)
        R = F0 * torch.conj(F1)
        R = R / (R.abs() + 1e-6)
        corr = torch.fft.irfft2(R, s=(Hp, Wp))
        corr_flat = corr.view(B, -1)
        peak, idx = corr_flat.max(dim=-1)
        dy = (idx // Wp).to(torch.long)
        dx = (idx % Wp).to(torch.long)
        dy = torch.where(dy > Hp // 2, dy - Hp, dy)
        dx = torch.where(dx > Wp // 2, dx - Wp, dx)
        if return_peak:
            return dx.to(dtype=f0.dtype), dy.to(dtype=f0.dtype), peak.to(dtype=f0.dtype)
        return dx.to(dtype=f0.dtype), dy.to(dtype=f0.dtype), None

    def _phasecorr_shift_multi_batch(
        self, f0: torch.Tensor, f1: torch.Tensor, *, return_peak: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # f0/f1: [B,C,Hp,Wp], returns dx/dy in token units (float)
        if f0.dim() != 4 or f1.dim() != 4:
            raise ValueError("phasecorr_shift_multi expects [B,C,H,W]")
        if f0.shape != f1.shape:
            raise ValueError("phasecorr_shift_multi expects f0 and f1 to have same shape")
        B, C, Hp, Wp = f0.shape
        # Remove per-channel means.
        f0 = f0 - f0.mean(dim=(2, 3), keepdim=True)
        f1 = f1 - f1.mean(dim=(2, 3), keepdim=True)
        F0 = torch.fft.rfft2(f0)
        F1 = torch.fft.rfft2(f1)
        # Sum cross-power across channels.
        R = (F0 * torch.conj(F1)).sum(dim=1)  # [B,Hp,Wp_rfft]
        R = R / (R.abs() + 1e-6)
        corr = torch.fft.irfft2(R, s=(Hp, Wp))
        corr_flat = corr.view(B, -1)
        peak, idx = corr_flat.max(dim=-1)
        dy = (idx // Wp).to(torch.long)
        dx = (idx % Wp).to(torch.long)
        dy = torch.where(dy > Hp // 2, dy - Hp, dy)
        dx = torch.where(dx > Wp // 2, dx - Wp, dx)
        if return_peak:
            return dx.to(dtype=f0.dtype), dy.to(dtype=f0.dtype), peak.to(dtype=f0.dtype)
        return dx.to(dtype=f0.dtype), dy.to(dtype=f0.dtype), None

    def _phasecorr_se2_batch(
        self, f0_score: torch.Tensor, f1_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # f0_score/f1_score: [B,Hp,Wp] (single-channel score maps)
        B, Hp, Wp = f0_score.shape
        best_score = torch.full((B,), -float("inf"), device=f0_score.device, dtype=f0_score.dtype)
        best_theta = torch.zeros((B,), device=f0_score.device, dtype=f0_score.dtype)
        best_dx = torch.zeros((B,), device=f0_score.device, dtype=f0_score.dtype)
        best_dy = torch.zeros((B,), device=f0_score.device, dtype=f0_score.dtype)

        for angle in self._angles_list:
            theta = torch.full((B,), float(angle), device=f0_score.device, dtype=f0_score.dtype)
            zeros = torch.zeros_like(theta)
            f1_rot = self._apply_se2_per_sample(f1_score.unsqueeze(-1), theta, zeros, zeros).squeeze(-1)
            # Phase correlation returns the shift `s` such that rolling `f1_rot` by `s` aligns it to `f0_score`.
            # Our SE(2) warp uses the mapping x_in = R x_out + t (grid_sample samples input at that coordinate),
            # so to apply a roll-shift `s` in output coordinates we need t = -R s.
            dx_s, dy_s, peak = self._phasecorr_shift_batch(f0_score, f1_rot, return_peak=True)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            dx = -(cos_t * dx_s - sin_t * dy_s)
            dy = -(sin_t * dx_s + cos_t * dy_s)
            better = peak > best_score
            best_score = torch.where(better, peak, best_score)
            best_theta = torch.where(better, theta, best_theta)
            best_dx = torch.where(better, dx, best_dx)
            best_dy = torch.where(better, dy, best_dy)

        return best_theta, best_dx, best_dy

    def _phasecorr_se2_multi_batch(
        self, f0: torch.Tensor, f1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # f0/f1: [B,Hp,Wp,D]
        B, Hp, Wp, _ = f0.shape
        f0c = f0.permute(0, 3, 1, 2).float()  # [B,D,Hp,Wp]
        f1c = f1.permute(0, 3, 1, 2).float()

        best_score = torch.full((B,), -float("inf"), device=f0.device, dtype=torch.float32)
        best_theta = torch.zeros((B,), device=f0.device, dtype=torch.float32)
        best_dx = torch.zeros((B,), device=f0.device, dtype=torch.float32)
        best_dy = torch.zeros((B,), device=f0.device, dtype=torch.float32)

        for angle in self._angles_list:
            theta = torch.full((B,), float(angle), device=f0.device, dtype=torch.float32)
            zeros = torch.zeros_like(theta)
            # Rotate f1 in token-grid space (no translation); keep channels intact.
            f1_rot = self._apply_se2_per_sample(f1, theta, zeros, zeros).permute(0, 3, 1, 2).float()
            dx_s, dy_s, peak = self._phasecorr_shift_multi_batch(f0c, f1_rot, return_peak=True)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            dx = -(cos_t * dx_s - sin_t * dy_s)
            dy = -(sin_t * dx_s + cos_t * dy_s)
            better = peak > best_score
            best_score = torch.where(better, peak, best_score)
            best_theta = torch.where(better, theta, best_theta)
            best_dx = torch.where(better, dx.to(dtype=torch.float32), best_dx)
            best_dy = torch.where(better, dy.to(dtype=torch.float32), best_dy)

        return best_theta.to(dtype=f0.dtype), best_dx.to(dtype=f0.dtype), best_dy.to(dtype=f0.dtype)

    def _phasecorr_se2_latent_batch(
        self, s0: torch.Tensor, s1: torch.Tensor, hp: int, wp: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # s0/s1: [B,C,H,W] straightened (or raw) latent maps at full spatial resolution.
        if s0.dim() != 4 or s1.dim() != 4:
            raise ValueError("phasecorr_se2_latent expects s0/s1 as [B,C,H,W]")
        if s0.shape != s1.shape:
            raise ValueError("phasecorr_se2_latent expects s0 and s1 to have same shape")
        B, C, H, W = s0.shape
        s0_f = s0.float()
        s1_f = s1.float()

        best_score = torch.full((B,), -float("inf"), device=s0.device, dtype=torch.float32)
        best_theta = torch.zeros((B,), device=s0.device, dtype=torch.float32)
        best_dx_tok = torch.zeros((B,), device=s0.device, dtype=torch.float32)
        best_dy_tok = torch.zeros((B,), device=s0.device, dtype=torch.float32)

        for angle in self._angles_list:
            theta = torch.full((B,), float(angle), device=s0.device, dtype=torch.float32)
            zeros = torch.zeros_like(theta)
            s1_rot = self._apply_se2_map_per_sample(s1_f, theta, zeros, zeros)
            if self.phasecorr_mode == "multi":
                dx_s, dy_s, peak = self._phasecorr_shift_multi_batch(s0_f, s1_rot, return_peak=True)
            else:
                dx_s, dy_s, peak = self._phasecorr_shift_batch(
                    s0_f.mean(dim=1), s1_rot.mean(dim=1), return_peak=True
                )
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            # Convert roll-shift `s` into affine translation `t = -R s` in pixel units.
            dx_pix = -(cos_t * dx_s - sin_t * dy_s)
            dy_pix = -(sin_t * dx_s + cos_t * dy_s)

            # Convert pixel translation into token-grid units.
            # Tokens live on a patch grid (stride = patch_size), so 1 token step corresponds to patch_size pixels.
            # Using (wp-1)/(W-1) here would be inconsistent with later token->pixel scaling by patch_size.
            ps = float(self.patch_size)
            dx_tok = dx_pix / ps
            dy_tok = dy_pix / ps

            better = peak > best_score
            best_score = torch.where(better, peak.to(dtype=torch.float32), best_score)
            best_theta = torch.where(better, theta, best_theta)
            best_dx_tok = torch.where(better, dx_tok.to(dtype=torch.float32), best_dx_tok)
            best_dy_tok = torch.where(better, dy_tok.to(dtype=torch.float32), best_dy_tok)

        return best_theta.to(dtype=s0.dtype), best_dx_tok.to(dtype=s0.dtype), best_dy_tok.to(dtype=s0.dtype)

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
                tau = self._tau(device=logits.device, dtype=logits.dtype)
                logits = logits / tau.clamp_min(1e-6)
                if self.spatial_gamma > 0.0 or self.spatial_radius > 0:
                    dist2 = self._spatial_dist2(h, w, device=logits.device, dtype=logits.dtype)
                    if self.spatial_gamma > 0.0:
                        logits = logits - float(self.spatial_gamma) * dist2
                    if self.spatial_radius > 0:
                        r2 = float(self.spatial_radius * self.spatial_radius)
                        logits = logits.masked_fill(dist2 > r2, -1e4)
                dust = self._dustbin(device=logits.device, dtype=logits.dtype)
                logp = torch.zeros((n + 1, n + 1), device=logits.device, dtype=logits.dtype) + dust
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
        stride = int(self.win_stride)

        # Overlapping windows (stride < win): use unfold/fold to stay vectorized and avoid seams.
        if stride < win:
            f0_map = f0.permute(0, 3, 1, 2).float()  # [B,D,Hp,Wp]
            f1_map = f1.permute(0, 3, 1, 2).float()
            D = f0_map.shape[1]

            # Pad bottom/right so every token participates in at least one window.
            if hp < win:
                pad_h = win - hp
            else:
                pad_h = (stride - ((hp - win) % stride)) % stride
            if wp < win:
                pad_w = win - wp
            else:
                pad_w = (stride - ((wp - win) % stride)) % stride
            if pad_h or pad_w:
                f0_map = F.pad(f0_map, (0, pad_w, 0, pad_h), mode="replicate")
                f1_map = F.pad(f1_map, (0, pad_w, 0, pad_h), mode="replicate")
            hp_p, wp_p = int(f0_map.shape[2]), int(f0_map.shape[3])

            cols0 = F.unfold(f0_map, kernel_size=win, stride=stride)  # [B, D*win*win, L]
            cols1 = F.unfold(f1_map, kernel_size=win, stride=stride)
            L = int(cols0.shape[-1])
            n = win * win

            x = cols0.view(B, D, n, L).permute(0, 3, 2, 1).reshape(B * L, n, D)
            y = cols1.view(B, D, n, L).permute(0, 3, 2, 1).reshape(B * L, n, D)

            logits = torch.bmm(x, y.transpose(1, 2)) / math.sqrt(max(1.0, float(D)))
            tau = self._tau(device=logits.device, dtype=logits.dtype)
            logits = logits / tau.clamp_min(1e-6)
            if self.spatial_gamma > 0.0 or self.spatial_radius > 0:
                dist2 = self._spatial_dist2(win, win, device=logits.device, dtype=logits.dtype)
                if self.spatial_gamma > 0.0:
                    logits = logits - float(self.spatial_gamma) * dist2
                if self.spatial_radius > 0:
                    r2 = float(self.spatial_radius * self.spatial_radius)
                    logits = logits.masked_fill(dist2 > r2, -1e4)

            dust = self._dustbin(device=logits.device, dtype=logits.dtype)
            logp = torch.zeros((B * L, n + 1, n + 1), device=logits.device, dtype=logits.dtype) + dust
            logp[:, :n, :n] = logits
            logp = _sinkhorn_log_batch(logp, self.sinkhorn_iters)
            p = torch.exp(logp)
            p_xy = p[:, :n, :n]
            mass = p_xy.sum(dim=2, keepdim=True).clamp_min(1e-8)
            yy, xx = torch.meshgrid(
                torch.arange(win, device=logits.device),
                torch.arange(win, device=logits.device),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=-1).view(n, 2).float()
            coords = coords.unsqueeze(0).expand(B * L, -1, -1)
            q = torch.bmm(p_xy, coords) / mass
            delta_blk = (q - coords).view(B * L, win, win, 2)
            conf_blk = (1.0 - p[:, :n, n]).view(B * L, win, win)

            # Weight delta by confidence when accumulating overlapping windows.
            dx_w = (delta_blk[..., 0] * conf_blk).view(B, L, n).permute(0, 2, 1)  # [B,n,L]
            dy_w = (delta_blk[..., 1] * conf_blk).view(B, L, n).permute(0, 2, 1)
            c_cols = conf_blk.view(B, L, n).permute(0, 2, 1)
            ones = torch.ones_like(c_cols)

            dx_sum = F.fold(dx_w, output_size=(hp_p, wp_p), kernel_size=win, stride=stride)
            dy_sum = F.fold(dy_w, output_size=(hp_p, wp_p), kernel_size=win, stride=stride)
            c_sum = F.fold(c_cols, output_size=(hp_p, wp_p), kernel_size=win, stride=stride)
            cnt = F.fold(ones, output_size=(hp_p, wp_p), kernel_size=win, stride=stride)

            # Convert to [B,Hp,Wp,2] / [B,Hp,Wp] and crop padding.
            c_sum = c_sum.clamp_min(1e-8)
            dx = (dx_sum / c_sum)[:, 0, :hp, :wp]
            dy = (dy_sum / c_sum)[:, 0, :hp, :wp]
            delta = torch.stack([dx, dy], dim=-1)
            conf = (c_sum / cnt.clamp_min(1.0))[:, 0, :hp, :wp].clamp(0.0, 1.0)
            return delta, conf

        def process_block(y0: int, x0: int, h: int, w: int) -> None:
            if h <= 0 or w <= 0:
                return
            x = f0[:, y0 : y0 + h, x0 : x0 + w].reshape(B, h * w, -1)
            y = f1[:, y0 : y0 + h, x0 : x0 + w].reshape(B, h * w, -1)
            n = x.shape[1]
            if n == 0:
                return
            logits = torch.bmm(x, y.transpose(1, 2)) / math.sqrt(max(1.0, float(x.shape[2])))
            tau = self._tau(device=logits.device, dtype=logits.dtype)
            logits = logits / tau.clamp_min(1e-6)
            if self.spatial_gamma > 0.0 or self.spatial_radius > 0:
                dist2 = self._spatial_dist2(h, w, device=logits.device, dtype=logits.dtype)
                if self.spatial_gamma > 0.0:
                    logits = logits - float(self.spatial_gamma) * dist2
                if self.spatial_radius > 0:
                    r2 = float(self.spatial_radius * self.spatial_radius)
                    logits = logits.masked_fill(dist2 > r2, -1e4)
            dust = self._dustbin(device=logits.device, dtype=logits.dtype)
            logp = torch.zeros((B, n + 1, n + 1), device=logits.device, dtype=logits.dtype) + dust
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
            tau = self._tau(device=logits.device, dtype=logits.dtype)
            logits = logits / tau.clamp_min(1e-6)
            if self.spatial_gamma > 0.0 or self.spatial_radius > 0:
                dist2 = self._spatial_dist2(win, win, device=logits.device, dtype=logits.dtype)
                if self.spatial_gamma > 0.0:
                    logits = logits - float(self.spatial_gamma) * dist2
                if self.spatial_radius > 0:
                    r2 = float(self.spatial_radius * self.spatial_radius)
                    logits = logits.masked_fill(dist2 > r2, -1e4)
            dust = self._dustbin(device=logits.device, dtype=logits.dtype)
            logp = (
                torch.zeros((B * nH * nW, win * win + 1, win * win + 1), device=logits.device, dtype=logits.dtype)
                + dust
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

    def _compute_flow_from_feats_batch(
        self, f0: torch.Tensor, f1: torch.Tensor, *, s0: torch.Tensor | None = None, s1: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # f0/f1: [B,Hp,Wp,D]
        hp, wp = f0.shape[1], f0.shape[2]
        if self.global_mode == "se2":
            theta, dx, dy = self._estimate_global_se2_batch(f0, f1)
        elif self.global_mode == "phasecorr":
            if self.phasecorr_level == "latent" and s0 is not None and s1 is not None:
                theta, dx, dy = self._phasecorr_se2_latent_batch(s0, s1, hp, wp)
            else:
                if self.phasecorr_mode == "multi":
                    theta, dx, dy = self._phasecorr_se2_multi_batch(f0, f1)
                else:
                    f0_score = f0.mean(dim=-1)
                    f1_score = f1.mean(dim=-1)
                    theta, dx, dy = self._phasecorr_se2_batch(f0_score, f1_score)
        else:
            theta = torch.zeros((f0.shape[0],), device=f0.device, dtype=f0.dtype)
            dx = torch.zeros_like(theta)
            dy = torch.zeros_like(theta)
        f1_aligned = self._apply_se2_per_sample(f1, theta, dx, dy)
        delta, conf = self._local_sinkhorn_delta_batch(f0, f1_aligned, hp, wp)
        flow_tok = self._compose_flow_batch(delta, theta, dx, dy, hp, wp)
        return flow_tok, conf

    def _fb_consistency_conf(
        self, flow01_tok: torch.Tensor, flow10_tok: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # flow*_tok: [B,Hp,Wp,2] in token units.
        if self.fb_sigma <= 0.0:
            B, Hp, Wp, _ = flow01_tok.shape
            ones = torch.ones((B, Hp, Wp), device=flow01_tok.device, dtype=torch.float32)
            zeros = torch.zeros((B, Hp, Wp), device=flow01_tok.device, dtype=torch.float32)
            return ones, ones, zeros, zeros

        flow01_map = flow01_tok.permute(0, 3, 1, 2).float()  # [B,2,Hp,Wp]
        flow10_map = flow10_tok.permute(0, 3, 1, 2).float()

        # err01 at frame0 coords: F01(x0) + F10(x0 + F01(x0))
        flow10_at_01 = _warp(flow10_map, flow01_map)
        fb01 = flow01_map + flow10_at_01
        err01 = torch.linalg.norm(fb01, dim=1)  # [B,Hp,Wp]

        # err10 at frame1 coords: F10(x1) + F01(x1 + F10(x1))
        flow01_at_10 = _warp(flow01_map, flow10_map)
        fb10 = flow10_map + flow01_at_10
        err10 = torch.linalg.norm(fb10, dim=1)

        sigma = float(self.fb_sigma)
        conf01 = torch.exp(-0.5 * (err01 / sigma) ** 2).clamp(0.0, 1.0)
        conf10 = torch.exp(-0.5 * (err10 / sigma) ** 2).clamp(0.0, 1.0)
        return conf01, conf10, err01, err10

    def compute_bidirectional_flow_and_confs_batch(
        self, f0: torch.Tensor, f1: torch.Tensor, *, s0: torch.Tensor | None = None, s1: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # f0/f1: [B,Hp,Wp,D]
        flow01_tok, conf01_dust = self._compute_flow_from_feats_batch(f0, f1, s0=s0, s1=s1)
        flow10_tok, conf10_dust = self._compute_flow_from_feats_batch(f1, f0, s0=s1, s1=s0)
        conf01_fb, conf10_fb, fb_err01, fb_err10 = self._fb_consistency_conf(flow01_tok, flow10_tok)
        conf01 = conf01_dust.float() * conf01_fb
        conf10 = conf10_dust.float() * conf10_fb
        return flow01_tok, flow10_tok, conf01, conf10, conf01_dust.float(), conf10_dust.float(), fb_err01, fb_err10

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
        # Precompute token features for all frames. Matching is always done in straightened space
        # (if available), while warping can happen in either z-space or s-space.
        latents_flat = latents.reshape(B * T, latents.shape[2], H, W)
        s_latents_flat = None
        if self.straightener is not None:
            with torch.no_grad():
                s_latents_flat = self.straightener.encode(latents_flat)
            feats, hp, wp = self._token_features(s_latents_flat, assume_straightened=True)
        else:
            feats, hp, wp = self._token_features(latents_flat, assume_straightened=True)
        feats = feats.view(B, T, hp, wp, -1)
        s_latents = None
        if self.warp_space == "s":
            # warp_space='s' requires straightener by construction.
            assert s_latents_flat is not None
            s_latents = s_latents_flat.view(B, T, latents.shape[2], H, W)

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

        # Global phase correlation can optionally run at full latent resolution.
        s0_pairs = None
        s1_pairs = None
        if self.global_mode == "phasecorr" and self.phasecorr_level == "latent":
            if s_latents_flat is not None:
                global_maps = s_latents_flat.view(B, T, latents.shape[2], H, W)
            else:
                global_maps = latents
            s0_pairs = global_maps[pair_b, pair_t0]
            s1_pairs = global_maps[pair_b, pair_t1]

        flow01_tok, flow10_tok, conf01_tok, conf10_tok, _, _, _, _ = self.compute_bidirectional_flow_and_confs_batch(
            f0, f1, s0=s0_pairs, s1=s1_pairs
        )

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

            gap = t1 - t0
            steps = gap - 1
            alpha = torch.linspace(1, steps, steps, device=latents.device, dtype=latents.dtype) / float(gap)
            alpha = alpha.view(-1, 1, 1, 1)

            # Apply confidence shrinkage to the displacement before warping.
            # This matches the semantics of the dustbin: if a token is likely unmatched, its expected motion is small
            # (mixture between "move" and "stay"). Empirically this also prevents catastrophic warps.
            flow01_eff = flow01_p * conf01_p
            flow10_eff = flow10_p * conf10_p
            if self.warp_space == "s":
                assert s_latents is not None
                s0 = s_latents[b : b + 1, t0]
                s1 = s_latents[b : b + 1, t1]
                s0_rep = s0.expand(steps, -1, -1, -1)
                s1_rep = s1.expand(steps, -1, -1, -1)
            else:
                z0_rep = z0.expand(steps, -1, -1, -1)
                z1_rep = z1.expand(steps, -1, -1, -1)
            flow01_rep = flow01_eff.expand(steps, -1, -1, -1) * alpha
            flow10_rep = flow10_eff.expand(steps, -1, -1, -1) * (1.0 - alpha)
            conf01_rep = conf01_p.expand(steps, -1, -1, -1)
            conf10_rep = conf10_p.expand(steps, -1, -1, -1)

            if self.warp_space == "s":
                s0_w = _warp(s0_rep, -flow01_rep)
                s1_w = _warp(s1_rep, -flow10_rep)
                conf0_w = _warp(conf01_rep, -flow01_rep)
                conf1_w = _warp(conf10_rep, -flow10_rep)
            else:
                z0_w = _warp(z0_rep, -flow01_rep)
                z1_w = _warp(z1_rep, -flow10_rep)
                conf0_w = _warp(conf01_rep, -flow01_rep)
                conf1_w = _warp(conf10_rep, -flow10_rep)

            w0 = (1.0 - alpha) * conf0_w
            w1 = alpha * conf1_w
            denom = w0 + w1
            if self.warp_space == "s":
                s_mix = (w0 * s0_w + w1 * s1_w) / denom.clamp_min(1e-6)
                mask = denom > 1e-6
                # If confidence is near-zero, do not apply the warp at all; fall back to unwarped LERP.
                # Using warped endpoints here can create catastrophic errors when the estimated flow is invalid.
                s_lerp = (1.0 - alpha) * s0_rep + alpha * s1_rep
                s_t = torch.where(mask, s_mix, s_lerp)
                with torch.no_grad():
                    z_t = self.straightener.decode(s_t.to(dtype=s0_w.dtype))
            else:
                z_mix = (w0 * z0_w + w1 * z1_w) / denom.clamp_min(1e-6)
                mask = denom > 1e-6
                # If confidence is near-zero, do not apply the warp at all; fall back to unwarped LERP.
                z_lerp = (1.0 - alpha) * z0_rep + alpha * z1_rep
                z_t = torch.where(mask, z_mix, z_lerp)
            out[b, t0 + 1 : t1] = z_t
            conf_step = torch.minimum(conf0_w, conf1_w)
            conf[b, t0 + 1 : t1] = conf_step[:, 0]
        return out, conf


__all__ = ["SinkhornWarpInterpolator"]
