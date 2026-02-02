from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from src.diffusion.schedules import make_alpha_bars, make_beta_schedule


@dataclass
class SegmentPrecompute:
    seg_i: torch.Tensor
    seg_j: torch.Tensor
    seg_len: torch.Tensor
    t_idx: torch.Tensor
    alpha: torch.Tensor
    weight: torch.Tensor
    seg_id: torch.Tensor


def build_snr_weights(
    schedule: str,
    n_train: int,
    s_min: float,
    s_max: float,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = make_beta_schedule(schedule, n_train)
    alpha_bar = make_alpha_bars(betas)["alpha_bar"]
    snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-8)
    snr_clipped = torch.clamp(snr, min=s_min, max=s_max)
    weights = snr_clipped.pow(gamma)
    return snr, weights


def sample_timesteps_log_snr(snr: torch.Tensor, num_steps: int) -> torch.Tensor:
    if num_steps <= 1:
        return torch.tensor([0], dtype=torch.long, device=snr.device)
    log_snr = torch.log(torch.clamp(snr, min=1e-12))
    targets = torch.linspace(log_snr.max(), log_snr.min(), num_steps, device=snr.device)
    idx = torch.abs(log_snr.unsqueeze(0) - targets.unsqueeze(1)).argmin(dim=1)
    idx = torch.unique(idx)
    if idx.numel() < num_steps:
        extra = torch.tensor([0, log_snr.shape[0] - 1], dtype=torch.long, device=snr.device)
        idx = torch.unique(torch.cat([idx, extra], dim=0))
    return torch.sort(idx).values


def build_segment_precompute(T: int, samples_per_seg: int, device: torch.device) -> SegmentPrecompute:
    seg_i_list = []
    seg_j_list = []
    t_idx_list = []
    alpha_list = []
    weight_list = []
    seg_len_list = []

    for i in range(T - 1):
        for j in range(i + 1, T):
            gap = j - i
            seg_i_list.append(i)
            seg_j_list.append(j)
            seg_len_list.append(gap)
            if gap <= 1:
                t_idx = torch.full((samples_per_seg,), i, dtype=torch.long)
                alpha = torch.zeros((samples_per_seg,), dtype=torch.float32)
                weight = 0.0
            else:
                interior = gap - 1
                offsets = (torch.arange(samples_per_seg, dtype=torch.float32) + 0.5) / samples_per_seg
                offsets = torch.floor(offsets * interior).long()
                t_idx = i + 1 + offsets
                alpha = (t_idx.float() - float(i)) / float(gap)
                weight = float(interior) / float(samples_per_seg)
            t_idx_list.append(t_idx)
            alpha_list.append(alpha)
            weight_list.append(weight)

    seg_i = torch.tensor(seg_i_list, dtype=torch.long, device=device)
    seg_j = torch.tensor(seg_j_list, dtype=torch.long, device=device)
    seg_len = torch.tensor(seg_len_list, dtype=torch.long, device=device)
    seg_id = torch.full((T, T), -1, dtype=torch.long, device=device)
    seg_id[seg_i, seg_j] = torch.arange(seg_i.shape[0], device=device, dtype=torch.long)
    t_idx = torch.stack(t_idx_list, dim=0).to(device)
    alpha = torch.stack(alpha_list, dim=0).to(device)
    weight = torch.tensor(weight_list, dtype=torch.float32, device=device)
    return SegmentPrecompute(
        seg_i=seg_i, seg_j=seg_j, seg_len=seg_len, t_idx=t_idx, alpha=alpha, weight=weight, seg_id=seg_id
    )


def build_segment_features(T: int, seg_i: torch.Tensor, seg_j: torch.Tensor) -> torch.Tensor:
    denom = float(max(1, T - 1))
    i_norm = seg_i.float() / denom
    j_norm = seg_j.float() / denom
    gap_norm = (seg_j.float() - seg_i.float()) / denom
    return torch.stack([i_norm, j_norm, gap_norm], dim=-1)


def build_segment_features_from_idx(idx: torch.Tensor, T: int, seg_feat_dim: int = 3) -> torch.Tensor:
    if idx.dim() != 2:
        raise ValueError("idx must be [B, K]")
    if seg_feat_dim <= 0:
        return torch.zeros((idx.shape[0], idx.shape[1] - 1, 0), device=idx.device)
    denom = float(max(1, T - 1))
    i = idx[:, :-1].float()
    j = idx[:, 1:].float()
    i_norm = i / denom
    j_norm = j / denom
    gap_norm = (j - i) / denom
    feat = torch.stack([i_norm, j_norm, gap_norm], dim=-1)
    if seg_feat_dim == 3:
        return feat
    if seg_feat_dim < 3:
        return feat[:, :, :seg_feat_dim]
    pad = torch.zeros((feat.shape[0], feat.shape[1], seg_feat_dim - 3), device=feat.device, dtype=feat.dtype)
    return torch.cat([feat, pad], dim=-1)


def compute_segment_costs_batch(
    x_pos: torch.Tensor,
    precomp: SegmentPrecompute,
    weight_scale: float,
) -> torch.Tensor:
    B, T, D = x_pos.shape
    if D < 2:
        raise ValueError("x_pos must have at least 2 dims")
    seg_i = precomp.seg_i
    seg_j = precomp.seg_j
    t_idx = precomp.t_idx
    alpha = precomp.alpha
    weight = precomp.weight

    x_i = x_pos[:, seg_i, :2]
    x_j = x_pos[:, seg_j, :2]
    diff_ij = x_j - x_i
    alpha_exp = alpha.unsqueeze(0).unsqueeze(-1)
    mu = x_i.unsqueeze(2) + alpha_exp * diff_ij.unsqueeze(2)

    t_idx_flat = t_idx.reshape(-1)
    x_t = x_pos[:, t_idx_flat, :2].reshape(B, t_idx.shape[0], t_idx.shape[1], 2)
    diff = x_t - mu
    sq = (diff * diff).sum(dim=-1)
    cost = sq.sum(dim=-1) * weight
    if weight_scale != 1.0:
        cost = cost * weight_scale
    return cost


def build_cost_matrix_from_segments(
    cost_seg: torch.Tensor, precomp: SegmentPrecompute, T: int
) -> torch.Tensor:
    device = cost_seg.device
    C = torch.full((T, T), float("inf"), device=device)
    C[precomp.seg_i, precomp.seg_j] = cost_seg
    return C


def build_cost_matrix_from_segments_batch(
    cost_seg: torch.Tensor, precomp: SegmentPrecompute, T: int
) -> torch.Tensor:
    if cost_seg.dim() != 2:
        raise ValueError("cost_seg must be [B, S]")
    B = cost_seg.shape[0]
    device = cost_seg.device
    C = torch.full((B, T, T), float("inf"), device=device)
    C[:, precomp.seg_i, precomp.seg_j] = cost_seg
    return C


def dp_select_indices(C: torch.Tensor, K: int) -> torch.Tensor:
    T = C.shape[0]
    if K < 2:
        raise ValueError("K must be >= 2")
    if K > T:
        K = T
    inf = float("inf")
    dp = torch.full((K, T), inf, device=C.device)
    parent = torch.full((K, T), -1, dtype=torch.long, device=C.device)
    dp[0, 0] = 0.0
    for k in range(1, K):
        for j in range(1, T):
            prev = dp[k - 1, :j] + C[:j, j]
            best = torch.argmin(prev)
            dp[k, j] = prev[best]
            parent[k, j] = best
    if not torch.isfinite(dp[K - 1, T - 1]):
        raise RuntimeError("DP failed to find a valid path to T-1.")
    idx = torch.zeros((K,), dtype=torch.long, device=C.device)
    idx[-1] = T - 1
    cur = T - 1
    for k in range(K - 1, 0, -1):
        cur = parent[k, cur].item()
        if cur < 0:
            raise RuntimeError("DP backtrack failed.")
        idx[k - 1] = cur
    return idx


def dp_select_indices_batch(C: torch.Tensor, K: int) -> torch.Tensor:
    if C.dim() != 3:
        raise ValueError("C must be [B,T,T]")
    B, T, _ = C.shape
    if K < 2:
        raise ValueError("K must be >= 2")
    if K > T:
        K = T
    inf = float("inf")
    dp = torch.full((B, K, T), inf, device=C.device)
    parent = torch.full((B, K, T), -1, dtype=torch.long, device=C.device)
    dp[:, 0, 0] = 0.0
    for k in range(1, K):
        for j in range(1, T):
            prev = dp[:, k - 1, :j] + C[:, :j, j]
            best = torch.argmin(prev, dim=1)
            dp[:, k, j] = prev.gather(1, best.unsqueeze(1)).squeeze(1)
            parent[:, k, j] = best
    if not torch.isfinite(dp[:, K - 1, T - 1]).all():
        raise RuntimeError("DP failed to find a valid path to T-1 for some samples.")
    idx = torch.zeros((B, K), dtype=torch.long, device=C.device)
    idx[:, -1] = T - 1
    cur = torch.full((B,), T - 1, device=C.device, dtype=torch.long)
    for k in range(K - 1, 0, -1):
        cur = parent[:, k].gather(1, cur.unsqueeze(1)).squeeze(1)
        if torch.any(cur < 0):
            raise RuntimeError("DP backtrack failed.")
        idx[:, k - 1] = cur
    return idx


def build_kp_feat(idx: torch.Tensor, T: int) -> torch.Tensor:
    K = idx.shape[0]
    feat = torch.zeros((K, 3), dtype=torch.float32, device=idx.device)
    denom = float(max(1, T - 1))
    feat[:, 2] = idx.float() / denom
    if K > 1:
        left = torch.zeros_like(idx, dtype=torch.float32)
        right = torch.zeros_like(idx, dtype=torch.float32)
        left[1:] = (idx[1:] - idx[:-1]).float() / denom
        right[:-1] = (idx[1:] - idx[:-1]).float() / denom
        feat[:, 0] = left
        feat[:, 1] = right
    return feat


def build_kp_feat_batch(idx: torch.Tensor, T: int) -> torch.Tensor:
    if idx.dim() != 2:
        raise ValueError("idx must be [B,K]")
    B, K = idx.shape
    feat = torch.zeros((B, K, 3), dtype=torch.float32, device=idx.device)
    denom = float(max(1, T - 1))
    feat[:, :, 2] = idx.float() / denom
    if K > 1:
        gaps = (idx[:, 1:] - idx[:, :-1]).float() / denom
        feat[:, 1:, 0] = gaps
        feat[:, :-1, 1] = gaps
    return feat
