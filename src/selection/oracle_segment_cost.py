from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OracleSegPrecompute:
    """Exact (non-subsampled) interior-frame precompute for all segments (i,j), i<j."""

    seg_i: torch.Tensor  # [S]
    seg_j: torch.Tensor  # [S]
    t_idx: torch.Tensor  # [S,Kmax] interior frame indices padded with 0
    alpha: torch.Tensor  # [S,Kmax] barycentric alpha for each interior frame (0 where padded)
    mask: torch.Tensor  # [S,Kmax] 1 for valid interior entries else 0


def build_oracle_seg_precompute(T: int, *, device: torch.device) -> OracleSegPrecompute:
    if T < 2:
        raise ValueError("T must be >= 2")
    seg_i_list = []
    seg_j_list = []
    for i in range(T - 1):
        for j in range(i + 1, T):
            seg_i_list.append(i)
            seg_j_list.append(j)
    seg_i = torch.tensor(seg_i_list, dtype=torch.long, device=device)
    seg_j = torch.tensor(seg_j_list, dtype=torch.long, device=device)
    S = int(seg_i.numel())
    Kmax = max(0, T - 2)
    t_idx = torch.zeros((S, Kmax), dtype=torch.long, device=device)
    alpha = torch.zeros((S, Kmax), dtype=torch.float32, device=device)
    mask = torch.zeros((S, Kmax), dtype=torch.float32, device=device)
    for s in range(S):
        i = int(seg_i[s].item())
        j = int(seg_j[s].item())
        gap = j - i
        interior = max(0, gap - 1)
        if interior <= 0:
            continue
        tt = torch.arange(1, interior + 1, device=device, dtype=torch.long)
        t_vals = i + tt
        a = tt.float() / float(gap)
        t_idx[s, :interior] = t_vals
        alpha[s, :interior] = a
        mask[s, :interior] = 1.0
    return OracleSegPrecompute(seg_i=seg_i, seg_j=seg_j, t_idx=t_idx, alpha=alpha, mask=mask)


@torch.no_grad()
def compute_oracle_cost_seg_mse(
    x: torch.Tensor,
    pre: OracleSegPrecompute,
    *,
    chunk_segments: int = 16,
) -> torch.Tensor:
    """Exact interior-frame sum of per-frame MSE for linear interpolation within each segment.

    x: [B,T,C,H,W] (float/bf16 ok; computes in float32)
    Returns: cost_seg [B,S] in float32.
    """
    if x.dim() != 5:
        raise ValueError("x must be [B,T,C,H,W]")
    B, T, C, H, W = x.shape
    if int(T) < 2:
        raise ValueError("T must be >= 2")
    seg_i = pre.seg_i
    seg_j = pre.seg_j
    t_idx = pre.t_idx
    alpha = pre.alpha
    mask = pre.mask
    S = int(seg_i.numel())
    out = torch.empty((B, S), device=x.device, dtype=torch.float32)

    denom = float(C * H * W)
    cs = max(1, int(chunk_segments))
    for s0 in range(0, S, cs):
        s1 = min(S, s0 + cs)
        si = seg_i[s0:s1]
        sj = seg_j[s0:s1]
        zi = x[:, si].float()  # [B,S',C,H,W]
        zj = x[:, sj].float()
        cost = torch.zeros((B, s1 - s0), device=x.device, dtype=torch.float32)
        for k in range(t_idx.shape[1]):
            mk = mask[s0:s1, k]  # [S']
            if not bool((mk > 0).any()):
                continue
            tk = t_idx[s0:s1, k]  # [S']
            zk = x[:, tk].float()
            ak = alpha[s0:s1, k].view(1, -1, 1, 1, 1)
            z_hat = (1.0 - ak) * zi + ak * zj
            diff2 = (z_hat - zk).pow(2).sum(dim=(2, 3, 4)) / denom  # [B,S']
            cost = cost + diff2 * mk.view(1, -1)
        out[:, s0:s1] = cost
    return out

