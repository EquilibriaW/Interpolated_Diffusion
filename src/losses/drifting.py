from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DriftingStats:
    dist_pos_mean: float
    dist_neg_mean: float
    a_pos_mass_mean: float
    a_neg_mass_mean: float
    drift_norm_mean: float
    eff_neighbors_pos: float
    eff_neighbors_neg: float


def _sample_entropy_to_effective_count(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute exp(H(p)) row-wise for probabilities p."""
    h = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)
    return torch.exp(h)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
    """Mask-aware softmax that returns zeros for fully-masked slices."""
    mask_f = mask.to(dtype=logits.dtype)
    neg_inf = torch.finfo(logits.dtype).min
    logits_m = torch.where(mask, logits, logits.new_full((), neg_inf))
    m = logits_m.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    ex = torch.exp(logits_m - m) * mask_f
    den = ex.sum(dim=dim, keepdim=True).clamp_min(eps)
    return ex / den


def _masked_mean(v: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Global mean over entries where mask=True."""
    mask_f = mask.to(dtype=v.dtype)
    return (v * mask_f).sum() / mask_f.sum().clamp_min(eps)


def _class_masked_mean(v: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-class mean over entries where mask=True, then average classes."""
    mask_f = mask.to(dtype=v.dtype)
    num = (v * mask_f).sum(dim=tuple(range(1, v.ndim)))
    den = mask_f.sum(dim=tuple(range(1, mask_f.ndim))).clamp_min(eps)
    return (num / den).mean()


def compute_drift_from_distances(
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    normalize_mode: str = "both",
    *,
    x_valid_mask: torch.Tensor | None = None,
    y_pos_valid_mask: torch.Tensor | None = None,
    y_neg_valid_mask: torch.Tensor | None = None,
    y_neg_weights: torch.Tensor | None = None,
    self_neg_k: int | torch.Tensor = 0,
) -> Tuple[torch.Tensor, DriftingStats]:
    """Compute drifting field V from precomputed pairwise distances.

    Supports 2D (N, P/Q) and class-batched 3D (C, N, P/Q) distance tensors.
    """
    tau = float(temperature)
    if tau <= 0.0:
        raise ValueError("temperature must be > 0")
    if dist_pos.ndim not in (2, 3):
        raise ValueError("dist_pos must be rank 2 or 3")
    if dist_neg.ndim != dist_pos.ndim:
        raise ValueError("dist_pos/dist_neg rank mismatch")
    if y_pos.ndim != dist_pos.ndim or y_neg.ndim != dist_pos.ndim:
        raise ValueError("y_pos/y_neg rank must match distance tensors")

    if dist_pos.ndim == 2:
        n, n_pos = dist_pos.shape
        n_neg = dist_neg.shape[1]
        if x_valid_mask is None:
            x_valid_mask = torch.ones((n,), dtype=torch.bool, device=dist_pos.device)
        if y_pos_valid_mask is None:
            y_pos_valid_mask = torch.ones((n_pos,), dtype=torch.bool, device=dist_pos.device)
        if y_neg_valid_mask is None:
            y_neg_valid_mask = torch.ones((n_neg,), dtype=torch.bool, device=dist_pos.device)
        x_mask = x_valid_mask[:, None]
        y_pos_mask = y_pos_valid_mask[None, :]
        y_neg_mask = y_neg_valid_mask[None, :]
        self_neg_mask = None
        if isinstance(self_neg_k, torch.Tensor):
            if self_neg_k.numel() != 1:
                raise ValueError("self_neg_k tensor must be scalar for 2D mode")
            self_neg_k = int(self_neg_k.item())
        k = max(0, min(int(self_neg_k), int(n), int(n_neg)))
        if k > 0:
            idx_n = torch.arange(n, device=dist_pos.device, dtype=torch.long)[:, None]
            idx_q = torch.arange(n_neg, device=dist_pos.device, dtype=torch.long)[None, :]
            self_neg_mask = (idx_n == idx_q) & (idx_n < k)
    else:
        c, n, n_pos = dist_pos.shape
        n_neg = dist_neg.shape[2]
        if x_valid_mask is None:
            x_valid_mask = torch.ones((c, n), dtype=torch.bool, device=dist_pos.device)
        if y_pos_valid_mask is None:
            y_pos_valid_mask = torch.ones((c, n_pos), dtype=torch.bool, device=dist_pos.device)
        if y_neg_valid_mask is None:
            y_neg_valid_mask = torch.ones((c, n_neg), dtype=torch.bool, device=dist_pos.device)
        x_mask = x_valid_mask[:, :, None]
        y_pos_mask = y_pos_valid_mask[:, None, :]
        y_neg_mask = y_neg_valid_mask[:, None, :]
        if isinstance(self_neg_k, torch.Tensor):
            if self_neg_k.ndim != 1 or self_neg_k.shape[0] != c:
                raise ValueError("self_neg_k tensor must be [C] for 3D mode")
            k_cls = self_neg_k.to(device=dist_pos.device, dtype=torch.long).clamp_min(0)
        else:
            k_cls = torch.full((c,), int(self_neg_k), device=dist_pos.device, dtype=torch.long).clamp_min(0)
        idx_n = torch.arange(n, device=dist_pos.device, dtype=torch.long)[None, :, None]
        idx_q = torch.arange(n_neg, device=dist_pos.device, dtype=torch.long)[None, None, :]
        k_view = torch.minimum(k_cls, torch.full_like(k_cls, min(n, n_neg))).view(c, 1, 1)
        self_mask = (idx_n == idx_q) & (idx_n < k_view)
        y_neg_mask = y_neg_mask & (~self_mask)

    if y_neg_weights is None:
        if dist_pos.ndim == 2:
            y_neg_weights = torch.ones((n_neg,), device=dist_pos.device, dtype=dist_pos.dtype)
        else:
            y_neg_weights = torch.ones((c, n_neg), device=dist_pos.device, dtype=dist_pos.dtype)
    else:
        y_neg_weights = y_neg_weights.to(device=dist_pos.device, dtype=dist_pos.dtype)
        if dist_pos.ndim == 2 and y_neg_weights.shape != (n_neg,):
            raise ValueError(f"y_neg_weights must be [N_neg], got {tuple(y_neg_weights.shape)}")
        if dist_pos.ndim == 3 and y_neg_weights.shape != (c, n_neg):
            raise ValueError(f"y_neg_weights must be [C,N_neg], got {tuple(y_neg_weights.shape)}")

    neg_w_mask = y_neg_weights > 0
    if dist_pos.ndim == 2:
        y_neg_mask = y_neg_mask & neg_w_mask[None, :]
    else:
        y_neg_mask = y_neg_mask & neg_w_mask[:, None, :]

    valid_pos = x_mask & y_pos_mask
    valid_neg = x_mask & y_neg_mask
    if dist_pos.ndim == 2 and self_neg_mask is not None:
        valid_neg = valid_neg & (~self_neg_mask)
    valid_all = torch.cat([valid_pos, valid_neg], dim=-1)

    logit_pos = -dist_pos / tau
    logit_neg = -dist_neg / tau
    if dist_pos.ndim == 2:
        logit_neg = logit_neg + torch.log(y_neg_weights.clamp_min(1e-12))[None, :]
    else:
        logit_neg = logit_neg + torch.log(y_neg_weights.clamp_min(1e-12))[:, None, :]
    logit = torch.cat([logit_pos, logit_neg], dim=-1)

    if normalize_mode == "none":
        a = torch.exp(logit) * valid_all.to(dtype=logit.dtype)
    elif normalize_mode == "y":
        a = _masked_softmax(logit, valid_all, dim=-1)
    elif normalize_mode == "both":
        a_row = _masked_softmax(logit, valid_all, dim=-1)
        a_col = _masked_softmax(logit, valid_all, dim=-2)
        a = torch.sqrt(a_row * a_col.clamp_min(1e-12))
    else:
        raise ValueError(f"unknown normalize_mode: {normalize_mode}")

    n_pos = y_pos.shape[-2]
    a_pos = a[..., :n_pos]
    a_neg = a[..., n_pos:]

    # Anti-symmetric weighting used in paper pseudocode.
    w_pos = a_pos * a_neg.sum(dim=-1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=-1, keepdim=True)

    drift_pos = torch.matmul(w_pos, y_pos)
    drift_neg = torch.matmul(w_neg, y_neg)
    v = drift_pos - drift_neg
    if dist_pos.ndim == 2:
        v = torch.where(x_valid_mask[:, None], v, torch.zeros_like(v))
    else:
        v = torch.where(x_valid_mask[:, :, None], v, torch.zeros_like(v))

    # Diagnostics.
    if normalize_mode == "both":
        p = _masked_softmax(logit, valid_all, dim=-1)
    elif normalize_mode == "none":
        den = a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p = a / den
    else:
        p = a
    p_pos = p[..., :n_pos]
    p_neg = p[..., n_pos:]

    if dist_pos.ndim == 2:
        pos_count = valid_pos.sum(dim=-1).clamp_min(1)
        neg_count = valid_neg.sum(dim=-1).clamp_min(1)
        eff_pos = _sample_entropy_to_effective_count(p_pos).clamp_max(pos_count.to(dtype=p_pos.dtype))
        eff_neg = _sample_entropy_to_effective_count(p_neg).clamp_max(neg_count.to(dtype=p_neg.dtype))
        v_norm = v.norm(dim=-1)
        stats = DriftingStats(
            dist_pos_mean=float(_masked_mean(dist_pos, valid_pos).item()),
            dist_neg_mean=float(_masked_mean(dist_neg, valid_neg).item()),
            a_pos_mass_mean=float(_masked_mean(a_pos.sum(dim=-1), x_valid_mask).item()),
            a_neg_mass_mean=float(_masked_mean(a_neg.sum(dim=-1), x_valid_mask).item()),
            drift_norm_mean=float(_masked_mean(v_norm, x_valid_mask).item()),
            eff_neighbors_pos=float(_masked_mean(eff_pos, x_valid_mask).item()),
            eff_neighbors_neg=float(_masked_mean(eff_neg, x_valid_mask).item()),
        )
    else:
        pos_count = valid_pos.sum(dim=-1).clamp_min(1)
        neg_count = valid_neg.sum(dim=-1).clamp_min(1)
        eff_pos = _sample_entropy_to_effective_count(p_pos).clamp_max(pos_count.to(dtype=p_pos.dtype))
        eff_neg = _sample_entropy_to_effective_count(p_neg).clamp_max(neg_count.to(dtype=p_neg.dtype))
        v_norm = v.norm(dim=-1)
        stats = DriftingStats(
            dist_pos_mean=float(_class_masked_mean(dist_pos, valid_pos).item()),
            dist_neg_mean=float(_class_masked_mean(dist_neg, valid_neg).item()),
            a_pos_mass_mean=float(_class_masked_mean(a_pos.sum(dim=-1), x_valid_mask).item()),
            a_neg_mass_mean=float(_class_masked_mean(a_neg.sum(dim=-1), x_valid_mask).item()),
            drift_norm_mean=float(_class_masked_mean(v_norm, x_valid_mask).item()),
            eff_neighbors_pos=float(_class_masked_mean(eff_pos, x_valid_mask).item()),
            eff_neighbors_neg=float(_class_masked_mean(eff_neg, x_valid_mask).item()),
        )
    return v, stats


def compute_drift_from_distances_nostats(
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    normalize_mode: str = "both",
    *,
    x_valid_mask: torch.Tensor | None = None,
    y_pos_valid_mask: torch.Tensor | None = None,
    y_neg_valid_mask: torch.Tensor | None = None,
    y_neg_weights: torch.Tensor | None = None,
    self_neg_k: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Compute drifting field V from precomputed distances, stats-free fast path."""
    tau = float(temperature)
    if tau <= 0.0:
        raise ValueError("temperature must be > 0")
    if dist_pos.ndim not in (2, 3):
        raise ValueError("dist_pos must be rank 2 or 3")
    if dist_neg.ndim != dist_pos.ndim:
        raise ValueError("dist_pos/dist_neg rank mismatch")
    if y_pos.ndim != dist_pos.ndim or y_neg.ndim != dist_pos.ndim:
        raise ValueError("y_pos/y_neg rank must match distance tensors")

    if dist_pos.ndim == 2:
        n, n_pos = dist_pos.shape
        n_neg = dist_neg.shape[1]
        if x_valid_mask is None:
            x_valid_mask = torch.ones((n,), dtype=torch.bool, device=dist_pos.device)
        if y_pos_valid_mask is None:
            y_pos_valid_mask = torch.ones((n_pos,), dtype=torch.bool, device=dist_pos.device)
        if y_neg_valid_mask is None:
            y_neg_valid_mask = torch.ones((n_neg,), dtype=torch.bool, device=dist_pos.device)
        x_mask = x_valid_mask[:, None]
        y_pos_mask = y_pos_valid_mask[None, :]
        y_neg_mask = y_neg_valid_mask[None, :]
        self_neg_mask = None
        if isinstance(self_neg_k, torch.Tensor):
            if self_neg_k.numel() != 1:
                raise ValueError("self_neg_k tensor must be scalar for 2D mode")
            self_neg_k = int(self_neg_k.item())
        k = max(0, min(int(self_neg_k), int(n), int(n_neg)))
        if k > 0:
            idx_n = torch.arange(n, device=dist_pos.device, dtype=torch.long)[:, None]
            idx_q = torch.arange(n_neg, device=dist_pos.device, dtype=torch.long)[None, :]
            self_neg_mask = (idx_n == idx_q) & (idx_n < k)
    else:
        c, n, n_pos = dist_pos.shape
        n_neg = dist_neg.shape[2]
        if x_valid_mask is None:
            x_valid_mask = torch.ones((c, n), dtype=torch.bool, device=dist_pos.device)
        if y_pos_valid_mask is None:
            y_pos_valid_mask = torch.ones((c, n_pos), dtype=torch.bool, device=dist_pos.device)
        if y_neg_valid_mask is None:
            y_neg_valid_mask = torch.ones((c, n_neg), dtype=torch.bool, device=dist_pos.device)
        x_mask = x_valid_mask[:, :, None]
        y_pos_mask = y_pos_valid_mask[:, None, :]
        y_neg_mask = y_neg_valid_mask[:, None, :]
        if isinstance(self_neg_k, torch.Tensor):
            if self_neg_k.ndim != 1 or self_neg_k.shape[0] != c:
                raise ValueError("self_neg_k tensor must be [C] for 3D mode")
            k_cls = self_neg_k.to(device=dist_pos.device, dtype=torch.long).clamp_min(0)
        else:
            k_cls = torch.full((c,), int(self_neg_k), device=dist_pos.device, dtype=torch.long).clamp_min(0)
        idx_n = torch.arange(n, device=dist_pos.device, dtype=torch.long)[None, :, None]
        idx_q = torch.arange(n_neg, device=dist_pos.device, dtype=torch.long)[None, None, :]
        k_view = torch.minimum(k_cls, torch.full_like(k_cls, min(n, n_neg))).view(c, 1, 1)
        self_mask = (idx_n == idx_q) & (idx_n < k_view)
        y_neg_mask = y_neg_mask & (~self_mask)

    if y_neg_weights is None:
        if dist_pos.ndim == 2:
            y_neg_weights = torch.ones((n_neg,), device=dist_pos.device, dtype=dist_pos.dtype)
        else:
            y_neg_weights = torch.ones((c, n_neg), device=dist_pos.device, dtype=dist_pos.dtype)
    else:
        y_neg_weights = y_neg_weights.to(device=dist_pos.device, dtype=dist_pos.dtype)
        if dist_pos.ndim == 2 and y_neg_weights.shape != (n_neg,):
            raise ValueError(f"y_neg_weights must be [N_neg], got {tuple(y_neg_weights.shape)}")
        if dist_pos.ndim == 3 and y_neg_weights.shape != (c, n_neg):
            raise ValueError(f"y_neg_weights must be [C,N_neg], got {tuple(y_neg_weights.shape)}")

    neg_w_mask = y_neg_weights > 0
    if dist_pos.ndim == 2:
        y_neg_mask = y_neg_mask & neg_w_mask[None, :]
    else:
        y_neg_mask = y_neg_mask & neg_w_mask[:, None, :]

    valid_pos = x_mask & y_pos_mask
    valid_neg = x_mask & y_neg_mask
    if dist_pos.ndim == 2 and self_neg_mask is not None:
        valid_neg = valid_neg & (~self_neg_mask)
    valid_all = torch.cat([valid_pos, valid_neg], dim=-1)

    logit_pos = -dist_pos / tau
    logit_neg = -dist_neg / tau
    if dist_pos.ndim == 2:
        logit_neg = logit_neg + torch.log(y_neg_weights.clamp_min(1e-12))[None, :]
    else:
        logit_neg = logit_neg + torch.log(y_neg_weights.clamp_min(1e-12))[:, None, :]
    logit = torch.cat([logit_pos, logit_neg], dim=-1)

    if normalize_mode == "none":
        a = torch.exp(logit) * valid_all.to(dtype=logit.dtype)
    elif normalize_mode == "y":
        a = _masked_softmax(logit, valid_all, dim=-1)
    elif normalize_mode == "both":
        a_row = _masked_softmax(logit, valid_all, dim=-1)
        a_col = _masked_softmax(logit, valid_all, dim=-2)
        a = torch.sqrt(a_row * a_col.clamp_min(1e-12))
    else:
        raise ValueError(f"unknown normalize_mode: {normalize_mode}")

    n_pos = y_pos.shape[-2]
    a_pos = a[..., :n_pos]
    a_neg = a[..., n_pos:]
    w_pos = a_pos * a_neg.sum(dim=-1, keepdim=True)
    w_neg = a_neg * a_pos.sum(dim=-1, keepdim=True)
    drift_pos = torch.matmul(w_pos, y_pos)
    drift_neg = torch.matmul(w_neg, y_neg)
    v = drift_pos - drift_neg
    if dist_pos.ndim == 2:
        v = torch.where(x_valid_mask[:, None], v, torch.zeros_like(v))
    else:
        v = torch.where(x_valid_mask[:, :, None], v, torch.zeros_like(v))
    return v


def compute_drift(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    normalize_mode: str = "both",
    mask_self_neg: bool = False,
) -> Tuple[torch.Tensor, DriftingStats]:
    """Compute drifting field V for one feature space.

    Args:
        x: [N, D] generated samples
        y_pos: [N_pos, D] positive samples
        y_neg: [N_neg, D] negative samples
        temperature: kernel temperature
        normalize_mode: one of {"none","y","both"}
        mask_self_neg: if True and N==N_neg, masks diagonal in dist_neg
    """
    if x.numel() == 0:
        z = torch.zeros_like(x)
        stats = DriftingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return z, stats
    if y_pos.numel() == 0 or y_neg.numel() == 0:
        z = torch.zeros_like(x)
        stats = DriftingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return z, stats

    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]
    if mask_self_neg and (x.shape[0] == y_neg.shape[0]):
        eye = torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(eye, 1e6)

    return compute_drift_from_distances(
        dist_pos=dist_pos,
        dist_neg=dist_neg,
        y_pos=y_pos,
        y_neg=y_neg,
        temperature=float(temperature),
        normalize_mode=normalize_mode,
    )


def _feature_normalization_scale(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """S_j from paper: avg distance / sqrt(C), stop-grad scalar."""
    c = float(x.shape[-1])
    d = torch.cdist(x, y)
    s = d.mean() / (c**0.5)
    return s.detach().clamp_min(eps)


def _drift_normalization_scale(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """lambda_j from paper: sqrt(E ||V||^2 / C), stop-grad scalar."""
    c = float(v.shape[-1])
    lam = torch.sqrt((v.pow(2).sum(dim=-1) / c).mean())
    return lam.detach().clamp_min(eps)


def drifting_loss_single_feature(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_mode: str = "both",
    mask_self_neg: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Drifting MSE loss for a single feature tensor.

    All inputs are [N, C]-style tensors in same feature space.
    """
    if x.numel() == 0:
        return x.new_zeros(()), {}
    if y_pos.numel() == 0 or y_neg.numel() == 0:
        return x.new_zeros(()), {}

    c = x.shape[-1]
    y_all = torch.cat([y_pos, y_neg], dim=0)
    s = _feature_normalization_scale(x, y_all)
    x_t = x / s
    y_pos_t = y_pos / s
    y_neg_t = y_neg / s

    dist_pos = torch.cdist(x_t, y_pos_t)
    dist_neg = torch.cdist(x_t, y_neg_t)
    if mask_self_neg and (x_t.shape[0] == y_neg_t.shape[0]):
        eye = torch.eye(x_t.shape[0], device=x_t.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(eye, 1e6)

    v_accum = torch.zeros_like(x_t)
    stats_out: Dict[str, float] = {
        "feature_scale": float(s.item()),
    }
    if drift_scale_overrides is not None and len(drift_scale_overrides) != len(temperatures):
        raise ValueError("drift_scale_overrides length must match temperatures length")
    for t_idx, tau in enumerate(temperatures):
        tau_eff = float(tau) * (float(c) ** 0.5)
        v_tau, stats_tau = compute_drift_from_distances(
            dist_pos=dist_pos,
            dist_neg=dist_neg,
            y_pos=y_pos_t,
            y_neg=y_neg_t,
            temperature=tau_eff,
            normalize_mode=normalize_mode,
        )
        lam = _drift_normalization_scale(v_tau)
        v_tau = v_tau / lam
        v_accum = v_accum + v_tau
        key_prefix = f"tau_{tau:g}"
        stats_out[f"{key_prefix}/drift_norm_mean"] = stats_tau.drift_norm_mean
        stats_out[f"{key_prefix}/dist_pos_mean"] = stats_tau.dist_pos_mean
        stats_out[f"{key_prefix}/dist_neg_mean"] = stats_tau.dist_neg_mean
        stats_out[f"{key_prefix}/a_pos_mass_mean"] = stats_tau.a_pos_mass_mean
        stats_out[f"{key_prefix}/a_neg_mass_mean"] = stats_tau.a_neg_mass_mean
        stats_out[f"{key_prefix}/eff_neighbors_pos"] = stats_tau.eff_neighbors_pos
        stats_out[f"{key_prefix}/eff_neighbors_neg"] = stats_tau.eff_neighbors_neg
        stats_out[f"{key_prefix}/drift_scale"] = float(lam.item())

    target = (x_t + v_accum).detach()
    loss = F.mse_loss(x_t, target)
    return loss, stats_out


def drifting_loss_multi_feature(
    x_feats: Sequence[torch.Tensor],
    y_pos_feats: Sequence[torch.Tensor],
    y_neg_feats: Sequence[torch.Tensor],
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_mode: str = "both",
    mask_self_neg: bool = False,
    names: Sequence[str] | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not (len(x_feats) == len(y_pos_feats) == len(y_neg_feats)):
        raise ValueError("feature list lengths must match")
    total = None
    stats: Dict[str, float] = {}
    m = len(x_feats)
    if names is None:
        names = [f"f{i}" for i in range(m)]
    for i in range(m):
        loss_i, stats_i = drifting_loss_single_feature(
            x_feats[i],
            y_pos_feats[i],
            y_neg_feats[i],
            temperatures=temperatures,
            normalize_mode=normalize_mode,
            mask_self_neg=mask_self_neg,
        )
        total = loss_i if total is None else total + loss_i
        prefix = names[i]
        stats[f"{prefix}/loss"] = float(loss_i.item())
        for k, v in stats_i.items():
            stats[f"{prefix}/{k}"] = float(v)
    if total is None:
        total = torch.tensor(0.0)
    return total, stats


def drifting_loss_single_feature_class_batched(
    x: torch.Tensor,
    x_valid_mask: torch.Tensor,
    y_pos: torch.Tensor,
    y_pos_valid_mask: torch.Tensor,
    y_neg: torch.Tensor,
    y_neg_valid_mask: torch.Tensor,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_mode: str = "both",
    y_neg_weights: torch.Tensor | None = None,
    self_neg_k: int | torch.Tensor = 0,
    feature_scale_override: torch.Tensor | None = None,
    drift_scale_overrides: Sequence[torch.Tensor] | None = None,
    return_stats: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Class-batched drifting loss for one feature space.

    Args:
        x: [C, N, D]
        x_valid_mask: [C, N]
        y_pos: [C, P, D]
        y_pos_valid_mask: [C, P]
        y_neg: [C, Q, D]
        y_neg_valid_mask: [C, Q]
    """
    if x.numel() == 0:
        return x.new_zeros(()), {}
    if y_pos.numel() == 0 or y_neg.numel() == 0:
        return x.new_zeros(()), {}
    if x.ndim != 3 or y_pos.ndim != 3 or y_neg.ndim != 3:
        raise ValueError("x/y_pos/y_neg must be rank-3 tensors")
    if x.shape[0] != y_pos.shape[0] or x.shape[0] != y_neg.shape[0]:
        raise ValueError("class dimension mismatch")

    cdim = x.shape[-1]
    dist_pos_raw = torch.cdist(x, y_pos)  # [C,N,P]
    dist_neg_raw = torch.cdist(x, y_neg)  # [C,N,Q]

    if feature_scale_override is None:
        y_all_valid = torch.cat([y_pos_valid_mask, y_neg_valid_mask], dim=1)  # [C,P+Q]
        dist_all_raw = torch.cat([dist_pos_raw, dist_neg_raw], dim=2)  # [C,N,P+Q]
        pair_mask = x_valid_mask[:, :, None] & y_all_valid[:, None, :]
        pair_mask_f = pair_mask.to(dtype=dist_all_raw.dtype)
        s_num = (dist_all_raw * pair_mask_f).sum(dim=(1, 2))
        s_den = pair_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
        s_cls = (s_num / s_den) / (float(cdim) ** 0.5)
        s_cls = s_cls.detach().clamp_min(1e-12)  # [C]
    else:
        if feature_scale_override.ndim != 1 or feature_scale_override.shape[0] != x.shape[0]:
            raise ValueError("feature_scale_override must be [C]")
        s_cls = feature_scale_override.to(device=x.device, dtype=x.dtype).detach().clamp_min(1e-12)

    x_t = x / s_cls[:, None, None]
    y_pos_t = y_pos / s_cls[:, None, None]
    y_neg_t = y_neg / s_cls[:, None, None]
    dist_pos = dist_pos_raw / s_cls[:, None, None]
    dist_neg = dist_neg_raw / s_cls[:, None, None]

    v_accum = torch.zeros_like(x_t)
    stats_out: Dict[str, float] = {}
    if return_stats:
        stats_out["feature_scale"] = float(s_cls.mean().item())
    for tau in temperatures:
        tau_eff = float(tau) * (float(cdim) ** 0.5)
        if return_stats:
            v_tau, stats_tau = compute_drift_from_distances(
                dist_pos=dist_pos,
                dist_neg=dist_neg,
                y_pos=y_pos_t,
                y_neg=y_neg_t,
                temperature=tau_eff,
                normalize_mode=normalize_mode,
                x_valid_mask=x_valid_mask,
                y_pos_valid_mask=y_pos_valid_mask,
                y_neg_valid_mask=y_neg_valid_mask,
                y_neg_weights=y_neg_weights,
                self_neg_k=self_neg_k,
            )
        else:
            v_tau = compute_drift_from_distances_nostats(
                dist_pos=dist_pos,
                dist_neg=dist_neg,
                y_pos=y_pos_t,
                y_neg=y_neg_t,
                temperature=tau_eff,
                normalize_mode=normalize_mode,
                x_valid_mask=x_valid_mask,
                y_pos_valid_mask=y_pos_valid_mask,
                y_neg_valid_mask=y_neg_valid_mask,
                y_neg_weights=y_neg_weights,
                self_neg_k=self_neg_k,
            )
        if drift_scale_overrides is None:
            v_tau_sq = (v_tau.pow(2).sum(dim=-1) / float(cdim))  # [C,N]
            x_mask_f = x_valid_mask.to(dtype=v_tau.dtype)
            lam_num = (v_tau_sq * x_mask_f).sum(dim=1)
            lam_den = x_mask_f.sum(dim=1).clamp_min(1.0)
            lam_cls = torch.sqrt(lam_num / lam_den).detach().clamp_min(1e-12)  # [C]
        else:
            lam_cls = drift_scale_overrides[t_idx]
            if lam_cls.ndim != 1 or lam_cls.shape[0] != x.shape[0]:
                raise ValueError("each drift_scale_overrides item must be [C]")
            lam_cls = lam_cls.to(device=x.device, dtype=x.dtype).detach().clamp_min(1e-12)
        v_tau = v_tau / lam_cls[:, None, None]
        v_accum = v_accum + v_tau
        if return_stats:
            key_prefix = f"tau_{tau:g}"
            stats_out[f"{key_prefix}/drift_norm_mean"] = stats_tau.drift_norm_mean
            stats_out[f"{key_prefix}/dist_pos_mean"] = stats_tau.dist_pos_mean
            stats_out[f"{key_prefix}/dist_neg_mean"] = stats_tau.dist_neg_mean
            stats_out[f"{key_prefix}/a_pos_mass_mean"] = stats_tau.a_pos_mass_mean
            stats_out[f"{key_prefix}/a_neg_mass_mean"] = stats_tau.a_neg_mass_mean
            stats_out[f"{key_prefix}/eff_neighbors_pos"] = stats_tau.eff_neighbors_pos
            stats_out[f"{key_prefix}/eff_neighbors_neg"] = stats_tau.eff_neighbors_neg
            stats_out[f"{key_prefix}/drift_scale"] = float(lam_cls.mean().item())

    target = (x_t + v_accum).detach()
    sq = (x_t - target).pow(2)
    x_mask = x_valid_mask.to(dtype=sq.dtype).unsqueeze(-1)
    num = (sq * x_mask).sum(dim=(1, 2))
    den = (x_valid_mask.sum(dim=1).to(dtype=sq.dtype) * float(cdim)).clamp_min(1e-12)
    loss = (num / den).mean()
    return loss, stats_out


def drifting_loss_single_feature_class_batched_lossonly(
    x: torch.Tensor,
    x_valid_mask: torch.Tensor,
    y_pos: torch.Tensor,
    y_pos_valid_mask: torch.Tensor,
    y_neg: torch.Tensor,
    y_neg_valid_mask: torch.Tensor,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_mode: str = "both",
    y_neg_weights: torch.Tensor | None = None,
    self_neg_k: int | torch.Tensor = 0,
    feature_scale_override: torch.Tensor | None = None,
    drift_scale_overrides: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Loss-only fast path for class-batched drifting."""
    loss, _ = drifting_loss_single_feature_class_batched(
        x=x,
        x_valid_mask=x_valid_mask,
        y_pos=y_pos,
        y_pos_valid_mask=y_pos_valid_mask,
        y_neg=y_neg,
        y_neg_valid_mask=y_neg_valid_mask,
        temperatures=temperatures,
        normalize_mode=normalize_mode,
        y_neg_weights=y_neg_weights,
        self_neg_k=self_neg_k,
        feature_scale_override=feature_scale_override,
        drift_scale_overrides=drift_scale_overrides,
        return_stats=False,
    )
    return loss


def drifting_loss_feature_group_class_batched(
    x_feats: Sequence[torch.Tensor],
    x_valid_masks: Sequence[torch.Tensor],
    y_pos_feats: Sequence[torch.Tensor],
    y_pos_valid_masks: Sequence[torch.Tensor],
    y_neg_feats: Sequence[torch.Tensor],
    y_neg_valid_masks: Sequence[torch.Tensor],
    *,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    normalize_mode: str = "both",
    y_neg_weights: Sequence[torch.Tensor | None] | None = None,
    self_neg_k: Sequence[int | torch.Tensor] | None = None,
    share_feature_scale: bool = True,
    share_drift_scale: bool = True,
    return_stats: bool = True,
    names: Sequence[str] | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, float]]:
    """Compute drifting loss for a feature group.

    When `share_feature_scale`/`share_drift_scale` are enabled, scales are shared
    across all features in this group (paper-style for multi-location features).
    """
    m = len(x_feats)
    if m == 0:
        z = torch.tensor(0.0)
        return z, [], {}
    if not (
        len(x_valid_masks) == len(y_pos_feats) == len(y_pos_valid_masks) == len(y_neg_feats) == len(y_neg_valid_masks) == m
    ):
        raise ValueError("feature list lengths must match")
    if names is None:
        names = [f"f{i}" for i in range(m)]
    if len(names) != m:
        raise ValueError("names length mismatch")

    if y_neg_weights is None:
        y_neg_weights = [None] * m
    if self_neg_k is None:
        self_neg_k = [0] * m
    if len(y_neg_weights) != m or len(self_neg_k) != m:
        raise ValueError("y_neg_weights/self_neg_k length mismatch")

    # Validate class dimension and infer shared dimensionality.
    c_count = int(x_feats[0].shape[0])
    cdim = int(x_feats[0].shape[-1])
    for i in range(m):
        if x_feats[i].shape[0] != c_count:
            raise ValueError("class dimension mismatch in x_feats")
        if y_pos_feats[i].shape[0] != c_count or y_neg_feats[i].shape[0] != c_count:
            raise ValueError("class dimension mismatch in y_pos_feats/y_neg_feats")
        if share_feature_scale or share_drift_scale:
            if x_feats[i].shape[-1] != cdim:
                raise ValueError("shared normalization requires same feature dim across group")

    # Precompute distances and feature scales.
    dist_pos_raws: List[torch.Tensor] = []
    dist_neg_raws: List[torch.Tensor] = []
    s_num_list: List[torch.Tensor] = []
    s_den_list: List[torch.Tensor] = []
    for i in range(m):
        x_i = x_feats[i]
        yp_i = y_pos_feats[i]
        yn_i = y_neg_feats[i]
        xp_mask_i = x_valid_masks[i]
        yp_mask_i = y_pos_valid_masks[i]
        yn_mask_i = y_neg_valid_masks[i]

        dist_pos_raw_i = torch.cdist(x_i, yp_i)
        dist_neg_raw_i = torch.cdist(x_i, yn_i)
        dist_pos_raws.append(dist_pos_raw_i)
        dist_neg_raws.append(dist_neg_raw_i)

        y_all_valid = torch.cat([yp_mask_i, yn_mask_i], dim=1)
        dist_all_raw = torch.cat([dist_pos_raw_i, dist_neg_raw_i], dim=2)
        pair_mask = xp_mask_i[:, :, None] & y_all_valid[:, None, :]
        pair_mask_f = pair_mask.to(dtype=dist_all_raw.dtype)
        s_num_list.append((dist_all_raw * pair_mask_f).sum(dim=(1, 2)))
        s_den_list.append(pair_mask_f.sum(dim=(1, 2)).clamp_min(1.0))

    if share_feature_scale:
        s_num = torch.stack(s_num_list, dim=0).sum(dim=0)
        s_den = torch.stack(s_den_list, dim=0).sum(dim=0).clamp_min(1.0)
        s_shared = ((s_num / s_den) / (float(cdim) ** 0.5)).detach().clamp_min(1e-12)
        s_list = [s_shared for _ in range(m)]
    else:
        s_list = [((s_num_list[i] / s_den_list[i]) / (float(x_feats[i].shape[-1]) ** 0.5)).detach().clamp_min(1e-12) for i in range(m)]

    # Scale features/distances once; reuse across temperatures.
    x_t_list: List[torch.Tensor] = []
    y_pos_t_list: List[torch.Tensor] = []
    y_neg_t_list: List[torch.Tensor] = []
    dist_pos_list: List[torch.Tensor] = []
    dist_neg_list: List[torch.Tensor] = []
    for i in range(m):
        s_i = s_list[i]
        x_t_list.append(x_feats[i] / s_i[:, None, None])
        y_pos_t_list.append(y_pos_feats[i] / s_i[:, None, None])
        y_neg_t_list.append(y_neg_feats[i] / s_i[:, None, None])
        dist_pos_list.append(dist_pos_raws[i] / s_i[:, None, None])
        dist_neg_list.append(dist_neg_raws[i] / s_i[:, None, None])

    v_accum = [torch.zeros_like(x_t_list[i]) for i in range(m)]
    feat_stats: Dict[str, float] = {}
    if return_stats:
        for i, n in enumerate(names):
            feat_stats[f"{n}/feature_scale"] = float(s_list[i].mean().item())
        if share_feature_scale:
            feat_stats["group/feature_scale"] = float(s_shared.mean().item())

    for t_idx, tau in enumerate(temperatures):
        tau_eff = float(tau) * (float(cdim) ** 0.5)
        v_tau_list: List[torch.Tensor] = []
        stats_tau_list: List[DriftingStats] = []
        lam_num_list: List[torch.Tensor] = []
        lam_den_list: List[torch.Tensor] = []
        for i in range(m):
            if return_stats:
                v_tau_i, s_tau_i = compute_drift_from_distances(
                    dist_pos=dist_pos_list[i],
                    dist_neg=dist_neg_list[i],
                    y_pos=y_pos_t_list[i],
                    y_neg=y_neg_t_list[i],
                    temperature=tau_eff,
                    normalize_mode=normalize_mode,
                    x_valid_mask=x_valid_masks[i],
                    y_pos_valid_mask=y_pos_valid_masks[i],
                    y_neg_valid_mask=y_neg_valid_masks[i],
                    y_neg_weights=y_neg_weights[i],
                    self_neg_k=self_neg_k[i],
                )
                stats_tau_list.append(s_tau_i)
            else:
                v_tau_i = compute_drift_from_distances_nostats(
                    dist_pos=dist_pos_list[i],
                    dist_neg=dist_neg_list[i],
                    y_pos=y_pos_t_list[i],
                    y_neg=y_neg_t_list[i],
                    temperature=tau_eff,
                    normalize_mode=normalize_mode,
                    x_valid_mask=x_valid_masks[i],
                    y_pos_valid_mask=y_pos_valid_masks[i],
                    y_neg_valid_mask=y_neg_valid_masks[i],
                    y_neg_weights=y_neg_weights[i],
                    self_neg_k=self_neg_k[i],
                )
            v_tau_list.append(v_tau_i)
            cdim_i = float(x_feats[i].shape[-1])
            v_tau_sq = (v_tau_i.pow(2).sum(dim=-1) / cdim_i)
            x_mask_f = x_valid_masks[i].to(dtype=v_tau_i.dtype)
            lam_num_list.append((v_tau_sq * x_mask_f).sum(dim=1))
            lam_den_list.append(x_mask_f.sum(dim=1).clamp_min(1.0))

        if share_drift_scale:
            lam_num = torch.stack(lam_num_list, dim=0).sum(dim=0)
            lam_den = torch.stack(lam_den_list, dim=0).sum(dim=0).clamp_min(1.0)
            lam_shared = torch.sqrt(lam_num / lam_den).detach().clamp_min(1e-12)
            lam_list = [lam_shared for _ in range(m)]
        else:
            lam_list = [torch.sqrt(lam_num_list[i] / lam_den_list[i]).detach().clamp_min(1e-12) for i in range(m)]

        for i in range(m):
            v_accum[i] = v_accum[i] + (v_tau_list[i] / lam_list[i][:, None, None])
            if return_stats:
                key_prefix = f"tau_{tau:g}"
                st = stats_tau_list[i]
                n = names[i]
                feat_stats[f"{n}/{key_prefix}/drift_norm_mean"] = st.drift_norm_mean
                feat_stats[f"{n}/{key_prefix}/dist_pos_mean"] = st.dist_pos_mean
                feat_stats[f"{n}/{key_prefix}/dist_neg_mean"] = st.dist_neg_mean
                feat_stats[f"{n}/{key_prefix}/a_pos_mass_mean"] = st.a_pos_mass_mean
                feat_stats[f"{n}/{key_prefix}/a_neg_mass_mean"] = st.a_neg_mass_mean
                feat_stats[f"{n}/{key_prefix}/eff_neighbors_pos"] = st.eff_neighbors_pos
                feat_stats[f"{n}/{key_prefix}/eff_neighbors_neg"] = st.eff_neighbors_neg
                feat_stats[f"{n}/{key_prefix}/drift_scale"] = float(lam_list[i].mean().item())
        if return_stats and share_drift_scale:
            feat_stats[f"group/tau_{tau:g}/drift_scale"] = float(lam_shared.mean().item())

    losses: List[torch.Tensor] = []
    total = None
    for i in range(m):
        target = (x_t_list[i] + v_accum[i]).detach()
        sq = (x_t_list[i] - target).pow(2)
        x_mask = x_valid_masks[i].to(dtype=sq.dtype).unsqueeze(-1)
        cdim_i = float(x_feats[i].shape[-1])
        num = (sq * x_mask).sum(dim=(1, 2))
        den = (x_valid_masks[i].sum(dim=1).to(dtype=sq.dtype) * cdim_i).clamp_min(1e-12)
        loss_i = (num / den).mean()
        losses.append(loss_i)
        total = loss_i if total is None else total + loss_i
        if return_stats:
            feat_stats[f"{names[i]}/loss"] = float(loss_i.item())

    if total is None:
        total = x_feats[0].new_zeros(())
    return total, losses, feat_stats


def merge_stats_mean(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for d in items:
        for k, v in d.items():
            acc.setdefault(k, []).append(float(v))
    out = {k: float(sum(vs) / max(1, len(vs))) for k, vs in acc.items()}
    return out


__all__ = [
    "compute_drift",
    "compute_drift_from_distances",
    "compute_drift_from_distances_nostats",
    "drifting_loss_single_feature",
    "drifting_loss_multi_feature",
    "drifting_loss_single_feature_class_batched",
    "drifting_loss_single_feature_class_batched_lossonly",
    "drifting_loss_feature_group_class_batched",
    "merge_stats_mean",
    "DriftingStats",
]
