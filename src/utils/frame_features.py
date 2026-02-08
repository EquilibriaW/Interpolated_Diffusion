from __future__ import annotations

import torch


def frame_features_from_mask(mask: torch.Tensor, *, include_time: bool = True) -> torch.Tensor:
    """Build per-frame geometric features from a boolean keyframe/anchor mask.

    mask: [B,T] bool (True = anchor/keyframe)
    Returns: feat [B,T,F] float32 where:
      - (optional) t_norm in [0,1]
      - is_anchor in {0,1}
      - alpha in [0,1] within the enclosing anchor segment
      - gap_norm in [0,1]
      - dist_mid in [0,1] where 0 at anchors and ~1 near segment midpoints
    """
    if mask.dim() != 2:
        raise ValueError("mask must be [B,T]")
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    B, T = mask.shape
    device = mask.device
    if T <= 1:
        # Degenerate; just emit zeros.
        feat_dim = 5 if include_time else 4
        return torch.zeros((B, T, feat_dim), device=device, dtype=torch.float32)

    # Ensure each sample has at least one anchor. If not, default to endpoints.
    has_any = mask.any(dim=1)
    if not bool(has_any.all()):
        mask = mask.clone()
        bad = ~has_any
        mask[bad, 0] = True
        mask[bad, -1] = True

    t = torch.arange(T, device=device, dtype=torch.float32).view(1, T).expand(B, T)
    neg_inf = torch.full((B, T), -1e9, device=device, dtype=torch.float32)
    pos_inf = torch.full((B, T), 1e9, device=device, dtype=torch.float32)

    first = mask.float().argmax(dim=1).float().view(B, 1)  # [B,1]
    last = (T - 1 - torch.flip(mask, dims=[1]).float().argmax(dim=1)).float().view(B, 1)

    left_vals = torch.where(mask, t, neg_inf)
    left = torch.cummax(left_vals, dim=1).values
    right_vals = torch.where(mask, t, pos_inf)
    right = torch.flip(torch.cummin(torch.flip(right_vals, dims=[1]), dim=1).values, dims=[1])

    # Fix invalid regions (before first anchor / after last anchor).
    left = torch.where(left < 0.0, first.expand_as(left), left)
    right = torch.where(right > float(T - 1), last.expand_as(right), right)

    gap = (right - left).clamp(min=1.0)
    alpha = ((t - left) / gap).clamp(0.0, 1.0)
    dist_left = (t - left).clamp(min=0.0)
    dist_right = (right - t).clamp(min=0.0)
    dist = torch.minimum(dist_left, dist_right)
    dist_mid = (2.0 * dist / gap).clamp(0.0, 1.0)
    gap_norm = gap / float(max(1, T - 1))
    is_anchor = mask.to(dtype=torch.float32)

    feats = [is_anchor, alpha, gap_norm, dist_mid]
    if include_time:
        t_norm = t / float(max(1, T - 1))
        feats = [t_norm] + feats
    return torch.stack(feats, dim=-1)


__all__ = ["frame_features_from_mask"]

