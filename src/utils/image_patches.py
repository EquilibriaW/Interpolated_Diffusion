from __future__ import annotations

from typing import List, Tuple

import torch


def patchify_images(x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Patchify images into tokens.

    Args:
        x: [B, C, H, W]
        patch_size: square patch size
    Returns:
        tokens: [B, N, D] where N = (H/P)*(W/P), D = C*P*P
        grid_hw: (H_p, W_p)
    """
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    b, c, h, w = x.shape
    p = int(patch_size)
    if p <= 0:
        raise ValueError("patch_size must be > 0")
    if (h % p) != 0 or (w % p) != 0:
        raise ValueError(f"image size must be divisible by patch_size: H={h}, W={w}, P={p}")
    hp, wp = h // p, w // p
    # [B, C, H/P, P, W/P, P] -> [B, H/P, W/P, C, P, P] -> [B, N, D]
    x_view = x.view(b, c, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    tokens = x_view.view(b, hp * wp, c * p * p)
    return tokens, (hp, wp)


def unpatchify_images(tokens: torch.Tensor, patch_size: int, grid_hw: Tuple[int, int], channels: int) -> torch.Tensor:
    """Inverse of patchify_images.

    Args:
        tokens: [B, N, D]
        patch_size: P
        grid_hw: (H/P, W/P)
        channels: C
    Returns:
        x: [B, C, H, W]
    """
    if tokens.dim() != 3:
        raise ValueError("tokens must be [B,N,D]")
    b, n, d = tokens.shape
    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    p = int(patch_size)
    c = int(channels)
    if hp * wp != n:
        raise ValueError(f"grid size mismatch: hp*wp={hp*wp}, n={n}")
    if c * p * p != d:
        raise ValueError(f"token dim mismatch: expected {c*p*p}, got {d}")
    x = tokens.view(b, hp, wp, c, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(b, c, hp * p, wp * p)
    return x


def _compute_k_schedule(
    n_tokens: int,
    k_min: int,
    levels: int,
    schedule: str = "geom",
    geom_gamma: float | None = None,
) -> List[int]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    k_min = max(1, min(int(k_min), int(n_tokens)))
    k_list = [0 for _ in range(levels + 1)]
    k_list[levels] = k_min
    if levels <= 0:
        return k_list
    if schedule == "doubling":
        for s in range(levels, 0, -1):
            k_prev = min(n_tokens, max(k_list[s] + 1, 2 * k_list[s]))
            k_list[s - 1] = k_prev
        return k_list
    if schedule == "linear":
        for s in range(levels - 1, -1, -1):
            frac = float(levels - s) / float(levels)
            target = int(round(k_min + frac * (n_tokens - k_min)))
            k_prev = min(n_tokens, max(k_list[s + 1] + 1, target))
            k_list[s] = k_prev
        return k_list
    if schedule == "geom":
        if geom_gamma is None:
            geom_gamma = (float(n_tokens) / float(k_min)) ** (1.0 / float(levels)) if k_min > 0 else 1.0
        for s in range(levels - 1, -1, -1):
            exp = float(levels - s)
            target = int(round(k_min * (geom_gamma ** exp)))
            k_prev = min(n_tokens, max(k_list[s + 1] + 1, target))
            k_list[s] = k_prev
        return k_list
    raise ValueError(f"unknown k schedule: {schedule}")


def build_nested_patch_masks_batch(
    batch_size: int,
    n_tokens: int,
    k_min: int,
    levels: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    k_schedule: str = "geom",
    k_geom_gamma: float | None = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Sample nested patch-anchor masks.

    Returns:
        masks_levels: [B, levels+1, N] bool
        idx_levels: list length levels+1, each [B, K_s] long
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    device = device or torch.device("cpu")
    b = int(batch_size)
    n = int(n_tokens)
    k_list = _compute_k_schedule(n, k_min, levels, schedule=k_schedule, geom_gamma=k_geom_gamma)

    scores = torch.rand((b, n), generator=generator, device=device)
    perm = torch.argsort(scores, dim=1)

    masks_levels = torch.zeros((b, levels + 1, n), dtype=torch.bool, device=device)
    idx_levels: List[torch.Tensor] = []
    for s in range(levels + 1):
        k_s = int(k_list[s])
        idx = torch.sort(perm[:, :k_s], dim=1).values
        idx_levels.append(idx)
        masks_levels[:, s].scatter_(1, idx, True)
    return masks_levels, idx_levels


def _grid_coords(hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    h, w = int(hw[0]), int(hw[1])
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    return coords  # [N,2]


def interpolate_patch_tokens_idw(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    grid_hw: Tuple[int, int],
    power: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Interpolate missing patch tokens from anchor tokens using inverse-distance weighting.

    Args:
        tokens: [B, N, D]
        mask: [B, N] bool (True = anchor token kept)
        grid_hw: patch grid (H_p, W_p)
    Returns:
        z_interp: [B, N, D]
    """
    if tokens.dim() != 3:
        raise ValueError("tokens must be [B,N,D]")
    if mask.dim() != 2:
        raise ValueError("mask must be [B,N]")
    if tokens.shape[:2] != mask.shape:
        raise ValueError("tokens/mask shape mismatch")
    b, n, _ = tokens.shape
    device = tokens.device
    mask = mask.to(dtype=torch.bool, device=device)

    coords = _grid_coords(grid_hw, device=device)  # [N,2]
    dist = torch.cdist(coords, coords).clamp_min(eps)  # [N,N]

    out = tokens.clone()
    for i in range(b):
        anchor_idx = torch.where(mask[i])[0]
        if anchor_idx.numel() == 0:
            # Fallback: no anchors, keep original tokens.
            continue
        if anchor_idx.numel() == n:
            continue
        vals = tokens[i, anchor_idx]  # [K,D]
        w = (dist[:, anchor_idx]).pow(-float(power))  # [N,K]
        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        interp = w @ vals  # [N,D]
        out[i] = torch.where(mask[i].unsqueeze(-1), tokens[i], interp)
    return out


def _grid_neighbor_pairs(
    grid_hw: Tuple[int, int],
    device: torch.device,
    include_diag: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return undirected neighbor edge list for a 2D grid as (u, v, dist2)."""
    h, w = int(grid_hw[0]), int(grid_hw[1])
    pairs_u: List[int] = []
    pairs_v: List[int] = []
    dist2: List[float] = []
    for y in range(h):
        for x in range(w):
            u = y * w + x
            if x + 1 < w:
                v = y * w + (x + 1)
                pairs_u.append(u)
                pairs_v.append(v)
                dist2.append(1.0)
            if y + 1 < h:
                v = (y + 1) * w + x
                pairs_u.append(u)
                pairs_v.append(v)
                dist2.append(1.0)
            if include_diag:
                if x + 1 < w and y + 1 < h:
                    v = (y + 1) * w + (x + 1)
                    pairs_u.append(u)
                    pairs_v.append(v)
                    dist2.append(2.0)
                if x - 1 >= 0 and y + 1 < h:
                    v = (y + 1) * w + (x - 1)
                    pairs_u.append(u)
                    pairs_v.append(v)
                    dist2.append(2.0)
    if not pairs_u:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    u = torch.tensor(pairs_u, dtype=torch.long, device=device)
    v = torch.tensor(pairs_v, dtype=torch.long, device=device)
    d2 = torch.tensor(dist2, dtype=torch.float32, device=device)
    return u, v, d2


def interpolate_patch_tokens_laplacian(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    grid_hw: Tuple[int, int],
    *,
    init_mode: str = "idw",
    idw_power: float = 2.0,
    sigma_feature: float = 0.20,
    sigma_space: float = 1.0,
    lambda_reg: float = 1e-4,
    include_diag_neighbors: bool = False,
    min_edge_weight: float = 1e-6,
) -> torch.Tensor:
    """Edge-aware harmonic interpolation on a 2D patch graph.

    Solve for unknown nodes U:
        (L_UU + lambda I) X_U = -L_UK X_K
    where graph weights are edge-aware from an initialization (IDW by default).
    """
    if tokens.dim() != 3:
        raise ValueError("tokens must be [B,N,D]")
    if mask.dim() != 2:
        raise ValueError("mask must be [B,N]")
    if tokens.shape[:2] != mask.shape:
        raise ValueError("tokens/mask shape mismatch")
    b, n, d = tokens.shape
    device = tokens.device
    dtype = tokens.dtype
    mask = mask.to(dtype=torch.bool, device=device)

    if init_mode == "idw":
        init = interpolate_patch_tokens_idw(tokens, mask, grid_hw=grid_hw, power=idw_power)
    elif init_mode == "copy":
        init = tokens.clone()
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")

    u_idx_edge, v_idx_edge, dist2 = _grid_neighbor_pairs(
        grid_hw=grid_hw,
        device=device,
        include_diag=bool(include_diag_neighbors),
    )
    if u_idx_edge.numel() == 0:
        return init

    out = init.clone()
    sigma_f2 = max(float(sigma_feature) ** 2, 1e-8)
    sigma_s2 = max(float(sigma_space) ** 2, 1e-8)
    min_w = float(min_edge_weight)
    reg = float(lambda_reg)

    for i in range(b):
        known = mask[i]
        unknown = ~known
        if not bool(unknown.any()):
            out[i] = tokens[i]
            continue
        if not bool(known.any()):
            # No anchors; keep initialized interpolation.
            out[i] = init[i]
            continue

        fi = init[i]  # [N,D]
        d_feat2 = (fi[u_idx_edge] - fi[v_idx_edge]).pow(2).mean(dim=-1)
        w_feat = torch.exp(-d_feat2 / (2.0 * sigma_f2))
        w_space = torch.exp(-dist2 / (2.0 * sigma_s2))
        w_edge = (w_feat * w_space).clamp_min(min_w)  # [E]

        # Build dense Laplacian (N is typically small: e.g., 16x16=256).
        L = torch.zeros((n, n), device=device, dtype=dtype)
        u = u_idx_edge
        v = v_idx_edge
        L[u, u] += w_edge
        L[v, v] += w_edge
        L[u, v] -= w_edge
        L[v, u] -= w_edge

        u_idx = torch.where(unknown)[0]
        k_idx = torch.where(known)[0]
        L_uu = L[u_idx][:, u_idx]
        L_uk = L[u_idx][:, k_idx]
        rhs = -L_uk @ tokens[i, k_idx]  # [U,D], anchors as hard constraints
        if reg > 0.0:
            L_uu = L_uu + reg * torch.eye(L_uu.shape[0], device=device, dtype=dtype)
        try:
            x_u = torch.linalg.solve(L_uu, rhs)  # [U,D]
        except RuntimeError:
            # Fallback to initialized values if system is singular/ill-conditioned.
            x_u = init[i, u_idx]

        out_i = init[i].clone()
        out_i[k_idx] = tokens[i, k_idx]
        out_i[u_idx] = x_u
        out[i] = out_i
    return out


__all__ = [
    "patchify_images",
    "unpatchify_images",
    "build_nested_patch_masks_batch",
    "interpolate_patch_tokens_idw",
    "interpolate_patch_tokens_laplacian",
]
