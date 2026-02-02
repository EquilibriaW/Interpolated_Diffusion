from typing import List, Tuple

import torch


def sample_fixed_k_mask(
    T: int,
    K: int,
    generator: torch.Generator = None,
    device: torch.device = None,
    ensure_endpoints: bool = True,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    if T <= 0:
        raise ValueError("T must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if ensure_endpoints:
        if T < 2:
            raise ValueError("T must be >= 2 when ensure_endpoints is True")
        if K < 2:
            raise ValueError("K must be >= 2 when ensure_endpoints is True")
    K = min(K, T)
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    if ensure_endpoints:
        mask[0] = True
        mask[T - 1] = True
        remaining = K - 2
        if remaining > 0 and T > 2:
            candidates = torch.arange(1, T - 1, device=device)
            perm = torch.randperm(candidates.numel(), generator=generator, device=device)
            chosen = candidates[perm[:remaining]]
            mask[chosen] = True
    else:
        candidates = torch.arange(0, T, device=device)
        perm = torch.randperm(candidates.numel(), generator=generator, device=device)
        chosen = candidates[perm[:K]]
        mask[chosen] = True
    return mask


def sample_fixed_k_indices_batch(
    B: int,
    T: int,
    K: int,
    generator: torch.Generator = None,
    device: torch.device = None,
    ensure_endpoints: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or torch.device("cpu")
    if T <= 0:
        raise ValueError("T must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if ensure_endpoints:
        if T < 2:
            raise ValueError("T must be >= 2 when ensure_endpoints is True")
        if K < 2:
            raise ValueError("K must be >= 2 when ensure_endpoints is True")
    K = min(K, T)
    if ensure_endpoints and T > 2 and K > 2:
        scores = torch.rand((B, T - 2), generator=generator, device=device)
        perm = torch.argsort(scores, dim=1)
        chosen = perm[:, : K - 2] + 1
        idx = torch.cat(
            [torch.zeros((B, 1), device=device, dtype=torch.long), chosen, torch.full((B, 1), T - 1, device=device, dtype=torch.long)],
            dim=1,
        )
    elif ensure_endpoints:
        idx = torch.cat(
            [torch.zeros((B, 1), device=device, dtype=torch.long), torch.full((B, 1), T - 1, device=device, dtype=torch.long)],
            dim=1,
        )
    else:
        scores = torch.rand((B, T), generator=generator, device=device)
        perm = torch.argsort(scores, dim=1)
        idx = perm[:, :K]
    idx = torch.sort(idx, dim=1).values
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return idx, mask


def sample_fixed_k_indices_uniform_batch(
    B: int,
    T: int,
    K: int,
    generator: torch.Generator = None,
    device: torch.device = None,
    ensure_endpoints: bool = True,
    jitter: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or torch.device("cpu")
    if T <= 0:
        raise ValueError("T must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if ensure_endpoints:
        if T < 2:
            raise ValueError("T must be >= 2 when ensure_endpoints is True")
        if K < 2:
            raise ValueError("K must be >= 2 when ensure_endpoints is True")
    K = min(K, T)
    if K > T:
        raise ValueError("K must be <= T for uniform spacing")
    base = torch.linspace(0, T - 1, K, device=device)
    if jitter and K > 2 and T > 2:
        spacing = float(T - 1) / float(K - 1)
        max_jitter = spacing * float(jitter) * 0.5
        noise = (torch.rand((B, K), generator=generator, device=device) - 0.5) * 2.0 * max_jitter
        noise[:, 0] = 0.0
        noise[:, -1] = 0.0
        pos = base.unsqueeze(0) + noise
    else:
        pos = base.unsqueeze(0).expand(B, -1)
    idx = torch.round(pos).long()
    idx = idx.clamp(0, T - 1)
    if ensure_endpoints and K >= 2:
        idx[:, 0] = 0
        idx[:, -1] = T - 1
    # Enforce strictly increasing indices.
    for k in range(1, K):
        idx[:, k] = torch.maximum(idx[:, k], idx[:, k - 1] + 1)
    for k in range(K - 2, -1, -1):
        idx[:, k] = torch.minimum(idx[:, k], idx[:, k + 1] - 1)
    idx = idx.clamp(0, T - 1)
    if ensure_endpoints and K >= 2:
        idx[:, 0] = 0
        idx[:, -1] = T - 1
    mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return idx, mask


def _compute_k_schedule(
    T: int,
    K_min: int,
    levels: int,
    schedule: str = "doubling",
    geom_gamma: float = None,
) -> List[int]:
    K_min = min(K_min, T)
    K_list = [0 for _ in range(levels + 1)]
    K_list[levels] = K_min
    if levels <= 0:
        return K_list
    if schedule == "doubling":
        for s in range(levels, 0, -1):
            K_prev = min(T, max(K_list[s] + 1, 2 * K_list[s]))
            K_list[s - 1] = K_prev
        return K_list
    if schedule == "linear":
        for s in range(levels - 1, -1, -1):
            frac = float(levels - s) / float(levels)
            target = int(round(K_min + frac * (T - K_min)))
            K_prev = min(T, max(K_list[s + 1] + 1, target))
            K_list[s] = K_prev
        return K_list
    if schedule == "geom":
        if geom_gamma is None:
            geom_gamma = (float(T) / float(K_min)) ** (1.0 / float(levels)) if K_min > 0 else 1.0
        for s in range(levels - 1, -1, -1):
            exp = float(levels - s)
            target = int(round(K_min * (geom_gamma ** exp)))
            K_prev = min(T, max(K_list[s + 1] + 1, target))
            K_list[s] = K_prev
        return K_list
    raise ValueError(f"Unknown k schedule: {schedule}")
    return K_list


def build_nested_masks_batch(
    B: int,
    T: int,
    K_min: int,
    levels: int,
    generator: torch.Generator = None,
    device: torch.device = None,
    k_schedule: str = "doubling",
    k_geom_gamma: float = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    device = device or torch.device("cpu")
    K_list = _compute_k_schedule(T, K_min, levels, schedule=k_schedule, geom_gamma=k_geom_gamma)
    if T < 2:
        raise ValueError("T must be >= 2 when using endpoints")
    scores = torch.rand((B, T - 2), generator=generator, device=device)
    perm = torch.argsort(scores, dim=1)

    masks_levels = torch.zeros((B, levels + 1, T), dtype=torch.bool, device=device)
    idx_levels: List[torch.Tensor] = []
    for s in range(levels + 1):
        K_s = K_list[s]
        if K_s <= 2 or T <= 2:
            idx = torch.cat(
                [torch.zeros((B, 1), device=device, dtype=torch.long), torch.full((B, 1), T - 1, device=device, dtype=torch.long)],
                dim=1,
            )
        else:
            interior = perm[:, : K_s - 2] + 1
            idx = torch.cat(
                [torch.zeros((B, 1), device=device, dtype=torch.long), interior, torch.full((B, 1), T - 1, device=device, dtype=torch.long)],
                dim=1,
            )
        idx = torch.sort(idx, dim=1).values
        idx_levels.append(idx)
        masks_levels[:, s].scatter_(1, idx, True)
    return masks_levels, idx_levels


def build_nested_masks_from_base(
    idx_base: torch.Tensor,
    T: int,
    levels: int,
    generator: torch.Generator = None,
    device: torch.device = None,
    k_schedule: str = "doubling",
    k_geom_gamma: float = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if idx_base.dim() != 2:
        raise ValueError("idx_base must be [B, K]")
    device = device or idx_base.device
    B, K_base = idx_base.shape
    K_list = _compute_k_schedule(T, K_base, levels, schedule=k_schedule, geom_gamma=k_geom_gamma)
    masks_levels = torch.zeros((B, levels + 1, T), dtype=torch.bool, device=device)
    idx_levels: List[torch.Tensor] = [None for _ in range(levels + 1)]
    # Level S (coarsest) fixed by idx_base.
    idx_s = torch.sort(idx_base, dim=1).values
    idx_levels[levels] = idx_s
    masks_levels[:, levels].scatter_(1, idx_s, True)
    # Add anchors as we move to finer levels.
    for s in range(levels - 1, -1, -1):
        K_s = K_list[s]
        prev_mask = masks_levels[:, s + 1]
        need = K_s - prev_mask.sum(dim=1)
        idx_list = []
        for b in range(B):
            k_need = int(need[b].item())
            if k_need <= 0:
                idx_list.append(idx_levels[s + 1][b])
                continue
            available = torch.where(~prev_mask[b])[0]
            if available.numel() == 0:
                idx_list.append(idx_levels[s + 1][b])
                continue
            perm = torch.randperm(available.numel(), generator=generator, device=device)
            chosen = available[perm[:k_need]]
            idx_new = torch.cat([idx_levels[s + 1][b], chosen], dim=0)
            idx_new = torch.sort(idx_new).values
            idx_list.append(idx_new)
        idx_tensor = torch.stack(idx_list, dim=0)
        idx_levels[s] = idx_tensor
        masks_levels[:, s].scatter_(1, idx_tensor, True)
    return masks_levels, idx_levels


def build_nested_masks_from_logits(
    logits: torch.Tensor,
    K_min: int,
    levels: int,
    k_schedule: str = "doubling",
    k_geom_gamma: float = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if logits.dim() != 2:
        raise ValueError("logits must be [B, T]")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    B, T = logits.shape
    if T < 2:
        raise ValueError("T must be >= 2 when using endpoints")
    device = logits.device
    K_list = _compute_k_schedule(T, K_min, levels, schedule=k_schedule, geom_gamma=k_geom_gamma)
    if K_list[levels] < 2:
        raise ValueError("K_min must be >= 2 to include endpoints")

    # Rank interior indices by descending logit. Endpoints are always included.
    interior = logits[:, 1:-1]
    order_interior = torch.argsort(interior, dim=1, descending=True) + 1
    endpoints = torch.zeros((B, 2), device=device, dtype=torch.long)
    endpoints[:, 1] = T - 1
    order = torch.cat([endpoints, order_interior], dim=1)  # [B, T]

    masks_levels = torch.zeros((B, levels + 1, T), dtype=torch.bool, device=device)
    idx_levels: List[torch.Tensor] = [None for _ in range(levels + 1)]
    for s in range(levels + 1):
        K_s = K_list[s]
        idx_s = order[:, :K_s]
        idx_s = torch.sort(idx_s, dim=1).values
        idx_levels[s] = idx_s
        masks_levels[:, s].scatter_(1, idx_s, True)
    return masks_levels, idx_levels


def build_nested_masks_from_level_logits(
    logits_levels: torch.Tensor,
    K_min: int,
    levels: int,
    k_schedule: str = "doubling",
    k_geom_gamma: float = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if logits_levels.dim() != 3:
        raise ValueError("logits_levels must be [B, L, T]")
    B, L, T = logits_levels.shape
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if L != levels + 1:
        raise ValueError(f"logits_levels second dim must be levels+1 ({levels+1}), got {L}")
    if T < 2:
        raise ValueError("T must be >= 2 when using endpoints")
    device = logits_levels.device
    K_list = _compute_k_schedule(T, K_min, levels, schedule=k_schedule, geom_gamma=k_geom_gamma)
    masks_levels = torch.zeros((B, levels + 1, T), dtype=torch.bool, device=device)
    idx_levels: List[torch.Tensor] = [None for _ in range(levels + 1)]

    selected = torch.zeros((B, T), dtype=torch.bool, device=device)
    selected[:, 0] = True
    selected[:, -1] = True

    for s in range(levels, -1, -1):
        K_s = K_list[s]
        need = K_s - selected.sum(dim=1)
        if torch.any(need < 0):
            raise ValueError("K_schedule produced decreasing K values; ensure nestedness.")
        if torch.any(need > 0):
            scores = logits_levels[:, s, :].clone()
            scores[selected] = -1e9
            max_need = int(need.max().item())
            if max_need > 0:
                top_idx = torch.topk(scores, max_need, dim=1).indices
                for b in range(B):
                    k_need = int(need[b].item())
                    if k_need <= 0:
                        continue
                    chosen = top_idx[b, :k_need]
                    selected[b].scatter_(0, chosen, True)
        masks_levels[:, s] = selected

    for s in range(levels + 1):
        K_s = K_list[s]
        idx_s = torch.topk(masks_levels[:, s].float(), K_s, dim=1).indices
        idx_levels[s] = torch.sort(idx_s, dim=1).values
    return masks_levels, idx_levels


def interpolate_from_indices(
    idx: torch.Tensor,
    vals: torch.Tensor,
    T: int,
    recompute_velocity: bool = False,
) -> torch.Tensor:
    if idx.dim() != 2:
        raise ValueError("idx must be [B, K]")
    if vals.dim() != 3:
        raise ValueError("vals must be [B, K, D]")
    B, K = idx.shape
    _, _, D = vals.shape
    idx = idx.contiguous()
    device = idx.device
    t_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    seg = torch.searchsorted(idx, t_grid, right=True) - 1
    seg = seg.clamp(0, K - 2)
    left_idx = idx.gather(1, seg)
    right_idx = idx.gather(1, seg + 1)
    left_val = vals.gather(1, seg.unsqueeze(-1).expand(B, T, D))
    right_val = vals.gather(1, (seg + 1).unsqueeze(-1).expand(B, T, D))
    denom = (right_idx - left_idx).clamp(min=1).to(vals.dtype).unsqueeze(-1)
    w = (t_grid - left_idx).to(vals.dtype).unsqueeze(-1) / denom
    y = left_val + w * (right_val - left_val)
    y.scatter_(1, idx.unsqueeze(-1).expand(B, K, D), vals)
    if recompute_velocity and D == 4:
        pos = y[:, :, :2]
        v = torch.zeros_like(pos)
        dt = 1.0 / float(T)
        v[:, :-1] = (pos[:, 1:] - pos[:, :-1]) / dt
        v[:, -1] = 0.0
        y = torch.cat([pos, v], dim=-1)
    return y


def build_nested_masks(
    T: int,
    K_min: int,
    levels: int,
    generator: torch.Generator = None,
    device: torch.device = None,
) -> List[torch.Tensor]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    device = device or torch.device("cpu")
    K_min = min(K_min, T)
    masks: List[torch.Tensor] = [None for _ in range(levels + 1)]
    K_s = K_min
    masks[levels] = sample_fixed_k_mask(T, K_s, generator=generator, device=device, ensure_endpoints=True)
    for s in range(levels, 0, -1):
        K_prev = min(T, max(K_s + 1, 2 * K_s))
        new_mask = masks[s].clone()
        need = int(K_prev - new_mask.sum().item())
        if need > 0:
            available = torch.where(~new_mask)[0]
            if need > available.numel():
                need = available.numel()
            perm = torch.randperm(available.numel(), generator=generator, device=device)
            chosen = available[perm[:need]]
            new_mask[chosen] = True
        masks[s - 1] = new_mask
        K_s = int(new_mask.sum().item())
    return masks


def interpolate_from_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    recompute_velocity: bool = False,
) -> torch.Tensor:
    if x.dim() == 2:
        return _interpolate_single(x, mask, recompute_velocity=recompute_velocity)
    if x.dim() != 3:
        raise ValueError("x must have shape [T, D] or [B, T, D]")
    B, T, D = x.shape
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(B, T)
    y = torch.zeros_like(x)
    for b in range(B):
        y[b] = _interpolate_single(x[b], mask[b], recompute_velocity=recompute_velocity)
    return y


def _interpolate_single(
    x: torch.Tensor,
    mask: torch.Tensor,
    recompute_velocity: bool = False,
) -> torch.Tensor:
    T, D = x.shape
    y = x.clone()
    keyframes = torch.where(mask)[0]
    if keyframes.numel() < 2:
        return y
    for i in range(keyframes.numel() - 1):
        a = int(keyframes[i].item())
        b = int(keyframes[i + 1].item())
        if b == a:
            continue
        w = torch.linspace(0.0, 1.0, b - a + 1, device=x.device).unsqueeze(-1)
        y[a : b + 1] = x[a] + w * (x[b] - x[a])
    if recompute_velocity and D == 4:
        pos = y[:, :2]
        v = torch.zeros_like(pos)
        dt = 1.0 / float(T)
        v[:-1] = (pos[1:] - pos[:-1]) / dt
        v[-1] = 0.0
        y = torch.cat([pos, v], dim=-1)
    return y


def sample_keyframe_mask(
    T: int,
    mode: str = "mixed",
    generator: torch.Generator = None,
    device: torch.device = None,
    max_gap: int = 16,
) -> Tuple[torch.Tensor, List[int]]:
    device = device or torch.device("cpu")
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    mask[0] = True
    mask[T - 1] = True
    if mode not in {"stride", "random", "mixed"}:
        raise ValueError(f"Unknown mode {mode}")
    if mode == "mixed":
        mode = "stride" if torch.rand((), generator=generator, device=device) < 0.5 else "random"
    if mode == "stride":
        stride = int(torch.randint(4, 17, (1,), generator=generator, device=device).item())
        mask[::stride] = True
        mask[T - 1] = True
    else:
        p = float(torch.rand((), generator=generator, device=device).item() * 0.15 + 0.05)
        if T > 2:
            mask[1:-1] = torch.rand(T - 2, generator=generator, device=device) < p
        keyframes = torch.where(mask)[0].tolist()
        i = 0
        while i < len(keyframes) - 1:
            a = keyframes[i]
            b = keyframes[i + 1]
            if b - a > max_gap:
                insert = a + max_gap
                mask[insert] = True
                keyframes.insert(i + 1, insert)
            else:
                i += 1
    keyframes = torch.where(mask)[0].tolist()
    return mask, keyframes


def interpolate_keyframes(x: torch.Tensor, mask: torch.Tensor, recompute_velocity: bool = False) -> torch.Tensor:
    """Linear interpolation between keyframes. x: [T, D]."""
    return interpolate_from_mask(x, mask, recompute_velocity=recompute_velocity)
