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


def _compute_k_schedule(T: int, K_min: int, levels: int) -> List[int]:
    K_min = min(K_min, T)
    K_list = [0 for _ in range(levels + 1)]
    K_list[levels] = K_min
    for s in range(levels, 0, -1):
        K_prev = min(T, max(K_list[s] + 1, 2 * K_list[s]))
        K_list[s - 1] = K_prev
    return K_list


def build_nested_masks_batch(
    B: int,
    T: int,
    K_min: int,
    levels: int,
    generator: torch.Generator = None,
    device: torch.device = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if levels < 1:
        raise ValueError("levels must be >= 1")
    device = device or torch.device("cpu")
    K_list = _compute_k_schedule(T, K_min, levels)
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
    device = idx.device
    t_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    seg = torch.searchsorted(idx, t_grid, right=True) - 1
    seg = seg.clamp(0, K - 2)
    left_idx = idx.gather(1, seg)
    right_idx = idx.gather(1, seg + 1)
    left_val = vals.gather(1, seg.unsqueeze(-1).expand(B, T, D))
    right_val = vals.gather(1, (seg + 1).unsqueeze(-1).expand(B, T, D))
    denom = (right_idx - left_idx).clamp(min=1).unsqueeze(-1)
    w = (t_grid - left_idx).unsqueeze(-1) / denom
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
