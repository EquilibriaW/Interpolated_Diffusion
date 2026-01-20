from typing import List, Tuple

import torch


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
