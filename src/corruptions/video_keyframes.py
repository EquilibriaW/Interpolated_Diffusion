from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .keyframes import build_nested_masks_batch, interpolate_from_indices
from ..models.video_interpolator import TinyTemporalInterpolator
from ..utils.video_tokens import patchify_latents, unpatchify_tokens


def _smooth_latents(z: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # z: [B,T,D] -> [B*D,1,T] conv1d
    B, T, D = z.shape
    z_in = z.permute(0, 2, 1).reshape(B * D, 1, T)
    kernel = kernel.to(z.device).view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    z_out = torch.nn.functional.conv1d(z_in, kernel, padding=pad)
    z_out = z_out.view(B, D, T).permute(0, 2, 1)
    return z_out


def interpolate_video_from_indices(
    idx: torch.Tensor,
    vals: torch.Tensor,
    T: int,
    mode: str = "linear",
    smooth_kernel: Optional[torch.Tensor] = None,
    interp_model: Optional[TinyTemporalInterpolator] = None,
) -> torch.Tensor:
    """Wrapper for interpolating flattened video latents."""
    z = interpolate_from_indices(idx, vals, T, recompute_velocity=False)
    if mode == "smooth":
        if smooth_kernel is None:
            smooth_kernel = torch.tensor([0.25, 0.5, 0.25], dtype=z.dtype)
        z = _smooth_latents(z, smooth_kernel)
        z = z.scatter(1, idx.unsqueeze(-1).expand_as(vals), vals)
        return z
    if mode == "learned":
        if interp_model is None:
            raise ValueError("interp_model is required for mode='learned'")
        z = interp_model(z)
        z = z.scatter(1, idx.unsqueeze(-1).expand_as(vals), vals)
        return z
    return z


def _patchify_conf(conf: torch.Tensor, patch_size: int) -> torch.Tensor:
    # conf: [B,T,H,W] -> [B,T,N]
    B, T, H, W = conf.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("conf H/W must be divisible by patch_size")
    H_p = H // patch_size
    W_p = W // patch_size
    conf = conf.view(B, T, H_p, patch_size, W_p, patch_size)
    conf = conf.mean(dim=(3, 5))
    return conf.view(B, T, H_p * W_p)


def _apply_anchor_noise_latents(
    latents: torch.Tensor,
    idx: torch.Tensor,
    replace_mask: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    if noise_std <= 0.0 or not torch.any(replace_mask):
        return latents
    latents = latents.clone()
    B, K = idx.shape
    for b in range(B):
        if not torch.any(replace_mask[b]):
            continue
        for k in range(K):
            if replace_mask[b, k]:
                latents[b, idx[b, k]] = latents[b, idx[b, k]] + torch.randn_like(latents[b, idx[b, k]]) * float(
                    noise_std
                )
    return latents


def _apply_anchor_conf_map(
    conf: torch.Tensor,
    idx: torch.Tensor,
    replace_mask: torch.Tensor,
    conf_anchor: float,
    conf_student: float,
    conf_endpoints: float,
    clamp_endpoints: bool,
) -> torch.Tensor:
    conf = conf.clone()
    B, K = idx.shape
    for b in range(B):
        for k in range(K):
            conf[b, idx[b, k]] = float(conf_student) if replace_mask[b, k] else float(conf_anchor)
    if clamp_endpoints:
        conf[:, 0] = float(conf_endpoints)
        conf[:, -1] = float(conf_endpoints)
    return conf


def _distance_alpha(idx: torch.Tensor, T: int) -> torch.Tensor:
    B, K = idx.shape
    device = idx.device
    idx = idx.contiguous()
    t_grid = torch.arange(T, device=device).unsqueeze(0).expand(B, T).contiguous()
    seg = torch.searchsorted(idx, t_grid, right=True) - 1
    seg = seg.clamp(0, K - 2)
    left_idx = idx.gather(1, seg)
    right_idx = idx.gather(1, seg + 1)
    gap = (right_idx - left_idx).clamp(min=1)
    dist = torch.minimum(t_grid - left_idx, right_idx - t_grid)
    alpha = (2.0 * dist.float() / gap.float()).clamp(0.0, 1.0)
    return alpha.unsqueeze(-1)


def build_video_interp_level_batch(
    z0_flat: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    masks_levels: Optional[torch.Tensor] = None,
    idx_levels: Optional[List[torch.Tensor]] = None,
    s_idx: Optional[torch.Tensor] = None,
    corrupt_mode: str = "gauss",
    corrupt_sigma: float = 0.02,
    anchor_noise_frac: float = 0.25,
    student_replace_prob: float = 0.5,
    student_noise_std: float = 0.02,
    anchor_values: Optional[torch.Tensor] = None,
    anchor_idx: Optional[torch.Tensor] = None,
    conf_anchor: float = 0.95,
    conf_student: float = 0.5,
    conf_endpoints: float = 1.0,
    conf_missing: float = 0.0,
    clamp_endpoints: bool = True,
    interp_mode: str = "linear",
    interp_model: Optional[TinyTemporalInterpolator] = None,
    smooth_kernel: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Build interpolation corruption for video latents.

    Returns:
        z_interp: [B,T,D]
        mask_s: [B,T] bool
        s_idx: [B]
        masks_levels: [B,levels+1,T]
        idx_levels: list of [B,K_s]
        conf_s: [B,T] float
    """
    B, T, D = z0_flat.shape
    device = z0_flat.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    if s_idx is None:
        s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)

    z_interp = torch.zeros_like(z0_flat)
    mask_s = torch.zeros((B, T), dtype=torch.bool, device=device)
    conf_s = torch.full((B, T), float(conf_missing), device=device)

    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        vals = z0_flat[sel].gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D)).clone()

        # Replace some anchors with "student" values (noisy teacher).
        if student_replace_prob > 0.0:
            replace_mask = torch.rand((idx.shape[0], idx.shape[1]), generator=generator, device=device) < float(
                student_replace_prob
            )
            if clamp_endpoints:
                replace_mask = replace_mask & ~(idx == 0) & ~(idx == (T - 1))
            student_noise = torch.randn_like(vals) * float(student_noise_std)
            vals = torch.where(replace_mask.unsqueeze(-1), vals + student_noise, vals)
        else:
            replace_mask = torch.zeros((idx.shape[0], idx.shape[1]), device=device, dtype=torch.bool)

        zs = interpolate_video_from_indices(idx, vals, T, mode=interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model)

        if corrupt_mode != "none" and corrupt_sigma > 0.0:
            noise = torch.randn_like(zs) * float(corrupt_sigma)
            if corrupt_mode == "dist":
                alpha = _distance_alpha(idx, T).to(zs.dtype)
                noise = noise * alpha
            if anchor_noise_frac < 1.0:
                anchor_mask = masks_levels[sel, s]
                scale = torch.where(
                    anchor_mask,
                    torch.full_like(anchor_mask, float(anchor_noise_frac), dtype=zs.dtype),
                    torch.ones_like(anchor_mask, dtype=zs.dtype),
                )
                zs = zs + noise * scale.unsqueeze(-1)
            else:
                zs = zs + noise

        z_interp[sel] = zs
        mask_s[sel] = masks_levels[sel, s]
        conf_vals = torch.full((idx.shape[0], idx.shape[1]), float(conf_anchor), device=device)
        if torch.any(replace_mask):
            conf_vals = torch.where(replace_mask, torch.full_like(conf_vals, float(conf_student)), conf_vals)
        conf_s[sel] = torch.full_like(conf_s[sel], float(conf_missing))
        conf_s[sel].scatter_(1, idx, conf_vals)
        if clamp_endpoints:
            conf_s[sel, 0] = float(conf_endpoints)
            conf_s[sel, -1] = float(conf_endpoints)

    return z_interp, mask_s, s_idx, masks_levels, idx_levels, conf_s


def build_video_interp_adjacent_batch(
    z0_flat: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    masks_levels: Optional[torch.Tensor] = None,
    idx_levels: Optional[List[torch.Tensor]] = None,
    s_idx: Optional[torch.Tensor] = None,
    corrupt_mode: str = "gauss",
    corrupt_sigma: float = 0.02,
    anchor_noise_frac: float = 0.25,
    student_replace_prob: float = 0.5,
    student_noise_std: float = 0.02,
    anchor_values: Optional[torch.Tensor] = None,
    anchor_idx: Optional[torch.Tensor] = None,
    conf_anchor: float = 0.95,
    conf_student: float = 0.5,
    conf_endpoints: float = 1.0,
    conf_missing: float = 0.0,
    clamp_endpoints: bool = True,
    interp_mode: str = "linear",
    interp_model: Optional[TinyTemporalInterpolator] = None,
    smooth_kernel: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    B, T, D = z0_flat.shape
    device = z0_flat.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    if s_idx is None:
        s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)

    z_s = torch.zeros_like(z0_flat)
    z_prev = torch.zeros_like(z0_flat)
    mask_s = torch.zeros((B, T), dtype=torch.bool, device=device)
    mask_prev = torch.zeros((B, T), dtype=torch.bool, device=device)
    conf_s = torch.full((B, T), float(conf_missing), device=device)
    conf_prev = torch.full((B, T), float(conf_missing), device=device)

    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        idx_p = idx_levels[s - 1][sel]

        vals = z0_flat[sel].gather(1, idx.unsqueeze(-1).expand(-1, idx.shape[1], D)).clone()
        vals_p = z0_flat[sel].gather(1, idx_p.unsqueeze(-1).expand(-1, idx_p.shape[1], D)).clone()

        replace_mask = torch.zeros((idx.shape[0], idx.shape[1]), device=device, dtype=torch.bool)
        replace_mask_p = torch.zeros((idx_p.shape[0], idx_p.shape[1]), device=device, dtype=torch.bool)
        if student_replace_prob > 0.0:
            replace_mask = torch.rand((idx.shape[0], idx.shape[1]), generator=generator, device=device) < float(
                student_replace_prob
            )
            replace_mask_p = torch.rand((idx_p.shape[0], idx_p.shape[1]), generator=generator, device=device) < float(
                student_replace_prob
            )
            if clamp_endpoints:
                replace_mask = replace_mask & ~(idx == 0) & ~(idx == (T - 1))
                replace_mask_p = replace_mask_p & ~(idx_p == 0) & ~(idx_p == (T - 1))
            student_noise = torch.randn_like(vals) * float(student_noise_std)
            student_noise_p = torch.randn_like(vals_p) * float(student_noise_std)
            vals = torch.where(replace_mask.unsqueeze(-1), vals + student_noise, vals)
            vals_p = torch.where(replace_mask_p.unsqueeze(-1), vals_p + student_noise_p, vals_p)

        zs = interpolate_video_from_indices(idx, vals, T, mode=interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model)
        zp = interpolate_video_from_indices(idx_p, vals_p, T, mode=interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model)

        if corrupt_mode != "none" and corrupt_sigma > 0.0:
            noise = torch.randn_like(zs) * float(corrupt_sigma)
            noise_p = torch.randn_like(zp) * float(corrupt_sigma)
            if corrupt_mode == "dist":
                alpha = _distance_alpha(idx, T).to(zs.dtype)
                alpha_p = _distance_alpha(idx_p, T).to(zp.dtype)
                noise = noise * alpha
                noise_p = noise_p * alpha_p
            if anchor_noise_frac < 1.0:
                anchor_mask = masks_levels[sel, s]
                anchor_mask_p = masks_levels[sel, s - 1]
                scale = torch.where(
                    anchor_mask,
                    torch.full_like(anchor_mask, float(anchor_noise_frac), dtype=zs.dtype),
                    torch.ones_like(anchor_mask, dtype=zs.dtype),
                )
                scale_p = torch.where(
                    anchor_mask_p,
                    torch.full_like(anchor_mask_p, float(anchor_noise_frac), dtype=zp.dtype),
                    torch.ones_like(anchor_mask_p, dtype=zp.dtype),
                )
                zs = zs + noise * scale.unsqueeze(-1)
                zp = zp + noise_p * scale_p.unsqueeze(-1)
            else:
                zs = zs + noise
                zp = zp + noise_p

        z_s[sel] = zs
        z_prev[sel] = zp
        mask_s[sel] = masks_levels[sel, s]
        mask_prev[sel] = masks_levels[sel, s - 1]

        conf_vals = torch.full((idx.shape[0], idx.shape[1]), float(conf_anchor), device=device)
        conf_vals_p = torch.full((idx_p.shape[0], idx_p.shape[1]), float(conf_anchor), device=device)
        if torch.any(replace_mask):
            conf_vals = torch.where(replace_mask, torch.full_like(conf_vals, float(conf_student)), conf_vals)
        if torch.any(replace_mask_p):
            conf_vals_p = torch.where(replace_mask_p, torch.full_like(conf_vals_p, float(conf_student)), conf_vals_p)
        conf_s[sel] = torch.full_like(conf_s[sel], float(conf_missing))
        conf_prev[sel] = torch.full_like(conf_prev[sel], float(conf_missing))
        conf_s[sel].scatter_(1, idx, conf_vals)
        conf_prev[sel].scatter_(1, idx_p, conf_vals_p)
        if clamp_endpoints:
            conf_s[sel, 0] = float(conf_endpoints)
            conf_s[sel, -1] = float(conf_endpoints)
            conf_prev[sel, 0] = float(conf_endpoints)
            conf_prev[sel, -1] = float(conf_endpoints)

    return z_s, z_prev, mask_s, mask_prev, s_idx, masks_levels, idx_levels, conf_s, conf_prev


def build_video_token_interp_level_batch(
    z0_tokens: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    masks_levels: Optional[torch.Tensor] = None,
    idx_levels: Optional[List[torch.Tensor]] = None,
    s_idx: Optional[torch.Tensor] = None,
    anchor_values: Optional[torch.Tensor] = None,
    anchor_idx: Optional[torch.Tensor] = None,
    corrupt_mode: str = "gauss",
    corrupt_sigma: float = 0.02,
    anchor_noise_frac: float = 0.25,
    student_replace_prob: float = 0.5,
    student_noise_std: float = 0.02,
    conf_anchor: float = 0.95,
    conf_student: float = 0.5,
    conf_endpoints: float = 1.0,
    conf_missing: float = 0.0,
    clamp_endpoints: bool = False,
    interp_mode: str = "linear",
    interp_model: Optional[TinyTemporalInterpolator] = None,
    smooth_kernel: Optional[torch.Tensor] = None,
    flow_warper: Optional[object] = None,
    sinkhorn_warper: Optional[object] = None,
    patch_size: Optional[int] = None,
    spatial_shape: Optional[Tuple[int, int]] = None,
    uncertainty_mode: str = "none",
    uncertainty_weight: float = 1.0,
    uncertainty_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Interpolation corruption for tokenized video latents.

    z0_tokens: [B,T,N,D] (N spatial tokens per frame)
    Returns:
        z_interp: [B,T,N,D]
        mask_s: [B,T,N] bool
        s_idx: [B]
        masks_levels: [B,levels+1,T]
        idx_levels: list of [B,K_s]
        conf_s: [B,T,N]
    """
    if z0_tokens.dim() != 4:
        raise ValueError("z0_tokens must have shape [B,T,N,D]")
    B, T, N, D = z0_tokens.shape
    device = z0_tokens.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    if s_idx is None:
        s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)

    z_interp = torch.zeros_like(z0_tokens)
    mask_s = torch.zeros((B, T, N), dtype=torch.bool, device=device)
    conf_s = torch.full((B, T, N), float(conf_missing), device=device)

    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        b_sel = idx.shape[0]
        if b_sel == 0:
            continue
        z0_sel = z0_tokens[sel]  # [B_sel,T,N,D]
        z0_flat = z0_sel.permute(0, 2, 1, 3).reshape(b_sel * N, T, D)
        idx_rep = idx.repeat_interleave(N, dim=0)
        vals = z0_flat.gather(1, idx_rep.unsqueeze(-1).expand(-1, idx_rep.shape[1], D)).clone()

        replace_mask = torch.zeros((b_sel, idx.shape[1]), device=device, dtype=torch.bool)
        if student_replace_prob > 0.0:
            replace_mask = torch.rand((b_sel, idx.shape[1]), generator=generator, device=device) < float(
                student_replace_prob
            )
            if clamp_endpoints:
                replace_mask = replace_mask & ~(idx == 0) & ~(idx == (T - 1))
            replace_mask_rep = replace_mask.repeat_interleave(N, dim=0)
            anchor_vals_rep = None
            if anchor_values is not None:
                if anchor_values.dim() != 4:
                    raise ValueError("anchor_values must be [B,T,N,D] or [B,K,N,D]")
                if anchor_values.shape[:3] == z0_tokens.shape[:3]:
                    anchor_full = anchor_values[sel]
                    anchor_flat = anchor_full.permute(0, 2, 1, 3).reshape(b_sel * N, T, D)
                    anchor_vals_rep = anchor_flat.gather(
                        1, idx_rep.unsqueeze(-1).expand(-1, idx_rep.shape[1], D)
                    )
                elif anchor_values.shape[2] == N:
                    anchor_vals = anchor_values[sel]
                    if anchor_idx is not None:
                        anchor_idx_sel = anchor_idx[sel]
                        if anchor_idx_sel.shape[1] != anchor_values.shape[1]:
                            raise ValueError("anchor_idx shape mismatch for anchor_values")
                        lookup = torch.full((b_sel, T), -1, device=device, dtype=torch.long)
                        pos_grid = torch.arange(anchor_idx_sel.shape[1], device=device).view(1, -1).expand(b_sel, -1)
                        lookup.scatter_(1, anchor_idx_sel, pos_grid)
                        pos = lookup.gather(1, idx)
                        valid = pos >= 0
                        pos = pos.clamp(min=0)
                        anchor_vals_sel = anchor_vals.gather(
                            1, pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
                        )
                        anchor_vals_rep = anchor_vals_sel.permute(0, 2, 1, 3).reshape(b_sel * N, idx.shape[1], D)
                        if not torch.all(valid):
                            valid_rep = valid.repeat_interleave(N, dim=0)
                            replace_mask_rep = replace_mask_rep & valid_rep.unsqueeze(1)
                    elif anchor_values.shape[1] == idx.shape[1]:
                        anchor_vals_rep = anchor_vals.permute(0, 2, 1, 3).reshape(b_sel * N, idx.shape[1], D)
                    else:
                        raise ValueError("anchor_values/anchor_idx mismatch for token anchors")
                else:
                    raise ValueError("anchor_values shape mismatch for token anchors")
                if anchor_idx is not None and anchor_values.shape[:3] == z0_tokens.shape[:3]:
                    anchor_idx_sel = anchor_idx[sel]
                    if anchor_idx_sel.shape != idx.shape:
                        raise ValueError("anchor_idx shape mismatch")
                    match = (anchor_idx_sel == idx).all(dim=1)
                    if not torch.all(match):
                        match_rep = match.repeat_interleave(N, dim=0)
                        replace_mask_rep = replace_mask_rep & match_rep.unsqueeze(1)
            if anchor_vals_rep is not None:
                vals = torch.where(replace_mask_rep.unsqueeze(-1), anchor_vals_rep, vals)
            else:
                student_noise = torch.randn_like(vals) * float(student_noise_std)
                vals = torch.where(replace_mask_rep.unsqueeze(-1), vals + student_noise, vals)

        if interp_mode in ("flow", "sinkhorn"):
            warper = flow_warper if interp_mode == "flow" else sinkhorn_warper
            if warper is None:
                raise ValueError(f"{interp_mode} interp requested but warper is None")
            if patch_size is None or spatial_shape is None:
                raise ValueError("flow interp requires patch_size and spatial_shape")
            latents_sel = unpatchify_tokens(z0_tokens[sel], patch_size, spatial_shape)
            latents_sel = _apply_anchor_noise_latents(latents_sel, idx, replace_mask, student_noise_std)
            try:
                flow_dtype = next(warper.parameters()).dtype
            except StopIteration:
                flow_dtype = latents_sel.dtype
            latents_sel = latents_sel.to(dtype=flow_dtype)
            zs_lat, conf_flow = warper.interpolate(latents_sel, idx)
            uncertainty = (1.0 - conf_flow).clamp(0.0, 1.0)
            if uncertainty_power != 1.0:
                uncertainty = uncertainty.pow(float(uncertainty_power))
            if uncertainty_weight != 1.0:
                uncertainty = (uncertainty * float(uncertainty_weight)).clamp(0.0, 1.0)
            conf_flow = _apply_anchor_conf_map(
                conf_flow,
                idx,
                replace_mask,
                conf_anchor,
                conf_student,
                conf_endpoints,
                clamp_endpoints,
            )
            if corrupt_mode != "none" and corrupt_sigma > 0.0:
                noise = torch.randn_like(zs_lat) * float(corrupt_sigma)
                if corrupt_mode == "dist":
                    alpha = _distance_alpha(idx, T).to(zs_lat.dtype)
                    noise = noise * alpha.unsqueeze(2).unsqueeze(-1)
                if anchor_noise_frac < 1.0:
                    anchor_mask = masks_levels[sel, s].view(b_sel, T)
                    scale = torch.where(
                        anchor_mask,
                        torch.full_like(anchor_mask, float(anchor_noise_frac), dtype=zs_lat.dtype),
                        torch.ones_like(anchor_mask, dtype=zs_lat.dtype),
                    )
                    noise = noise * scale[:, :, None, None, None]
                if uncertainty_mode == "replace":
                    zs_lat = zs_lat * (1.0 - uncertainty.unsqueeze(2)) + noise * uncertainty.unsqueeze(2)
                elif uncertainty_mode == "add":
                    zs_lat = zs_lat + noise * uncertainty.unsqueeze(2)
                else:
                    zs_lat = zs_lat + noise
            zs, _ = patchify_latents(zs_lat, patch_size)
            zs = zs.to(dtype=z_interp.dtype)
            z_interp[sel] = zs
            mask_t = masks_levels[sel, s]
            mask_s[sel] = mask_t.unsqueeze(-1).expand(-1, -1, N)
            conf_s[sel] = _patchify_conf(conf_flow, patch_size).to(conf_s.dtype)
            continue
        zs_flat = interpolate_video_from_indices(
            idx_rep, vals, T, mode=interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model
        )
        zs = zs_flat.view(b_sel, N, T, D).permute(0, 2, 1, 3)

        if corrupt_mode != "none" and corrupt_sigma > 0.0:
            noise = torch.randn_like(zs) * float(corrupt_sigma)
            if corrupt_mode == "dist":
                alpha = _distance_alpha(idx, T).to(zs.dtype)  # [B_sel,T,1]
                noise = noise * alpha.unsqueeze(2)
            if anchor_noise_frac < 1.0:
                anchor_mask = masks_levels[sel, s]
                scale = torch.where(
                    anchor_mask,
                    torch.full_like(anchor_mask, float(anchor_noise_frac), dtype=zs.dtype),
                    torch.ones_like(anchor_mask, dtype=zs.dtype),
                )
                noise = noise * scale.unsqueeze(-1).unsqueeze(-1)
            zs = zs + noise

        z_interp[sel] = zs
        mask_t = masks_levels[sel, s]
        mask_s[sel] = mask_t.unsqueeze(-1).expand(-1, -1, N)

        conf_vals = torch.full((b_sel, idx.shape[1]), float(conf_anchor), device=device)
        if torch.any(replace_mask):
            conf_vals = torch.where(replace_mask, torch.full_like(conf_vals, float(conf_student)), conf_vals)
        conf_t = torch.full((b_sel, T), float(conf_missing), device=device)
        conf_t.scatter_(1, idx, conf_vals)
        if clamp_endpoints:
            conf_t[:, 0] = float(conf_endpoints)
            conf_t[:, -1] = float(conf_endpoints)
        conf_s[sel] = conf_t.unsqueeze(-1).expand(-1, -1, N)

    return z_interp, mask_s, s_idx, masks_levels, idx_levels, conf_s


def build_video_token_interp_adjacent_batch(
    z0_tokens: torch.Tensor,
    K_min: int,
    levels: int,
    generator: torch.Generator,
    masks_levels: Optional[torch.Tensor] = None,
    idx_levels: Optional[List[torch.Tensor]] = None,
    s_idx: Optional[torch.Tensor] = None,
    anchor_values: Optional[torch.Tensor] = None,
    anchor_idx: Optional[torch.Tensor] = None,
    corrupt_mode: str = "gauss",
    corrupt_sigma: float = 0.02,
    anchor_noise_frac: float = 0.25,
    student_replace_prob: float = 0.5,
    student_noise_std: float = 0.02,
    conf_anchor: float = 0.95,
    conf_student: float = 0.5,
    conf_endpoints: float = 1.0,
    conf_missing: float = 0.0,
    clamp_endpoints: bool = False,
    interp_mode: str = "linear",
    interp_model: Optional[TinyTemporalInterpolator] = None,
    smooth_kernel: Optional[torch.Tensor] = None,
    flow_warper: Optional[object] = None,
    patch_size: Optional[int] = None,
    spatial_shape: Optional[Tuple[int, int]] = None,
    uncertainty_mode: str = "none",
    uncertainty_weight: float = 1.0,
    uncertainty_power: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    if z0_tokens.dim() != 4:
        raise ValueError("z0_tokens must have shape [B,T,N,D]")
    B, T, N, D = z0_tokens.shape
    device = z0_tokens.device
    if masks_levels is None or idx_levels is None:
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, generator=generator, device=device)
    if s_idx is None:
        s_idx = torch.randint(1, levels + 1, (B,), generator=generator, device=device, dtype=torch.long)

    z_s = torch.zeros_like(z0_tokens)
    z_prev = torch.zeros_like(z0_tokens)
    mask_s = torch.zeros((B, T, N), dtype=torch.bool, device=device)
    mask_prev = torch.zeros((B, T, N), dtype=torch.bool, device=device)
    conf_s = torch.full((B, T, N), float(conf_missing), device=device)
    conf_prev = torch.full((B, T, N), float(conf_missing), device=device)

    for s in range(1, levels + 1):
        sel = s_idx == s
        if not torch.any(sel):
            continue
        idx = idx_levels[s][sel]
        idx_p = idx_levels[s - 1][sel]
        b_sel = idx.shape[0]
        if b_sel == 0:
            continue

        z0_sel = z0_tokens[sel]
        z0_flat = z0_sel.permute(0, 2, 1, 3).reshape(b_sel * N, T, D)
        idx_rep = idx.repeat_interleave(N, dim=0)
        idx_p_rep = idx_p.repeat_interleave(N, dim=0)

        vals = z0_flat.gather(1, idx_rep.unsqueeze(-1).expand(-1, idx_rep.shape[1], D)).clone()
        vals_p = z0_flat.gather(1, idx_p_rep.unsqueeze(-1).expand(-1, idx_p_rep.shape[1], D)).clone()

        replace_mask = torch.zeros((b_sel, idx.shape[1]), device=device, dtype=torch.bool)
        replace_mask_p = torch.zeros((b_sel, idx_p.shape[1]), device=device, dtype=torch.bool)
        if student_replace_prob > 0.0:
            replace_mask = torch.rand((b_sel, idx.shape[1]), generator=generator, device=device) < float(
                student_replace_prob
            )
            replace_mask_p = torch.rand((b_sel, idx_p.shape[1]), generator=generator, device=device) < float(
                student_replace_prob
            )
            if clamp_endpoints:
                replace_mask = replace_mask & ~(idx == 0) & ~(idx == (T - 1))
                replace_mask_p = replace_mask_p & ~(idx_p == 0) & ~(idx_p == (T - 1))
            replace_mask_rep = replace_mask.repeat_interleave(N, dim=0)
            replace_mask_p_rep = replace_mask_p.repeat_interleave(N, dim=0)
            anchor_vals_rep = None
            anchor_vals_rep_p = None
            if anchor_values is not None:
                if anchor_values.dim() != 4:
                    raise ValueError("anchor_values must be [B,T,N,D] or [B,K,N,D]")
                if anchor_values.shape[:3] == z0_tokens.shape[:3]:
                    anchor_full = anchor_values[sel]
                    anchor_flat = anchor_full.permute(0, 2, 1, 3).reshape(b_sel * N, T, D)
                    anchor_vals_rep = anchor_flat.gather(
                        1, idx_rep.unsqueeze(-1).expand(-1, idx_rep.shape[1], D)
                    )
                    anchor_vals_rep_p = anchor_flat.gather(
                        1, idx_p_rep.unsqueeze(-1).expand(-1, idx_p_rep.shape[1], D)
                    )
                elif anchor_values.shape[2] == N:
                    anchor_vals = anchor_values[sel]
                    if anchor_idx is not None:
                        anchor_idx_sel = anchor_idx[sel]
                        if anchor_idx_sel.shape[1] != anchor_values.shape[1]:
                            raise ValueError("anchor_idx shape mismatch for anchor_values")
                        lookup = torch.full((b_sel, T), -1, device=device, dtype=torch.long)
                        pos_grid = torch.arange(anchor_idx_sel.shape[1], device=device).view(1, -1).expand(b_sel, -1)
                        lookup.scatter_(1, anchor_idx_sel, pos_grid)
                        pos = lookup.gather(1, idx)
                        pos_p = lookup.gather(1, idx_p)
                        valid = pos >= 0
                        valid_p = pos_p >= 0
                        pos = pos.clamp(min=0)
                        pos_p = pos_p.clamp(min=0)
                        anchor_vals_sel = anchor_vals.gather(
                            1, pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
                        )
                        anchor_vals_sel_p = anchor_vals.gather(
                            1, pos_p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, D)
                        )
                        anchor_vals_rep = anchor_vals_sel.permute(0, 2, 1, 3).reshape(b_sel * N, idx.shape[1], D)
                        anchor_vals_rep_p = anchor_vals_sel_p.permute(0, 2, 1, 3).reshape(
                            b_sel * N, idx_p.shape[1], D
                        )
                        if not torch.all(valid):
                            valid_rep = valid.repeat_interleave(N, dim=0)
                            replace_mask_rep = replace_mask_rep & valid_rep.unsqueeze(1)
                        if not torch.all(valid_p):
                            valid_rep_p = valid_p.repeat_interleave(N, dim=0)
                            replace_mask_p_rep = replace_mask_p_rep & valid_rep_p.unsqueeze(1)
                    elif anchor_values.shape[1] == idx.shape[1]:
                        anchor_vals_rep = anchor_vals.permute(0, 2, 1, 3).reshape(b_sel * N, idx.shape[1], D)
                        if idx_p.shape[1] == idx.shape[1]:
                            anchor_vals_rep_p = anchor_vals_rep
                        elif anchor_values.shape[1] == idx_p.shape[1]:
                            anchor_vals_rep_p = anchor_vals.permute(0, 2, 1, 3).reshape(
                                b_sel * N, idx_p.shape[1], D
                            )
                    elif anchor_values.shape[1] == idx_p.shape[1]:
                        anchor_vals_rep_p = anchor_vals.permute(0, 2, 1, 3).reshape(b_sel * N, idx_p.shape[1], D)
                    else:
                        raise ValueError("anchor_values/anchor_idx mismatch for token anchors")
                else:
                    raise ValueError("anchor_values shape mismatch for token anchors")
                if anchor_idx is not None and anchor_values.shape[:3] == z0_tokens.shape[:3]:
                    anchor_idx_sel = anchor_idx[sel]
                    if anchor_idx_sel.shape != idx.shape:
                        raise ValueError("anchor_idx shape mismatch")
                    match = (anchor_idx_sel == idx).all(dim=1)
                    if not torch.all(match):
                        match_rep = match.repeat_interleave(N, dim=0)
                        replace_mask_rep = replace_mask_rep & match_rep.unsqueeze(1)
                        replace_mask_p_rep = replace_mask_p_rep & match_rep.unsqueeze(1)
            if anchor_vals_rep is not None and anchor_vals_rep_p is not None:
                vals = torch.where(replace_mask_rep.unsqueeze(-1), anchor_vals_rep, vals)
                vals_p = torch.where(replace_mask_p_rep.unsqueeze(-1), anchor_vals_rep_p, vals_p)
            else:
                student_noise = torch.randn_like(vals) * float(student_noise_std)
                student_noise_p = torch.randn_like(vals_p) * float(student_noise_std)
                vals = torch.where(replace_mask_rep.unsqueeze(-1), vals + student_noise, vals)
                vals_p = torch.where(replace_mask_p_rep.unsqueeze(-1), vals_p + student_noise_p, vals_p)

        if interp_mode in ("flow", "sinkhorn"):
            warper = flow_warper if interp_mode == "flow" else sinkhorn_warper
            if warper is None:
                raise ValueError(f"{interp_mode} interp requested but warper is None")
            if patch_size is None or spatial_shape is None:
                raise ValueError("flow interp requires patch_size and spatial_shape")
            latents_sel = unpatchify_tokens(z0_tokens[sel], patch_size, spatial_shape)
            latents_sel_s = _apply_anchor_noise_latents(latents_sel, idx, replace_mask, student_noise_std)
            latents_sel_p = _apply_anchor_noise_latents(latents_sel, idx_p, replace_mask_p, student_noise_std)
            try:
                flow_dtype = next(warper.parameters()).dtype
            except StopIteration:
                flow_dtype = latents_sel.dtype
            latents_sel_s = latents_sel_s.to(dtype=flow_dtype)
            latents_sel_p = latents_sel_p.to(dtype=flow_dtype)
            zs_lat, conf_flow = warper.interpolate(latents_sel_s, idx)
            zp_lat, conf_flow_p = warper.interpolate(latents_sel_p, idx_p)
            uncertainty = (1.0 - conf_flow).clamp(0.0, 1.0)
            uncertainty_p = (1.0 - conf_flow_p).clamp(0.0, 1.0)
            if uncertainty_power != 1.0:
                uncertainty = uncertainty.pow(float(uncertainty_power))
                uncertainty_p = uncertainty_p.pow(float(uncertainty_power))
            if uncertainty_weight != 1.0:
                uncertainty = (uncertainty * float(uncertainty_weight)).clamp(0.0, 1.0)
                uncertainty_p = (uncertainty_p * float(uncertainty_weight)).clamp(0.0, 1.0)
            conf_flow = _apply_anchor_conf_map(
                conf_flow,
                idx,
                replace_mask,
                conf_anchor,
                conf_student,
                conf_endpoints,
                clamp_endpoints,
            )
            conf_flow_p = _apply_anchor_conf_map(
                conf_flow_p,
                idx_p,
                replace_mask_p,
                conf_anchor,
                conf_student,
                conf_endpoints,
                clamp_endpoints,
            )
            if corrupt_mode != "none" and corrupt_sigma > 0.0:
                noise = torch.randn_like(zs_lat) * float(corrupt_sigma)
                noise_p = torch.randn_like(zp_lat) * float(corrupt_sigma)
                if corrupt_mode == "dist":
                    alpha = _distance_alpha(idx, T).to(zs_lat.dtype)
                    alpha_p = _distance_alpha(idx_p, T).to(zp_lat.dtype)
                    noise = noise * alpha.unsqueeze(2).unsqueeze(-1)
                    noise_p = noise_p * alpha_p.unsqueeze(2).unsqueeze(-1)
                if anchor_noise_frac < 1.0:
                    anchor_mask = masks_levels[sel, s].view(b_sel, T)
                    anchor_mask_p = masks_levels[sel, s - 1].view(b_sel, T)
                    scale = torch.where(
                        anchor_mask,
                        torch.full_like(anchor_mask, float(anchor_noise_frac), dtype=zs_lat.dtype),
                        torch.ones_like(anchor_mask, dtype=zs_lat.dtype),
                    )
                    scale_p = torch.where(
                        anchor_mask_p,
                        torch.full_like(anchor_mask_p, float(anchor_noise_frac), dtype=zp_lat.dtype),
                        torch.ones_like(anchor_mask_p, dtype=zp_lat.dtype),
                    )
                    noise = noise * scale[:, :, None, None, None]
                    noise_p = noise_p * scale_p[:, :, None, None, None]
                if uncertainty_mode == "replace":
                    zs_lat = zs_lat * (1.0 - uncertainty.unsqueeze(2)) + noise * uncertainty.unsqueeze(2)
                    zp_lat = zp_lat * (1.0 - uncertainty_p.unsqueeze(2)) + noise_p * uncertainty_p.unsqueeze(2)
                elif uncertainty_mode == "add":
                    zs_lat = zs_lat + noise * uncertainty.unsqueeze(2)
                    zp_lat = zp_lat + noise_p * uncertainty_p.unsqueeze(2)
                else:
                    zs_lat = zs_lat + noise
                    zp_lat = zp_lat + noise_p
            zs, _ = patchify_latents(zs_lat, patch_size)
            zp, _ = patchify_latents(zp_lat, patch_size)
            zs = zs.to(dtype=z_s.dtype)
            zp = zp.to(dtype=z_prev.dtype)
            z_s[sel] = zs
            z_prev[sel] = zp
            mask_t = masks_levels[sel, s]
            mask_t_p = masks_levels[sel, s - 1]
            mask_s[sel] = mask_t.unsqueeze(-1).expand(-1, -1, N)
            mask_prev[sel] = mask_t_p.unsqueeze(-1).expand(-1, -1, N)
            conf_s[sel] = _patchify_conf(conf_flow, patch_size).to(conf_s.dtype)
            conf_prev[sel] = _patchify_conf(conf_flow_p, patch_size).to(conf_prev.dtype)
            continue
        zs_flat = interpolate_video_from_indices(
            idx_rep, vals, T, mode=interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model
        )
        zp_flat = interpolate_video_from_indices(
            idx_p_rep, vals_p, T, mode=interp_mode, smooth_kernel=smooth_kernel, interp_model=interp_model
        )
        zs = zs_flat.view(b_sel, N, T, D).permute(0, 2, 1, 3)
        zp = zp_flat.view(b_sel, N, T, D).permute(0, 2, 1, 3)

        if corrupt_mode != "none" and corrupt_sigma > 0.0:
            noise = torch.randn_like(zs) * float(corrupt_sigma)
            noise_p = torch.randn_like(zp) * float(corrupt_sigma)
            if corrupt_mode == "dist":
                alpha = _distance_alpha(idx, T).to(zs.dtype)
                alpha_p = _distance_alpha(idx_p, T).to(zp.dtype)
                noise = noise * alpha.unsqueeze(2)
                noise_p = noise_p * alpha_p.unsqueeze(2)
            if anchor_noise_frac < 1.0:
                anchor_mask = masks_levels[sel, s]
                anchor_mask_p = masks_levels[sel, s - 1]
                scale = torch.where(
                    anchor_mask,
                    torch.full_like(anchor_mask, float(anchor_noise_frac), dtype=zs.dtype),
                    torch.ones_like(anchor_mask, dtype=zs.dtype),
                )
                scale_p = torch.where(
                    anchor_mask_p,
                    torch.full_like(anchor_mask_p, float(anchor_noise_frac), dtype=zp.dtype),
                    torch.ones_like(anchor_mask_p, dtype=zp.dtype),
                )
                noise = noise * scale.unsqueeze(-1).unsqueeze(-1)
                noise_p = noise_p * scale_p.unsqueeze(-1).unsqueeze(-1)
            zs = zs + noise
            zp = zp + noise_p

        z_s[sel] = zs
        z_prev[sel] = zp
        mask_t = masks_levels[sel, s]
        mask_t_p = masks_levels[sel, s - 1]
        mask_s[sel] = mask_t.unsqueeze(-1).expand(-1, -1, N)
        mask_prev[sel] = mask_t_p.unsqueeze(-1).expand(-1, -1, N)

        conf_vals = torch.full((b_sel, idx.shape[1]), float(conf_anchor), device=device)
        conf_vals_p = torch.full((b_sel, idx_p.shape[1]), float(conf_anchor), device=device)
        if torch.any(replace_mask):
            conf_vals = torch.where(replace_mask, torch.full_like(conf_vals, float(conf_student)), conf_vals)
        if torch.any(replace_mask_p):
            conf_vals_p = torch.where(replace_mask_p, torch.full_like(conf_vals_p, float(conf_student)), conf_vals_p)
        conf_t = torch.full((b_sel, T), float(conf_missing), device=device)
        conf_t_p = torch.full((b_sel, T), float(conf_missing), device=device)
        conf_t.scatter_(1, idx, conf_vals)
        conf_t_p.scatter_(1, idx_p, conf_vals_p)
        if clamp_endpoints:
            conf_t[:, 0] = float(conf_endpoints)
            conf_t[:, -1] = float(conf_endpoints)
            conf_t_p[:, 0] = float(conf_endpoints)
            conf_t_p[:, -1] = float(conf_endpoints)
        conf_s[sel] = conf_t.unsqueeze(-1).expand(-1, -1, N)
        conf_prev[sel] = conf_t_p.unsqueeze(-1).expand(-1, -1, N)

    return z_s, z_prev, mask_s, mask_prev, s_idx, masks_levels, idx_levels, conf_s, conf_prev
