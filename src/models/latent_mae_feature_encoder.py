from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_mae import LatentMAE


class FrozenLatentMAEMultiFeature(nn.Module):
    """Frozen multi-feature extractor from a trained LatentMAE encoder."""

    def __init__(
        self,
        ckpt_path: str,
        in_channels: int = 4,
        base_width: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = LatentMAE(in_channels=int(in_channels), base_width=int(base_width))
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif isinstance(state, dict):
            sd = state
        else:
            raise ValueError("invalid checkpoint format")
        # Support checkpoints saved from DDP wrappers.
        clean_sd = {}
        for k, v in sd.items():
            kk = str(k)
            if kk.startswith("module."):
                kk = kk[len("module.") :]
            clean_sd[kk] = v
        self.encoder.load_state_dict(clean_sd, strict=False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @staticmethod
    def _select_location_indices(hw: int, max_locations: int, device: torch.device) -> torch.Tensor:
        if max_locations <= 0 or hw <= max_locations:
            return torch.arange(hw, device=device, dtype=torch.long)
        idx = torch.linspace(0, hw - 1, steps=max_locations, device=device)
        return torch.round(idx).long().unique(sorted=True)

    def _append_spatial_vectors(
        self,
        fmap: torch.Tensor,
        prefix: str,
        out_feats: List[torch.Tensor],
        out_names: List[str],
        max_locations: int,
    ) -> None:
        b, c, h, w = fmap.shape
        f = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)
        idx = self._select_location_indices(h * w, max_locations=max_locations, device=f.device)
        f = f[:, idx, :]
        for i in range(f.shape[1]):
            out_feats.append(f[:, i, :])
            out_names.append(f"{prefix}_loc_{int(i):04d}")

    def extract_feature_vectors(
        self,
        x: torch.Tensor,
        *,
        include_global: bool = True,
        include_input_sq: bool = True,
        include_local: bool = False,
        include_patch_stats: bool = True,
        max_locations: int = 64,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        feats: List[torch.Tensor] = []
        names: List[str] = []
        maps = self.encoder.forward_maps(x)

        if include_input_sq:
            x_sq = (x**2).mean(dim=(2, 3))
            feats.append(x_sq)
            names.append("input_x2_mean")

        for s, fmap in enumerate(maps, start=1):
            if include_global:
                mu = fmap.mean(dim=(2, 3))
                var = (fmap**2).mean(dim=(2, 3)) - mu.pow(2)
                feats.append(mu)
                names.append(f"s{s}_global_mean")
                feats.append(var.clamp_min(1e-6).sqrt())
                names.append(f"s{s}_global_std")

            if include_local:
                self._append_spatial_vectors(
                    fmap,
                    prefix=f"s{s}",
                    out_feats=feats,
                    out_names=names,
                    max_locations=max_locations,
                )

            if include_patch_stats:
                for k in (2, 4):
                    if fmap.shape[-2] < k or fmap.shape[-1] < k:
                        continue
                    mean_map = F.avg_pool2d(fmap, kernel_size=k, stride=k)
                    mean_sq_map = F.avg_pool2d(fmap.pow(2), kernel_size=k, stride=k)
                    std_map = (mean_sq_map - mean_map.pow(2)).clamp_min(1e-6).sqrt()
                    self._append_spatial_vectors(
                        mean_map,
                        prefix=f"s{s}_k{k}_mean",
                        out_feats=feats,
                        out_names=names,
                        max_locations=max_locations,
                    )
                    self._append_spatial_vectors(
                        std_map,
                        prefix=f"s{s}_k{k}_std",
                        out_feats=feats,
                        out_names=names,
                        max_locations=max_locations,
                    )
        return feats, names


__all__ = ["FrozenLatentMAEMultiFeature"]

