from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenResNetMultiFeature(nn.Module):
    """Frozen ResNet feature extractor with paper-style multi-feature vectors."""

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        normalize_imagenet: bool = True,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import (
                ResNet18_Weights,
                ResNet34_Weights,
                ResNet50_Weights,
                resnet18,
                resnet34,
                resnet50,
            )
        except Exception as exc:  # pragma: no cover
            raise ImportError("torchvision is required for FrozenResNetMultiFeature") from exc

        arch = str(arch).lower()
        if arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=weights)
        elif arch == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet34(weights=weights)
        elif arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = resnet50(weights=weights)
        else:
            raise ValueError(f"unsupported arch: {arch}")

        self.arch = arch
        self.normalize_imagenet = bool(normalize_imagenet)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if self.normalize_imagenet:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("norm_mean", mean, persistent=False)
            self.register_buffer("norm_std", std, persistent=False)
        else:
            self.register_buffer("norm_mean", torch.zeros(1, 3, 1, 1), persistent=False)
            self.register_buffer("norm_std", torch.ones(1, 3, 1, 1), persistent=False)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = (x - self.norm_mean.to(dtype=x.dtype, device=x.device)) / self.norm_std.to(dtype=x.dtype, device=x.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

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
        f = fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]
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
        """Extract per-sample feature vectors [B,C_j] across multiple spaces."""
        feats: List[torch.Tensor] = []
        names: List[str] = []

        maps = self.forward_maps(x)

        if include_input_sq:
            x_sq = (x**2).mean(dim=(2, 3))
            feats.append(x_sq)
            names.append("input_x2_mean")

        for s, fmap in enumerate(maps, start=1):
            if include_global:
                feats.append(fmap.mean(dim=(2, 3)))
                names.append(f"s{s}_global_mean")
                var = (fmap**2).mean(dim=(2, 3)) - fmap.mean(dim=(2, 3)).pow(2)
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


__all__ = ["FrozenResNetMultiFeature"]

