from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.gn2(h)
        y = h + self.skip(x)
        y = self.act(y)
        return y


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act(x)
        return x


class LatentMAE(nn.Module):
    """ResNet-style latent MAE with multi-scale encoder maps and U-Net-like decoder."""

    def __init__(
        self,
        in_channels: int = 4,
        base_width: int = 256,
        blocks_per_stage: Sequence[int] = (3, 4, 6, 3),
        norm_groups: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_width = int(base_width)
        b = self.base_width
        c1, c2, c3, c4 = b, 2 * b, 4 * b, 8 * b

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(norm_groups, c1), num_channels=c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, int(blocks_per_stage[0]), stride=1, norm_groups=norm_groups)
        self.stage2 = self._make_stage(c1, c2, int(blocks_per_stage[1]), stride=2, norm_groups=norm_groups)
        self.stage3 = self._make_stage(c2, c3, int(blocks_per_stage[2]), stride=2, norm_groups=norm_groups)
        self.stage4 = self._make_stage(c3, c4, int(blocks_per_stage[3]), stride=2, norm_groups=norm_groups)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(norm_groups, c4), num_channels=c4),
            nn.ReLU(inplace=True),
        )
        self.up3 = UpBlock(in_ch=c4, skip_ch=c3, out_ch=c3, groups=norm_groups)
        self.up2 = UpBlock(in_ch=c3, skip_ch=c2, out_ch=c2, groups=norm_groups)
        self.up1 = UpBlock(in_ch=c2, skip_ch=c1, out_ch=c1, groups=norm_groups)
        self.head = nn.Conv2d(c1, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int, norm_groups: int) -> nn.Sequential:
        blocks: List[nn.Module] = [BasicResBlock(in_ch, out_ch, stride=stride, groups=norm_groups)]
        for _ in range(1, int(num_blocks)):
            blocks.append(BasicResBlock(out_ch, out_ch, stride=1, groups=norm_groups))
        return nn.Sequential(*blocks)

    def forward_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_maps(x)[-1]

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self.forward_maps(x_masked)
        h = self.bottleneck(f4)
        h = self.up3(h, f3)
        h = self.up2(h, f2)
        h = self.up1(h, f1)
        out = self.head(h)
        return out


__all__ = ["LatentMAE"]

