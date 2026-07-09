from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DecoderBlock


class MambaLikeBlock(nn.Module):
    """Dependency-light Mamba-inspired 2D block.

    Official U-Mamba implementations usually depend on selective-scan / mamba_ssm kernels.
    This block is a runnable proxy: depthwise spatial mixing + gated channel mixing + residual.
    Use it for quick controlled experiments; use the official implementation for final SOTA claims.
    """

    def __init__(self, channels: int, expansion: int = 2, kernel_size: int = 7):
        super().__init__()
        hidden = channels * expansion
        pad = kernel_size // 2
        self.norm = nn.BatchNorm2d(channels)
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.act = nn.SiLU(inplace=False)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dwconv(y)
        a, gate = self.pw1(y).chunk(2, dim=1)
        y = self.pw2(self.act(a) * torch.sigmoid(gate))
        return x + self.gamma * y


class MambaConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = ConvBNAct(in_ch, out_ch)
        self.conv = ConvBNAct(out_ch, out_ch)
        self.mamba = MambaLikeBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.conv(x)
        return self.mamba(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = MambaConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UMamba(nn.Module):
    """2D U-Mamba-inspired baseline with Mamba-like blocks in encoder and decoder."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_channels: int = 32, channels: List[int] = None):
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.inc = MambaConvBlock(in_channels, channels[0])
        self.downs = nn.ModuleList([DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        rev = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        self.post_mamba = nn.ModuleList()
        for i in range(len(rev) - 1):
            out_ch = rev[i + 1]
            self.up_blocks.append(DecoderBlock(rev[i] + out_ch, out_ch))
            self.post_mamba.append(MambaLikeBlock(out_ch))
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        skips = []
        y = self.inc(x)
        skips.append(y)
        for down in self.downs:
            y = down(y)
            skips.append(y)
        y = skips[-1]
        skips = list(reversed(skips[:-1]))
        for skip, block, mamba in zip(skips, self.up_blocks, self.post_mamba):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = mamba(block(y))
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        return {"seg": self.seg_head(y), "boundary": None, "deep_outputs": []}
