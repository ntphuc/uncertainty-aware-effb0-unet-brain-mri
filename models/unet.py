from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, DecoderBlock


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard 2D U-Net baseline.

    Output format is compatible with the training/evaluation pipeline:
        {"seg": logits, "boundary": None, "deep_outputs": []}
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_channels: int = 32, channels: List[int] = None):
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.inc = nn.Sequential(
            ConvBNAct(in_channels, channels[0]),
            ConvBNAct(channels[0], channels[0]),
        )
        self.downs = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i] + rev_channels[i + 1]
            out_ch = rev_channels[i + 1]
            self.up_blocks.append(DecoderBlock(in_ch, out_ch))
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
        for skip, block in zip(skips, self.up_blocks):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        seg = self.seg_head(y)
        return {"seg": seg, "boundary": None, "deep_outputs": []}
