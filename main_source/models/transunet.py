from typing import Dict, List

import math
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


class TransformerBottleneck(nn.Module):
    """Transformer encoder applied to the bottleneck feature map.

    This is a compact TransUNet-style bottleneck: CNN encoder extracts local features,
    the bottleneck is flattened into tokens, transformer layers model long-range context,
    and a CNN decoder restores the segmentation map.
    """

    def __init__(self, dim: int, depth: int = 2, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _sincos_2d_pos_embed(c: int, h: int, w: int, device, dtype) -> torch.Tensor:
        if c % 4 != 0:
            return torch.zeros(1, h * w, c, device=device, dtype=dtype)
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        omega = torch.arange(c // 4, device=device, dtype=dtype) / max(c // 4 - 1, 1)
        omega = 1.0 / (10000 ** omega)
        out_y = y.reshape(-1, 1) * omega.reshape(1, -1)
        out_x = x.reshape(-1, 1) * omega.reshape(1, -1)
        pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
        return pos.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        tokens = tokens + self._sincos_2d_pos_embed(c, h, w, x.device, x.dtype)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TransUNet(nn.Module):
    """Compact TransUNet-style baseline.

    It is intentionally dependency-light and keeps the same output dict as other models.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        channels: List[int] = None,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8]
        self.deep_supervision = deep_supervision
        self.inc = nn.Sequential(ConvBNAct(in_channels, channels[0]), ConvBNAct(channels[0], channels[0]))
        self.downs = nn.ModuleList([DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.transformer = TransformerBottleneck(channels[-1], depth=transformer_depth, num_heads=transformer_heads)

        rev = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        for i in range(len(rev) - 1):
            self.up_blocks.append(DecoderBlock(rev[i] + rev[i + 1], rev[i + 1]))
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv2d(ch, num_classes, kernel_size=1) for ch in reversed(channels[:-1])])
        else:
            self.ds_heads = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        skips = []
        y = self.inc(x)
        skips.append(y)
        for down in self.downs:
            y = down(y)
            skips.append(y)
        y = self.transformer(skips[-1])
        skips = list(reversed(skips[:-1]))
        deep_outputs = []
        for i, (skip, block) in enumerate(zip(skips, self.up_blocks)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)
            if self.deep_supervision:
                aux = self.ds_heads[i](y)
                aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
                deep_outputs.append(aux)
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        seg = self.seg_head(y)
        return {"seg": seg, "boundary": None, "deep_outputs": deep_outputs}
