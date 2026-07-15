"""Faithful, non-proxy segmentation baselines used for paper comparisons.

Architectures:
- Original-style U-Net (Ronneberger et al., 2015)
- U-Net++ / Nested U-Net (Zhou et al., 2018)
- Attention U-Net (Oktay et al., 2018)
- DeepLabV3+ with a ResNet-50 encoder and output stride 16

Every model returns the dictionary expected by this repository:
    {"seg": logits, "boundary": None, "deep_outputs": [...]}
The returned segmentation maps are raw logits (no sigmoid/softmax).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ModelOutput = Dict[str, object]


def _init_conv_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions followed by ReLU, as in the canonical U-Net family."""

    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for cin, cout in [(in_channels, out_channels), (out_channels, out_channels)]:
            layers.append(nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpConcatBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, batch_norm: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class OriginalUNet(nn.Module):
    """Canonical 2D U-Net encoder-decoder with transposed-convolution upsampling."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        batch_norm: bool = False,
    ):
        super().__init__()
        c = [base_channels * (2**i) for i in range(5)]
        self.enc1 = DoubleConv(in_channels, c[0], batch_norm)
        self.enc2 = DoubleConv(c[0], c[1], batch_norm)
        self.enc3 = DoubleConv(c[1], c[2], batch_norm)
        self.enc4 = DoubleConv(c[2], c[3], batch_norm)
        self.bottleneck = DoubleConv(c[3], c[4], batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = UpConcatBlock(c[4], c[3], c[3], batch_norm)
        self.up3 = UpConcatBlock(c[3], c[2], c[2], batch_norm)
        self.up2 = UpConcatBlock(c[2], c[1], c[1], batch_norm)
        self.up1 = UpConcatBlock(c[1], c[0], c[0], batch_norm)
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)
        self.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        logits = self.head(d1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"seg": logits, "boundary": None, "deep_outputs": []}


class NestedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat(features, dim=1))


class OriginalUNetPlusPlus(nn.Module):
    """U-Net++ with nested dense skip pathways and optional deep supervision."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        deep_supervision: bool = True,
        batch_norm: bool = False,
    ):
        super().__init__()
        f = [base_channels * (2**i) for i in range(5)]
        self.deep_supervision = bool(deep_supervision)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = DoubleConv(in_channels, f[0], batch_norm)
        self.conv1_0 = DoubleConv(f[0], f[1], batch_norm)
        self.conv2_0 = DoubleConv(f[1], f[2], batch_norm)
        self.conv3_0 = DoubleConv(f[2], f[3], batch_norm)
        self.conv4_0 = DoubleConv(f[3], f[4], batch_norm)

        self.conv0_1 = NestedConvBlock(f[0] + f[1], f[0], batch_norm)
        self.conv1_1 = NestedConvBlock(f[1] + f[2], f[1], batch_norm)
        self.conv2_1 = NestedConvBlock(f[2] + f[3], f[2], batch_norm)
        self.conv3_1 = NestedConvBlock(f[3] + f[4], f[3], batch_norm)

        self.conv0_2 = NestedConvBlock(f[0] * 2 + f[1], f[0], batch_norm)
        self.conv1_2 = NestedConvBlock(f[1] * 2 + f[2], f[1], batch_norm)
        self.conv2_2 = NestedConvBlock(f[2] * 2 + f[3], f[2], batch_norm)

        self.conv0_3 = NestedConvBlock(f[0] * 3 + f[1], f[0], batch_norm)
        self.conv1_3 = NestedConvBlock(f[1] * 3 + f[2], f[1], batch_norm)

        self.conv0_4 = NestedConvBlock(f[0] * 4 + f[1], f[0], batch_norm)

        self.final = nn.Conv2d(f[0], num_classes, kernel_size=1)
        self.ds_heads = nn.ModuleList([nn.Conv2d(f[0], num_classes, 1) for _ in range(3)])
        self.apply(_init_conv_weights)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(x0_0, self._up(x1_0, x0_0))
        x1_1 = self.conv1_1(x1_0, self._up(x2_0, x1_0))
        x2_1 = self.conv2_1(x2_0, self._up(x3_0, x2_0))
        x3_1 = self.conv3_1(x3_0, self._up(x4_0, x3_0))

        x0_2 = self.conv0_2(x0_0, x0_1, self._up(x1_1, x0_0))
        x1_2 = self.conv1_2(x1_0, x1_1, self._up(x2_1, x1_0))
        x2_2 = self.conv2_2(x2_0, x2_1, self._up(x3_1, x2_0))

        x0_3 = self.conv0_3(x0_0, x0_1, x0_2, self._up(x1_2, x0_0))
        x1_3 = self.conv1_3(x1_0, x1_1, x1_2, self._up(x2_2, x1_0))

        x0_4 = self.conv0_4(x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0))
        logits = self.final(x0_4)
        deep_outputs: List[torch.Tensor] = []
        if self.deep_supervision:
            deep_outputs = [head(node) for head, node in zip(self.ds_heads, (x0_1, x0_2, x0_3))]
        return {"seg": logits, "boundary": None, "deep_outputs": deep_outputs}


class AttentionGate(nn.Module):
    """Additive grid-attention gate from Attention U-Net."""

    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.theta_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.phi_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gating: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.phi_g(gating)
        x = self.theta_x(skip)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(g + x))
        return skip * alpha


class AttentionUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 1))
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attention(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class OriginalAttentionUNet(nn.Module):
    """Attention U-Net with additive attention gates on every decoder skip connection."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        batch_norm: bool = True,
    ):
        super().__init__()
        c = [base_channels * (2**i) for i in range(5)]
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = DoubleConv(in_channels, c[0], batch_norm)
        self.enc2 = DoubleConv(c[0], c[1], batch_norm)
        self.enc3 = DoubleConv(c[1], c[2], batch_norm)
        self.enc4 = DoubleConv(c[2], c[3], batch_norm)
        self.bottleneck = DoubleConv(c[3], c[4], batch_norm)

        self.up4 = AttentionUpBlock(c[4], c[3], c[3], batch_norm)
        self.up3 = AttentionUpBlock(c[3], c[2], c[2], batch_norm)
        self.up2 = AttentionUpBlock(c[2], c[1], c[1], batch_norm)
        self.up1 = AttentionUpBlock(c[1], c[0], c[0], batch_norm)
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)
        self.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        logits = self.head(d1)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"seg": logits, "boundary": None, "deep_outputs": []}


class AtrousSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        padding = dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # GroupNorm is stable even when training with batch size 1.
            nn.GroupNorm(32 if out_channels >= 32 else 1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = super().forward(x)
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates: Sequence[int] = (6, 12, 18)):
        super().__init__()
        branches: List[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        branches.extend(ASPPConv(in_channels, out_channels, rate) for rate in rates)
        branches.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(branches), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature extractor exposing low-level (/4) and high-level (/16) maps."""

    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise ImportError("DeepLabV3+ requires torchvision, which is already listed in requirements.txt") from exc

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        if in_channels != 3:
            old = backbone.conv1
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    if in_channels == 1:
                        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    else:
                        repeat = (in_channels + 2) // 3
                        expanded = old.weight.repeat(1, repeat, 1, 1)[:, :in_channels]
                        new.weight.copy_(expanded * (3.0 / float(in_channels)))
            backbone.conv1 = new

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        high = self.layer4(x)
        return low, high


class DeepLabV3PlusResNet50(nn.Module):
    """DeepLabV3+ with ASPP, a low-level decoder, and a ResNet-50 OS=16 encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        pretrained: bool = False,
        aspp_channels: int = 256,
        low_level_channels: int = 48,
        atrous_rates: Sequence[int] = (6, 12, 18),
    ):
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=pretrained)
        self.aspp = ASPP(2048, out_channels=aspp_channels, rates=atrous_rates)
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(256, low_level_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_channels),
            nn.ReLU(inplace=True),
        )
        decoder_in = aspp_channels + low_level_channels
        self.decoder = nn.Sequential(
            AtrousSeparableConv(decoder_in, 256, dilation=1),
            AtrousSeparableConv(256, 256, dilation=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        # Keep torchvision's initialization for the encoder. Initialize only newly added layers.
        self.aspp.apply(_init_conv_weights)
        self.low_level_projection.apply(_init_conv_weights)
        self.decoder.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        input_size = x.shape[-2:]
        low, high = self.encoder(x)
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_level_projection(low)
        logits = self.decoder(torch.cat([high, low], dim=1))
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"seg": logits, "boundary": None, "deep_outputs": []}
