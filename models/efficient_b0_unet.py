from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    timm = None
    _TIMM_ERROR = exc

from .attention import SEBlock, IdentityAttention
from .blocks import ConvBNAct, DecoderBlock
from .boundary_head import BoundaryHead
from .rgamf import RGAMF


class MultiScaleContext(nn.Module):
    """Project all encoder features to a small channel size, resize them to one target scale, then fuse."""

    def __init__(self, encoder_channels: List[int], ms_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(ch, ms_channels, kernel_size=1, bias=False)
            for ch in encoder_channels
        ])
        self.fuse = nn.Sequential(
            ConvBNAct(ms_channels * len(encoder_channels), out_channels, kernel_size=1, padding=0),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, features: List[torch.Tensor], target_size) -> torch.Tensor:
        resized = []
        for feat, proj in zip(features, self.proj):
            y = proj(feat)
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            resized.append(y)
        return self.fuse(torch.cat(resized, dim=1))


class EfficientB0UNetBoundary(nn.Module):
    """
    EfficientNet-B0 U-Net with:
    - multi-scale context fusion at each decoder stage
    - SE-filtered skip connections
    - optional boundary head
    - optional deep supervision heads

    Output dict:
        seg: final segmentation logits, shape [B, 1, H, W]
        boundary: boundary logits or None
        deep_outputs: list of auxiliary segmentation logits
        rgamf_weights: list of four [B,5] tensors or None
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 1,
        num_classes: int = 1,
        decoder_channels: List[int] = None,
        ms_channels: int = 24,
        use_multiscale: bool = True,
        fusion_type: str = "concat",
        rgamf_channels: int = 64,
        rgamf_hidden_channels: int = 32,
        rgamf_temperature: float = 1.0,
        expected_encoder_channels: List[int] = None,
        use_se: bool = True,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("Package 'timm' is required. Install with: pip install timm") from _TIMM_ERROR

        if decoder_channels is None:
            decoder_channels = [256, 160, 96, 64, 32]

        self.use_boundary_head = use_boundary_head
        self.use_deep_supervision = use_deep_supervision
        self.use_multiscale = use_multiscale
        self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in {"concat", "rgamf"}:
            raise ValueError("fusion_type must be either 'concat' or 'rgamf'")
        self.rgamf_channels = int(rgamf_channels)
        self.rgamf_hidden_channels = int(rgamf_hidden_channels)
        self.rgamf_temperature = float(rgamf_temperature)
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),
        )
        encoder_channels = list(self.encoder.feature_info.channels())
        self.encoder_channels = encoder_channels
        if expected_encoder_channels is not None:
            expected = [int(ch) for ch in expected_encoder_channels]
            if encoder_channels != expected:
                raise ValueError(
                    "The encoder feature channels do not match the configured RGAMF "
                    f"assumption. Expected {expected}, timm returned {encoder_channels}. "
                    "Adjust encoder out_indices/model version or remove "
                    "expected_encoder_channels to use the channels reported by timm."
                )

        # Bottleneck projection
        self.center = nn.Sequential(
            ConvBNAct(encoder_channels[-1], decoder_channels[0]),
            ConvBNAct(decoder_channels[0], decoder_channels[0]),
        )

        # We decode from deepest skip to shallowest skip: f3, f2, f1, f0.
        skip_channels = list(reversed(encoder_channels[:-1]))
        stage_out = decoder_channels[1:]
        self.skip_attention = nn.ModuleList([
            SEBlock(ch) if use_se else IdentityAttention()
            for ch in skip_channels
        ])
        # Decoder-context channel count before every stage:
        # stage 0 receives the center output; later stages receive the previous
        # decoder block output.
        stage_prev_channels = [decoder_channels[0], *stage_out[:-1]]

        if self.use_multiscale and self.fusion_type == "concat":
            self.ms_contexts = nn.ModuleList([
                MultiScaleContext(encoder_channels, ms_channels, out_ch)
                for out_ch in stage_out
            ])
            context_channels = list(stage_out)
        elif self.use_multiscale and self.fusion_type == "rgamf":
            self.ms_contexts = nn.ModuleList([
                RGAMF(
                    encoder_channels=encoder_channels,
                    decoder_channels=prev_ch,
                    fusion_channels=self.rgamf_channels,
                    reliability_hidden_channels=self.rgamf_hidden_channels,
                    temperature=self.rgamf_temperature,
                )
                for prev_ch in stage_prev_channels
            ])
            context_channels = [self.rgamf_channels] * len(stage_out)
        else:
            self.ms_contexts = nn.ModuleList([IdentityAttention() for _ in stage_out])
            context_channels = [0] * len(stage_out)

        blocks = []
        for prev_ch, skip_ch, context_ch, out_ch in zip(
            stage_prev_channels, skip_channels, context_channels, stage_out
        ):
            # Decoder update:
            # D_s = psi_s(Up(D_{s+1}) || E_s_SE || M_s)
            in_ch = prev_ch + skip_ch + context_ch
            blocks.append(DecoderBlock(in_ch, out_ch))
        self.decoder_blocks = nn.ModuleList(blocks)

        self.final_refine = nn.Sequential(
            ConvBNAct(decoder_channels[-1], decoder_channels[-1]),
            ConvBNAct(decoder_channels[-1], decoder_channels[-1]),
        )
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        if self.use_boundary_head:
            self.boundary_head = BoundaryHead(decoder_channels[-1], out_ch=1)
        else:
            self.boundary_head = None

        if self.use_deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv2d(ch, num_classes, kernel_size=1)
                for ch in stage_out
            ])
        else:
            self.deep_heads = None

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        x = self.center(features[-1])

        skips = list(reversed(features[:-1]))
        deep_outputs = []
        rgamf_weights = []

        for i, (skip, attn, ms_ctx, block) in enumerate(
            zip(skips, self.skip_attention, self.ms_contexts, self.decoder_blocks)
        ):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = attn(skip)
            if self.use_multiscale and self.fusion_type == "rgamf":
                # x: upsampled decoder context [B, C_d, H_s, W_s]
                # ctx: RGAMF feature [B, C_f=64, H_s, W_s]
                # alpha: adaptive scale weights [B, 5], sum(alpha, dim=1)=1
                ctx, alpha = ms_ctx(
                    encoder_features=features,
                    decoder_feature=x,
                    target_size=skip.shape[-2:],
                )
                rgamf_weights.append(alpha)
                x = torch.cat([x, skip, ctx], dim=1)
            elif self.use_multiscale:
                ctx = ms_ctx(features, target_size=skip.shape[-2:])
                x = torch.cat([x, skip, ctx], dim=1)
            else:
                x = torch.cat([x, skip], dim=1)
            x = block(x)
            if self.use_deep_supervision:
                aux = self.deep_heads[i](x)
                aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
                deep_outputs.append(aux)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.final_refine(x)
        seg = self.seg_head(x)
        boundary = self.boundary_head(x) if self.boundary_head is not None else None

        return {
            "seg": seg,
            "boundary": boundary,
            "deep_outputs": deep_outputs,
            # List ordered from deepest decoder stage to shallowest stage.
            # Each item is [B, 5] for encoder scales E1--E5.
            "rgamf_weights": rgamf_weights if self.fusion_type == "rgamf" else None,
        }
