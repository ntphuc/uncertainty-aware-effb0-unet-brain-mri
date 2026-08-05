from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    timm = None
    _TIMM_ERROR = exc

from .attention import IdentityAttention, SEBlock
from .blocks import ConvBNAct, DecoderBlock
from .boundary_head import BoundaryHead
from .hcsaf_br import BoundaryRefinementBlock, HCSAF, LearnedResidualUpsample
from .rgamf import RGAMF


class MultiScaleContext(nn.Module):
    """Original concat-based multi-scale context fusion."""

    def __init__(self, encoder_channels: List[int], ms_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Conv2d(ch, ms_channels, kernel_size=1, bias=False)
                for ch in encoder_channels
            ]
        )
        self.fuse = nn.Sequential(
            ConvBNAct(
                ms_channels * len(encoder_channels),
                out_channels,
                kernel_size=1,
                padding=0,
            ),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, features: List[torch.Tensor], target_size) -> torch.Tensor:
        resized = []
        for feat, proj in zip(features, self.proj):
            projected = proj(feat)
            projected = F.interpolate(
                projected,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            resized.append(projected)
        return self.fuse(torch.cat(resized, dim=1))


class EfficientB0UNetBoundary(nn.Module):
    """EfficientNet-B0 U-Net supporting Concat, RGAMF, and HCSAF-BR.

    Decoder update at stage ``s``::

        D_s = psi_s(Up(D_{s+1}) || SkipSE(E_s) || M_s)

    Output dictionary
    -----------------
    seg:
        Final segmentation logits ``[B, num_classes, H, W]``.
    boundary:
        Final boundary logits or ``None``.
    deep_outputs:
        Auxiliary segmentation logits ordered deepest to shallowest.
    rgamf_weights:
        Four ``[B, 5]`` tensors for RGAMF, otherwise ``None``.
    hcsaf_weights:
        Four diagnostic dictionaries for HCSAF-BR, otherwise ``None``.
    boundary_guide:
        Preliminary full-resolution boundary guide for HCSAF-BR, otherwise
        ``None``. This is diagnostic only; the existing final boundary head is
        still used by the training loss.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 1,
        num_classes: int = 1,
        decoder_channels: List[int] | None = None,
        ms_channels: int = 24,
        use_multiscale: bool = True,
        fusion_type: str = "concat",
        rgamf_channels: int = 64,
        rgamf_hidden_channels: int = 32,
        rgamf_temperature: float = 1.0,
        hcsaf_channels: int = 64,
        hcsaf_channel_hidden_channels: int = 32,
        hcsaf_spatial_hidden_channels: int = 8,
        hcsaf_temperature: float = 1.0,
        hcsaf_adaptive_init: float = 0.8,
        hcsaf_spatial_stage_indices: Sequence[int] = (2, 3),
        hcsaf_learned_upsample_stage_indices: Sequence[int] = (3,),
        hcsaf_use_learned_final_upsample: bool = True,
        hcsaf_use_boundary_refinement: bool = True,
        hcsaf_upsample_residual_init: float = 0.1,
        hcsaf_boundary_residual_init: float = 0.1,
        expected_encoder_channels: List[int] | None = None,
        use_se: bool = True,
        use_boundary_head: bool = True,
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(
                "Package 'timm' is required. Install with: pip install timm"
            ) from _TIMM_ERROR

        if decoder_channels is None:
            decoder_channels = [256, 160, 96, 64, 32]
        if len(decoder_channels) != 5:
            raise ValueError(
                "decoder_channels must contain [center, D4, D3, D2, D1] channels"
            )

        self.use_boundary_head = bool(use_boundary_head)
        self.use_deep_supervision = bool(use_deep_supervision)
        self.use_multiscale = bool(use_multiscale)
        self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in {"concat", "rgamf", "hcsaf_br"}:
            raise ValueError(
                "fusion_type must be one of: 'concat', 'rgamf', 'hcsaf_br'"
            )
        self.rgamf_channels = int(rgamf_channels)
        self.rgamf_hidden_channels = int(rgamf_hidden_channels)
        self.rgamf_temperature = float(rgamf_temperature)
        self.hcsaf_channels = int(hcsaf_channels)
        self.hcsaf_spatial_stage_indices = {
            int(index) for index in hcsaf_spatial_stage_indices
        }
        self.hcsaf_learned_upsample_stage_indices = {
            int(index) for index in hcsaf_learned_upsample_stage_indices
        }
        self.hcsaf_use_learned_final_upsample = bool(
            hcsaf_use_learned_final_upsample
        )
        self.hcsaf_use_boundary_refinement = bool(hcsaf_use_boundary_refinement)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        invalid_stage_indices = (
            self.hcsaf_spatial_stage_indices
            | self.hcsaf_learned_upsample_stage_indices
        ) - {0, 1, 2, 3}
        if invalid_stage_indices:
            raise ValueError(
                "Decoder stage indices must be within 0..3 (D4,D3,D2,D1); "
                f"got {sorted(invalid_stage_indices)}"
            )

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
                    "The encoder feature channels do not match the configured "
                    f"assumption. Expected {expected}, timm returned "
                    f"{encoder_channels}. Adjust encoder taps/version or remove "
                    "expected_encoder_channels."
                )

        self.center = nn.Sequential(
            ConvBNAct(encoder_channels[-1], decoder_channels[0]),
            ConvBNAct(decoder_channels[0], decoder_channels[0]),
        )

        # Decode from E4 to E1. E5 is consumed by the center and remains an
        # input to all multi-scale fusion modules.
        skip_channels = list(reversed(encoder_channels[:-1]))
        stage_out = decoder_channels[1:]  # D4, D3, D2, D1
        stage_prev_channels = [decoder_channels[0], *stage_out[:-1]]

        self.skip_attention = nn.ModuleList(
            [
                SEBlock(ch) if use_se else IdentityAttention()
                for ch in skip_channels
            ]
        )

        if self.use_multiscale and self.fusion_type == "concat":
            self.ms_contexts = nn.ModuleList(
                [
                    MultiScaleContext(encoder_channels, ms_channels, out_ch)
                    for out_ch in stage_out
                ]
            )
            context_channels = list(stage_out)
        elif self.use_multiscale and self.fusion_type == "rgamf":
            self.ms_contexts = nn.ModuleList(
                [
                    RGAMF(
                        encoder_channels=encoder_channels,
                        decoder_channels=prev_ch,
                        fusion_channels=self.rgamf_channels,
                        reliability_hidden_channels=self.rgamf_hidden_channels,
                        temperature=self.rgamf_temperature,
                    )
                    for prev_ch in stage_prev_channels
                ]
            )
            context_channels = [self.rgamf_channels] * len(stage_out)
        elif self.use_multiscale and self.fusion_type == "hcsaf_br":
            self.ms_contexts = nn.ModuleList(
                [
                    HCSAF(
                        encoder_channels=encoder_channels,
                        decoder_channels=prev_ch,
                        fusion_channels=self.hcsaf_channels,
                        channel_hidden_channels=hcsaf_channel_hidden_channels,
                        use_spatial=stage_index in self.hcsaf_spatial_stage_indices,
                        spatial_hidden_channels=hcsaf_spatial_hidden_channels,
                        temperature=hcsaf_temperature,
                        adaptive_init=hcsaf_adaptive_init,
                    )
                    for stage_index, prev_ch in enumerate(stage_prev_channels)
                ]
            )
            context_channels = [self.hcsaf_channels] * len(stage_out)
        else:
            self.ms_contexts = nn.ModuleList(
                [IdentityAttention() for _ in stage_out]
            )
            context_channels = [0] * len(stage_out)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(prev_ch + skip_ch + context_ch, out_ch)
                for prev_ch, skip_ch, context_ch, out_ch in zip(
                    stage_prev_channels,
                    skip_channels,
                    context_channels,
                    stage_out,
                )
            ]
        )

        # Only HCSAF-BR uses learned residual upsampling. Other variants retain
        # the original bilinear pipeline for clean ablation comparisons.
        self.decoder_upsamplers = nn.ModuleList()
        for stage_index, prev_ch in enumerate(stage_prev_channels):
            if (
                self.fusion_type == "hcsaf_br"
                and stage_index in self.hcsaf_learned_upsample_stage_indices
            ):
                self.decoder_upsamplers.append(
                    LearnedResidualUpsample(
                        prev_ch,
                        residual_init=hcsaf_upsample_residual_init,
                    )
                )
            else:
                self.decoder_upsamplers.append(IdentityAttention())

        if self.fusion_type == "hcsaf_br" and self.hcsaf_use_learned_final_upsample:
            self.final_upsampler: nn.Module | None = LearnedResidualUpsample(
                decoder_channels[-1],
                residual_init=hcsaf_upsample_residual_init,
            )
        else:
            self.final_upsampler = None

        self.final_refine = nn.Sequential(
            ConvBNAct(decoder_channels[-1], decoder_channels[-1]),
            ConvBNAct(decoder_channels[-1], decoder_channels[-1]),
        )

        if self.fusion_type == "hcsaf_br" and self.hcsaf_use_boundary_refinement:
            self.boundary_refinement: BoundaryRefinementBlock | None = (
                BoundaryRefinementBlock(
                    decoder_channels[-1],
                    residual_init=hcsaf_boundary_residual_init,
                )
            )
        else:
            self.boundary_refinement = None

        self.seg_head = nn.Conv2d(
            decoder_channels[-1], num_classes, kernel_size=1
        )
        self.boundary_head = (
            BoundaryHead(decoder_channels[-1], out_ch=1)
            if self.use_boundary_head
            else None
        )
        self.deep_heads = (
            nn.ModuleList(
                [nn.Conv2d(ch, num_classes, kernel_size=1) for ch in stage_out]
            )
            if self.use_deep_supervision
            else None
        )

    @staticmethod
    def _upsample_decoder(
        x: torch.Tensor,
        target_size,
        upsampler: nn.Module,
    ) -> torch.Tensor:
        if isinstance(upsampler, LearnedResidualUpsample):
            return upsampler(x, target_size=target_size)
        return F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        input_size = tuple(int(v) for v in x.shape[-2:])
        features = self.encoder(x)
        decoder = self.center(features[-1])  # [B, 256, 8, 8]

        skips = list(reversed(features[:-1]))  # E4, E3, E2, E1
        deep_outputs: List[torch.Tensor] = []
        rgamf_weights: List[torch.Tensor] = []
        hcsaf_weights: List[Dict[str, torch.Tensor | None]] = []

        for stage_index, (skip, attention, context_module, block, upsampler) in enumerate(
            zip(
                skips,
                self.skip_attention,
                self.ms_contexts,
                self.decoder_blocks,
                self.decoder_upsamplers,
            )
        ):
            decoder = self._upsample_decoder(
                decoder,
                target_size=skip.shape[-2:],
                upsampler=upsampler,
            )
            filtered_skip = attention(skip)

            if self.use_multiscale and self.fusion_type == "rgamf":
                context, alpha = context_module(
                    encoder_features=features,
                    decoder_feature=decoder,
                    target_size=skip.shape[-2:],
                )
                rgamf_weights.append(alpha)
                decoder_input = torch.cat(
                    [decoder, filtered_skip, context], dim=1
                )
            elif self.use_multiscale and self.fusion_type == "hcsaf_br":
                context, diagnostics = context_module(
                    encoder_features=features,
                    decoder_feature=decoder,
                    target_size=skip.shape[-2:],
                )
                hcsaf_weights.append(diagnostics)
                decoder_input = torch.cat(
                    [decoder, filtered_skip, context], dim=1
                )
            elif self.use_multiscale:
                context = context_module(features, target_size=skip.shape[-2:])
                decoder_input = torch.cat(
                    [decoder, filtered_skip, context], dim=1
                )
            else:
                decoder_input = torch.cat([decoder, filtered_skip], dim=1)

            decoder = block(decoder_input)
            if self.use_deep_supervision:
                auxiliary = self.deep_heads[stage_index](decoder)
                auxiliary = F.interpolate(
                    auxiliary,
                    size=input_size,
                    mode="bilinear",
                    align_corners=False,
                )
                deep_outputs.append(auxiliary)

        if self.final_upsampler is not None:
            decoder = self.final_upsampler(decoder, target_size=input_size)
        else:
            decoder = F.interpolate(
                decoder,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        decoder = self.final_refine(decoder)

        boundary_guide = None
        if self.boundary_refinement is not None:
            decoder, boundary_guide = self.boundary_refinement(decoder)

        segmentation = self.seg_head(decoder)
        boundary = (
            self.boundary_head(decoder)
            if self.boundary_head is not None
            else None
        )

        return {
            "seg": segmentation,
            "boundary": boundary,
            "deep_outputs": deep_outputs,
            "rgamf_weights": (
                rgamf_weights if self.fusion_type == "rgamf" else None
            ),
            "hcsaf_weights": (
                hcsaf_weights if self.fusion_type == "hcsaf_br" else None
            ),
            "boundary_guide": boundary_guide,
        }
