"""Hierarchical Channel-Spatial Adaptive Fusion with Boundary Refinement.

This module implements the HCSAF-BR blocks used by the EfficientNet-B0 U-Net
variant in this repository.

HCSAF performs image- and decoder-conditioned multi-scale fusion in two steps:

1. Channel reliability: every encoder scale receives a per-channel weight.
2. Spatial reliability: selected high-resolution decoder stages additionally
   receive a per-pixel weight for every scale.

The factorized channel/spatial formulation avoids materializing a dense
``[B, L, C, H, W]`` attention tensor. Fusion is accumulated scale by scale,
which keeps peak memory practical at 128 x 128 resolution.

Boundary refinement is implemented as a lightweight residual block guided by
an auxiliary boundary probability map. It does not require deformable-conv
extensions and therefore remains portable across standard PyTorch setups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, preferred_groups: int = 8) -> int:
    """Return the largest practical GroupNorm group count dividing channels."""
    if channels <= 0:
        raise ValueError("channels must be positive")
    for groups in range(min(preferred_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _probability_to_logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return math.log(probability / (1.0 - probability))


class ConvGNAct(nn.Module):
    """Convolution followed by GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        groups: int = 1,
        preferred_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(
                _valid_group_count(out_channels, preferred_norm_groups),
                out_channels,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HCSAFFeatureProjection(nn.Module):
    """Project an encoder feature to a normalized shared feature space.

    Input
    -----
    ``[B, C_l, H_l, W_l]``

    Output
    ------
    ``[B, C_f, H_l, W_l]``
    """

    def __init__(
        self,
        in_channels: int,
        fusion_channels: int = 64,
        preferred_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or fusion_channels <= 0:
            raise ValueError("in_channels and fusion_channels must be positive")
        self.in_channels = int(in_channels)
        self.fusion_channels = int(fusion_channels)
        self.projection = nn.Sequential(
            nn.Conv2d(self.in_channels, self.fusion_channels, kernel_size=1, bias=False),
            nn.GroupNorm(
                _valid_group_count(self.fusion_channels, preferred_norm_groups),
                self.fusion_channels,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}"
            )
        return self.projection(x)


class ChannelReliabilityEstimator(nn.Module):
    """Estimate per-channel scale reliability from encoder and decoder context.

    For scale ``l`` and stage ``s``::

        z_l = [GAP(E_l_tilde), GAP(Q_s)]                  # [B, 2C]
        u_l = Linear(SiLU(Linear(z_l)))                   # [B, C]

    Softmax normalization across scales is performed by :class:`HCSAF`.
    """

    def __init__(self, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("channels and hidden_channels must be positive")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.channels, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels, self.channels),
        )

    def forward(
        self,
        projected_feature: torch.Tensor,
        decoder_query: torch.Tensor,
    ) -> torch.Tensor:
        if projected_feature.ndim != 4 or decoder_query.ndim != 4:
            raise ValueError("ChannelReliabilityEstimator expects two 4D tensors")
        if projected_feature.shape[0] != decoder_query.shape[0]:
            raise ValueError("Encoder and decoder batch sizes must match")
        if projected_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected projected feature with {self.channels} channels, "
                f"got {projected_feature.shape[1]}"
            )
        if decoder_query.shape[1] != self.channels:
            raise ValueError(
                f"Expected decoder query with {self.channels} channels, "
                f"got {decoder_query.shape[1]}"
            )

        encoder_vector = self.pool(projected_feature).flatten(1)  # [B, C]
        decoder_vector = self.pool(decoder_query).flatten(1)      # [B, C]
        return self.mlp(torch.cat([encoder_vector, decoder_vector], dim=1))


class SpatialReliabilityEstimator(nn.Module):
    """Estimate a lightweight per-pixel reliability logit for one scale.

    Instead of concatenating full 64-channel tensors, the estimator uses six
    inexpensive statistics:

    - mean and max over channels for the encoder feature,
    - mean and max over channels for the decoder query,
    - absolute mean-feature difference,
    - mean-feature product.

    Input features are ``[B, C, H, W]`` and output is ``[B, 1, H, W]``.
    """

    def __init__(self, hidden_channels: int = 8) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_valid_group_count(hidden_channels, 4), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(
        self,
        projected_resized: torch.Tensor,
        decoder_query: torch.Tensor,
    ) -> torch.Tensor:
        if projected_resized.shape != decoder_query.shape:
            raise ValueError(
                "Spatial reliability inputs must have identical shapes; "
                f"got {tuple(projected_resized.shape)} and {tuple(decoder_query.shape)}"
            )
        encoder_mean = projected_resized.mean(dim=1, keepdim=True)
        encoder_max = projected_resized.amax(dim=1, keepdim=True)
        decoder_mean = decoder_query.mean(dim=1, keepdim=True)
        decoder_max = decoder_query.amax(dim=1, keepdim=True)
        difference = torch.abs(encoder_mean - decoder_mean)
        product = encoder_mean * decoder_mean
        statistics = torch.cat(
            [encoder_mean, encoder_max, decoder_mean, decoder_max, difference, product],
            dim=1,
        )
        return self.net(statistics)


class DepthwiseSeparableFusion(nn.Module):
    """Low-cost 3x3 spatial mixing followed by 1x1 channel mixing."""

    def __init__(self, channels: int, preferred_norm_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
                preferred_norm_groups=preferred_norm_groups,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
                preferred_norm_groups=preferred_norm_groups,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class HCSAFOutputs:
    """Structured HCSAF diagnostics returned during inference."""

    channel_weights: torch.Tensor       # [B, L, C]
    spatial_weights: torch.Tensor | None  # [B, L, H, W]
    scale_summary: torch.Tensor         # [B, L]
    adaptive_mix: torch.Tensor          # scalar tensor in [0,1]

    def as_dict(self) -> Dict[str, torch.Tensor | None]:
        return {
            "channel_weights": self.channel_weights,
            "spatial_weights": self.spatial_weights,
            "scale_summary": self.scale_summary,
            "adaptive_mix": self.adaptive_mix,
        }


class HCSAF(nn.Module):
    """Hierarchical Channel-Spatial Adaptive Fusion for one decoder stage.

    Parameters
    ----------
    encoder_channels:
        Channels of E1--E5.
    decoder_channels:
        Channels of the upsampled decoder feature entering this stage.
    fusion_channels:
        Shared projection/output width; 64 in the paper configuration.
    channel_hidden_channels:
        Hidden width of each scale-specific channel reliability MLP.
    use_spatial:
        Enable spatial reliability at this stage. Recommended only for D2/D1.
    spatial_hidden_channels:
        Width of the lightweight spatial score network.
    temperature:
        Softmax temperature for both channel and spatial weights.
    adaptive_init:
        Initial contribution of adaptive fusion versus uniform fusion.
        ``0.8`` means the module starts with 80% adaptive and 20% uniform.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        fusion_channels: int = 64,
        channel_hidden_channels: int = 32,
        use_spatial: bool = False,
        spatial_hidden_channels: int = 8,
        temperature: float = 1.0,
        adaptive_init: float = 0.8,
        preferred_norm_groups: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("HCSAF requires at least two encoder scales")
        if decoder_channels <= 0 or fusion_channels <= 0:
            raise ValueError("decoder_channels and fusion_channels must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.encoder_channels = tuple(int(ch) for ch in encoder_channels)
        self.decoder_channels = int(decoder_channels)
        self.fusion_channels = int(fusion_channels)
        self.num_scales = len(self.encoder_channels)
        self.use_spatial = bool(use_spatial)
        self.temperature = float(temperature)
        self.eps = float(eps)

        self.projections = nn.ModuleList(
            HCSAFFeatureProjection(
                in_channels=ch,
                fusion_channels=self.fusion_channels,
                preferred_norm_groups=preferred_norm_groups,
            )
            for ch in self.encoder_channels
        )
        self.decoder_query = nn.Sequential(
            nn.Conv2d(
                self.decoder_channels,
                self.fusion_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _valid_group_count(self.fusion_channels, preferred_norm_groups),
                self.fusion_channels,
            ),
            nn.SiLU(inplace=True),
        )
        self.channel_estimators = nn.ModuleList(
            ChannelReliabilityEstimator(
                channels=self.fusion_channels,
                hidden_channels=channel_hidden_channels,
            )
            for _ in self.encoder_channels
        )
        self.spatial_estimator = (
            SpatialReliabilityEstimator(hidden_channels=spatial_hidden_channels)
            if self.use_spatial
            else None
        )
        self.fusion = DepthwiseSeparableFusion(
            self.fusion_channels,
            preferred_norm_groups=preferred_norm_groups,
        )
        self.adaptive_logit = nn.Parameter(
            torch.tensor(_probability_to_logit(float(adaptive_init)), dtype=torch.float32)
        )

    def forward(
        self,
        encoder_features: Sequence[torch.Tensor],
        decoder_feature: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | None]]:
        if len(encoder_features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} encoder features, got {len(encoder_features)}"
            )
        if decoder_feature.ndim != 4:
            raise ValueError("decoder_feature must be [B,C,H,W]")
        if decoder_feature.shape[1] != self.decoder_channels:
            raise ValueError(
                f"Expected decoder feature with {self.decoder_channels} channels, "
                f"got {decoder_feature.shape[1]}"
            )
        if target_size is None:
            target_size = tuple(int(v) for v in decoder_feature.shape[-2:])
        else:
            target_size = tuple(int(v) for v in target_size)

        if tuple(decoder_feature.shape[-2:]) != target_size:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        query = self.decoder_query(decoder_feature)  # [B, C_f, H_s, W_s]
        projected_original: List[torch.Tensor] = []
        projected_resized: List[torch.Tensor] = []
        channel_logits: List[torch.Tensor] = []
        spatial_logits: List[torch.Tensor] = []

        for index, (feature, expected_channels, projection, estimator) in enumerate(
            zip(
                encoder_features,
                self.encoder_channels,
                self.projections,
                self.channel_estimators,
            )
        ):
            if feature.ndim != 4:
                raise ValueError(f"Encoder scale {index} is not a 4D tensor")
            if feature.shape[0] != decoder_feature.shape[0]:
                raise ValueError("All features must share the same batch size")
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"Encoder scale {index} expected {expected_channels} channels, "
                    f"got {feature.shape[1]}"
                )

            projected = projection(feature)  # [B, C_f, H_l, W_l]
            resized = F.interpolate(
                projected,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )  # [B, C_f, H_s, W_s]
            projected_original.append(projected)
            projected_resized.append(resized)
            channel_logits.append(estimator(projected, query))  # [B, C_f]
            if self.spatial_estimator is not None:
                spatial_logits.append(self.spatial_estimator(resized, query).squeeze(1))

        channel_logit_tensor = torch.stack(channel_logits, dim=1)  # [B, L, C]
        channel_weights = torch.softmax(
            channel_logit_tensor.float() / self.temperature,
            dim=1,
        ).to(dtype=projected_resized[0].dtype)

        uniform_feature = torch.zeros_like(projected_resized[0])
        for feature in projected_resized:
            uniform_feature = uniform_feature + feature
        uniform_feature = uniform_feature / float(self.num_scales)

        spatial_weights: torch.Tensor | None = None
        if self.spatial_estimator is None:
            adaptive_feature = torch.zeros_like(projected_resized[0])
            for scale_index, feature in enumerate(projected_resized):
                weight = channel_weights[:, scale_index, :, None, None]
                adaptive_feature = adaptive_feature + weight * feature
            scale_summary = channel_weights.mean(dim=2)  # [B, L]
        else:
            spatial_logit_tensor = torch.stack(spatial_logits, dim=1)  # [B, L, H, W]
            spatial_weights = torch.softmax(
                spatial_logit_tensor.float() / self.temperature,
                dim=1,
            ).to(dtype=projected_resized[0].dtype)

            numerator = torch.zeros_like(projected_resized[0])
            denominator = torch.zeros_like(projected_resized[0])
            for scale_index, feature in enumerate(projected_resized):
                channel_weight = channel_weights[:, scale_index, :, None, None]
                spatial_weight = spatial_weights[:, scale_index, None, :, :]
                factor = channel_weight * spatial_weight
                numerator = numerator + factor * feature
                denominator = denominator + factor
            adaptive_feature = numerator / denominator.clamp_min(self.eps)

            channel_summary = channel_weights.mean(dim=2)  # [B, L]
            spatial_summary = spatial_weights.mean(dim=(2, 3))  # [B, L]
            scale_summary = channel_summary * spatial_summary
            scale_summary = scale_summary / scale_summary.sum(dim=1, keepdim=True).clamp_min(
                self.eps
            )

        adaptive_mix = torch.sigmoid(self.adaptive_logit).to(adaptive_feature.dtype)
        mixed_feature = (
            adaptive_mix * adaptive_feature
            + (1.0 - adaptive_mix) * uniform_feature
        )
        fused_feature = self.fusion(mixed_feature)  # [B, C_f, H_s, W_s]

        diagnostics = HCSAFOutputs(
            channel_weights=channel_weights,
            spatial_weights=spatial_weights,
            scale_summary=scale_summary,
            adaptive_mix=adaptive_mix,
        ).as_dict()
        return fused_feature, diagnostics


class LearnedResidualUpsample(nn.Module):
    """Content-conditioned residual refinement on top of bilinear upsampling.

    The module starts close to standard bilinear interpolation. A learnable
    gate decides where the depthwise-separable residual should modify the base
    upsampled feature.

    Input:  ``[B, C, H, W]``
    Output: ``[B, C, H_t, W_t]``
    """

    def __init__(
        self,
        channels: int,
        residual_init: float = 0.1,
        preferred_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.refine = DepthwiseSeparableFusion(
            self.channels,
            preferred_norm_groups=preferred_norm_groups,
        )
        self.gate = nn.Conv2d(self.channels, 1, kernel_size=1, bias=True)
        self.residual_logit = nn.Parameter(
            torch.tensor(_probability_to_logit(float(residual_init)), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        base = F.interpolate(
            x,
            size=tuple(int(v) for v in target_size),
            mode="bilinear",
            align_corners=False,
        )
        residual = self.refine(base)
        gate = torch.sigmoid(self.gate(base))
        strength = torch.sigmoid(self.residual_logit).to(base.dtype)
        return base + strength * gate * residual


class BoundaryRefinementBlock(nn.Module):
    """Boundary-guided residual refinement at full image resolution.

    A lightweight guide head predicts a preliminary boundary probability. The
    probability gates a depthwise-separable residual, emphasizing corrections
    near uncertain contours while preserving a stable identity path.

    Returns
    -------
    refined_feature:
        ``[B, C, H, W]``
    guide_logits:
        Preliminary boundary logits ``[B, 1, H, W]`` for visualization.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        residual_init: float = 0.1,
        preferred_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden_channels = int(hidden_channels or max(channels // 2, 16))
        self.channels = int(channels)
        self.guide_head = nn.Sequential(
            ConvGNAct(
                self.channels,
                hidden_channels,
                kernel_size=3,
                preferred_norm_groups=preferred_norm_groups,
            ),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.refine = DepthwiseSeparableFusion(
            self.channels,
            preferred_norm_groups=preferred_norm_groups,
        )
        self.residual_logit = nn.Parameter(
            torch.tensor(_probability_to_logit(float(residual_init)), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )
        guide_logits = self.guide_head(x)
        boundary_gate = torch.sigmoid(guide_logits)
        residual = self.refine(x)
        strength = torch.sigmoid(self.residual_logit).to(x.dtype)
        # 0.5 preserves some global refinement; + boundary_gate emphasizes edges.
        refined = x + strength * (0.5 + boundary_gate) * residual
        return refined, guide_logits
