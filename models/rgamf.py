"""Reliability-Guided Adaptive Multi-scale Fusion (RGAMF).

The module implements, at decoder stage ``s``::

    M_s = phi_s(sum_l alpha_{l,s}(x) * E_tilde_l)

where each encoder feature is projected to a common channel dimension,
resized to the current decoder resolution, assigned an image-conditioned
reliability weight, and fused by a weighted sum.

The implementation is deliberately independent of EfficientNet so that the
fusion module can be unit-tested and reused with other encoders.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    """Project one encoder feature to the shared RGAMF channel dimension.

    Parameters
    ----------
    in_channels:
        Number of channels in the encoder feature.
    out_channels:
        Shared fusion width. The paper configuration uses 64.

    Notes
    -----
    A single bias-free 1x1 convolution is used to keep the projection directly
    comparable with the original concat-based multi-scale implementation.
    Normalization and non-linearity are applied only after adaptive fusion.
    """

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.projection = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the channel-projected feature.

        Parameters
        ----------
        x:
            Encoder feature ``[B, C_l, H_l, W_l]``.

        Returns
        -------
        torch.Tensor
            Projected feature ``[B, C_f, H_l, W_l]``.
        """
        if x.ndim != 4:
            raise ValueError(f"FeatureProjection expects a 4D tensor, got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        return self.projection(x)


class ReliabilityEstimator(nn.Module):
    """Estimate one scalar reliability score for one encoder scale.

    The estimator pools both the projected encoder feature and the current
    decoder context, concatenates their global descriptors, and applies a
    two-layer MLP:

        score_l = Linear(ReLU(Linear([GAP(E_tilde_l), GAP(D_s)])))

    The score is intentionally left unnormalized. ``RGAMF`` applies a softmax
    jointly across all scales.
    """

    def __init__(
        self,
        projected_channels: int,
        decoder_channels: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if min(projected_channels, decoder_channels, hidden_channels) <= 0:
            raise ValueError("All channel dimensions must be positive")

        self.projected_channels = int(projected_channels)
        self.decoder_channels = int(decoder_channels)
        self.hidden_channels = int(hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(
                self.projected_channels + self.decoder_channels,
                self.hidden_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, 1),
        )

    def forward(
        self,
        projected_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Return an unnormalized per-sample reliability score.

        Parameters
        ----------
        projected_feature:
            Projected encoder feature ``[B, C_f, H_l, W_l]``.
        decoder_feature:
            Current decoder context, normally already upsampled to the target
            decoder stage resolution, ``[B, C_d, H_s, W_s]``.

        Returns
        -------
        torch.Tensor
            Reliability score ``[B]``.
        """
        if projected_feature.ndim != 4 or decoder_feature.ndim != 4:
            raise ValueError("ReliabilityEstimator expects two 4D tensors")
        if projected_feature.shape[0] != decoder_feature.shape[0]:
            raise ValueError("Encoder and decoder batch sizes must match")
        if projected_feature.shape[1] != self.projected_channels:
            raise ValueError(
                f"Expected projected feature with {self.projected_channels} channels, "
                f"got {projected_feature.shape[1]}"
            )
        if decoder_feature.shape[1] != self.decoder_channels:
            raise ValueError(
                f"Expected decoder feature with {self.decoder_channels} channels, "
                f"got {decoder_feature.shape[1]}"
            )

        # [B, C_f, H_l, W_l] -> [B, C_f]
        encoder_vector = self.pool(projected_feature).flatten(1)
        # [B, C_d, H_s, W_s] -> [B, C_d]
        decoder_vector = self.pool(decoder_feature).flatten(1)
        # [B, C_f + C_d]
        joint_vector = torch.cat([encoder_vector, decoder_vector], dim=1)
        # [B, 1] -> [B]
        return self.mlp(joint_vector).squeeze(-1)


class RGAMF(nn.Module):
    """Reliability-Guided Adaptive Multi-scale Fusion for one decoder stage.

    Parameters
    ----------
    encoder_channels:
        Channel dimensions of all encoder features E1--E5.
    decoder_channels:
        Channel dimension of the current decoder feature before concatenation.
    fusion_channels:
        Common projection and output width. The requested configuration uses 64.
    reliability_hidden_channels:
        Hidden width of every reliability MLP.
    temperature:
        Positive softmax temperature. ``1.0`` is the default paper setting.

    Returns
    -------
    fused_feature:
        RGAMF context tensor ``[B, C_f, H_s, W_s]``.
    scale_weights:
        Adaptive weights ``[B, L]`` where ``L`` is the number of encoder scales;
        every row is non-negative and sums to one.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: int,
        fusion_channels: int = 64,
        reliability_hidden_channels: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("RGAMF requires at least two encoder scales")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.encoder_channels = tuple(int(ch) for ch in encoder_channels)
        self.decoder_channels = int(decoder_channels)
        self.fusion_channels = int(fusion_channels)
        self.reliability_hidden_channels = int(reliability_hidden_channels)
        self.temperature = float(temperature)
        self.num_scales = len(self.encoder_channels)

        # One independent projection for every encoder level.
        self.projections = nn.ModuleList(
            FeatureProjection(ch, self.fusion_channels)
            for ch in self.encoder_channels
        )

        # One small scale-specific MLP per encoder level. Scale-specific MLPs let
        # each level learn its own reliability prior while remaining conditioned
        # on the current image feature and decoder context.
        self.reliability_estimators = nn.ModuleList(
            ReliabilityEstimator(
                projected_channels=self.fusion_channels,
                decoder_channels=self.decoder_channels,
                hidden_channels=self.reliability_hidden_channels,
            )
            for _ in self.encoder_channels
        )

        # phi_s: 3x3 Conv + BN + ReLU, as requested.
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.fusion_channels,
                self.fusion_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        encoder_features: Sequence[torch.Tensor],
        decoder_feature: torch.Tensor,
        target_size: Tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse all encoder scales with decoder-conditioned adaptive weights.

        Shape convention for EfficientNet-B0 at 256x256 input::

            E1: [B,  24, 128, 128]
            E2: [B,  40,  64,  64]
            E3: [B,  80,  32,  32]
            E4: [B, 112,  16,  16]
            E5: [B, 320,   8,   8]
            D : [B, C_d, H_s, W_s]

        All encoder features are projected and resized to ``[B, 64, H_s, W_s]``.
        """
        if len(encoder_features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} encoder features, got {len(encoder_features)}"
            )
        if decoder_feature.ndim != 4:
            raise ValueError("decoder_feature must be a 4D tensor [B,C,H,W]")
        if decoder_feature.shape[1] != self.decoder_channels:
            raise ValueError(
                f"Expected decoder context with {self.decoder_channels} channels, "
                f"got {decoder_feature.shape[1]}"
            )

        if target_size is None:
            target_size = tuple(int(v) for v in decoder_feature.shape[-2:])
        else:
            target_size = tuple(int(v) for v in target_size)

        projected_resized: List[torch.Tensor] = []
        reliability_scores: List[torch.Tensor] = []

        for scale_idx, (feature, projection, estimator, expected_channels) in enumerate(
            zip(
                encoder_features,
                self.projections,
                self.reliability_estimators,
                self.encoder_channels,
            )
        ):
            if feature.ndim != 4:
                raise ValueError(f"Encoder scale {scale_idx} is not 4D")
            if feature.shape[0] != decoder_feature.shape[0]:
                raise ValueError("All encoder and decoder tensors must share batch size")
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"Encoder scale {scale_idx} expected {expected_channels} channels, "
                    f"got {feature.shape[1]}"
                )

            # [B, C_l, H_l, W_l] -> [B, C_f, H_l, W_l]
            projected = projection(feature)

            # Score the original projected feature. GAP makes the score spatially
            # size-independent and avoids reliability changes caused only by resize.
            # [B, C_f, H_l, W_l] + [B, C_d, H_s, W_s] -> [B]
            score = estimator(projected, decoder_feature)
            reliability_scores.append(score)

            # [B, C_f, H_l, W_l] -> [B, C_f, H_s, W_s]
            resized = F.interpolate(
                projected,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            projected_resized.append(resized)

        # [B] x L -> [B, L]
        score_tensor = torch.stack(reliability_scores, dim=1)

        # Compute softmax in float32 for numerical stability under AMP, then cast
        # back to the feature dtype for the weighted sum.
        scale_weights = torch.softmax(
            score_tensor.float() / self.temperature,
            dim=1,
        ).to(dtype=projected_resized[0].dtype)

        # [B, C_f, H_s, W_s] x L -> [B, L, C_f, H_s, W_s]
        stacked_features = torch.stack(projected_resized, dim=1)
        # [B, L] -> [B, L, 1, 1, 1]
        broadcast_weights = scale_weights[:, :, None, None, None]
        # Weighted sum across scales: [B, C_f, H_s, W_s]
        weighted_feature = torch.sum(stacked_features * broadcast_weights, dim=1)
        # phi_s: [B, C_f, H_s, W_s] -> [B, C_f, H_s, W_s]
        fused_feature = self.fusion(weighted_feature)

        return fused_feature, scale_weights
