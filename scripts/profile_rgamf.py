#!/usr/bin/env python3
"""Analytical parameter and MAC comparison: concat fusion vs RGAMF.

The estimate covers the four multi-scale modules and the four decoder blocks.
Encoder, center, final heads, interpolation, BN, activations, and pooling are
shared or comparatively small and are not included in the analytical MACs.
For a full-model profile, use a single profiler and identical input settings for
both configurations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.blocks import DecoderBlock
from models.efficient_b0_unet import MultiScaleContext
from models.rgamf import RGAMF


def trainable_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def conv2d_macs(height: int, width: int, in_ch: int, out_ch: int, kernel: int) -> int:
    return height * width * in_ch * out_ch * kernel * kernel


def linear_macs(in_features: int, out_features: int) -> int:
    return in_features * out_features


def projection_macs(
    encoder_channels: Sequence[int],
    encoder_sizes: Sequence[int],
    projected_channels: int,
) -> int:
    return sum(
        size * size * in_ch * projected_channels
        for in_ch, size in zip(encoder_channels, encoder_sizes)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-channels", type=int, nargs=5, default=[24, 40, 80, 112, 320])
    parser.add_argument("--encoder-sizes", type=int, nargs=5, default=[128, 64, 32, 16, 8])
    parser.add_argument("--decoder-context-channels", type=int, nargs=4, default=[256, 160, 96, 64])
    parser.add_argument("--skip-channels", type=int, nargs=4, default=[112, 80, 40, 24])
    parser.add_argument("--decoder-out-channels", type=int, nargs=4, default=[160, 96, 64, 32])
    parser.add_argument("--decoder-sizes", type=int, nargs=4, default=[16, 32, 64, 128])
    parser.add_argument("--concat-projection-channels", type=int, default=24)
    parser.add_argument("--rgamf-channels", type=int, default=64)
    parser.add_argument("--rgamf-hidden", type=int, default=32)
    args = parser.parse_args()

    concat_params = 0
    rgamf_params = 0
    concat_macs = 0
    rgamf_macs = 0

    for target_size, prev_ch, skip_ch, out_ch in zip(
        args.decoder_sizes,
        args.decoder_context_channels,
        args.skip_channels,
        args.decoder_out_channels,
    ):
        concat_context = MultiScaleContext(
            args.encoder_channels,
            args.concat_projection_channels,
            out_ch,
        )
        rgamf_context = RGAMF(
            args.encoder_channels,
            decoder_channels=prev_ch,
            fusion_channels=args.rgamf_channels,
            reliability_hidden_channels=args.rgamf_hidden,
        )

        concat_decoder = DecoderBlock(prev_ch + skip_ch + out_ch, out_ch)
        rgamf_decoder = DecoderBlock(prev_ch + skip_ch + args.rgamf_channels, out_ch)

        concat_params += trainable_params(concat_context) + trainable_params(concat_decoder)
        rgamf_params += trainable_params(rgamf_context) + trainable_params(rgamf_decoder)

        # Original concat context: projections + 1x1 fusion + 3x3 fusion.
        concat_macs += projection_macs(
            args.encoder_channels,
            args.encoder_sizes,
            args.concat_projection_channels,
        )
        concat_macs += conv2d_macs(
            target_size,
            target_size,
            args.concat_projection_channels * len(args.encoder_channels),
            out_ch,
            1,
        )
        concat_macs += conv2d_macs(target_size, target_size, out_ch, out_ch, 3)

        # RGAMF context: projections + five reliability MLPs + weighted product
        # + 3x3 fusion. GAP/additions/interpolation are not included.
        rgamf_macs += projection_macs(
            args.encoder_channels,
            args.encoder_sizes,
            args.rgamf_channels,
        )
        for _ in args.encoder_channels:
            rgamf_macs += linear_macs(
                args.rgamf_channels + prev_ch,
                args.rgamf_hidden,
            )
            rgamf_macs += linear_macs(args.rgamf_hidden, 1)
        rgamf_macs += target_size * target_size * len(args.encoder_channels) * args.rgamf_channels
        rgamf_macs += conv2d_macs(
            target_size,
            target_size,
            args.rgamf_channels,
            args.rgamf_channels,
            3,
        )

        # Two-convolution decoder block.
        concat_in = prev_ch + skip_ch + out_ch
        rgamf_in = prev_ch + skip_ch + args.rgamf_channels
        concat_macs += conv2d_macs(target_size, target_size, concat_in, out_ch, 3)
        concat_macs += conv2d_macs(target_size, target_size, out_ch, out_ch, 3)
        rgamf_macs += conv2d_macs(target_size, target_size, rgamf_in, out_ch, 3)
        rgamf_macs += conv2d_macs(target_size, target_size, out_ch, out_ch, 3)

    delta_params = rgamf_params - concat_params
    delta_macs = rgamf_macs - concat_macs

    print("Fusion + decoder comparison under the requested channel assumptions")
    print(f"Concat parameters : {concat_params:,}")
    print(f"RGAMF parameters  : {rgamf_params:,}")
    print(f"Delta parameters  : {delta_params:+,} ({100.0 * delta_params / concat_params:+.2f}%)")
    print()
    print(f"Concat MACs       : {concat_macs / 1e9:.4f} GMACs")
    print(f"RGAMF MACs        : {rgamf_macs / 1e9:.4f} GMACs")
    print(f"Delta MACs        : {delta_macs / 1e9:+.4f} GMACs ({100.0 * delta_macs / concat_macs:+.2f}%)")
    print(f"Approx. FLOP delta: {2.0 * delta_macs / 1e9:+.4f} GFLOPs when one MAC is counted as 2 FLOPs")
    print()
    print("Note: interpolation, pooling, BN, ReLU, softmax, and elementwise additions are excluded.")
    print("Use the same full-model profiler, input, precision, and profiler version for paper numbers.")


if __name__ == "__main__":
    main()
