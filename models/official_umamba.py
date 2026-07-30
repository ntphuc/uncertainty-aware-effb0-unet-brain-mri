"""Standalone 2D U-Mamba_Bot and U-Mamba_Enc adapters.

This module is a clean PyTorch adaptation of the architecture described in
"U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation"
and the authors' Apache-2.0 reference implementation.

The original project is coupled to nnU-Net's planning/training stack.  The
classes below preserve the defining architecture while exposing this
repository's standard output contract:

    {"seg": logits, "boundary": None, "deep_outputs": [...]}

Architectural variants
----------------------
* U-Mamba_Bot: residual CNN encoder/decoder with one Mamba layer at the
  bottleneck.
* U-Mamba_Enc: residual CNN encoder/decoder with Mamba layers on alternating
  encoder stages, always including the deepest stage, matching the official
  2D implementation's placement rule.

For paper comparisons, use ``backend='mamba_ssm'``.  No silent proxy fallback
is used: a missing Mamba dependency raises an actionable ImportError.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaDependencyError(ImportError):
    """Raised when the official selective-scan dependency is unavailable."""


def _load_mamba_class():
    try:
        from mamba_ssm import Mamba  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CUDA environment
        raise MambaDependencyError(
            "U-Mamba_Bot/U-Mamba_Enc require the official `mamba_ssm` package. "
            "Install it with `bash scripts/setup_official_umamba.sh` on a "
            "compatible Linux + NVIDIA CUDA environment. The legacy `umamba` "
            "model remains a proxy and must not be reported as official U-Mamba."
        ) from exc
    return Mamba


def _as_list(value, length: int, name: str) -> List[int]:
    if isinstance(value, int):
        return [int(value)] * length
    result = [int(v) for v in value]
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} entries, got {len(result)}")
    return result


class BasicResidualBlock2D(nn.Module):
    """Two-convolution residual block used by the official U-Mamba encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        conv_bias: bool = True,
        norm_eps: float = 1e-5,
        negative_slope: float = 1e-2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=conv_bias,
        )
        self.norm1 = nn.InstanceNorm2d(out_channels, eps=norm_eps, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=conv_bias,
        )
        self.norm2 = nn.InstanceNorm2d(out_channels, eps=norm_eps, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=conv_bias,
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act2(y + residual)


class ResidualStage2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_blocks: int,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError("Each residual stage must have at least one block")
        blocks: List[nn.Module] = [
            BasicResidualBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv_bias=conv_bias,
            )
        ]
        blocks.extend(
            BasicResidualBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                conv_bias=conv_bias,
            )
            for _ in range(n_blocks - 1)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MambaLayer2D(nn.Module):
    """Apply Mamba to flattened spatial tokens or flattened channel tokens.

    The official U-Mamba_Enc code switches to channel-token mode when the
    spatial feature-map size is not larger than the channel count.  That rule is
    preserved here because it keeps the deepest sequence dimension compact.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"Mamba dimension must be positive, got {dim}")
        Mamba = _load_mamba_class()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )

    def _forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        if channels != self.dim:
            raise RuntimeError(
                f"Patch-token Mamba expected {self.dim} channels, got {channels}"
            )
        spatial_shape = x.shape[2:]
        token_count = math.prod(spatial_shape)
        tokens = x.reshape(batch, channels, token_count).transpose(1, 2)
        tokens = self.mamba(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, *spatial_shape)

    def _forward_channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        spatial_shape = x.shape[2:]
        flattened_spatial = math.prod(spatial_shape)
        if flattened_spatial != self.dim:
            raise RuntimeError(
                "Channel-token Mamba was configured for flattened spatial "
                f"dimension {self.dim}, but received {flattened_spatial}. "
                "Keep evaluation/training image_size equal to the model config."
            )
        tokens = x.flatten(2)  # [B, C, H*W], channels are the sequence tokens
        tokens = self.mamba(self.norm(tokens))
        return tokens.reshape(batch, channels, *spatial_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The official implementation explicitly executes Mamba in fp32 because
        # AMP may otherwise produce NaNs in selective scan.
        device_type = x.device.type
        autocast_supported = device_type in {"cuda", "cpu", "xpu", "mps"}
        context = (
            torch.autocast(device_type=device_type, enabled=False)
            if autocast_supported
            else torch.no_grad()  # replaced immediately below; unusual devices only
        )
        if not autocast_supported:  # avoid disabling gradients on uncommon devices
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            x_fp32 = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
            if self.channel_token:
                return self._forward_channel_tokens(x_fp32)
            return self._forward_patch_tokens(x_fp32)


class UResEncoder2D(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        encoder_blocks: Sequence[int],
        mamba_stage_indices: Sequence[int],
        d_state: int,
        d_conv: int,
        expand: int,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        n_stages = len(channels)
        if n_stages < 2:
            raise ValueError("U-Mamba needs at least two encoder stages")
        if len(strides) != n_stages or len(encoder_blocks) != n_stages:
            raise ValueError("channels, strides and encoder_blocks must have equal length")

        self.channels = [int(v) for v in channels]
        self.strides = [int(v) for v in strides]
        self.mamba_stage_indices = set(int(v) for v in mamba_stage_indices)

        self.stem = BasicResidualBlock2D(
            in_channels=in_channels,
            out_channels=self.channels[0],
            stride=1,
            conv_bias=conv_bias,
        )

        feature_map_sizes: List[Tuple[int, int]] = []
        current_h, current_w = int(input_size[0]), int(input_size[1])
        stage_in_channels = self.channels[0]
        stages: List[nn.Module] = []
        mamba_layers: List[nn.Module] = []

        for stage_index, (out_channels, stride, n_blocks) in enumerate(
            zip(self.channels, self.strides, encoder_blocks)
        ):
            if current_h % stride != 0 or current_w % stride != 0:
                raise ValueError(
                    f"input_size {input_size} is incompatible with stride {stride} "
                    f"at encoder stage {stage_index}"
                )
            current_h //= stride
            current_w //= stride
            feature_map_sizes.append((current_h, current_w))
            stages.append(
                ResidualStage2D(
                    in_channels=stage_in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    n_blocks=int(n_blocks),
                    conv_bias=conv_bias,
                )
            )
            if stage_index in self.mamba_stage_indices:
                spatial_dim = current_h * current_w
                channel_token = spatial_dim <= out_channels
                mamba_dim = spatial_dim if channel_token else out_channels
                mamba_layers.append(
                    MambaLayer2D(
                        dim=mamba_dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        channel_token=channel_token,
                    )
                )
            else:
                mamba_layers.append(nn.Identity())
            stage_in_channels = out_channels

        self.feature_map_sizes = feature_map_sizes
        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for stage, mamba in zip(self.stages, self.mamba_layers):
            x = stage(x)
            x = mamba(x)
            skips.append(x)
        return skips


class UResDecoder2D(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        encoder_strides: Sequence[int],
        decoder_blocks: Sequence[int],
        num_classes: int,
        deep_supervision: bool,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        n_stages = len(encoder_channels)
        if len(decoder_blocks) != n_stages - 1:
            raise ValueError(
                f"decoder_blocks must have {n_stages - 1} entries, got {len(decoder_blocks)}"
            )
        self.deep_supervision = bool(deep_supervision)
        self.upsample_projections = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_heads = nn.ModuleList()

        # Mirrors the official decoder: all decoder stages except the final one
        # concatenate the corresponding encoder skip. The final stage does not
        # concatenate stage-0 because the stem already operates at full scale.
        for decoder_index in range(n_stages - 1):
            source_index = n_stages - 1 - decoder_index
            target_index = source_index - 1
            in_channels = int(encoder_channels[source_index])
            out_channels = int(encoder_channels[target_index])
            stride = int(encoder_strides[source_index])
            self.upsample_projections.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=conv_bias)
            )
            has_skip = decoder_index < (n_stages - 2)
            stage_in_channels = out_channels * 2 if has_skip else out_channels
            self.stages.append(
                ResidualStage2D(
                    in_channels=stage_in_channels,
                    out_channels=out_channels,
                    stride=1,
                    n_blocks=int(decoder_blocks[decoder_index]),
                    conv_bias=conv_bias,
                )
            )
            self.seg_heads.append(nn.Conv2d(out_channels, num_classes, kernel_size=1))

    def forward(
        self,
        skips: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = skips[-1]
        deep_outputs: List[torch.Tensor] = []
        n_decoder_stages = len(self.stages)

        for decoder_index, (projection, stage, seg_head) in enumerate(
            zip(self.upsample_projections, self.stages, self.seg_heads)
        ):
            source_index = len(skips) - 1 - decoder_index
            target_index = source_index - 1
            target_size = skips[target_index].shape[-2:]
            x = F.interpolate(x, size=target_size, mode="nearest")
            x = projection(x)
            if decoder_index < n_decoder_stages - 1:
                x = torch.cat([x, skips[target_index]], dim=1)
            x = stage(x)
            logits = seg_head(x)
            if self.deep_supervision and decoder_index < n_decoder_stages - 1:
                deep_outputs.append(
                    F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
                )

        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        # Highest-resolution auxiliary output first, matching typical deep
        # supervision ordering used elsewhere in this repository.
        deep_outputs.reverse()
        return logits, deep_outputs


class _UMambaBase2D(nn.Module):
    variant_name = "U-Mamba"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        input_size: int | Tuple[int, int] = 256,
        channels: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        encoder_blocks: int | Sequence[int] = 2,
        decoder_blocks: int | Sequence[int] = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        deep_supervision: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.channels = list(channels or [32, 64, 128, 256, 320, 320])
        self.strides = list(strides or [1, 2, 2, 2, 2, 2])
        n_stages = len(self.channels)
        if len(self.strides) != n_stages:
            raise ValueError("channels and strides must contain the same number of stages")

        encoder_block_list = _as_list(encoder_blocks, n_stages, "encoder_blocks")
        decoder_block_list = _as_list(decoder_blocks, n_stages - 1, "decoder_blocks")

        # The official code reduces block counts in deeper stages to keep the
        # network efficient. Preserve that behavior even when scalar values are
        # supplied in YAML.
        for stage_index in range(math.ceil(n_stages / 2), n_stages):
            encoder_block_list[stage_index] = 1
        decoder_reduce_start = math.ceil((n_stages - 1) / 2 + 0.5)
        for decoder_index in range(decoder_reduce_start, n_stages - 1):
            decoder_block_list[decoder_index] = 1

        mamba_stages = self._mamba_stage_indices(n_stages)
        self.encoder = UResEncoder2D(
            input_size=self.input_size,
            in_channels=int(in_channels),
            channels=self.channels,
            strides=self.strides,
            encoder_blocks=encoder_block_list,
            mamba_stage_indices=mamba_stages,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            conv_bias=bool(conv_bias),
        )
        self.decoder = UResDecoder2D(
            encoder_channels=self.channels,
            encoder_strides=self.strides,
            decoder_blocks=decoder_block_list,
            num_classes=int(num_classes),
            deep_supervision=bool(deep_supervision),
            conv_bias=bool(conv_bias),
        )
        self._initialize_weights()

    def _mamba_stage_indices(self, n_stages: int) -> Sequence[int]:
        raise NotImplementedError

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=1e-2,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor] | None]:
        input_size = tuple(int(v) for v in x.shape[-2:])
        if input_size != self.input_size:
            raise ValueError(
                f"{self.variant_name} was configured for input_size={self.input_size}, "
                f"but received {input_size}. Set model.input_size and "
                "training.image_size to the same value."
            )
        skips = self.encoder(x)
        logits, deep_outputs = self.decoder(skips, output_size=input_size)
        return {"seg": logits, "boundary": None, "deep_outputs": deep_outputs}


class UMambaBot2D(_UMambaBase2D):
    """U-Mamba_Bot: Mamba is applied only at the deepest encoder feature map."""

    variant_name = "U-Mamba_Bot"

    def _mamba_stage_indices(self, n_stages: int) -> Sequence[int]:
        return [n_stages - 1]


class UMambaEnc2D(_UMambaBase2D):
    """U-Mamba_Enc: Mamba on alternating encoder stages including bottleneck."""

    variant_name = "U-Mamba_Enc"

    def _mamba_stage_indices(self, n_stages: int) -> Sequence[int]:
        # Exact placement rule from the official 2D implementation:
        # bool(stage % 2) XOR bool(n_stages % 2), guaranteeing the final stage.
        return [
            stage
            for stage in range(n_stages)
            if bool(stage % 2) ^ bool(n_stages % 2)
        ]
