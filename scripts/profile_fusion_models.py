#!/usr/bin/env python3
"""Profile Concat, RGAMF, and HCSAF-BR with one identical input.

The script reports trainable parameters and, when ``thop`` is available,
full-model MACs/FLOPs. Pretrained downloads are disabled for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model


class SegmentationOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["seg"]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_variant(base_cfg: dict, fusion_type: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    model_cfg = cfg["model"]
    model_cfg["pretrained"] = False
    model_cfg["use_multiscale"] = True
    model_cfg["fusion_type"] = fusion_type
    if fusion_type == "concat":
        model_cfg["name"] = "efficientnet_b0_unet_boundary"
    elif fusion_type == "rgamf":
        model_cfg["name"] = "efficientnet_b0_unet_rgamf"
    elif fusion_type == "hcsaf_br":
        model_cfg["name"] = "efficientnet_b0_unet_hcsaf_br"
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/efficient_b0_hcsaf_br_boundary.yaml",
        help="Base configuration used to keep encoder/decoder settings identical.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    try:
        from thop import profile
    except ImportError:
        profile = None

    device = torch.device(args.device)
    in_channels = int(base_cfg["model"].get("in_channels", 1))
    dummy = torch.randn(
        args.batch_size,
        in_channels,
        args.image_size,
        args.image_size,
        device=device,
    )

    rows = []
    for fusion_type in ("concat", "rgamf", "hcsaf_br"):
        cfg = make_variant(base_cfg, fusion_type)
        model = build_model(cfg).to(device).eval()
        params = count_trainable_parameters(model)
        macs = None
        if profile is not None:
            wrapped = SegmentationOnly(model)
            with torch.no_grad():
                macs, _ = profile(wrapped, inputs=(dummy,), verbose=False)
        rows.append((fusion_type, params, macs))

    print("\nFull-model complexity at identical input settings")
    print(f"Input: [{args.batch_size}, {in_channels}, {args.image_size}, {args.image_size}]")
    print("-" * 78)
    print(f"{'Fusion':<14}{'Params (M)':>14}{'MACs (G)':>14}{'FLOPs* (G)':>16}")
    print("-" * 78)
    for fusion_type, params, macs in rows:
        if macs is None:
            mac_text = "n/a"
            flop_text = "n/a"
        else:
            mac_text = f"{macs / 1e9:.4f}"
            flop_text = f"{2.0 * macs / 1e9:.4f}"
        print(
            f"{fusion_type:<14}{params / 1e6:>14.4f}{mac_text:>14}{flop_text:>16}"
        )
    print("\n*Approximation: 1 multiply-accumulate = 2 floating-point operations.")
    print("Use one profiler/version/hardware for every row reported in the paper.")


if __name__ == "__main__":
    main()
