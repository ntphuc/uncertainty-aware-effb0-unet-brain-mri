#!/usr/bin/env python3
"""Instantiate paper baselines and run one forward pass without touching BRISC data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_model


CONFIGS = {
    "unet": "configs/paper_baselines/unet_original.yaml",
    "unetpp": "configs/paper_baselines/unetpp_original.yaml",
    "attention_unet": "configs/paper_baselines/attention_unet_original.yaml",
    "deeplabv3plus": "configs/paper_baselines/deeplabv3plus_resnet50.yaml",
    "vmunet": "configs/paper_baselines/vmunet_official.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)

    failed = []
    for key in args.models:
        cfg = yaml.safe_load((ROOT / CONFIGS[key]).read_text(encoding="utf-8"))
        try:
            model = build_model(cfg).to(device).eval()
            in_channels = int(cfg["model"].get("in_channels", 1))
            x = torch.randn(1, in_channels, args.image_size, args.image_size, device=device)
            with torch.no_grad():
                output = model(x)
            logits = output["seg"]
            assert logits.shape == (1, int(cfg["model"].get("num_classes", 1)), args.image_size, args.image_size)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"OK  {key:18s} output={tuple(logits.shape)} params={params:,}")
        except Exception as exc:
            failed.append((key, exc))
            print(f"FAIL {key:18s} {type(exc).__name__}: {exc}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
