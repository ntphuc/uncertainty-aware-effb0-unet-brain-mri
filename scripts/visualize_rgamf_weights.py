#!/usr/bin/env python3
"""Export and visualize RGAMF scale weights during inference.

Example
-------
python scripts/visualize_rgamf_weights.py \
  --config configs/efficient_b0_rgamf_boundary.yaml \
  --checkpoint outputs/rgamf/checkpoints/best_validation.pth \
  --output-dir outputs/rgamf/alpha_analysis \
  --max-samples 860
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.npy_dataset import NPYSliceDataset
from models import build_model


STAGE_LABELS = ["D4 (16x16)", "D3 (32x32)", "D2 (64x64)", "D1 (128x128)"]
SCALE_LABELS = ["E1", "E2", "E3", "E4", "E5"]


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)


def save_mean_heatmap(weights: np.ndarray, output_path: Path) -> None:
    """weights: [N, 4 decoder stages, 5 encoder scales]."""
    mean_weights = weights.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(mean_weights, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(SCALE_LABELS)), SCALE_LABELS)
    ax.set_yticks(range(len(STAGE_LABELS)), STAGE_LABELS)
    ax.set_xlabel("Encoder scale")
    ax.set_ylabel("Decoder stage")
    ax.set_title("Mean RGAMF reliability weights")
    for row in range(mean_weights.shape[0]):
        for col in range(mean_weights.shape[1]):
            ax.text(col, row, f"{mean_weights[row, col]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Mean alpha")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_sample_heatmap(sample_weights: np.ndarray, sample_index: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(sample_weights, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(SCALE_LABELS)), SCALE_LABELS)
    ax.set_yticks(range(len(STAGE_LABELS)), STAGE_LABELS)
    ax.set_xlabel("Encoder scale")
    ax.set_ylabel("Decoder stage")
    ax.set_title(f"RGAMF weights - sample {sample_index}")
    for row in range(sample_weights.shape[0]):
        for col in range(sample_weights.shape[1]):
            ax.text(col, row, f"{sample_weights[row, col]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Alpha")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=[],
        help="Dataset indices for which individual alpha heatmaps are saved.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as handle:
        cfg: Dict = yaml.safe_load(handle)

    model_cfg = cfg["model"]
    if str(model_cfg.get("fusion_type", "")).lower() != "rgamf" and "rgamf" not in str(
        model_cfg.get("name", "")
    ).lower():
        raise ValueError("The selected configuration does not enable RGAMF")

    paths = cfg["paths"]
    if args.split == "test":
        x_path, y_path = paths["x_test"], paths["y_test"]
    else:
        x_path, y_path = paths["x_train"], paths["y_train"]

    dataset = NPYSliceDataset(
        x_path=x_path,
        y_path=y_path,
        image_size=cfg.get("training", {}).get("image_size", 256),
        in_channels=model_cfg.get("in_channels", 1),
        normalize="zscore",
        augment=False,
        return_index=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_weights: List[np.ndarray] = []
    all_indices: List[int] = []
    sample_weight_map: Dict[int, np.ndarray] = {}
    processed = 0

    with torch.inference_mode():
        for images, _masks, indices in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            stage_weights = outputs.get("rgamf_weights")
            if not stage_weights:
                raise RuntimeError("Model output does not contain RGAMF weights")

            # Four [B,5] tensors -> [B,4,5].
            batch_weights = torch.stack(stage_weights, dim=1).float().cpu().numpy()
            batch_indices = indices.cpu().numpy().astype(int)

            if args.max_samples is not None:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    break
                batch_weights = batch_weights[:remaining]
                batch_indices = batch_indices[:remaining]

            all_weights.append(batch_weights)
            all_indices.extend(batch_indices.tolist())
            for idx, weights in zip(batch_indices.tolist(), batch_weights):
                if idx in args.sample_indices:
                    sample_weight_map[idx] = weights

            processed += len(batch_indices)
            if args.max_samples is not None and processed >= args.max_samples:
                break

    if not all_weights:
        raise RuntimeError("No samples were processed")

    weights = np.concatenate(all_weights, axis=0)  # [N,4,5]
    np.save(output_dir / "rgamf_weights.npy", weights)
    np.save(output_dir / "sample_indices.npy", np.asarray(all_indices, dtype=np.int64))

    with (output_dir / "rgamf_weights_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "decoder_stage", "encoder_scale", "alpha"])
        for sample_idx, sample_weights in zip(all_indices, weights):
            for stage_idx, stage_name in enumerate(STAGE_LABELS):
                for scale_idx, scale_name in enumerate(SCALE_LABELS):
                    writer.writerow(
                        [sample_idx, stage_name, scale_name, float(sample_weights[stage_idx, scale_idx])]
                    )

    save_mean_heatmap(weights, output_dir / "rgamf_mean_weights.png")
    for sample_idx, sample_weights in sample_weight_map.items():
        save_sample_heatmap(
            sample_weights,
            sample_idx,
            output_dir / f"rgamf_weights_sample_{sample_idx}.png",
        )

    print(f"Processed {weights.shape[0]} samples")
    print(f"Weights shape: {weights.shape} [samples, decoder_stages, encoder_scales]")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
