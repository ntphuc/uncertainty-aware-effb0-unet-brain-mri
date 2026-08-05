#!/usr/bin/env python3
"""Export HCSAF channel/spatial reliability diagnostics during inference.

Outputs
-------
- ``hcsaf_scale_summary.npy``: [N, 4, 5]
- ``hcsaf_scale_summary_long.csv``
- ``hcsaf_mean_scale_summary.png``
- optional per-sample stage/scale spatial maps for D2/D1
- optional boundary-guide overlays
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
import torch.nn.functional as F
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


def save_heatmap(values: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(SCALE_LABELS)), SCALE_LABELS)
    ax.set_yticks(range(len(STAGE_LABELS)), STAGE_LABELS)
    ax.set_xlabel("Encoder scale")
    ax.set_ylabel("Decoder stage")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Reliability summary")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_spatial_grid(
    spatial_weights: np.ndarray,
    sample_index: int,
    stage_index: int,
    output_path: Path,
) -> None:
    """spatial_weights: [5, H, W]."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for scale_index, ax in enumerate(axes):
        image = ax.imshow(spatial_weights[scale_index], vmin=0.0, vmax=1.0)
        ax.set_title(SCALE_LABELS[scale_index])
        ax.axis("off")
    fig.suptitle(
        f"HCSAF spatial reliability - sample {sample_index} - {STAGE_LABELS[stage_index]}"
    )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Spatial weight")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_boundary_guide(
    image: np.ndarray,
    guide: np.ndarray,
    sample_index: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI")
    axes[1].imshow(guide, vmin=0.0, vmax=1.0)
    axes[1].set_title("Boundary guide")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(guide, alpha=0.55, vmin=0.0, vmax=1.0)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"HCSAF-BR boundary guide - sample {sample_index}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as handle:
        cfg: Dict = yaml.safe_load(handle)

    model_cfg = cfg["model"]
    if str(model_cfg.get("fusion_type", "")).lower() != "hcsaf_br" and "hcsaf" not in str(
        model_cfg.get("name", "")
    ).lower():
        raise ValueError("The selected configuration does not enable HCSAF-BR")

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

    all_summaries: List[np.ndarray] = []
    all_indices: List[int] = []
    selected = set(args.sample_indices)
    selected_spatial: Dict[tuple[int, int], np.ndarray] = {}
    selected_guides: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    processed = 0

    with torch.inference_mode():
        for images, _masks, indices in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            diagnostics = outputs.get("hcsaf_weights")
            if not diagnostics:
                raise RuntimeError("Model output does not contain HCSAF diagnostics")

            summaries = torch.stack(
                [stage["scale_summary"] for stage in diagnostics], dim=1
            ).float().cpu().numpy()  # [B,4,5]
            batch_indices = indices.cpu().numpy().astype(int)

            if args.max_samples is not None:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    break
                summaries = summaries[:remaining]
                batch_indices = batch_indices[:remaining]
                images = images[:remaining]

            all_summaries.append(summaries)
            all_indices.extend(batch_indices.tolist())

            boundary_guide = outputs.get("boundary_guide")
            guide_prob = (
                torch.sigmoid(boundary_guide).float().cpu().numpy()
                if boundary_guide is not None
                else None
            )
            image_np = images[:, 0].float().cpu().numpy()

            for local_index, sample_index in enumerate(batch_indices.tolist()):
                if sample_index not in selected:
                    continue
                for stage_index, stage in enumerate(diagnostics):
                    spatial = stage.get("spatial_weights")
                    if spatial is not None:
                        selected_spatial[(sample_index, stage_index)] = (
                            spatial[local_index].float().cpu().numpy()
                        )
                if guide_prob is not None:
                    selected_guides[sample_index] = (
                        image_np[local_index],
                        guide_prob[local_index, 0],
                    )

            processed += len(batch_indices)
            if args.max_samples is not None and processed >= args.max_samples:
                break

    if not all_summaries:
        raise RuntimeError("No samples were processed")

    summaries = np.concatenate(all_summaries, axis=0)
    np.save(output_dir / "hcsaf_scale_summary.npy", summaries)
    np.save(output_dir / "sample_indices.npy", np.asarray(all_indices, dtype=np.int64))

    with (output_dir / "hcsaf_scale_summary_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "decoder_stage", "encoder_scale", "weight"])
        for sample_index, sample_summary in zip(all_indices, summaries):
            for stage_index, stage_name in enumerate(STAGE_LABELS):
                for scale_index, scale_name in enumerate(SCALE_LABELS):
                    writer.writerow(
                        [
                            sample_index,
                            stage_name,
                            scale_name,
                            float(sample_summary[stage_index, scale_index]),
                        ]
                    )

    save_heatmap(
        summaries.mean(axis=0),
        "Mean HCSAF scale reliability",
        output_dir / "hcsaf_mean_scale_summary.png",
    )

    for sample_index in selected:
        positions = np.where(np.asarray(all_indices) == sample_index)[0]
        if len(positions):
            save_heatmap(
                summaries[positions[0]],
                f"HCSAF scale reliability - sample {sample_index}",
                output_dir / f"hcsaf_scale_summary_sample_{sample_index}.png",
            )

    for (sample_index, stage_index), spatial in selected_spatial.items():
        save_spatial_grid(
            spatial,
            sample_index,
            stage_index,
            output_dir / f"hcsaf_spatial_sample_{sample_index}_stage_{stage_index}.png",
        )

    for sample_index, (image, guide) in selected_guides.items():
        save_boundary_guide(
            image,
            guide,
            sample_index,
            output_dir / f"hcsaf_boundary_guide_sample_{sample_index}.png",
        )

    print(f"Processed {summaries.shape[0]} samples")
    print(f"Summary shape: {summaries.shape} [samples, decoder_stages, encoder_scales]")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
