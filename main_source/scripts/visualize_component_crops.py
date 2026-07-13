"""Visualize how component-aware crop changes training samples.

Example:
    python scripts/visualize_component_crops.py \
      --config configs/efficient_b0_recall_ms_boundary_component_ftv.yaml \
      --num_samples 12
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets import NPYSliceDataset
from utils.misc import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_recall_ms_boundary_component_ftv.yaml")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    acfg = cfg.get("augmentation", {})

    ds = NPYSliceDataset(
        paths["x_train"], paths["y_train"],
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=True,
        horizontal_flip=False,
        vertical_flip=False,
        random_rotate90=False,
        lesion_crop_prob=acfg.get("lesion_crop_prob", 0.8),
        lesion_crop_scale_min=acfg.get("lesion_crop_scale_min", 1.6),
        lesion_crop_scale_max=acfg.get("lesion_crop_scale_max", 3.2),
        lesion_crop_min_size=acfg.get("lesion_crop_min_size", 96),
        lesion_crop_mode=acfg.get("lesion_crop_mode", "mixed"),
        component_crop_prob=acfg.get("component_crop_prob", 0.75),
        component_min_area=acfg.get("component_min_area", 5),
        component_connectivity=acfg.get("component_connectivity", 8),
        small_component_bias=acfg.get("small_component_bias", 0.75),
        component_jitter_ratio=acfg.get("component_jitter_ratio", 0.15),
        intensity_aug=False,
    )

    n = min(args.num_samples, len(ds))
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = [axes]

    for row in range(n):
        idx = int(torch.randint(0, len(ds), (1,)).item())
        image, mask = ds[idx]
        img = image[0].numpy()
        m = mask[0].numpy()
        axes[row][0].imshow(img, cmap="gray")
        axes[row][0].set_title(f"Cropped image idx={idx}")
        axes[row][0].axis("off")
        axes[row][1].imshow(img, cmap="gray")
        axes[row][1].imshow(m, alpha=0.45, cmap="Reds")
        axes[row][1].set_title("Mask overlay")
        axes[row][1].axis("off")

    out = Path(args.out) if args.out else Path(cfg["paths"].get("output_dir", "outputs")) / "component_crop_samples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
