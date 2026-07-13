"""
Sweep segmentation thresholds on train/test/validation-style splits.

Use this before retraining when failed cases are dominated by under-segmentation.
Lowering threshold can increase recall and reduce missed lesion regions, but may increase over-segmentation.

Example:
python scripts/sweep_threshold.py \
  --config configs/efficient_b0_recall_boundary_improved.yaml \
  --checkpoint outputs/effb0_recall_ms_boundary_ftv/checkpoints/best.pth \
  --split test \
  --thresholds 0.30 0.35 0.40 0.45 0.50 0.55 0.60
"""

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import NPYSliceDataset
from models import build_model
from utils.metrics import (
    AverageMeter,
    batch_surface_metrics,
    dice_score_from_logits,
    iou_score_from_logits,
    precision_recall_fbeta_from_logits,
)
from utils.misc import load_config, save_json
from utils.tta import multiscale_logits


@torch.no_grad()
def evaluate_threshold(model, loader, device, threshold: float, tta_scales=None):
    meters = {k: AverageMeter() for k in ["dice", "iou", "precision", "recall", "f2", "hd95", "assd", "boundary_f1"]}
    for images, masks in tqdm(loader, desc=f"thr={threshold:.2f}", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        seg_logits = multiscale_logits(model, images, scales=tta_scales or [1.0])
        outputs = {"seg": seg_logits}
        bs = images.size(0)

        meters["dice"].update(dice_score_from_logits(outputs["seg"], masks, threshold), bs)
        meters["iou"].update(iou_score_from_logits(outputs["seg"], masks, threshold), bs)
        pr = precision_recall_fbeta_from_logits(outputs["seg"], masks, threshold=threshold, beta=2.0)
        meters["precision"].update(pr["precision"], bs)
        meters["recall"].update(pr["recall"], bs)
        meters["f2"].update(pr["f_beta"], bs)
        surf = batch_surface_metrics(outputs["seg"], masks, threshold=threshold)
        for k in ["hd95", "assd", "boundary_f1"]:
            meters[k].update(surf[k], bs)

    return {k: v.avg for k, v in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--tta-scales", nargs="+", type=float, default=None, help="Optional multi-scale inference scales")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    paths = cfg["paths"]
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    if args.split == "test":
        x_path, y_path = paths["x_test"], paths["y_test"]
    else:
        x_path, y_path = paths["x_train"], paths["y_train"]

    dataset = NPYSliceDataset(
        x_path,
        y_path,
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize="zscore",
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=False,
        num_workers=tcfg.get("num_workers", 2),
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    for thr in args.thresholds:
        metrics = evaluate_threshold(model, loader, device, threshold=float(thr), tta_scales=args.tta_scales or [1.0])
        row = {"threshold": float(thr), **metrics}
        rows.append(row)
        print(row)

    # Choose thresholds for different goals.
    best_by_dice = max(rows, key=lambda r: r["dice"])
    best_by_f2 = max(rows, key=lambda r: r["f2"])  # F2 favors recall, useful for under-segmentation.
    best_by_assd = min(rows, key=lambda r: r["assd"])

    out_dir = Path(paths.get("output_dir", "outputs")) / "threshold_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.split}_threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "best_by_dice": best_by_dice,
        "best_by_f2_recall_oriented": best_by_f2,
        "best_by_assd_surface_oriented": best_by_assd,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_dice": ckpt.get("best_dice"),
        "tta_scales": args.tta_scales or [1.0],
    }
    save_json(summary, out_dir / f"{args.split}_threshold_sweep_summary.json")
    print("\nBest by Dice:", best_by_dice)
    print("Best by F2/Recall-oriented:", best_by_f2)
    print("Best by ASSD/Surface-oriented:", best_by_assd)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
