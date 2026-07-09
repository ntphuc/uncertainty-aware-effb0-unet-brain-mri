#!/usr/bin/env python3
"""
Prediction-level probe for failed cases.

This script answers questions like:
  - Did the model assign any probability inside the true GT lesion?
  - Is the failure caused by threshold, or did the model never "see" the lesion?
  - Is the model predicting a high-confidence false-positive region elsewhere?
  - How many GT components are missed at component level?

It runs the model on selected failed case indices and exports per-case CSV + visual overlays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import NPYSliceDataset
from models import build_model
from utils.misc import load_config
from utils.tta import multiscale_logits
from utils.postprocess import postprocess_batch_predictions

EPS = 1e-8


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fg_zscore(image_chw: torch.Tensor) -> np.ndarray:
    img = image_chw[0].detach().cpu().numpy().astype(np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    cut = np.percentile(finite, 5)
    fg = finite[finite > cut]
    if fg.size < 16:
        fg = finite
    return ((img - float(fg.mean())) / (float(fg.std()) + EPS)).astype(np.float32)


def _components(mask: np.ndarray, connectivity: int = 8, min_area: int = 1):
    mask = mask.astype(bool)
    if ndimage is None:
        ys, xs = np.where(mask)
        if len(xs) < min_area:
            return [], np.zeros_like(mask, dtype=np.int32)
        labeled = mask.astype(np.int32)
        return [{"id": 1, "area": int(len(xs)), "mask": mask}], labeled
    conn = 2 if connectivity == 8 else 1
    structure = ndimage.generate_binary_structure(2, conn)
    labeled, num = ndimage.label(mask, structure=structure)
    comps = []
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        comps.append({
            "id": comp_id,
            "area": area,
            "mask": comp,
            "cy": float(ys.mean()),
            "cx": float(xs.mean()),
            "y1": int(ys.min()), "y2": int(ys.max() + 1),
            "x1": int(xs.min()), "x2": int(xs.max() + 1),
        })
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps, labeled


def _component_matching(gt_mask: np.ndarray, pred_mask: np.ndarray, connectivity: int = 8) -> Dict[str, float]:
    gt_comps, _ = _components(gt_mask, connectivity=connectivity, min_area=1)
    pred_comps, _ = _components(pred_mask, connectivity=connectivity, min_area=1)
    matched_gt = 0
    for g in gt_comps:
        if np.logical_and(g["mask"], pred_mask).sum() > 0:
            matched_gt += 1
    fp_pred = 0
    for p in pred_comps:
        if np.logical_and(p["mask"], gt_mask).sum() == 0:
            fp_pred += 1
    gt_count = len(gt_comps)
    pred_count = len(pred_comps)
    return {
        "gt_component_count": gt_count,
        "pred_component_count": pred_count,
        "matched_component_count": matched_gt,
        "missed_component_count": max(0, gt_count - matched_gt),
        "fp_component_count": fp_pred,
        "component_recall": float(matched_gt / (gt_count + EPS)),
        "component_precision": float((pred_count - fp_pred) / (pred_count + EPS)),
    }


def _safe_region_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _classify_failure(tp: int, fp: int, fn: int, gt_area: int, pred_area: int, max_prob_gt: float, threshold: float) -> Tuple[str, str]:
    if gt_area == 0 and pred_area > 0:
        return "false_positive_on_empty_gt", "GT has no lesion; prediction is all FP. Check empty-mask handling/labels."
    if gt_area > 0 and pred_area == 0:
        if max_prob_gt >= threshold:
            return "postprocess_removed_or_binary_issue", "GT region has high probability but final mask is empty; check threshold/postprocess."
        if max_prob_gt >= 0.5 * threshold:
            return "below_threshold_miss", "Model weakly sees GT; threshold/calibration/TTA may help."
        return "complete_miss_empty_prediction", "Model gives very low probability inside GT; data coverage/representation issue likely."
    if tp == 0 and pred_area > 0 and gt_area > 0:
        if max_prob_gt >= threshold:
            return "complete_miss_with_fp_postprocess_or_spatial_issue", "GT has high probability somewhere but no overlap after mask; inspect postprocess/spatial alignment."
        if max_prob_gt >= 0.5 * threshold:
            return "complete_miss_with_fp_below_threshold", "Model partially sees GT but prefers another FP region; TTA/calibration plus hard negatives may help."
        return "complete_miss_with_false_positive", "Model does not see GT and predicts elsewhere; data distribution/annotation/context issue likely."
    if fn > fp * 1.5:
        return "under_segmentation", "Recall is the main issue."
    if fp > fn * 1.5:
        return "over_segmentation", "Precision/FP control is the main issue."
    return "mixed_or_boundary_error", "Both FP and FN exist; inspect boundary/component metrics."


def _visualize_case(out_path: Path, image_z: np.ndarray, gt: np.ndarray, pred: np.ndarray, prob: np.ndarray, title: str):
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    err = np.zeros((*gt.shape, 3), dtype=np.float32)
    err[..., 0] = fp.astype(np.float32)  # red
    err[..., 1] = tp.astype(np.float32)  # green
    err[..., 2] = fn.astype(np.float32)  # blue

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    lo, hi = np.percentile(image_z, [1, 99])
    axes[0].imshow(image_z, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("Image")
    axes[1].imshow(image_z, cmap="gray", vmin=lo, vmax=hi)
    axes[1].contour(gt.astype(float), levels=[0.5], linewidths=1.2)
    axes[1].set_title("GT contour")
    axes[2].imshow(image_z, cmap="gray", vmin=lo, vmax=hi)
    axes[2].contour(pred.astype(float), levels=[0.5], linewidths=1.2)
    axes[2].set_title("Prediction contour")
    axes[3].imshow(err)
    axes[3].set_title("Error: FP red, TP green, FN blue")
    im = axes[4].imshow(prob, vmin=0.0, vmax=1.0)
    axes[4].set_title("Probability")
    fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_hard_negative_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--failed-csv", default=None, help="CSV with case_idx. If omitted, probes all test cases up to --max-cases.")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tta-scales", nargs="+", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-vis", type=int, default=40)
    parser.add_argument("--connectivity", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    threshold = float(args.threshold if args.threshold is not None else tcfg.get("threshold", 0.5))
    tta_scales = args.tta_scales or [1.0]
    out_dir = Path(args.out_dir or (Path(paths.get("output_dir", "outputs")) / "prediction_failure_probe"))
    _ensure_dir(out_dir)
    vis_dir = _ensure_dir(out_dir / "visualizations")

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

    if args.failed_csv is not None and Path(args.failed_csv).exists():
        failed_raw = pd.read_csv(args.failed_csv)
        case_indices = failed_raw["case_idx"].astype(int).tolist()
    else:
        failed_raw = pd.DataFrame()
        case_indices = list(range(len(dataset)))
    if args.max_cases is not None:
        case_indices = case_indices[:args.max_cases]

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    pcfg = cfg.get("postprocess", {})
    use_pp = bool(pcfg.get("enabled", False)) and not args.no_postprocess

    rows: List[Dict] = []
    vis_count = 0
    for idx in tqdm(case_indices, desc="prediction-probe"):
        image, mask = dataset[idx]
        image_b = image.unsqueeze(0).to(device)
        mask_np = (mask[0].numpy() > 0.5)

        logits = multiscale_logits(model, image_b, scales=tta_scales)
        prob_t = torch.sigmoid(logits).detach().cpu()
        prob_np = prob_t[0, 0].numpy().astype(np.float32)

        if use_pp:
            pred_b, st = postprocess_batch_predictions(image_b.detach().cpu(), prob_t, pcfg, threshold=threshold)
            pred_np = pred_b[0, 0].numpy().astype(bool)
        else:
            st = {"components_before": 0, "components_after": 0, "removed_small": 0, "removed_intensity": 0}
            pred_np = prob_np > threshold

        gt = mask_np.astype(bool)
        pred = pred_np.astype(bool)
        tp_mask = np.logical_and(pred, gt)
        fp_mask = np.logical_and(pred, ~gt)
        fn_mask = np.logical_and(~pred, gt)
        tn_mask = np.logical_and(~pred, ~gt)

        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())
        tn = int(tn_mask.sum())
        gt_area = int(gt.sum())
        pred_area = int(pred.sum())

        image_z = _fg_zscore(image.detach().cpu())
        comp_metrics = _component_matching(gt, pred, connectivity=args.connectivity)

        region_stats = {}
        region_stats.update(_safe_region_stats(prob_np[gt], "prob_in_gt"))
        region_stats.update(_safe_region_stats(prob_np[fp_mask], "prob_in_fp"))
        region_stats.update(_safe_region_stats(prob_np[fn_mask], "prob_in_fn"))
        region_stats.update(_safe_region_stats(image_z[gt], "intensity_z_in_gt"))
        region_stats.update(_safe_region_stats(image_z[fp_mask], "intensity_z_in_fp"))
        region_stats.update(_safe_region_stats(image_z[fn_mask], "intensity_z_in_fn"))

        max_prob_gt = region_stats.get("prob_in_gt_max", float("nan"))
        subtype, hint = _classify_failure(tp, fp, fn, gt_area, pred_area, max_prob_gt, threshold)
        dice = float((2 * tp + EPS) / (2 * tp + fp + fn + EPS))
        iou = float((tp + EPS) / (tp + fp + fn + EPS))
        precision = float((tp + EPS) / (tp + fp + EPS))
        recall = float((tp + EPS) / (tp + fn + EPS))

        row = {
            "case_idx": int(idx),
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "gt_area": gt_area,
            "pred_area": pred_area,
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "tn_pixels": tn,
            "mean_prob": float(prob_np.mean()),
            "max_prob": float(prob_np.max()),
            "threshold": threshold,
            "postprocess_enabled": use_pp,
            "post_components_before": int(st.get("components_before", 0)),
            "post_components_after": int(st.get("components_after", 0)),
            "post_removed_small": int(st.get("removed_small", 0)),
            "post_removed_intensity": int(st.get("removed_intensity", 0)),
            "failure_subtype": subtype,
            "diagnostic_hint": hint,
        }
        row.update(comp_metrics)
        row.update(region_stats)
        rows.append(row)

        # Visualize the most important cases: TP=0, component miss, or low dice.
        should_vis = (tp == 0 and gt_area > 0) or (dice < 0.2) or (comp_metrics["missed_component_count"] > 0)
        if should_vis and vis_count < args.max_vis:
            _visualize_case(
                vis_dir / f"case_{idx:04d}_{subtype}.png",
                image_z, gt, pred, prob_np,
                title=f"case={idx} | {subtype} | TP={tp}, FP={fp}, FN={fn}, maxProbGT={max_prob_gt:.3f}",
            )
            vis_count += 1

    result_df = pd.DataFrame(rows)
    # Merge original failed CSV columns for convenience.
    if not failed_raw.empty:
        keep_cols = [c for c in failed_raw.columns if c != "case_idx"]
        result_df = result_df.merge(failed_raw[["case_idx"] + keep_cols], on="case_idx", how="left", suffixes=("", "_input"))
    result_df.to_csv(out_dir / "prediction_probe_cases.csv", index=False)

    summary = {
        "cases_probed": int(len(result_df)),
        "threshold": threshold,
        "tta_scales": tta_scales,
        "postprocess_enabled": use_pp,
        "tp_zero_cases": int(((result_df["tp_pixels"] == 0) & (result_df["gt_area"] > 0)).sum()),
        "complete_miss_with_fp_cases": int((result_df["failure_subtype"].str.contains("complete_miss_with", na=False)).sum()),
        "below_threshold_miss_cases": int((result_df["failure_subtype"] == "below_threshold_miss").sum()),
        "mean_prob_in_gt_mean": float(result_df["prob_in_gt_mean"].mean(skipna=True)),
        "max_prob_in_gt_median": float(result_df["prob_in_gt_max"].median(skipna=True)),
        "mean_prob_in_fp_mean": float(result_df["prob_in_fp_mean"].mean(skipna=True)),
        "component_recall_mean": float(result_df["component_recall"].mean(skipna=True)),
        "subtype_counts": result_df["failure_subtype"].value_counts().to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown report
    lines = ["# Prediction Failure Probe Report\n"]
    lines.append("## Summary\n")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## How to interpret\n")
    lines.extend([
        "- If `prob_in_gt_max` is very low, the model did not see the GT lesion at all. This supports a data coverage / OOD / representation issue.",
        "- If `prob_in_gt_max` is close to the threshold, threshold tuning, TTA, or calibration may help.",
        "- If `prob_in_fp_max` is high while `prob_in_gt_max` is low, the model prefers a wrong region. Inspect annotation consistency and nearest train matches.",
        "- If `missed_component_count` is high, report component-level recall, not only Dice.",
    ])
    (out_dir / "prediction_probe_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved prediction probe outputs to: {out_dir}")


if __name__ == "__main__":
    main()
