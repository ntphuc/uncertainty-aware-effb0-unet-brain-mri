#!/usr/bin/env python3
"""Uncertainty-aware failure analysis for binary brain MRI segmentation.

What this script produces:
  - case-level segmentation metrics;
  - region-level uncertainty maps;
  - automatic textual reliability warning for each case;
  - failure-detection AUROC/AUPRC for uncertainty score;
  - risk-coverage curve: whether removing high-uncertainty cases improves mean Dice;
  - visualizations for the most uncertain / most failed cases.

This script does not need doctor interaction at inference. It uses ground-truth masks
only for offline evaluation of whether the uncertainty score actually detects failures.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import NPYSliceDataset
from models import build_model
from utils.metrics import hd95, assd, boundary_f1
from utils.misc import load_config, save_json
from utils.postprocess import postprocess_batch_predictions
from utils.text_report import build_text_explanation, failure_label_from_metrics, risk_level
from utils.uncertainty import (
    build_train_references_from_arrays,
    component_feature,
    component_matching_stats,
    component_table,
    dice_iou_from_binary_np,
    image_global_feature,
    mc_dropout_probability_stack,
    normalized_binary_entropy_from_prob,
    summarize_uncertainty_maps,
    tta_probability_stack,
)


def _load_checkpoint(model, checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    return ckpt


def _safe_float(x, default: float = 0.0) -> float:
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _minmax_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if not finite.any():
        return out
    v = values[finite]
    lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
    if abs(hi - lo) < 1e-8:
        out[finite] = 0.0
    else:
        out[finite] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out


def compute_case_uncertainty_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized uncertainty score columns.

    The score combines complementary signals:
      entropy: probability ambiguity;
      TTA variance: instability under augmentation;
      component instability: predicted components not stable / many false-like regions;
      train distance: image/predicted component far from training reference.
    """
    df = df.copy()
    entropy_n = _minmax_norm(df["entropy_p95"].values)
    var_n = _minmax_norm(df["tta_var_p95"].values)
    img_dist_n = _minmax_norm(df.get("train_image_distance", pd.Series(np.zeros(len(df)))).values)
    comp_dist_n = _minmax_norm(df.get("pred_component_train_distance", pd.Series(np.zeros(len(df)))).values)
    comp_instab_n = _minmax_norm(df.get("component_instability", pd.Series(np.zeros(len(df)))).values)

    # Conservative weights. Entropy alone is not enough for high-confidence wrong predictions,
    # so train-distance and TTA variance are given comparable weight.
    score = (
        0.25 * entropy_n +
        0.25 * var_n +
        0.20 * img_dist_n +
        0.20 * comp_dist_n +
        0.10 * comp_instab_n
    )
    df["entropy_norm"] = entropy_n
    df["tta_var_norm"] = var_n
    df["train_image_distance_norm"] = img_dist_n
    df["pred_component_train_distance_norm"] = comp_dist_n
    df["component_instability_norm"] = comp_instab_n
    df["case_uncertainty_score"] = score.astype(float)
    df["risk_level"] = [risk_level(float(s)) for s in score]
    return df


def _prob_stats_in_regions(prob: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    fp = pred & (~gt)
    fn = (~pred) & gt
    out: Dict[str, float] = {}
    for name, mask in [("gt", gt), ("pred", pred), ("fp", fp), ("fn", fn)]:
        if mask.sum() > 0:
            vals = prob[mask]
            out[f"prob_in_{name}_mean"] = float(vals.mean())
            out[f"prob_in_{name}_max"] = float(vals.max())
            out[f"prob_in_{name}_p95"] = float(np.percentile(vals, 95))
        else:
            out[f"prob_in_{name}_mean"] = float("nan")
            out[f"prob_in_{name}_max"] = float("nan")
            out[f"prob_in_{name}_p95"] = float("nan")
    return out


def _map_stats(name: str, m: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    boundary_like = np.logical_xor(ndimage.binary_dilation(pred, iterations=2), ndimage.binary_erosion(pred, iterations=1)) if pred.any() else pred
    regions = {
        "all": np.ones_like(m, dtype=bool),
        "pred": pred,
        "gt": gt,
        "boundary": boundary_like,
    }
    out: Dict[str, float] = {}
    for rname, mask in regions.items():
        if mask.sum() > 0:
            vals = m[mask]
            out[f"{name}_{rname}_mean"] = float(vals.mean())
            out[f"{name}_{rname}_p95"] = float(np.percentile(vals, 95))
        else:
            out[f"{name}_{rname}_mean"] = float("nan")
            out[f"{name}_{rname}_p95"] = float("nan")
    return out


def _case_figure(
    save_path: Path,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    unc: np.ndarray,
    text: str,
    title: str,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fp = pred.astype(bool) & (~gt.astype(bool))
    fn = (~pred.astype(bool)) & gt.astype(bool)
    tp = pred.astype(bool) & gt.astype(bool)

    error_rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    error_rgb[..., 0] = fp.astype(np.float32)  # red FP
    error_rgb[..., 1] = tp.astype(np.float32)  # green TP
    error_rgb[..., 2] = fn.astype(np.float32)  # blue FN

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]
    fig.suptitle(title, fontsize=13)

    panels = [
        (image, "MRI", "gray"),
        (gt, "GT mask", "gray"),
        (pred, "Prediction", "gray"),
        (prob, "Probability", "magma"),
        (unc, "Uncertainty map", "inferno"),
        (error_rgb, "Error map: FP red / TP green / FN blue", None),
    ]
    for ax, (arr, name, cmap) in zip(axes[:6], panels):
        if cmap is None:
            ax.imshow(image, cmap="gray")
            ax.imshow(arr, alpha=0.65)
        else:
            ax.imshow(arr, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")

    # Overlay uncertainty on MRI
    axes[6].imshow(image, cmap="gray")
    axes[6].imshow(unc, cmap="inferno", alpha=0.55)
    axes[6].set_title("Uncertainty overlay")
    axes[6].axis("off")

    axes[7].axis("off")
    wrapped = "\n".join([text[i:i + 72] for i in range(0, len(text), 72)])
    axes[7].text(0.0, 1.0, wrapped, va="top", ha="left", fontsize=10)
    axes[7].set_title("Automatic textual warning")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _risk_coverage(df: pd.DataFrame, out_path: Path):
    rows = []
    if "dice" not in df.columns:
        return
    ordered = df.sort_values("case_uncertainty_score", ascending=True).reset_index(drop=True)
    n = len(ordered)
    for cov in np.linspace(0.1, 1.0, 10):
        k = max(1, int(round(n * cov)))
        keep = ordered.iloc[:k]
        rows.append({
            "coverage": float(k / n),
            "kept_cases": int(k),
            "rejected_cases": int(n - k),
            "mean_dice_retained": float(keep["dice"].mean()),
            "mean_hd95_retained": float(keep["hd95"].replace([np.inf, -np.inf], np.nan).mean()) if "hd95" in keep else float("nan"),
            "failure_rate_retained": float(keep["failure_binary"].mean()) if "failure_binary" in keep else float("nan"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_hard_negative_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out-dir", default=None)

    parser.add_argument("--tta-scales", nargs="+", type=float, default=[1.0, 1.25, 1.5])
    parser.add_argument("--no-hflip", action="store_true")
    parser.add_argument("--vflip", action="store_true")
    parser.add_argument("--mc-runs", type=int, default=0, help="Optional MC Dropout runs. 0 disables MC dropout.")
    parser.add_argument("--postprocess", action="store_true", help="Apply config postprocess before metrics/text. Default: raw threshold mask.")
    parser.add_argument("--train-coverage", action="store_true", help="Build train reference and compute train-distance uncertainty.")
    parser.add_argument("--train-ref-max-samples", type=int, default=2000)

    parser.add_argument("--failure-dice-threshold", type=float, default=0.70)
    parser.add_argument("--failure-hd95-threshold", type=float, default=20.0)
    parser.add_argument("--save-top-k", type=int, default=30)
    parser.add_argument("--language", choices=["vi", "en"], default="vi")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    paths = cfg["paths"]
    tcfg = cfg.get("training", {})
    mcfg = cfg.get("model", {})

    threshold = float(args.threshold if args.threshold is not None else tcfg.get("threshold", 0.5))
    batch_size = int(args.batch_size if args.batch_size is not None else tcfg.get("batch_size", 8))
    out_dir = Path(args.out_dir or (Path(paths.get("output_dir", "outputs")) / "uncertainty_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visualizations").mkdir(exist_ok=True)
    (out_dir / "text_reports").mkdir(exist_ok=True)

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=tcfg.get("num_workers", 2), pin_memory=True)

    train_refs = None
    if args.train_coverage:
        print("[INFO] Building train coverage reference from X_train/Y_train...")
        Xtr = np.load(paths["x_train"], mmap_mode="r")
        Ytr = np.load(paths["y_train"], mmap_mode="r")
        train_refs = build_train_references_from_arrays(
            Xtr, Ytr,
            max_samples=args.train_ref_max_samples,
            min_component_area=5,
            seed=int(cfg.get("seed", 42)),
        )
        np.save(out_dir / "train_image_reference_mean.npy", train_refs["image"].mean)
        np.save(out_dir / "train_component_reference_mean.npy", train_refs["component"].mean)

    model = build_model(cfg).to(device)
    ckpt = _load_checkpoint(model, args.checkpoint, device)
    model.eval()

    rows: List[Dict[str, object]] = []
    case_artifacts: List[Tuple[int, float]] = []
    global_idx = 0

    for images, masks in tqdm(loader, desc=f"uncertainty-{args.split}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bs = images.size(0)

        tta_stack = tta_probability_stack(
            model,
            images,
            scales=args.tta_scales,
            hflip=not args.no_hflip,
            vflip=args.vflip,
        )
        mean_prob = tta_stack.mean(dim=0)
        tta_var = tta_stack.var(dim=0, unbiased=False)

        if args.mc_runs and args.mc_runs > 1:
            mc_stack = mc_dropout_probability_stack(model, images, runs=args.mc_runs)
            mc_mean = mc_stack.mean(dim=0)
            mc_var = mc_stack.var(dim=0, unbiased=False)
            mean_prob = 0.5 * mean_prob + 0.5 * mc_mean
            tta_var = 0.5 * tta_var + 0.5 * mc_var

        unc_maps = summarize_uncertainty_maps(mean_prob, tta_var, threshold=threshold)
        pred_mask_t = (mean_prob > threshold).float()

        if args.postprocess:
            pcfg = cfg.get("postprocess", {})
            pred_mask_t, post_stats = postprocess_batch_predictions(images, mean_prob, pcfg, threshold=threshold)
        else:
            post_stats = {}

        entropy_t = unc_maps["entropy"]
        combined_unc_t = unc_maps["combined_uncertainty"]

        for b in range(bs):
            idx = global_idx + b
            img_np = images[b, 0].detach().cpu().numpy().astype(np.float32)
            gt_np = (masks[b, 0].detach().cpu().numpy() > 0.5)
            pred_np = (pred_mask_t[b, 0].detach().cpu().numpy() > 0.5)
            prob_np = mean_prob[b, 0].detach().cpu().numpy().astype(np.float32)
            entropy_np = entropy_t[b, 0].detach().cpu().numpy().astype(np.float32)
            tta_var_np = tta_var[b, 0].detach().cpu().numpy().astype(np.float32)
            combined_unc_np = combined_unc_t[b, 0].detach().cpu().numpy().astype(np.float32)

            metrics = dice_iou_from_binary_np(pred_np, gt_np)
            try:
                metrics["hd95"] = hd95(pred_np, gt_np)
                metrics["assd"] = assd(pred_np, gt_np)
                metrics["boundary_f1"] = boundary_f1(pred_np, gt_np, tolerance=2)
            except Exception:
                metrics["hd95"] = float("inf")
                metrics["assd"] = float("inf")
                metrics["boundary_f1"] = 0.0

            comp_stats = component_matching_stats(pred_np, gt_np)
            gt_area = int(gt_np.sum())
            pred_area = int(pred_np.sum())

            row: Dict[str, object] = {
                "case_idx": idx,
                "threshold": threshold,
                "gt_area": gt_area,
                "pred_area": pred_area,
                **metrics,
                **comp_stats,
            }
            row.update(_prob_stats_in_regions(prob_np, pred_np, gt_np))
            row.update(_map_stats("entropy", entropy_np, pred_np, gt_np))
            row.update(_map_stats("tta_var", tta_var_np, pred_np, gt_np))
            row.update(_map_stats("combined_unc", combined_unc_np, pred_np, gt_np))

            # Component instability: many predicted components + unstable pixels around prediction.
            pred_comps = component_table(pred_np, img_np, min_area=1)
            if pred_comps:
                comp_var_vals = []
                comp_train_dists = []
                for comp in pred_comps:
                    # Use bbox-centered component approximation: collect actual connected comp by nearest not stored here.
                    comp_train_dists.append(float("nan"))
                row["largest_pred_component_area"] = max([c["area"] for c in pred_comps])
            else:
                row["largest_pred_component_area"] = 0

            row["component_instability"] = float(
                _safe_float(row.get("tta_var_pred_p95"), 0.0) * (1.0 + math.log1p(comp_stats["pred_component_count"]))
            )

            if train_refs is not None:
                img_feat = image_global_feature(img_np)
                row["train_image_distance"] = train_refs["image"].nearest_distance(img_feat)
                pred_comp_dists: List[float] = []
                for comp in pred_comps:
                    pred_comp_dists.append(train_refs["component"].nearest_distance(component_feature(comp)))
                row["pred_component_train_distance"] = float(np.nanmax(pred_comp_dists)) if pred_comp_dists else float("nan")
            else:
                row["train_image_distance"] = float("nan")
                row["pred_component_train_distance"] = float("nan")

            # Basic failure label before uncertainty score; score added later after normalization.
            row["failure_subtype"] = failure_label_from_metrics(row)
            hd95_clean = _safe_float(row.get("hd95"), default=1e9)
            row["failure_binary"] = int(
                float(row["dice"]) < args.failure_dice_threshold or
                hd95_clean > args.failure_hd95_threshold or
                row["failure_subtype"] in {"complete_miss_with_false_positive", "complete_miss"}
            )
            rows.append(row)
        global_idx += bs

    df = pd.DataFrame(rows)
    df = compute_case_uncertainty_score(df)
    # Generate text after score is available.
    texts = []
    for _, r in df.iterrows():
        d = r.to_dict()
        text = build_text_explanation(d, language=args.language)
        texts.append(text)
    df["text_warning"] = texts

    # Save per-case text reports.
    for _, r in df.iterrows():
        txt_path = out_dir / "text_reports" / f"case_{int(r['case_idx']):04d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(str(r["text_warning"]) + "\n")

    cases_csv = out_dir / "uncertainty_cases.csv"
    df.to_csv(cases_csv, index=False)

    # Failure detection metrics.
    summary: Dict[str, object] = {
        "num_cases": int(len(df)),
        "threshold": threshold,
        "tta_scales": args.tta_scales,
        "hflip": not args.no_hflip,
        "vflip": bool(args.vflip),
        "mc_runs": int(args.mc_runs),
        "postprocess_used": bool(args.postprocess),
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "checkpoint_best_dice": ckpt.get("best_dice") if isinstance(ckpt, dict) else None,
        "mean_dice": float(df["dice"].mean()),
        "mean_iou": float(df["iou"].mean()),
        "mean_hd95": float(df["hd95"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_assd": float(df["assd"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_boundary_f1": float(df["boundary_f1"].mean()),
        "failure_rate": float(df["failure_binary"].mean()),
        "high_risk_cases": int((df["risk_level"] == "high").sum()),
        "medium_risk_cases": int((df["risk_level"] == "medium").sum()),
        "low_risk_cases": int((df["risk_level"] == "low").sum()),
    }
    try:
        y = df["failure_binary"].astype(int).values
        s = df["case_uncertainty_score"].astype(float).values
        if len(np.unique(y)) > 1:
            summary["failure_detection_auroc"] = float(roc_auc_score(y, s))
            summary["failure_detection_auprc"] = float(average_precision_score(y, s))
        else:
            summary["failure_detection_auroc"] = None
            summary["failure_detection_auprc"] = None
    except Exception as e:
        summary["failure_detection_error"] = str(e)

    # Risk groups.
    risk_summary = []
    for level in ["low", "medium", "high"]:
        sub = df[df["risk_level"] == level]
        if len(sub) == 0:
            continue
        risk_summary.append({
            "risk_level": level,
            "n": int(len(sub)),
            "mean_dice": float(sub["dice"].mean()),
            "failure_rate": float(sub["failure_binary"].mean()),
            "mean_uncertainty_score": float(sub["case_uncertainty_score"].mean()),
        })
    pd.DataFrame(risk_summary).to_csv(out_dir / "risk_group_summary.csv", index=False)
    summary["risk_group_summary"] = risk_summary

    _risk_coverage(df, out_dir / "risk_coverage.csv")
    save_json(summary, out_dir / "summary.json")

    # Save visualizations for the most uncertain cases and severe failures.
    save_candidates = df.sort_values(["case_uncertainty_score", "failure_binary"], ascending=[False, False]).head(args.save_top_k)
    # Re-run saved cases one by one to avoid keeping all maps in memory.
    save_indices = set(int(x) for x in save_candidates["case_idx"].tolist())
    if save_indices:
        model.eval()
        for idx in tqdm(sorted(save_indices), desc="saving-uncertainty-figures"):
            image, mask = dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            tta_stack = tta_probability_stack(model, image_b, scales=args.tta_scales, hflip=not args.no_hflip, vflip=args.vflip)
            mean_prob = tta_stack.mean(dim=0)
            tta_var = tta_stack.var(dim=0, unbiased=False)
            unc_maps = summarize_uncertainty_maps(mean_prob, tta_var, threshold=threshold)
            pred_t = (mean_prob > threshold).float()
            if args.postprocess:
                pred_t, _ = postprocess_batch_predictions(image_b, mean_prob, cfg.get("postprocess", {}), threshold=threshold)
            row = df[df["case_idx"] == idx].iloc[0]
            _case_figure(
                out_dir / "visualizations" / f"case_{idx:04d}_uncertainty.png",
                image[0].detach().cpu().numpy(),
                (mask[0].detach().cpu().numpy() > 0.5),
                (pred_t[0, 0].detach().cpu().numpy() > 0.5),
                mean_prob[0, 0].detach().cpu().numpy(),
                unc_maps["combined_uncertainty"][0, 0].detach().cpu().numpy(),
                str(row["text_warning"]),
                title=(
                    f"Case #{idx} | risk={row['risk_level']} | U={row['case_uncertainty_score']:.3f} | "
                    f"Dice={row['dice']:.3f} | subtype={row['failure_subtype']}"
                ),
            )

    # Markdown report.
    report = out_dir / "uncertainty_report.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Uncertainty-aware Segmentation Failure Report\n\n")
        f.write(f"- Cases: {summary['num_cases']}\n")
        f.write(f"- Mean Dice: {summary['mean_dice']:.4f}\n")
        f.write(f"- Mean HD95: {summary['mean_hd95']:.4f}\n")
        f.write(f"- Failure rate: {summary['failure_rate']:.4f}\n")
        f.write(f"- Failure detection AUROC: {summary.get('failure_detection_auroc')}\n")
        f.write(f"- Failure detection AUPRC: {summary.get('failure_detection_auprc')}\n\n")
        f.write("## Risk groups\n\n")
        f.write(pd.DataFrame(risk_summary).to_markdown(index=False) if risk_summary else "No risk groups.\n")
        f.write("\n\n## Main files\n\n")
        f.write("- `uncertainty_cases.csv`: per-case metrics, uncertainty scores, text warning.\n")
        f.write("- `summary.json`: overall performance and failure-detection metrics.\n")
        f.write("- `risk_coverage.csv`: risk-coverage curve data.\n")
        f.write("- `visualizations/`: MRI/GT/pred/error/uncertainty/text figures.\n")
        f.write("- `text_reports/`: automatic text warning for each case.\n")

    print(f"[DONE] Saved uncertainty analysis to: {out_dir}")
    print(f"[DONE] Main CSV: {cases_csv}")
    print(f"[DONE] Report: {report}")
    print(summary)


if __name__ == "__main__":
    main()
