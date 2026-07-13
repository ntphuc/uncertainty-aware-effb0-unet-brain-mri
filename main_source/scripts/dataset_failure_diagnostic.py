#!/usr/bin/env python3
"""
Dataset-centric diagnostic for segmentation failures.

Goal
----
Check the hypothesis:
    "The model fails because the training set does not contain cases similar to the failed test cases."

This script does NOT train a model. It compares train/test/failed samples using:
  - lesion area distribution
  - connected-component count distribution
  - lesion position/shape distribution
  - lesion intensity distribution
  - nearest train samples by lesion metadata
  - optional nearest train lesion crops by image appearance

Outputs are saved under outputs/dataset_failure_diagnostic by default.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from scipy import ndimage
except Exception as exc:  # pragma: no cover
    ndimage = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.misc import load_config

EPS = 1e-8


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_chw_image(arr: np.ndarray, in_channels: int = 1) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # HWC if last dim is small and first dim is spatial-like.
        if arr.shape[-1] <= 4 and arr.shape[0] > 4:
            arr = np.transpose(arr, (2, 0, 1))
        # else assume CHW
    else:
        raise ValueError(f"Unsupported image sample shape: {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.shape[0] == 1 and in_channels == 3:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] != in_channels:
        if in_channels == 1:
            arr = arr[:1]
        else:
            arr = arr[:in_channels]
    return arr


def _to_hw_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        mask = arr
    elif arr.ndim == 3:
        if arr.shape[-1] <= 4 and arr.shape[0] > 4:  # HWC
            arr = np.transpose(arr, (2, 0, 1))
        # CHW, merge foreground channels
        mask = arr.max(axis=0)
    else:
        raise ValueError(f"Unsupported mask sample shape: {arr.shape}")
    return (mask.astype(np.float32) > 0.5)


def _image_plane(image_chw: np.ndarray) -> np.ndarray:
    # Use the first channel for intensity diagnostics. For 2.5D/3-channel inputs,
    # channel 0 is still a meaningful approximation for image appearance.
    return image_chw[0].astype(np.float32)


def _foreground_zscore(image: np.ndarray, foreground_percentile: float = 5.0) -> Tuple[np.ndarray, Dict[str, float]]:
    finite = image[np.isfinite(image)].astype(np.float32)
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32), {"fg_mean": 0.0, "fg_std": 1.0, "fg_cut": 0.0}
    fg_cut = float(np.percentile(finite, foreground_percentile))
    fg = finite[finite > fg_cut]
    if fg.size < 16:
        fg = finite
    mean = float(np.mean(fg))
    std = float(np.std(fg) + EPS)
    z = (image.astype(np.float32) - mean) / std
    return z.astype(np.float32), {"fg_mean": mean, "fg_std": std, "fg_cut": fg_cut}


def _components(mask: np.ndarray, connectivity: int = 8, min_area: int = 1) -> List[Dict[str, float]]:
    if ndimage is None:
        ys, xs = np.where(mask)
        if len(xs) < min_area:
            return []
        return [{
            "area": float(len(xs)),
            "y1": float(ys.min()), "y2": float(ys.max() + 1),
            "x1": float(xs.min()), "x2": float(xs.max() + 1),
            "cy": float(ys.mean()), "cx": float(xs.mean()),
        }]
    conn = 2 if connectivity == 8 else 1
    structure = ndimage.generate_binary_structure(2, conn)
    labeled, num = ndimage.label(mask.astype(bool), structure=structure)
    comps: List[Dict[str, float]] = []
    for comp_id in range(1, num + 1):
        ys, xs = np.where(labeled == comp_id)
        area = len(xs)
        if area < min_area:
            continue
        comps.append({
            "area": float(area),
            "y1": float(ys.min()), "y2": float(ys.max() + 1),
            "x1": float(xs.min()), "x2": float(xs.max() + 1),
            "cy": float(ys.mean()), "cx": float(xs.mean()),
        })
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def _percentile_rank(train_values: np.ndarray, value: float) -> float:
    values = train_values[np.isfinite(train_values)]
    if values.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(100.0 * np.mean(values <= value))


def _sample_stats(x_arr: np.ndarray, y_arr: np.ndarray, idx: int, split: str, in_channels: int, connectivity: int) -> Dict[str, float]:
    image_chw = _to_chw_image(x_arr[idx], in_channels=in_channels)
    image = _image_plane(image_chw)
    mask = _to_hw_mask(y_arr[idx])
    h, w = mask.shape
    z_img, z_meta = _foreground_zscore(image)
    comps = _components(mask, connectivity=connectivity, min_area=1)

    gt_area = int(mask.sum())
    area_ratio = float(gt_area / (h * w + EPS))
    bbox = _bbox_from_mask(mask)

    if gt_area > 0:
        raw_vals = image[mask]
        z_vals = z_img[mask]
        lesion_raw_mean = float(np.mean(raw_vals))
        lesion_raw_std = float(np.std(raw_vals))
        lesion_raw_p75 = float(np.percentile(raw_vals, 75))
        lesion_raw_max = float(np.max(raw_vals))
        lesion_z_mean = float(np.mean(z_vals))
        lesion_z_std = float(np.std(z_vals))
        lesion_z_p75 = float(np.percentile(z_vals, 75))
        lesion_z_max = float(np.max(z_vals))
    else:
        lesion_raw_mean = lesion_raw_std = lesion_raw_p75 = lesion_raw_max = float("nan")
        lesion_z_mean = lesion_z_std = lesion_z_p75 = lesion_z_max = float("nan")

    if bbox is not None:
        y1, y2, x1, x2 = bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        bbox_area = bbox_h * bbox_w
        bbox_fill = float(gt_area / (bbox_area + EPS))
        centroid_y = float(np.where(mask)[0].mean())
        centroid_x = float(np.where(mask)[1].mean())
    else:
        bbox_h = bbox_w = bbox_area = 0
        bbox_fill = float("nan")
        centroid_y = centroid_x = float("nan")

    comp_areas = np.asarray([c["area"] for c in comps], dtype=np.float32)
    largest = comps[0] if comps else None
    smallest_area = float(comp_areas.min()) if comp_areas.size else 0.0
    largest_area = float(comp_areas.max()) if comp_areas.size else 0.0
    mean_comp_area = float(comp_areas.mean()) if comp_areas.size else 0.0

    return {
        "split": split,
        "case_idx": int(idx),
        "height": int(h),
        "width": int(w),
        "gt_area": int(gt_area),
        "gt_area_ratio": area_ratio,
        "gt_component_count": int(len(comps)),
        "component_area_min": smallest_area,
        "component_area_mean": mean_comp_area,
        "component_area_max": largest_area,
        "largest_component_area": largest_area,
        "bbox_h": float(bbox_h),
        "bbox_w": float(bbox_w),
        "bbox_area": float(bbox_area),
        "bbox_fill_ratio": bbox_fill,
        "bbox_h_norm": float(bbox_h / (h + EPS)),
        "bbox_w_norm": float(bbox_w / (w + EPS)),
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "centroid_y_norm": float(centroid_y / (h + EPS)) if np.isfinite(centroid_y) else float("nan"),
        "centroid_x_norm": float(centroid_x / (w + EPS)) if np.isfinite(centroid_x) else float("nan"),
        "lesion_raw_mean": lesion_raw_mean,
        "lesion_raw_std": lesion_raw_std,
        "lesion_raw_p75": lesion_raw_p75,
        "lesion_raw_max": lesion_raw_max,
        "lesion_z_mean": lesion_z_mean,
        "lesion_z_std": lesion_z_std,
        "lesion_z_p75": lesion_z_p75,
        "lesion_z_max": lesion_z_max,
        "image_fg_mean": z_meta["fg_mean"],
        "image_fg_std": z_meta["fg_std"],
        "image_fg_cut": z_meta["fg_cut"],
    }


def _scan_split(x_path: str, y_path: str, split: str, in_channels: int, connectivity: int, limit: Optional[int] = None) -> pd.DataFrame:
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    n = len(x) if limit is None else min(int(limit), len(x))
    rows = []
    for idx in tqdm(range(n), desc=f"scan-{split}"):
        rows.append(_sample_stats(x, y, idx, split, in_channels, connectivity))
    return pd.DataFrame(rows)


def _vectorize(df: pd.DataFrame, train_ref: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    d = df.copy()
    d["log_gt_area"] = np.log1p(d["gt_area"].astype(float))
    feature_cols = [
        "log_gt_area", "gt_area_ratio", "gt_component_count",
        "component_area_min", "component_area_max",
        "bbox_h_norm", "bbox_w_norm", "bbox_fill_ratio",
        "centroid_y_norm", "centroid_x_norm",
        "lesion_z_mean", "lesion_z_p75", "lesion_z_max",
    ]
    if train_ref is None:
        ref = d
    else:
        ref = train_ref.copy()
        ref["log_gt_area"] = np.log1p(ref["gt_area"].astype(float))
    X = d[feature_cols].to_numpy(dtype=np.float32)
    R = ref[feature_cols].to_numpy(dtype=np.float32)
    means = np.nanmean(R, axis=0)
    stds = np.nanstd(R, axis=0) + EPS
    # replace nan by train mean before standardization
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(means, inds[1])
    Z = (X - means) / stds
    return Z.astype(np.float32), feature_cols, means.astype(np.float32), stds.astype(np.float32)


def _nearest_by_stats(train_df: pd.DataFrame, failed_df: pd.DataFrame, topk: int) -> pd.DataFrame:
    train_pos = train_df[train_df["gt_area"] > 0].copy()
    if train_pos.empty or failed_df.empty:
        return pd.DataFrame()
    train_vec, feature_cols, means, stds = _vectorize(train_pos, train_ref=train_pos)
    failed_vec, _, _, _ = _vectorize(failed_df, train_ref=train_pos)
    rows = []
    for i, row in failed_df.reset_index(drop=True).iterrows():
        diff = train_vec - failed_vec[i:i+1]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(dist)[:topk]
        for rank, pos in enumerate(order, start=1):
            tr = train_pos.iloc[int(pos)]
            rows.append({
                "failed_case_idx": int(row["case_idx"]),
                "rank": rank,
                "train_case_idx": int(tr["case_idx"]),
                "stats_distance": float(dist[pos]),
                "failed_gt_area": float(row["gt_area"]),
                "train_gt_area": float(tr["gt_area"]),
                "failed_components": int(row["gt_component_count"]),
                "train_components": int(tr["gt_component_count"]),
                "failed_lesion_z_mean": float(row["lesion_z_mean"]),
                "train_lesion_z_mean": float(tr["lesion_z_mean"]),
            })
    return pd.DataFrame(rows)


def _resize_crop_to_vector(image: np.ndarray, mask: np.ndarray, crop_size: int = 32, scale: float = 2.5) -> Optional[np.ndarray]:
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return None
    y1, y2, x1, x2 = bbox
    h, w = mask.shape
    cy = (y1 + y2) / 2.0
    cx = (x1 + x2) / 2.0
    box_h = y2 - y1
    box_w = x2 - x1
    side = int(max(box_h, box_w) * scale)
    side = max(side, 32)
    side = min(side, h, w)
    yy1 = int(round(cy - side / 2))
    xx1 = int(round(cx - side / 2))
    yy1 = max(0, min(yy1, h - side))
    xx1 = max(0, min(xx1, w - side))
    crop = image[yy1:yy1 + side, xx1:xx1 + side].astype(np.float32)
    z, _ = _foreground_zscore(crop, foreground_percentile=5.0)
    t = torch.from_numpy(z)[None, None]
    t = F.interpolate(t, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    v = t.flatten().numpy().astype(np.float32)
    v = v - float(v.mean())
    v = v / (float(np.linalg.norm(v)) + EPS)
    return v


def _nearest_by_crop(
    cfg: dict,
    failed_df: pd.DataFrame,
    out_dir: Path,
    topk: int,
    crop_size: int,
    max_train_candidates: int,
    seed: int,
) -> pd.DataFrame:
    paths = cfg["paths"]
    in_channels = int(cfg.get("model", {}).get("in_channels", 1))
    x_train = np.load(paths["x_train"], mmap_mode="r")
    y_train = np.load(paths["y_train"], mmap_mode="r")
    x_test = np.load(paths["x_test"], mmap_mode="r")
    y_test = np.load(paths["y_test"], mmap_mode="r")

    train_positive_indices = []
    for idx in range(len(y_train)):
        if _to_hw_mask(y_train[idx]).sum() > 0:
            train_positive_indices.append(idx)
    if len(train_positive_indices) == 0:
        return pd.DataFrame()
    if len(train_positive_indices) > max_train_candidates:
        rng = random.Random(seed)
        train_positive_indices = rng.sample(train_positive_indices, max_train_candidates)

    train_vecs = []
    train_ids = []
    for idx in tqdm(train_positive_indices, desc="build-train-crop-vectors"):
        img = _image_plane(_to_chw_image(x_train[idx], in_channels=in_channels))
        m = _to_hw_mask(y_train[idx])
        v = _resize_crop_to_vector(img, m, crop_size=crop_size)
        if v is not None:
            train_vecs.append(v)
            train_ids.append(idx)
    if not train_vecs:
        return pd.DataFrame()
    train_mat = np.stack(train_vecs, axis=0).astype(np.float32)

    rows = []
    for _, row in tqdm(failed_df.iterrows(), total=len(failed_df), desc="crop-nn-failed"):
        fidx = int(row["case_idx"])
        img = _image_plane(_to_chw_image(x_test[fidx], in_channels=in_channels))
        m = _to_hw_mask(y_test[fidx])
        q = _resize_crop_to_vector(img, m, crop_size=crop_size)
        if q is None:
            continue
        sim = train_mat @ q
        order = np.argsort(-sim)[:topk]
        for rank, pos in enumerate(order, start=1):
            rows.append({
                "failed_case_idx": fidx,
                "rank": rank,
                "train_case_idx": int(train_ids[int(pos)]),
                "cosine_similarity": float(sim[int(pos)]),
            })
    crop_nn = pd.DataFrame(rows)
    crop_nn.to_csv(out_dir / "nearest_train_by_lesion_crop.csv", index=False)
    return crop_nn


def _plot_distributions(train_df: pd.DataFrame, test_df: pd.DataFrame, failed_df: pd.DataFrame, out_dir: Path):
    plot_dir = _ensure_dir(out_dir / "plots")
    specs = [
        ("gt_area", True, "GT lesion area"),
        ("gt_component_count", False, "GT component count"),
        ("lesion_z_mean", False, "Lesion mean z-intensity"),
        ("bbox_h_norm", False, "GT bbox height / image height"),
        ("bbox_w_norm", False, "GT bbox width / image width"),
    ]
    for col, logx, title in specs:
        plt.figure(figsize=(8, 5))
        for name, df in [("train", train_df), ("test", test_df), ("failed", failed_df)]:
            vals = df[col].to_numpy(dtype=float) if col in df else np.array([])
            vals = vals[np.isfinite(vals)]
            if col == "gt_area":
                vals = vals[vals > 0]
            if vals.size == 0:
                continue
            if logx:
                vals = np.log1p(vals)
                xlabel = f"log1p({col})"
            else:
                xlabel = col
            plt.hist(vals, bins=40, alpha=0.45, label=name, density=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}_distribution.png", dpi=160)
        plt.close()


def _save_nearest_visuals(cfg: dict, failed_df: pd.DataFrame, nn_df: pd.DataFrame, out_dir: Path, max_cases: int = 12):
    if nn_df.empty or failed_df.empty:
        return
    vis_dir = _ensure_dir(out_dir / "nearest_train_visuals")
    paths = cfg["paths"]
    in_channels = int(cfg.get("model", {}).get("in_channels", 1))
    x_train = np.load(paths["x_train"], mmap_mode="r")
    y_train = np.load(paths["y_train"], mmap_mode="r")
    x_test = np.load(paths["x_test"], mmap_mode="r")
    y_test = np.load(paths["y_test"], mmap_mode="r")

    selected = failed_df.head(max_cases)
    for _, frow in selected.iterrows():
        fidx = int(frow["case_idx"])
        matches = nn_df[nn_df["failed_case_idx"] == fidx].sort_values("rank").head(5)
        if matches.empty:
            continue
        cols = 1 + len(matches)
        fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 6.0))
        if cols == 1:
            axes = np.asarray([[axes[0]], [axes[1]]])

        def show(ax_img, ax_mask, x_arr, y_arr, idx, title):
            img = _image_plane(_to_chw_image(x_arr[idx], in_channels=in_channels))
            mask = _to_hw_mask(y_arr[idx])
            z, _ = _foreground_zscore(img)
            lo, hi = np.percentile(z, [1, 99])
            ax_img.imshow(z, cmap="gray", vmin=lo, vmax=hi)
            ax_img.contour(mask.astype(float), levels=[0.5], linewidths=1.0)
            ax_img.set_title(title, fontsize=9)
            ax_img.axis("off")
            ax_mask.imshow(mask.astype(float), cmap="gray")
            ax_mask.set_title("GT mask", fontsize=9)
            ax_mask.axis("off")

        show(axes[0, 0], axes[1, 0], x_test, y_test, fidx, f"FAILED test #{fidx}")
        for c, (_, mrow) in enumerate(matches.iterrows(), start=1):
            tidx = int(mrow["train_case_idx"])
            dist_txt = f"d={mrow.get('stats_distance', np.nan):.2f}" if "stats_distance" in mrow else ""
            sim_txt = f"sim={mrow.get('cosine_similarity', np.nan):.2f}" if "cosine_similarity" in mrow else ""
            show(axes[0, c], axes[1, c], x_train, y_train, tidx, f"train #{tidx}\n{dist_txt}{sim_txt}")
        plt.tight_layout()
        fig.savefig(vis_dir / f"failed_{fidx:04d}_nearest_train.png", dpi=160)
        plt.close(fig)


def _build_report(out_dir: Path, summary: Dict, hints: List[str]):
    lines = []
    lines.append("# Dataset Failure Diagnostic Report\n")
    lines.append("## Key counts\n")
    for k, v in summary.items():
        if isinstance(v, (int, float, str)):
            lines.append(f"- {k}: {v}")
    lines.append("\n## Interpretation hints\n")
    if hints:
        for h in hints:
            lines.append(f"- {h}")
    else:
        lines.append("- No strong distribution red flag was detected by the automatic rules. Inspect nearest-neighbor visuals and labels manually.")
    lines.append("\n## Files to inspect\n")
    lines.extend([
        "- train_slice_stats.csv",
        "- test_slice_stats.csv",
        "- failed_cases_enriched.csv",
        "- nearest_train_by_stats.csv",
        "- plots/*.png",
        "- nearest_train_visuals/*.png",
    ])
    (out_dir / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficient_b0_hard_negative_v3.yaml")
    parser.add_argument("--failed-csv", default=None, help="CSV from diagnose_failed_cases.py/test_failed_cases.csv. Must contain case_idx.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit for train/test scan.")
    parser.add_argument("--with-crop-nn", action="store_true", help="Also search nearest train lesion crops by appearance.")
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--max-train-crop-candidates", type=int, default=5000)
    parser.add_argument("--max-visual-cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    in_channels = int(cfg.get("model", {}).get("in_channels", 1))
    out_dir = Path(args.out_dir or (Path(paths.get("output_dir", "outputs")) / "dataset_failure_diagnostic"))
    _ensure_dir(out_dir)

    train_df = _scan_split(paths["x_train"], paths["y_train"], "train", in_channels, args.connectivity, args.limit)
    test_df = _scan_split(paths["x_test"], paths["y_test"], "test", in_channels, args.connectivity, args.limit)
    train_df.to_csv(out_dir / "train_slice_stats.csv", index=False)
    test_df.to_csv(out_dir / "test_slice_stats.csv", index=False)

    if args.failed_csv is not None and Path(args.failed_csv).exists():
        failed_raw = pd.read_csv(args.failed_csv)
        if "case_idx" not in failed_raw.columns:
            raise ValueError("failed CSV must contain a case_idx column")
        failed_raw["case_idx"] = failed_raw["case_idx"].astype(int)
        failed_df = failed_raw.merge(test_df, on="case_idx", how="left", suffixes=("", "_from_test"))
    else:
        # If no failed CSV is provided, use the lowest-area positive test cases as diagnostic targets.
        failed_df = test_df[test_df["gt_area"] > 0].sort_values("gt_area").head(30).copy()

    train_pos = train_df[train_df["gt_area"] > 0].copy()
    for col in ["gt_area", "gt_component_count", "lesion_z_mean", "lesion_z_p75", "bbox_h_norm", "bbox_w_norm"]:
        train_vals = train_pos[col].to_numpy(dtype=float) if col in train_pos else np.array([])
        failed_df[f"{col}_percentile_in_train"] = failed_df[col].apply(lambda v: _percentile_rank(train_vals, float(v)))

    # automatic flags
    failed_df["flag_tiny_lesion_vs_train"] = failed_df["gt_area_percentile_in_train"] < 10
    failed_df["flag_large_component_count_vs_train"] = failed_df["gt_component_count_percentile_in_train"] > 90
    failed_df["flag_low_lesion_intensity_vs_train"] = failed_df["lesion_z_mean_percentile_in_train"] < 10
    failed_df["flag_high_lesion_intensity_vs_train"] = failed_df["lesion_z_mean_percentile_in_train"] > 90
    if "tp_pixels" in failed_df.columns:
        failed_df["is_tp_zero"] = failed_df["tp_pixels"].fillna(0).astype(float) <= 0
        if "pred_area" in failed_df.columns:
            failed_df["is_complete_miss_with_fp"] = failed_df["is_tp_zero"] & (failed_df["pred_area"].fillna(0).astype(float) > 0)
        else:
            failed_df["is_complete_miss_with_fp"] = failed_df["is_tp_zero"]
    else:
        failed_df["is_tp_zero"] = False
        failed_df["is_complete_miss_with_fp"] = False

    failed_df.to_csv(out_dir / "failed_cases_enriched.csv", index=False)

    nn_stats = _nearest_by_stats(train_df, failed_df, topk=args.topk)
    nn_stats.to_csv(out_dir / "nearest_train_by_stats.csv", index=False)
    nn_for_vis = nn_stats

    if args.with_crop_nn:
        nn_crop = _nearest_by_crop(
            cfg, failed_df, out_dir,
            topk=args.topk,
            crop_size=args.crop_size,
            max_train_candidates=args.max_train_crop_candidates,
            seed=args.seed,
        )
        if not nn_crop.empty:
            nn_for_vis = nn_crop

    _plot_distributions(train_df, test_df, failed_df, out_dir)
    _save_nearest_visuals(cfg, failed_df, nn_for_vis, out_dir, max_cases=args.max_visual_cases)

    def pct(series: pd.Series) -> float:
        return float(100.0 * series.mean()) if len(series) else 0.0

    summary = {
        "train_slices": int(len(train_df)),
        "test_slices": int(len(test_df)),
        "train_positive_slices": int((train_df["gt_area"] > 0).sum()),
        "test_positive_slices": int((test_df["gt_area"] > 0).sum()),
        "failed_cases": int(len(failed_df)),
        "failed_tp_zero_cases": int(failed_df.get("is_tp_zero", pd.Series(dtype=bool)).sum()),
        "failed_complete_miss_with_fp_cases": int(failed_df.get("is_complete_miss_with_fp", pd.Series(dtype=bool)).sum()),
        "failed_tiny_lesion_pct": round(pct(failed_df["flag_tiny_lesion_vs_train"]), 2),
        "failed_large_component_count_pct": round(pct(failed_df["flag_large_component_count_vs_train"]), 2),
        "failed_low_lesion_intensity_pct": round(pct(failed_df["flag_low_lesion_intensity_vs_train"]), 2),
        "failed_high_lesion_intensity_pct": round(pct(failed_df["flag_high_lesion_intensity_vs_train"]), 2),
        "train_gt_area_median": float(train_pos["gt_area"].median()) if len(train_pos) else 0.0,
        "failed_gt_area_median": float(failed_df["gt_area"].median()) if len(failed_df) else 0.0,
        "train_component_count_median": float(train_pos["gt_component_count"].median()) if len(train_pos) else 0.0,
        "failed_component_count_median": float(failed_df["gt_component_count"].median()) if len(failed_df) else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hints = []
    if summary["failed_tiny_lesion_pct"] >= 40:
        hints.append("Many failed cases have lesion area below the 10th percentile of train. This supports a small-lesion data imbalance hypothesis.")
    if summary["failed_large_component_count_pct"] >= 30:
        hints.append("Many failed cases have unusually many lesion components compared with train. This supports a multi-lesion distribution gap hypothesis.")
    if summary["failed_low_lesion_intensity_pct"] >= 30 or summary["failed_high_lesion_intensity_pct"] >= 30:
        hints.append("Many failed lesions have intensity outside the common train range. This supports an intensity/domain-shift hypothesis.")
    if summary["failed_complete_miss_with_fp_cases"] > 0:
        hints.append("There are complete-miss-with-false-positive cases. Inspect prediction_probe outputs to decide whether the model gives low probability inside GT or predicts the wrong high-confidence region.")
    _build_report(out_dir, summary, hints)

    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostic outputs to: {out_dir}")


if __name__ == "__main__":
    main()
