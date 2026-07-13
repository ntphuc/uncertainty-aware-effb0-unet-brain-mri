#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import NPYSliceDataset
from models import build_model
from utils.boundary import mask_to_boundary
from utils.misc import count_parameters, estimate_flops, load_config, save_json

_METRIC_FNS = None


def get_surface_metric_fns():
    global _METRIC_FNS
    if _METRIC_FNS is None:
        try:
            from utils.metrics import assd as _assd
            from utils.metrics import boundary_f1 as _boundary_f1
            from utils.metrics import hd95 as _hd95
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scipy is required for hd95/assd/boundary metrics. "
                "Install it with: pip install scipy"
            ) from exc
        _METRIC_FNS = (_hd95, _assd, _boundary_f1)
    return _METRIC_FNS


def dice_iou_from_masks(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = float(np.logical_and(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = pred_sum + gt_sum - intersection
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


def clamp_inf(values: np.ndarray, scale: float = 1.5) -> np.ndarray:
    arr = values.copy().astype(np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        arr[:] = 1e6
        return arr
    max_finite = arr[finite].max()
    arr[~finite] = max_finite * scale
    return arr


def robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    lo = np.percentile(arr, 5)
    hi = np.percentile(arr, 95)
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def infer_failure_type(row: Dict[str, float]) -> str:
    gt_area = float(row["gt_area"])
    pred_area = float(row["pred_area"])
    fp = float(row["fp_pixels"])
    fn = float(row["fn_pixels"])
    bf1 = float(row["boundary_f1"])

    if gt_area == 0 and pred_area > 0:
        return "false_positive_on_negative_slice"
    if gt_area > 0 and pred_area == 0:
        return "missed_lesion_false_negative"

    fn_ratio = fn / max(gt_area, 1.0)
    fp_ratio = fp / max(pred_area, 1.0)

    if fn_ratio >= 0.7:
        return "under_segmentation"
    if fp_ratio >= 0.7:
        return "over_segmentation"
    if bf1 < 0.7:
        return "boundary_mismatch"
    return "mixed_error"


def infer_recommendation(error_type: str) -> str:
    mapping = {
        "false_positive_on_negative_slice": "Tang threshold prediction hoac bo sung hard negative slices trong train.",
        "missed_lesion_false_negative": "Giam threshold, tang trong so Dice/Boundary loss, va bo sung ca lesion nho.",
        "under_segmentation": "Tang recall bang threshold thap hon hoac augment zoom/crop de giu lesion.",
        "over_segmentation": "Tang precision bang threshold cao hon va regularization boundary.",
        "boundary_mismatch": "Tang lambda boundary, tinh chinh tolerance boundary metric va train them edge-focused augment.",
        "mixed_error": "Xem lai can bang du lieu, loss weights va calibration threshold theo validation.",
    }
    return mapping.get(error_type, mapping["mixed_error"])


def write_csv(rows: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_case_figure(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    row: Dict[str, object],
    save_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to save failed-case visualizations. "
            "Install it with: pip install matplotlib"
        ) from exc

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)

    image_norm = image.astype(np.float32)
    image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min() + 1e-6)

    gt_t = torch.from_numpy(gt.astype(np.float32))[None, None]
    pred_t = torch.from_numpy(pred.astype(np.float32))[None, None]
    gt_boundary = mask_to_boundary(gt_t)[0, 0].numpy() > 0.5
    pred_boundary = mask_to_boundary(pred_t)[0, 0].numpy() > 0.5

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(image_norm, cmap="gray")
    axes[0, 0].set_title("MRI")

    axes[0, 1].imshow(image_norm, cmap="gray")
    axes[0, 1].imshow(np.ma.masked_where(~gt, gt), cmap="Greens", alpha=0.55)
    axes[0, 1].set_title("GT Overlay")

    axes[0, 2].imshow(image_norm, cmap="gray")
    axes[0, 2].imshow(np.ma.masked_where(~pred, pred), cmap="Reds", alpha=0.55)
    axes[0, 2].set_title("Prediction Overlay")

    error_rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    error_rgb[..., 0] = fp.astype(np.float32)
    error_rgb[..., 2] = fn.astype(np.float32)
    axes[1, 0].imshow(image_norm, cmap="gray")
    axes[1, 0].imshow(error_rgb, alpha=0.7)
    axes[1, 0].set_title("Error Map (FP=Red, FN=Blue)")

    boundary_rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    boundary_rgb[..., 1] = gt_boundary.astype(np.float32)
    boundary_rgb[..., 0] = pred_boundary.astype(np.float32)
    axes[1, 1].imshow(image_norm, cmap="gray")
    axes[1, 1].imshow(boundary_rgb, alpha=0.75)
    axes[1, 1].set_title("Boundary (Pred=Red, GT=Green)")

    im = axes[1, 2].imshow(prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Probability Map")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    title = (
        f"Case #{row['case_idx']} | failure={row['failure_score']:.4f} | "
        f"dice={row['dice']:.4f} iou={row['iou']:.4f} hd95={row['hd95']:.3f} assd={row['assd']:.3f} "
        f"bf1={row['boundary_f1']:.4f}"
    )
    fig.suptitle(title, fontsize=11)

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def compute_case_rows(
    model,
    loader,
    device: torch.device,
    threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    case_idx = 0
    hd95_fn, assd_fn, bf1_fn = get_surface_metric_fns()

    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="analyze-cases"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)

            probs = torch.sigmoid(outputs["seg"]).detach().cpu().numpy()
            preds = probs > threshold
            gts = (masks.detach().cpu().numpy() > 0.5)

            for i in range(images.size(0)):
                pred = preds[i, 0]
                gt = gts[i, 0]
                prob = probs[i, 0]

                dice, iou = dice_iou_from_masks(pred, gt)
                hd = hd95_fn(pred, gt)
                ad = assd_fn(pred, gt)
                bf1 = bf1_fn(pred, gt)

                fp = int(np.logical_and(pred, ~gt).sum())
                fn = int(np.logical_and(~pred, gt).sum())
                tp = int(np.logical_and(pred, gt).sum())
                tn = int(np.logical_and(~pred, ~gt).sum())

                row: Dict[str, object] = {
                    "case_idx": case_idx,
                    "dice": float(dice),
                    "iou": float(iou),
                    "hd95": float(hd),
                    "assd": float(ad),
                    "boundary_f1": float(bf1),
                    "gt_area": int(gt.sum()),
                    "pred_area": int(pred.sum()),
                    "fp_pixels": fp,
                    "fn_pixels": fn,
                    "tp_pixels": tp,
                    "tn_pixels": tn,
                    "mean_prob": float(prob.mean()),
                    "max_prob": float(prob.max()),
                }
                rows.append(row)
                case_idx += 1

    return rows


def add_failure_score(rows: List[Dict[str, object]]) -> None:
    dice_arr = np.array([float(r["dice"]) for r in rows], dtype=np.float64)
    iou_arr = np.array([float(r["iou"]) for r in rows], dtype=np.float64)
    bf1_arr = np.array([float(r["boundary_f1"]) for r in rows], dtype=np.float64)
    hd_arr = clamp_inf(np.array([float(r["hd95"]) for r in rows], dtype=np.float64))
    assd_arr = clamp_inf(np.array([float(r["assd"]) for r in rows], dtype=np.float64))

    hd_norm = robust_normalize(hd_arr)
    assd_norm = robust_normalize(assd_arr)

    score = (1.0 - dice_arr) + (1.0 - iou_arr) + (1.0 - bf1_arr) + hd_norm + assd_norm
    for i, r in enumerate(rows):
        r["failure_score"] = float(score[i])
        r["error_type"] = infer_failure_type(r)  # type: ignore[arg-type]
        r["recommendation"] = infer_recommendation(str(r["error_type"]))


def select_failed_cases(
    rows: List[Dict[str, object]],
    fail_metric: str,
    top_k: int,
    dice_below: float,
) -> List[Dict[str, object]]:
    filtered = rows
    if dice_below >= 0.0:
        filtered = [r for r in filtered if float(r["dice"]) < dice_below]

    if fail_metric in {"dice", "iou", "boundary_f1"}:
        filtered.sort(key=lambda r: float(r[fail_metric]))
    else:
        filtered.sort(key=lambda r: float(r[fail_metric]), reverse=True)

    return filtered[:top_k]


def export_report(
    failed_rows: List[Dict[str, object]],
    out_path: Path,
    meta: Dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    for r in failed_rows:
        key = str(r["error_type"])
        counts[key] = counts.get(key, 0) + 1

    lines = []
    lines.append("# Failed Cases Diagnostic Report")
    lines.append("")
    lines.append("## Model")
    lines.append(f"- config: {meta['config']}")
    lines.append(f"- checkpoint: {meta['checkpoint']}")
    lines.append(f"- split: {meta['split']}")
    lines.append(f"- threshold: {meta['threshold']}")
    lines.append(f"- params: {meta['params']}")
    lines.append(f"- gflops: {meta['gflops']}")
    lines.append("")
    lines.append("## Failed Case Distribution")
    for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Suggested Fix Directions")
    seen = set()
    for r in failed_rows:
        et = str(r["error_type"])
        rec = str(r["recommendation"])
        if et in seen:
            continue
        seen.add(et)
        lines.append(f"- {et}: {rec}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Xem file failed_cases.csv de loc va sap xep theo metric.")
    lines.append("- Xem thu muc failed_images/ de phan tich tung case theo overlay loi.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-case failures for one segmentation model.")
    parser.add_argument("--config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fail_metric", choices=["failure_score", "dice", "iou", "hd95", "assd", "boundary_f1"], default="failure_score")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--dice_below", type=float, default=-1.0, help="Only keep cases with dice below this value; <0 disables.")
    parser.add_argument("--output_dir", default=None, help="Default: <output_dir>/eval_failed_cases")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    tcfg = cfg["training"]
    mcfg = cfg["model"]
    paths = cfg["paths"]

    if args.split == "test":
        x_path = paths["x_test"]
        y_path = paths["y_test"]
    else:
        x_path = paths["x_train"]
        y_path = paths["y_train"]

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    threshold = float(tcfg.get("threshold", 0.5))
    rows = compute_case_rows(model, loader, device, threshold)
    if not rows:
        raise RuntimeError("No cases found in dataset")

    add_failure_score(rows)

    params = count_parameters(model)
    flops, _ = estimate_flops(
        model,
        input_shape=(1, mcfg.get("in_channels", 1), tcfg.get("image_size", 256), tcfg.get("image_size", 256)),
        device=device,
    )
    gflops = None if flops is None else flops / 1e9

    failed_rows = select_failed_cases(rows, args.fail_metric, args.top_k, args.dice_below)

    base_out = Path(args.output_dir) if args.output_dir else Path(paths.get("output_dir", "outputs")) / "eval_failed_cases"
    base_out.mkdir(parents=True, exist_ok=True)

    all_csv = base_out / f"{args.split}_all_cases.csv"
    failed_csv = base_out / f"{args.split}_failed_cases.csv"
    write_csv(rows, all_csv)
    write_csv(failed_rows, failed_csv)

    image_dir = base_out / f"{args.split}_failed_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for row in tqdm(failed_rows, desc="save-failed-images"):
            idx = int(row["case_idx"])
            image_t, mask_t = dataset[idx]
            image_t = image_t.unsqueeze(0).to(device)
            out = model(image_t)
            prob = torch.sigmoid(out["seg"])[0, 0].detach().cpu().numpy()
            pred = prob > threshold
            gt = (mask_t[0].detach().cpu().numpy() > 0.5)
            image = image_t[0, 0].detach().cpu().numpy()

            fig_path = image_dir / f"case_{idx:04d}.png"
            save_case_figure(image, gt, pred, prob, row, fig_path)

    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "threshold": threshold,
        "params": params,
        "gflops": gflops,
        "top_k": args.top_k,
        "fail_metric": args.fail_metric,
        "dice_below": args.dice_below,
        "num_cases": len(rows),
        "num_failed_saved": len(failed_rows),
    }

    save_json(meta, base_out / f"{args.split}_analysis_meta.json")
    export_report(failed_rows, base_out / f"{args.split}_failed_report.md", meta)

    print(f"Saved all-cases CSV: {all_csv}")
    print(f"Saved failed-cases CSV: {failed_csv}")
    print(f"Saved failed images: {image_dir}")
    print(f"Saved report: {base_out / f'{args.split}_failed_report.md'}")


if __name__ == "__main__":
    main()
