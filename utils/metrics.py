from typing import Dict, List

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_dilation


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    cardinality = torch.sum(preds + targets, dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return dice.mean().item()


def iou_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    union = torch.sum(preds + targets, dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def precision_recall_fbeta_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    beta: float = 2.0,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Pixel-level precision/recall/F-beta for binary segmentation."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)

    tp = torch.sum(preds * targets, dims)
    fp = torch.sum(preds * (1.0 - targets), dims)
    fn = torch.sum((1.0 - preds) * targets, dims)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f_beta": fbeta.mean().item(),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask ^ eroded


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return np.array([0.0])
    if pred.sum() == 0 or gt.sum() == 0:
        return np.array([np.inf])

    pred_surf = _surface(pred)
    gt_surf = _surface(gt)

    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    d = surface_distances(pred, gt)
    if np.isinf(d).any():
        return float("inf")
    return float(np.percentile(d, 95))


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    d = surface_distances(pred, gt)
    if np.isinf(d).any():
        return float("inf")
    return float(np.mean(d))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pred_s = _surface(pred)
    gt_s = _surface(gt)
    if pred_s.sum() == 0 and gt_s.sum() == 0:
        return 1.0
    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return 0.0
    struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    gt_dil = binary_dilation(gt_s, structure=struct)
    pred_dil = binary_dilation(pred_s, structure=struct)
    precision = (pred_s & gt_dil).sum() / (pred_s.sum() + 1e-6)
    recall = (gt_s & pred_dil).sum() / (gt_s.sum() + 1e-6)
    return float(2 * precision * recall / (precision + recall + 1e-6))


def batch_surface_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    tgts = targets.detach().cpu().numpy()
    preds = probs > threshold
    tgts = tgts > 0.5

    hd95_values, assd_values, bf1_values = [], [], []
    for p, g in zip(preds, tgts):
        p2 = p[0]
        g2 = g[0]
        hd95_values.append(hd95(p2, g2))
        assd_values.append(assd(p2, g2))
        bf1_values.append(boundary_f1(p2, g2))

    def finite_mean(values: List[float]) -> float:
        arr = np.array(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) > 0 else float("inf")

    return {
        "hd95": finite_mean(hd95_values),
        "assd": finite_mean(assd_values),
        "boundary_f1": finite_mean(bf1_values),
    }


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)


def dice_score_from_binary(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    cardinality = torch.sum(preds + targets, dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return dice.mean().item()


def iou_score_from_binary(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    union = torch.sum(preds + targets, dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def precision_recall_fbeta_from_binary(
    preds: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 2.0,
    eps: float = 1e-6,
) -> Dict[str, float]:
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    tp = torch.sum(preds * targets, dims)
    fp = torch.sum(preds * (1.0 - targets), dims)
    fn = torch.sum((1.0 - preds) * targets, dims)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    beta2 = beta ** 2
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    return {"precision": precision.mean().item(), "recall": recall.mean().item(), "f_beta": fbeta.mean().item()}


def batch_surface_metrics_from_binary(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    p_np = (preds.detach().cpu().numpy() > 0.5)
    t_np = (targets.detach().cpu().numpy() > 0.5)
    hd95_values, assd_values, bf1_values = [], [], []
    for p, g in zip(p_np, t_np):
        p2 = p[0]
        g2 = g[0]
        hd95_values.append(hd95(p2, g2))
        assd_values.append(assd(p2, g2))
        bf1_values.append(boundary_f1(p2, g2))

    def finite_mean(values: List[float]) -> float:
        arr = np.array(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) > 0 else float("inf")

    return {"hd95": finite_mean(hd95_values), "assd": finite_mean(assd_values), "boundary_f1": finite_mean(bf1_values)}


def segmentation_metrics_per_sample_from_binary(
    preds: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 2.0,
    eps: float = 1e-6,
) -> Dict[str, List[float]]:
    """Return one metric value per image instead of only a batch average.

    This is used to report mean ± sample standard deviation across test cases.
    HD95/ASSD keep ``inf`` for complete foreground mismatches; the evaluation
    script reports how many non-finite cases were excluded from finite summary
    statistics instead of silently treating them as ordinary distances.
    """
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)

    tp = torch.sum(preds * targets, dims)
    fp = torch.sum(preds * (1.0 - targets), dims)
    fn = torch.sum((1.0 - preds) * targets, dims)
    intersection = tp
    cardinality = torch.sum(preds + targets, dims)
    union = cardinality - intersection

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    beta2 = beta ** 2
    fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + eps)

    p_np = preds.detach().cpu().numpy() > 0.5
    t_np = targets.detach().cpu().numpy() > 0.5
    hd95_values: List[float] = []
    assd_values: List[float] = []
    boundary_f1_values: List[float] = []
    for pred, target in zip(p_np, t_np):
        pred_2d = pred[0]
        target_2d = target[0]
        hd95_values.append(hd95(pred_2d, target_2d))
        assd_values.append(assd(pred_2d, target_2d))
        boundary_f1_values.append(boundary_f1(pred_2d, target_2d))

    return {
        "dice": dice.detach().cpu().tolist(),
        "iou": iou.detach().cpu().tolist(),
        "precision": precision.detach().cpu().tolist(),
        "recall": recall.detach().cpu().tolist(),
        "f2": fbeta.detach().cpu().tolist(),
        "hd95": hd95_values,
        "assd": assd_values,
        "boundary_f1": boundary_f1_values,
    }


def segmentation_metrics_per_sample_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    beta: float = 2.0,
    eps: float = 1e-6,
) -> Dict[str, List[float]]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    return segmentation_metrics_per_sample_from_binary(
        preds=preds,
        targets=targets,
        beta=beta,
        eps=eps,
    )


def summarize_metric_values(values: List[float], ddof: int = 1) -> Dict[str, float | int | str]:
    """Summarize metric values with mean, sample std, SEM and 95% CI half-width."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n_total = int(arr.size)
    n_finite = int(finite.size)
    n_nonfinite = n_total - n_finite
    if n_finite == 0:
        return {
            "mean": float("inf"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_half_width": float("nan"),
            "n": n_total,
            "n_finite": 0,
            "n_nonfinite": n_nonfinite,
            "mean_pm_std": "NA",
        }

    mean = float(finite.mean())
    effective_ddof = int(ddof) if n_finite > int(ddof) else 0
    std = float(finite.std(ddof=effective_ddof))
    sem = float(std / np.sqrt(n_finite)) if n_finite > 0 else float("nan")
    ci95 = float(1.96 * sem)
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": ci95,
        "n": n_total,
        "n_finite": n_finite,
        "n_nonfinite": n_nonfinite,
        "mean_pm_std": f"{mean:.6f} ± {std:.6f}",
    }
