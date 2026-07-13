"""Uncertainty utilities for binary medical image segmentation.

This module is intentionally dependency-light and works with the existing project
model API where ``model(images)`` returns a dict with key ``seg``.

It supports three uncertainty signals:
  1) probability entropy: pixel-level ambiguity around p=0.5;
  2) TTA variance/disagreement: instability under safe test-time transforms;
  3) train-coverage distance: how similar a predicted component/image is to the
     lesion patterns seen in the training set.

The first two do not require labels at inference. The train-coverage reference is
built from the training masks once, then used at test time without using test labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.distance import cdist


EPS = 1e-7


def binary_entropy_from_prob(prob: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Binary predictive entropy map in [0, ~0.693].

    Args:
        prob: Tensor [B,1,H,W] with values in [0,1].
    """
    p = prob.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def normalized_binary_entropy_from_prob(prob: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Binary entropy normalized to [0,1]."""
    return binary_entropy_from_prob(prob, eps=eps) / np.log(2.0)


def _model_seg_logits(model, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    if isinstance(out, dict):
        return out["seg"]
    return out


@torch.no_grad()
def predict_prob(model, images: torch.Tensor) -> torch.Tensor:
    """Single deterministic probability map."""
    return torch.sigmoid(_model_seg_logits(model, images))


def _maybe_resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def tta_probability_stack(
    model,
    images: torch.Tensor,
    scales: Sequence[float] = (1.0, 1.25, 1.5),
    hflip: bool = True,
    vflip: bool = False,
) -> torch.Tensor:
    """Return probability stack for TTA predictions.

    Args:
        images: [B,C,H,W]
        scales: input scale factors.
        hflip/vflip: safe flips. For brain MRI, horizontal flip may be acceptable
            depending on task; vertical flip is disabled by default.

    Returns:
        probs: [T,B,1,H,W], aligned back to original image orientation/size.
    """
    model.eval()
    orig_size = images.shape[-2:]
    probs: List[torch.Tensor] = []
    transforms: List[Tuple[float, bool, bool]] = []
    for s in scales:
        transforms.append((float(s), False, False))
        if hflip:
            transforms.append((float(s), True, False))
        if vflip:
            transforms.append((float(s), False, True))
        if hflip and vflip:
            transforms.append((float(s), True, True))

    seen = set()
    unique_transforms = []
    for t in transforms:
        if t not in seen:
            unique_transforms.append(t)
            seen.add(t)

    for scale, do_h, do_v in unique_transforms:
        x = images
        if do_h:
            x = torch.flip(x, dims=[-1])
        if do_v:
            x = torch.flip(x, dims=[-2])
        if abs(scale - 1.0) > 1e-6:
            new_h = max(16, int(round(orig_size[0] * scale)))
            new_w = max(16, int(round(orig_size[1] * scale)))
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        p = torch.sigmoid(_model_seg_logits(model, x))
        p = _maybe_resize(p, orig_size)
        if do_v:
            p = torch.flip(p, dims=[-2])
        if do_h:
            p = torch.flip(p, dims=[-1])
        probs.append(p)

    return torch.stack(probs, dim=0)


def enable_mc_dropout(model) -> None:
    """Enable Dropout modules while keeping the rest of the model in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)):
            module.train()


@torch.no_grad()
def mc_dropout_probability_stack(
    model,
    images: torch.Tensor,
    runs: int = 8,
) -> torch.Tensor:
    """Return probability stack from MC Dropout inference.

    This works only if the model contains Dropout modules. If it has no Dropout,
    repeated predictions may be identical, which is still safe but not informative.
    """
    runs = int(max(1, runs))
    enable_mc_dropout(model)
    probs = [torch.sigmoid(_model_seg_logits(model, images)) for _ in range(runs)]
    model.eval()
    return torch.stack(probs, dim=0)


def summarize_uncertainty_maps(
    mean_prob: torch.Tensor,
    var_map: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Build common uncertainty maps from mean probability and optional variance."""
    entropy = normalized_binary_entropy_from_prob(mean_prob)
    pred = (mean_prob > float(threshold)).float()

    # Boundary ambiguity: uncertain pixels around the decision boundary. This is
    # high when p is close to 0.5, low near 0/1.
    margin_unc = 1.0 - torch.abs(mean_prob - 0.5) * 2.0
    margin_unc = margin_unc.clamp(0.0, 1.0)

    result = {
        "mean_prob": mean_prob,
        "entropy": entropy,
        "margin_uncertainty": margin_unc,
        "pred_mask": pred,
    }
    if var_map is not None:
        result["variance"] = var_map
        # Composite region map for visualization. Entropy captures p~0.5; variance
        # captures instability under TTA/MC. Normalize variance per batch to avoid scale issues.
        v = var_map
        vmax = v.flatten(1).amax(dim=1).view(-1, 1, 1, 1).clamp_min(EPS)
        result["combined_uncertainty"] = (0.5 * entropy + 0.5 * (v / vmax).clamp(0, 1)).clamp(0, 1)
    else:
        result["combined_uncertainty"] = entropy
    return result


def binary_confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {"tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "tn_pixels": tn}


def dice_iou_from_binary_np(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    c = binary_confusion_counts(pred, gt)
    tp, fp, fn = c["tp_pixels"], c["fp_pixels"], c["fn_pixels"]
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f2 = (5 * precision * recall) / (4 * precision + recall + eps)
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall), "f2": float(f2), **c}


def connected_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    mask = mask.astype(bool)
    conn = 2 if int(connectivity) == 8 else 1
    structure = ndimage.generate_binary_structure(2, conn)
    return ndimage.label(mask, structure=structure)


def component_table(mask: np.ndarray, image: Optional[np.ndarray] = None, min_area: int = 1) -> List[Dict[str, float]]:
    labeled, num = connected_components(mask)
    comps: List[Dict[str, float]] = []
    h, w = mask.shape[-2:]
    for cid in range(1, num + 1):
        comp = labeled == cid
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        row = {
            "component_id": int(cid),
            "area": area,
            "area_ratio": float(area / max(1, h * w)),
            "centroid_y": float(ys.mean() / max(1, h - 1)),
            "centroid_x": float(xs.mean() / max(1, w - 1)),
            "bbox_h": float((ys.max() - ys.min() + 1) / max(1, h)),
            "bbox_w": float((xs.max() - xs.min() + 1) / max(1, w)),
            "bbox_fill": float(area / max(1, (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))),
        }
        if image is not None:
            vals = image[comp]
            row.update({
                "intensity_mean": float(vals.mean()) if vals.size else 0.0,
                "intensity_std": float(vals.std()) if vals.size else 0.0,
                "intensity_p75": float(np.percentile(vals, 75)) if vals.size else 0.0,
                "intensity_p95": float(np.percentile(vals, 95)) if vals.size else 0.0,
            })
        comps.append(row)
    return comps


def component_matching_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """Component-level matching using any overlap as a match."""
    pred_lab, pred_n = connected_components(pred)
    gt_lab, gt_n = connected_components(gt)
    matched_gt = set()
    matched_pred = set()
    for pi in range(1, pred_n + 1):
        overlap_gt_ids = np.unique(gt_lab[pred_lab == pi])
        overlap_gt_ids = [int(x) for x in overlap_gt_ids if int(x) != 0]
        if overlap_gt_ids:
            matched_pred.add(pi)
            matched_gt.update(overlap_gt_ids)
    return {
        "gt_component_count": int(gt_n),
        "pred_component_count": int(pred_n),
        "matched_gt_components": int(len(matched_gt)),
        "matched_pred_components": int(len(matched_pred)),
        "missed_component_count": int(max(0, gt_n - len(matched_gt))),
        "fp_component_count": int(max(0, pred_n - len(matched_pred))),
    }


def image_global_feature(image: np.ndarray) -> np.ndarray:
    """Small feature vector from raw MRI image, usable without a mask."""
    x = image.astype(np.float32)
    flat = x.reshape(-1)
    p = np.percentile(flat, [1, 5, 10, 25, 50, 75, 90, 95, 99]).astype(np.float32)
    hist, _ = np.histogram(flat, bins=16, range=(float(flat.min()), float(flat.max()) + 1e-6), density=True)
    hist = hist.astype(np.float32)
    stats = np.asarray([flat.mean(), flat.std(), flat.min(), flat.max()], dtype=np.float32)
    return np.concatenate([stats, p, hist], axis=0)


COMPONENT_FEATURE_KEYS = [
    "area_ratio", "centroid_y", "centroid_x", "bbox_h", "bbox_w", "bbox_fill",
    "intensity_mean", "intensity_std", "intensity_p75", "intensity_p95",
]


def component_feature(row: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(row.get(k, 0.0)) for k in COMPONENT_FEATURE_KEYS], dtype=np.float32)


@dataclass
class FeatureReference:
    features: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_features(cls, features: np.ndarray) -> "FeatureReference":
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-6
        return cls(features=features, mean=mean, std=std)

    def nearest_distance(self, feature: np.ndarray) -> float:
        if self.features.size == 0:
            return float("nan")
        z_train = (self.features - self.mean) / self.std
        z = ((np.asarray(feature, dtype=np.float32) - self.mean) / self.std)[None, :]
        d = cdist(z, z_train, metric="euclidean")
        return float(d.min())


def build_train_references_from_arrays(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = 2000,
    min_component_area: int = 5,
    seed: int = 42,
) -> Dict[str, FeatureReference]:
    """Build lightweight train distribution references.

    - image reference: raw image feature, no mask needed at inference.
    - lesion component reference: GT lesion components in train; test uses predicted
      components to see whether the predicted region resembles training lesions.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idxs = np.arange(n)
    if n > max_samples:
        idxs = rng.choice(idxs, size=max_samples, replace=False)

    image_feats: List[np.ndarray] = []
    comp_feats: List[np.ndarray] = []
    for idx in idxs:
        img = np.asarray(X[idx]).astype(np.float32)
        m = np.asarray(Y[idx]).astype(np.float32)
        if img.ndim == 3:
            img2 = img[..., 0] if img.shape[-1] <= 4 else img[0]
        else:
            img2 = img
        if m.ndim == 3:
            mask2 = m[..., 0] if m.shape[-1] <= 4 else m[0]
        else:
            mask2 = m
        # z-score like dataset normalization for consistency.
        img2 = (img2 - img2.mean()) / (img2.std() + 1e-6)
        image_feats.append(image_global_feature(img2))
        for comp in component_table(mask2 > 0.5, img2, min_area=min_component_area):
            comp_feats.append(component_feature(comp))

    if not comp_feats:
        comp_feats = [np.zeros(len(COMPONENT_FEATURE_KEYS), dtype=np.float32)]
    return {
        "image": FeatureReference.from_features(np.stack(image_feats, axis=0)),
        "component": FeatureReference.from_features(np.stack(comp_feats, axis=0)),
    }
