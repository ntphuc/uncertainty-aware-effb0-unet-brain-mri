"""Post-processing utilities for binary segmentation probability maps.

The main goal is to remove false-positive connected components that look like
low-intensity / gray background structures rather than bright enhancing lesions.
This does not use ground-truth masks at inference time.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy import ndimage


def _as_numpy_2d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[0]
    return x.astype(np.float32)


def postprocess_single_prediction(
    prob: np.ndarray,
    image: np.ndarray,
    threshold: float = 0.5,
    enabled: bool = True,
    remove_small_components: bool = True,
    min_component_area: int = 20,
    intensity_filter: bool = False,
    intensity_percentile: float = 70.0,
    foreground_percentile: float = 5.0,
    min_component_mean_z: float | None = None,
    min_component_p75_z: float | None = None,
    connectivity: int = 8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return post-processed binary mask and simple diagnostic counts.

    Args:
        prob: [H, W] probability map.
        image: [H, W] normalized MRI image, usually z-score normalized.
        threshold: segmentation threshold.
        intensity_filter: remove a predicted component if its intensity is too low.
            The low-intensity criterion is relative to each image and/or absolute z-score.
    """
    prob = _as_numpy_2d(prob)
    image = _as_numpy_2d(image)
    pred = prob > float(threshold)

    stats = {
        "components_before": 0,
        "components_after": 0,
        "removed_small": 0,
        "removed_intensity": 0,
    }
    if not enabled:
        stats["components_before"] = int(ndimage.label(pred)[1])
        stats["components_after"] = stats["components_before"]
        return pred.astype(bool), stats

    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    labeled, num = ndimage.label(pred, structure=structure)
    stats["components_before"] = int(num)
    if num == 0:
        return pred.astype(bool), stats

    # Reference intensity: use non-empty foreground-like pixels to avoid black borders dominating percentiles.
    fg_cut = np.percentile(image, foreground_percentile)
    fg_pixels = image[image > fg_cut]
    if fg_pixels.size == 0:
        fg_pixels = image.reshape(-1)
    ref_intensity = float(np.percentile(fg_pixels, intensity_percentile))

    keep = np.zeros_like(pred, dtype=bool)
    for label_id in range(1, num + 1):
        comp = labeled == label_id
        area = int(comp.sum())
        if remove_small_components and area < int(min_component_area):
            stats["removed_small"] += 1
            continue

        if intensity_filter:
            values = image[comp]
            comp_mean = float(values.mean())
            comp_p75 = float(np.percentile(values, 75))

            # A component is removed only when it is consistently too gray/low-intensity.
            # The default condition uses both relative and optional absolute z-score criteria.
            too_low_relative = comp_mean < ref_intensity
            # Conservative removal: if an absolute z-score threshold is provided,
            # require the component to be low relative to the image AND low in absolute terms.
            # This avoids deleting true lesions that are slightly below the image percentile
            # but still clearly bright after z-score normalization.
            too_low_mean = True if min_component_mean_z is None else comp_mean < float(min_component_mean_z)
            too_low_p75 = True if min_component_p75_z is None else comp_p75 < float(min_component_p75_z)

            if too_low_relative and too_low_mean and too_low_p75:
                stats["removed_intensity"] += 1
                continue

        keep |= comp

    stats["components_after"] = int(ndimage.label(keep, structure=structure)[1])
    return keep.astype(bool), stats


def postprocess_batch_predictions(
    images: torch.Tensor,
    probs: torch.Tensor,
    cfg: Dict,
    threshold: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Post-process a batch of probability maps.

    Args:
        images: [B, C, H, W], normalized image tensor.
        probs: [B, 1, H, W], probability tensor.
        cfg: cfg.get("postprocess", {}).
        threshold: segmentation threshold.

    Returns:
        pred_tensor: [B, 1, H, W] float binary tensor on same device as probs.
    """
    pcfg = cfg or {}
    enabled = bool(pcfg.get("enabled", False))
    preds = []
    total_stats = {"components_before": 0, "components_after": 0, "removed_small": 0, "removed_intensity": 0}

    for i in range(probs.shape[0]):
        pred_np, st = postprocess_single_prediction(
            prob=probs[i, 0],
            image=images[i, 0],
            threshold=threshold,
            enabled=enabled,
            remove_small_components=pcfg.get("remove_small_components", True),
            min_component_area=pcfg.get("min_component_area", 20),
            intensity_filter=pcfg.get("intensity_filter", False),
            intensity_percentile=pcfg.get("intensity_percentile", 70.0),
            foreground_percentile=pcfg.get("foreground_percentile", 5.0),
            min_component_mean_z=pcfg.get("min_component_mean_z", None),
            min_component_p75_z=pcfg.get("min_component_p75_z", None),
            connectivity=pcfg.get("connectivity", 8),
        )
        for k in total_stats:
            total_stats[k] += st[k]
        preds.append(torch.from_numpy(pred_np.astype(np.float32))[None, ...])

    pred_tensor = torch.stack(preds, dim=0).to(device=probs.device, dtype=torch.float32)
    return pred_tensor, total_stats


def binary_mask_to_logits(mask: torch.Tensor, high: float = 20.0) -> torch.Tensor:
    """Convert a binary mask to pseudo-logits for existing visualization code."""
    return torch.where(mask > 0.5, torch.full_like(mask, high), torch.full_like(mask, -high))
