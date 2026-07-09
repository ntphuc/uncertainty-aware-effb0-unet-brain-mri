"""Component-weighted segmentation loss.

Why this exists:
    Component-aware crop lets the model *see* small separated lesions more often.
    But the normal pixel/region loss can still be dominated by the largest lesion.

This loss creates a foreground weight map from the ground-truth mask:
    - background weight = 1.0
    - largest lesion component weight ~= 1.0
    - smaller lesion components receive higher weights up to `max_weight`

It is designed for multi-lesion slices where the model detects the main lesion but
misses one or two small separated lesions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None


class ComponentWeightedSegLoss(nn.Module):
    """Weighted BCE + weighted soft Dice using connected components from GT masks.

    Args:
        max_weight: Maximum pixel weight assigned to the smallest lesion component.
        power: Controls how aggressively small lesions are up-weighted.
            Larger values push small components closer to max_weight.
        min_area: Ignore components smaller than this many pixels when constructing weights.
        bce_weight: Weight of the BCE term.
        dice_weight: Weight of the weighted Dice term.
        eps: Numerical stability.

    Notes:
        The weight map is computed from the target only, so it does not leak test
        information during training. It is used only inside the training loss.
    """

    def __init__(
        self,
        max_weight: float = 4.0,
        power: float = 0.5,
        min_area: int = 5,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.max_weight = float(max_weight)
        self.power = float(power)
        self.min_area = int(min_area)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    @torch.no_grad()
    def make_weight_map(self, targets: torch.Tensor) -> torch.Tensor:
        """Return a [B,1,H,W] component-aware weight map on the same device."""
        if self.max_weight <= 1.0 or ndimage is None:
            return torch.ones_like(targets)

        device = targets.device
        dtype = targets.dtype
        masks_np = (targets.detach().float().cpu().numpy() > 0.5)
        weights_np = np.ones_like(masks_np, dtype=np.float32)

        structure = np.ones((3, 3), dtype=np.uint8)
        for b in range(masks_np.shape[0]):
            mask2d = masks_np[b, 0]
            if mask2d.sum() == 0:
                continue

            labeled, num = ndimage.label(mask2d, structure=structure)
            if num == 0:
                continue

            areas = []
            for comp_id in range(1, num + 1):
                area = int((labeled == comp_id).sum())
                if area >= self.min_area:
                    areas.append((comp_id, area))

            if not areas:
                continue

            max_area = max(a for _, a in areas)
            for comp_id, area in areas:
                # Largest component gets weight near 1.0; smaller components get higher weights.
                ratio = float(area) / float(max_area + self.eps)
                smallness = 1.0 - (ratio ** self.power)
                comp_weight = 1.0 + (self.max_weight - 1.0) * smallness
                weights_np[b, 0][labeled == comp_id] = max(1.0, min(self.max_weight, comp_weight))

        return torch.from_numpy(weights_np).to(device=device, dtype=dtype)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        targets = (targets > 0.5).float()
        if weight_map is None:
            weight_map = self.make_weight_map(targets)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted_bce = (bce * weight_map).mean()

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(weight_map * probs * targets, dims)
        denom = torch.sum(weight_map * probs, dims) + torch.sum(weight_map * targets, dims)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)
        weighted_dice = dice_loss.mean()

        return self.bce_weight * weighted_bce + self.dice_weight * weighted_dice
