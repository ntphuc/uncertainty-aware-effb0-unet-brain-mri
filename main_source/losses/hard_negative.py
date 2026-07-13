"""Hard-negative loss for binary medical image segmentation.

This loss is designed for cases where the model predicts a high-confidence lesion
inside background regions, e.g. a gray anatomical structure is mistaken as tumor.

It only looks at background pixels according to the ground-truth mask and penalizes
pixels whose predicted probability is high. A small top-k option is included so the
loss still has signal even when few pixels exceed the threshold early in training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeLoss(nn.Module):
    def __init__(
        self,
        hard_threshold: float = 0.60,
        topk_percent: float = 0.01,
        min_pixels: int = 16,
        background_value: float = 0.0,
    ):
        super().__init__()
        self.hard_threshold = float(hard_threshold)
        self.topk_percent = float(topk_percent)
        self.min_pixels = int(min_pixels)
        self.background_value = float(background_value)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return BCE on hard false-positive-like background pixels.

        Args:
            logits: [B, 1, H, W] raw segmentation logits.
            targets: [B, 1, H, W] binary GT mask.
        """
        targets = (targets > 0.5).float()
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")

        losses = []
        for i in range(logits.shape[0]):
            bg = targets[i] < 0.5
            if bg.sum() == 0:
                continue

            bg_prob = probs[i][bg]
            bg_bce = bce[i][bg]

            # Pixels already above a probability threshold are hard negatives.
            hard_mask = bg_prob.detach() > self.hard_threshold

            # Top-k pixels provide a stable signal even when no pixel crosses the threshold.
            if self.topk_percent > 0:
                k = max(self.min_pixels, int(bg_prob.numel() * self.topk_percent))
                k = min(k, bg_prob.numel())
                _, idx = torch.topk(bg_prob.detach(), k=k, largest=True)
                topk_mask = torch.zeros_like(bg_prob, dtype=torch.bool)
                topk_mask[idx] = True
                hard_mask = hard_mask | topk_mask

            if hard_mask.sum() > 0:
                losses.append(bg_bce[hard_mask].mean())

        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()
