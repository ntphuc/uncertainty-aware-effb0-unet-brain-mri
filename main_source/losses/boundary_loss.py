import torch
import torch.nn as nn
from .dice_ce import DiceLoss


class BoundaryLoss(nn.Module):
    """BCE + Dice on predicted boundary map."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, boundary_logits: torch.Tensor, boundary_targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(boundary_logits, boundary_targets.float())
            + self.dice_weight * self.dice(boundary_logits, boundary_targets.float())
        )
