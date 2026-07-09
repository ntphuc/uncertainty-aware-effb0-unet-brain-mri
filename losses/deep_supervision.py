import torch
import torch.nn as nn
from .dice_ce import DiceCELoss


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = DiceCELoss()

    def forward(self, deep_outputs, targets: torch.Tensor) -> torch.Tensor:
        if deep_outputs is None or len(deep_outputs) == 0:
            return targets.new_tensor(0.0)
        losses = [self.base(out, targets) for out in deep_outputs]
        return torch.stack(losses).mean()
