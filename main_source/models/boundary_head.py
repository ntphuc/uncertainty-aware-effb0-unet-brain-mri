import torch.nn as nn
from .blocks import ConvBNAct


class BoundaryHead(nn.Module):
    """Auxiliary branch that predicts lesion/tumor boundary map."""

    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_ch, max(in_ch // 2, 16)),
            nn.Conv2d(max(in_ch // 2, 16), out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)
