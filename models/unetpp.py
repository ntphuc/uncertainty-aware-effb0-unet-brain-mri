from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct


def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class DenseConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch),
        )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat(xs, dim=1))


class UNetPlusPlus(nn.Module):
    """2D U-Net++ baseline with nested dense skip connections.

    This implementation uses 5 encoder depths and the final x0_4 node for prediction.
    If deep_supervision=True, auxiliary outputs from x0_1, x0_2, x0_3 are returned.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        channels: List[int] = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        nb_filter = channels
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)

        self.conv0_0 = DenseConv(in_channels, nb_filter[0])
        self.conv1_0 = DenseConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DenseConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DenseConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DenseConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DenseConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DenseConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DenseConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DenseConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DenseConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DenseConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DenseConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DenseConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DenseConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DenseConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(nb_filter[0], num_classes, kernel_size=1) for _ in range(3)
            ])
        else:
            self.ds_heads = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(x0_0, _up_to(x1_0, x0_0))
        x1_1 = self.conv1_1(x1_0, _up_to(x2_0, x1_0))
        x2_1 = self.conv2_1(x2_0, _up_to(x3_0, x2_0))
        x3_1 = self.conv3_1(x3_0, _up_to(x4_0, x3_0))

        x0_2 = self.conv0_2(x0_0, x0_1, _up_to(x1_1, x0_0))
        x1_2 = self.conv1_2(x1_0, x1_1, _up_to(x2_1, x1_0))
        x2_2 = self.conv2_2(x2_0, x2_1, _up_to(x3_1, x2_0))

        x0_3 = self.conv0_3(x0_0, x0_1, x0_2, _up_to(x1_2, x0_0))
        x1_3 = self.conv1_3(x1_0, x1_1, x1_2, _up_to(x2_2, x1_0))

        x0_4 = self.conv0_4(x0_0, x0_1, x0_2, x0_3, _up_to(x1_3, x0_0))
        seg = self.final(x0_4)
        seg = F.interpolate(seg, size=input_size, mode="bilinear", align_corners=False)

        deep_outputs = []
        if self.deep_supervision:
            for node, head in zip([x0_1, x0_2, x0_3], self.ds_heads):
                aux = head(node)
                aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)
                deep_outputs.append(aux)
        return {"seg": seg, "boundary": None, "deep_outputs": deep_outputs}
