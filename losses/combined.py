from typing import Dict

import torch
import torch.nn as nn

from .dice_ce import DiceCELoss
from .boundary_loss import BoundaryLoss
from .deep_supervision import DeepSupervisionLoss
from .focal_tversky import FocalTverskyLoss, TverskyLoss
from .component_weighted import ComponentWeightedSegLoss
from .hard_negative import HardNegativeLoss
from utils.boundary import mask_to_boundary, mask_to_soft_boundary


class CombinedSegBoundaryLoss(nn.Module):
    """
    Combined loss for the proposed lightweight segmentation model.

    Default behavior is compatible with the old project:
        DiceCE + lambda_boundary * BoundaryLoss + beta_deep_supervision * DeepSupervisionLoss

    New optional recall-aware terms:
        gamma_tversky * FocalTverskyLoss
        component_weight_lambda * ComponentWeightedSegLoss

    Use beta_tversky > alpha_tversky when failed cases are dominated by under-segmentation,
    because this penalizes false negatives more strongly.
    """

    def __init__(
        self,
        lambda_boundary: float = 0.15,
        lambda_boundary_guide: float = 0.0,
        beta_deep_supervision: float = 0.25,
        boundary_kernel_size: int = 3,
        gamma_tversky: float = 0.0,
        alpha_tversky: float = 0.3,
        beta_tversky: float = 0.7,
        focal_tversky_gamma: float = 0.75,
        use_focal_tversky: bool = True,
        use_soft_boundary: bool = False,
        soft_boundary_sigma: float = 2.0,
        soft_boundary_radius: int = 5,
        use_component_weight: bool = False,
        component_weight_lambda: float = 0.0,
        component_weight_max: float = 4.0,
        component_weight_power: float = 0.5,
        component_weight_min_area: int = 5,
        use_hard_negative_loss: bool = False,
        lambda_hard_negative: float = 0.0,
        hard_negative_threshold: float = 0.60,
        hard_negative_topk_percent: float = 0.01,
        hard_negative_min_pixels: int = 16,
    ):
        super().__init__()
        self.seg_loss = DiceCELoss()
        self.boundary_loss = BoundaryLoss()
        self.deep_loss = DeepSupervisionLoss()
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_boundary_guide = float(lambda_boundary_guide)
        self.beta_deep_supervision = float(beta_deep_supervision)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.gamma_tversky = float(gamma_tversky)
        self.use_soft_boundary = bool(use_soft_boundary)
        self.soft_boundary_sigma = float(soft_boundary_sigma)
        self.soft_boundary_radius = int(soft_boundary_radius)
        self.use_component_weight = bool(use_component_weight)
        self.component_weight_lambda = float(component_weight_lambda)
        self.component_weight_loss = ComponentWeightedSegLoss(
            max_weight=component_weight_max,
            power=component_weight_power,
            min_area=component_weight_min_area,
        )
        self.use_hard_negative_loss = bool(use_hard_negative_loss)
        self.lambda_hard_negative = float(lambda_hard_negative)
        self.hard_negative_loss = HardNegativeLoss(
            hard_threshold=hard_negative_threshold,
            topk_percent=hard_negative_topk_percent,
            min_pixels=hard_negative_min_pixels,
        )

        if use_focal_tversky:
            self.tversky_loss = FocalTverskyLoss(
                alpha=alpha_tversky,
                beta=beta_tversky,
                gamma=focal_tversky_gamma,
            )
        else:
            self.tversky_loss = TverskyLoss(alpha=alpha_tversky, beta=beta_tversky)

    def forward(self, outputs: Dict[str, torch.Tensor], masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_seg = self.seg_loss(outputs["seg"], masks)

        if self.gamma_tversky > 0:
            loss_tversky = self.tversky_loss(outputs["seg"], masks)
        else:
            loss_tversky = masks.new_tensor(0.0)

        if self.use_component_weight and self.component_weight_lambda > 0:
            loss_component_weight = self.component_weight_loss(outputs["seg"], masks)
        else:
            loss_component_weight = masks.new_tensor(0.0)

        if self.use_hard_negative_loss and self.lambda_hard_negative > 0:
            loss_hard_negative = self.hard_negative_loss(outputs["seg"], masks)
        else:
            loss_hard_negative = masks.new_tensor(0.0)

        needs_boundary_target = (
            (outputs.get("boundary") is not None and self.lambda_boundary > 0)
            or (
                outputs.get("boundary_guide") is not None
                and self.lambda_boundary_guide > 0
            )
        )
        if needs_boundary_target:
            if self.use_soft_boundary:
                boundary_gt = mask_to_soft_boundary(
                    masks,
                    kernel_size=self.boundary_kernel_size,
                    sigma=self.soft_boundary_sigma,
                    radius=self.soft_boundary_radius,
                )
            else:
                boundary_gt = mask_to_boundary(masks, kernel_size=self.boundary_kernel_size)
        else:
            boundary_gt = None

        if outputs.get("boundary") is not None and self.lambda_boundary > 0:
            loss_boundary = self.boundary_loss(outputs["boundary"], boundary_gt)
        else:
            loss_boundary = masks.new_tensor(0.0)

        if (
            outputs.get("boundary_guide") is not None
            and self.lambda_boundary_guide > 0
        ):
            loss_boundary_guide = self.boundary_loss(
                outputs["boundary_guide"], boundary_gt
            )
        else:
            loss_boundary_guide = masks.new_tensor(0.0)

        loss_deep = self.deep_loss(outputs.get("deep_outputs", []), masks)

        total = (
            loss_seg
            + self.gamma_tversky * loss_tversky
            + self.component_weight_lambda * loss_component_weight
            + self.lambda_hard_negative * loss_hard_negative
            + self.lambda_boundary * loss_boundary
            + self.lambda_boundary_guide * loss_boundary_guide
            + self.beta_deep_supervision * loss_deep
        )
        return {
            "loss": total,
            "seg_loss": loss_seg.detach(),
            "tversky_loss": loss_tversky.detach(),
            "component_weight_loss": loss_component_weight.detach(),
            "hard_negative_loss": loss_hard_negative.detach(),
            "boundary_loss": loss_boundary.detach(),
            "boundary_guide_loss": loss_boundary_guide.detach(),
            "deep_loss": loss_deep.detach(),
        }
