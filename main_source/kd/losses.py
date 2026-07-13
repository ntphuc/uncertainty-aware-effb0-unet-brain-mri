"""Knowledge distillation losses for binary medical image segmentation."""
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.boundary import mask_to_soft_boundary


def binary_entropy_from_prob(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    ent = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    return ent / 0.69314718056


def soft_dice_loss_from_probs(student_prob: torch.Tensor, teacher_prob: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    if weight is None:
        weight = torch.ones_like(student_prob)
    sp = student_prob * weight
    tp = teacher_prob * weight
    dims = (1, 2, 3)
    inter = (sp * tp).sum(dim=dims)
    denom = sp.sum(dim=dims) + tp.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def weighted_bce_with_soft_targets(logits: torch.Tensor, targets: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


class SegKDLoss(nn.Module):
    """Selective / uncertainty-weighted segmentation KD.

    teacher_output_type: 'logit' if teacher returns mask logits, 'prob' if it returns probabilities.
    selective=True keeps only teacher-confident pixels.
    uncertainty_weighted=True downweights high-entropy teacher pixels.
    """
    def __init__(
        self,
        alpha_logit: float = 0.5,
        alpha_dice: float = 0.5,
        alpha_boundary: float = 0.0,
        temperature: float = 1.0,
        teacher_output_type: str = "logit",
        selective: bool = True,
        confidence_threshold: float = 0.65,
        uncertainty_weighted: bool = True,
        uncertainty_power: float = 1.0,
        boundary_kernel_size: int = 3,
        soft_boundary_sigma: float = 2.0,
        soft_boundary_radius: int = 5,
    ):
        super().__init__()
        self.alpha_logit = float(alpha_logit)
        self.alpha_dice = float(alpha_dice)
        self.alpha_boundary = float(alpha_boundary)
        self.temperature = float(temperature)
        self.teacher_output_type = str(teacher_output_type).lower()
        self.selective = bool(selective)
        self.confidence_threshold = float(confidence_threshold)
        self.uncertainty_weighted = bool(uncertainty_weighted)
        self.uncertainty_power = float(uncertainty_power)
        self.boundary_kernel_size = int(boundary_kernel_size)
        self.soft_boundary_sigma = float(soft_boundary_sigma)
        self.soft_boundary_radius = int(soft_boundary_radius)

    def _teacher_prob(self, teacher_pred: torch.Tensor) -> torch.Tensor:
        if self.teacher_output_type in ["prob", "probs", "probability"]:
            return teacher_pred.clamp(0, 1)
        if self.teacher_output_type in ["logit", "logits"]:
            return torch.sigmoid(teacher_pred / max(self.temperature, 1e-6))
        raise ValueError(f"Unknown teacher_output_type={self.teacher_output_type}")

    def _weight(self, teacher_prob: torch.Tensor, teacher_uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = torch.ones_like(teacher_prob)
        if self.selective:
            confident = (teacher_prob >= self.confidence_threshold) | (teacher_prob <= (1.0 - self.confidence_threshold))
            weight = weight * confident.float()
        if self.uncertainty_weighted:
            if teacher_uncertainty is None:
                teacher_uncertainty = binary_entropy_from_prob(teacher_prob)
            teacher_uncertainty = teacher_uncertainty.clamp(0, 1)
            weight = weight * (1.0 - teacher_uncertainty).pow(self.uncertainty_power)
        return weight.detach()

    def forward(self, student_logits: torch.Tensor, teacher_pred: torch.Tensor, teacher_uncertainty: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        teacher_prob = self._teacher_prob(teacher_pred).detach()
        if teacher_prob.shape[-2:] != student_logits.shape[-2:]:
            teacher_prob = F.interpolate(teacher_prob, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)
        if teacher_prob.shape[1] != student_logits.shape[1]:
            teacher_prob = teacher_prob[:, :1]
        weight = self._weight(teacher_prob, teacher_uncertainty=teacher_uncertainty)
        student_prob = torch.sigmoid(student_logits)
        loss_logit = weighted_bce_with_soft_targets(student_logits / max(self.temperature, 1e-6), teacher_prob, weight=weight)
        loss_dice = soft_dice_loss_from_probs(student_prob, teacher_prob, weight=weight)
        if self.alpha_boundary > 0:
            teacher_boundary = mask_to_soft_boundary(teacher_prob, kernel_size=self.boundary_kernel_size, sigma=self.soft_boundary_sigma, radius=self.soft_boundary_radius)
            student_boundary = mask_to_soft_boundary(student_prob, kernel_size=self.boundary_kernel_size, sigma=self.soft_boundary_sigma, radius=self.soft_boundary_radius)
            loss_boundary = F.l1_loss(student_boundary, teacher_boundary)
        else:
            loss_boundary = student_logits.new_tensor(0.0)
        total = self.alpha_logit * loss_logit + self.alpha_dice * loss_dice + self.alpha_boundary * loss_boundary
        return {
            "loss": total,
            "kd_logit_loss": loss_logit.detach(),
            "kd_dice_loss": loss_dice.detach(),
            "kd_boundary_loss": loss_boundary.detach(),
            "kd_weight_mean": weight.mean().detach(),
        }
