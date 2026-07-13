import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """
    Tversky loss for binary segmentation.

    alpha controls false-positive penalty.
    beta controls false-negative penalty.

    For under-segmentation, use beta > alpha, e.g. alpha=0.3, beta=0.7,
    because false negatives are pixels where GT is lesion but prediction misses it.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = (targets > 0.5).float()
        dims = (1, 2, 3)

        tp = torch.sum(probs * targets, dims)
        fp = torch.sum(probs * (1.0 - targets), dims)
        fn = torch.sum((1.0 - probs) * targets, dims)

        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - score.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss.

    This is useful when the dominant failed case is under-segmentation.
    It focuses training on hard foreground pixels and can increase lesion recall.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, eps=eps)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tv = self.tversky(logits, targets)
        return torch.pow(tv, self.gamma)
