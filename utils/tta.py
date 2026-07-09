"""Test-time augmentation / multi-scale inference helpers."""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn.functional as F


@torch.no_grad()
def multiscale_logits(model, images: torch.Tensor, scales: Iterable[float] = (1.0,)) -> torch.Tensor:
    """Average segmentation logits over multiple input scales.

    This does not use ground-truth masks. It is safe for validation/test inference.
    It can help tiny lesions because the model also sees zoomed versions of the full image.

    Args:
        model: segmentation model returning a dict with key ``seg``.
        images: [B,C,H,W] tensor.
        scales: e.g. [1.0, 1.25, 1.5].

    Returns:
        Averaged logits at original [H,W].
    """
    scales = [float(s) for s in scales]
    if len(scales) == 0:
        scales = [1.0]
    orig_size = images.shape[-2:]
    logits_accum: List[torch.Tensor] = []

    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            x = images
        else:
            new_h = max(16, int(round(orig_size[0] * scale)))
            new_w = max(16, int(round(orig_size[1] * scale)))
            x = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

        out = model(x)["seg"]
        if out.shape[-2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)
        logits_accum.append(out)

    return torch.stack(logits_accum, dim=0).mean(dim=0)
