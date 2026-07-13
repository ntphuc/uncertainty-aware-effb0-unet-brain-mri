"""Prompt helpers for Medical-SAM3/Medical Samba teacher distillation."""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch


def masks_to_boxes(masks: torch.Tensor, pad: int = 4, normalized: bool = False) -> torch.Tensor:
    """Convert binary masks [B,1,H,W] to xyxy boxes [B,4]. Empty masks -> full image box."""
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"masks must be [B,1,H,W], got {tuple(masks.shape)}")
    b, _, h, w = masks.shape
    boxes = []
    for i in range(b):
        fg = (masks[i, 0] > 0.5).nonzero(as_tuple=False)
        if fg.numel() == 0:
            x1, y1, x2, y2 = 0, 0, w - 1, h - 1
        else:
            y1 = max(0, int(fg[:, 0].min().item()) - pad)
            y2 = min(h - 1, int(fg[:, 0].max().item()) + pad)
            x1 = max(0, int(fg[:, 1].min().item()) - pad)
            x2 = min(w - 1, int(fg[:, 1].max().item()) + pad)
        boxes.append([x1, y1, x2, y2])
    out = torch.tensor(boxes, dtype=torch.float32, device=masks.device)
    if normalized:
        denom = torch.tensor([w - 1, h - 1, w - 1, h - 1], dtype=out.dtype, device=out.device).clamp_min(1)
        out = out / denom
    return out


def make_text_prompts(batch_size: int, text: Optional[str] = "tumor") -> Optional[List[str]]:
    """Create a list of text prompts, or None for box-only prompting.

    Use text=None, text="", or text=false in YAML when you want Medical-SAM3/Medical Samba
    to run with box prompt only. This avoids accidentally sending the literal string "None".
    """
    if text is None:
        return None
    if isinstance(text, bool) and text is False:
        return None
    text = str(text).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return [text] * int(batch_size)


def build_teacher_prompts(
    images: torch.Tensor,
    masks: Optional[torch.Tensor],
    text: Optional[str] = "tumor",
    use_box: bool = True,
    box_pad: int = 4,
    normalized_box: bool = False,
) -> Tuple[Optional[List[str]], Optional[torch.Tensor]]:
    """Create text and/or box prompts for the promptable teacher only during training/precompute."""
    texts = make_text_prompts(images.shape[0], text=text)
    boxes = None
    if use_box:
        if masks is None:
            b, _, h, w = images.shape
            boxes = torch.tensor([[0, 0, w - 1, h - 1]] * b, dtype=torch.float32, device=images.device)
            if normalized_box:
                boxes = boxes / torch.tensor([w - 1, h - 1, w - 1, h - 1], dtype=boxes.dtype, device=boxes.device).clamp_min(1)
        else:
            boxes = masks_to_boxes(masks, pad=box_pad, normalized=normalized_box)
    return texts, boxes
