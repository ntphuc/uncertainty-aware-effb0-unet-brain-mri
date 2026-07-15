import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _SegmentationTensorOutput(nn.Module):
    """Expose the primary segmentation logits to FLOP profilers.

    Most models in this repository return a dictionary, while THOP expects a
    tensor-like output. This wrapper does not alter the executed operations; it
    only selects ``output["seg"]`` after the forward pass.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output["seg"]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


def estimate_flops(model, input_shape, device):
    was_training = model.training
    try:
        from thop import profile

        dummy = torch.randn(*input_shape, device=device)
        model.eval()
        wrapped = _SegmentationTensorOutput(model)
        with torch.no_grad():
            flops, params = profile(wrapped, inputs=(dummy,), verbose=False)
        return int(flops), int(params)
    except Exception as exc:
        # Selective-scan kernels (VM-UNet) may not have a THOP handler. Keep the
        # evaluation usable and report NA rather than silently inventing FLOPs.
        print(f"Warning: FLOP estimation unavailable: {type(exc).__name__}: {exc}")
        return None, count_parameters(model)
    finally:
        model.train(was_training)
