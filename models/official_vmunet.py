"""Adapter for the *official* VM-UNet implementation.

This module intentionally does not reimplement VM-UNet with a convolutional proxy.
It loads VSSM directly from the authors' official repository:
    https://github.com/JCruan519/VM-UNet

Run scripts/setup_official_vmunet.sh before instantiating this model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialVMUNet(nn.Module):
    """Paper-faithful VM-UNet wrapper returning raw segmentation logits."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        repo_path: str = "external/VM-UNet",
        depths: Sequence[int] = (2, 2, 9, 2),
        depths_decoder: Sequence[int] = (2, 9, 2, 2),
        drop_path_rate: float = 0.2,
        repeat_grayscale_to_rgb: bool = True,
        load_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.input_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.repeat_grayscale_to_rgb = bool(repeat_grayscale_to_rgb and in_channels == 1)
        official_in_channels = 3 if self.repeat_grayscale_to_rgb else in_channels

        module = self._load_official_vmamba_module(repo_path)
        if not hasattr(module, "VSSM"):
            raise ImportError(f"Official VM-UNet module at {repo_path!r} does not expose VSSM")

        self.vmunet = module.VSSM(
            in_chans=official_in_channels,
            num_classes=num_classes,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            drop_path_rate=float(drop_path_rate),
        )
        if load_ckpt_path:
            self.load_official_pretrained(load_ckpt_path)

    @staticmethod
    def _load_official_vmamba_module(repo_path: str) -> ModuleType:
        repo = Path(repo_path).expanduser().resolve()
        module_file = repo / "models" / "vmunet" / "vmamba.py"
        if not module_file.exists():
            raise FileNotFoundError(
                "Official VM-UNet source was not found. Expected:\n"
                f"  {module_file}\n"
                "Run: bash scripts/setup_official_vmunet.sh"
            )

        try:
            import mamba_ssm  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "The official VM-UNet selective-scan dependency is missing. "
                "Run: bash scripts/setup_official_vmunet.sh\n"
                "VM-UNet is CUDA-oriented; use an NVIDIA CUDA environment."
            ) from exc

        module_name = "brisc_official_vmunet_vmamba"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official VM-UNet module from {module_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ImportError(
                "Failed to import the official VM-UNet implementation. "
                "Check torch/CUDA, causal-conv1d, mamba-ssm and timm versions."
            ) from exc
        return module

    def load_official_pretrained(self, checkpoint_path: str) -> None:
        """Load an official VMamba/VM-UNet checkpoint and mirror encoder weights to decoder.

        This follows the loading strategy in the authors' official vmunet.py wrapper.
        It also accepts this project's checkpoints and common state-dict layouts.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, Mapping):
            state = checkpoint.get("model", checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint)))
        else:
            state = checkpoint
        if not isinstance(state, Mapping):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

        cleaned = {}
        for key, value in state.items():
            key = str(key)
            for prefix in ("module.", "vmunet.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value

        model_state = self.vmunet.state_dict()
        direct = {k: v for k, v in cleaned.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(direct)
        self.vmunet.load_state_dict(model_state, strict=False)

        mirrored = {}
        layer_map = {"layers.0": "layers_up.3", "layers.1": "layers_up.2", "layers.2": "layers_up.1", "layers.3": "layers_up.0"}
        for key, value in cleaned.items():
            for src, dst in layer_map.items():
                if src in key:
                    new_key = key.replace(src, dst)
                    if new_key in model_state and model_state[new_key].shape == value.shape:
                        mirrored[new_key] = value
                    break
        model_state.update(mirrored)
        self.vmunet.load_state_dict(model_state, strict=False)
        print(
            f"Loaded official VM-UNet checkpoint: direct={len(direct)}, "
            f"encoder-to-decoder mirrored={len(mirrored)}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        input_size = x.shape[-2:]
        if self.repeat_grayscale_to_rgb:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not torch.is_tensor(logits):
            raise TypeError(f"Official VM-UNet returned unsupported output type: {type(logits)!r}")
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return {"seg": logits, "boundary": None, "deep_outputs": []}
