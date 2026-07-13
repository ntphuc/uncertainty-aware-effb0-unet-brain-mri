"""Medical-SAM3 / Medical Samba teacher adapter.

This file is a thin adapter layer because Medical-SAM3/Medical-Samba repos differ.
Usually you only edit `_build_external_teacher` and `_call_external_teacher` to match
how your 10GB checkpoint is loaded and called.

Expected output inside this project:
    {"logits": Tensor[B,1,H,W]} or {"prob": Tensor[B,1,H,W]}
Optional:
    {"uncertainty": Tensor[B,1,H,W] in [0,1]}
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prompting import build_teacher_prompts


class MedicalSambaTeacher(nn.Module):
    def __init__(self, cfg: Dict[str, Any], image_size: int = 256):
        super().__init__()
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.image_size = int(image_size)
        self.mode = str(self.cfg.get("mode", "external_callable")).lower()
        self.checkpoint = self.cfg.get("checkpoint", None)
        self.repo_path = self.cfg.get("repo_path", None)
        self.builder = self.cfg.get("builder", None)
        self.output_key = self.cfg.get("output_key", "logits")
        self.output_type = str(self.cfg.get("output_type", "logit")).lower()
        self.freeze = bool(self.cfg.get("freeze", True))
        pcfg = self.cfg.get("prompt", {})
        self.text = pcfg.get("text", "tumor")
        self.use_box = bool(pcfg.get("use_box", True))
        self.box_pad = int(pcfg.get("box_pad", 4))
        self.normalized_box = bool(pcfg.get("normalized_box", False))

        if not self.enabled:
            self.teacher = None
        elif self.mode == "external_callable":
            self.teacher = self._build_external_teacher()
        elif self.mode == "dummy":
            self.teacher = None
        else:
            raise ValueError(f"Unsupported teacher.mode={self.mode}. Use external_callable or dummy.")
        if self.teacher is not None and self.freeze:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def _build_external_teacher(self) -> Optional[nn.Module]:
        if self.repo_path:
            repo_path = str(Path(self.repo_path).expanduser().resolve())

            if not Path(repo_path).exists():
                raise FileNotFoundError(
                    f"Medical-SAM3 repository not found: {repo_path}"
                )

            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        if not self.builder:
            raise ValueError(
                "teacher.builder is required. "
                "Use sam3.model_builder:build_sam3_image_model"
            )

        module_name, func_name = str(self.builder).split(":", maxsplit=1)

        module = importlib.import_module(module_name)
        build_fn = getattr(module, func_name)

        print(f"[Teacher] Builder: {self.builder}")
        print(f"[Teacher] Checkpoint: {self.checkpoint}")

        # Official Medical-SAM3 / SAM3 builder
        if (
            module_name == "sam3.model_builder"
            and func_name == "build_sam3_image_model"
        ):
            model = build_fn(
                checkpoint_path=self.checkpoint,
                load_from_HF=False,
                device="cpu",
                eval_mode=True,
                enable_segmentation=True,
                enable_inst_interactivity=True,
            )
        else:
            # Generic fallback for other repositories
            try:
                model = build_fn(
                    checkpoint_path=self.checkpoint,
                    device="cpu",
                    eval_mode=True,
                )
            except TypeError:
                try:
                    model = build_fn(checkpoint=self.checkpoint)
                except TypeError:
                    model = build_fn()

                    if self.checkpoint:
                        checkpoint = torch.load(
                            self.checkpoint,
                            map_location="cpu",
                            weights_only=False,
                        )

                        state = checkpoint.get(
                            "model",
                            checkpoint.get(
                                "model_state",
                                checkpoint.get(
                                    "state_dict",
                                    checkpoint,
                                ),
                            ),
                        )

                        missing, unexpected = model.load_state_dict(
                            state,
                            strict=False,
                        )

                        print(
                            "[Teacher] Loaded checkpoint with "
                            f"missing={len(missing)}, "
                            f"unexpected={len(unexpected)}"
                        )

        model.eval()

        if self.freeze:
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        return model

    @torch.no_grad()
    def _call_external_teacher(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        assert self.teacher is not None
        texts, boxes = build_teacher_prompts(
            images=images,
            masks=masks,
            text=self.text,
            use_box=self.use_box,
            box_pad=self.box_pad,
            normalized_box=self.normalized_box,
        )
        if hasattr(self.teacher, "predict"):
            out = self.teacher.predict(images, text_prompts=texts, boxes=boxes)
        else:
            try:
                out = self.teacher(images, text_prompts=texts, boxes=boxes)
            except TypeError:
                try:
                    out = self.teacher(images, texts, boxes)
                except TypeError as exc:
                    raise RuntimeError(
                        "Cannot call Medical Samba teacher. Edit kd/medical_samba_teacher.py::_call_external_teacher "
                        "to match your repo API."
                    ) from exc
        if isinstance(out, torch.Tensor):
            result = {self.output_key: out}
        elif isinstance(out, dict):
            result = out
        else:
            raise TypeError(f"Teacher output must be Tensor or dict, got {type(out)}")
        if "logits" not in result and "prob" not in result:
            for key in [self.output_key, "mask_logits", "masks", "pred_masks", "seg"]:
                if key in result:
                    result["logits" if self.output_type == "logit" else "prob"] = result[key]
                    break
        if "logits" not in result and "prob" not in result:
            raise KeyError(f"Cannot find teacher mask in keys: {list(result.keys())}")
        return result

    @torch.no_grad()
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            raise RuntimeError("Teacher disabled but forward() was called")
        if self.mode == "dummy":
            if masks is None:
                raise ValueError("dummy teacher requires masks")
            prob = F.avg_pool2d(masks.float(), kernel_size=5, stride=1, padding=2).clamp(0, 1)
            logits = torch.logit(prob.clamp(1e-4, 1 - 1e-4))
            return {"logits": logits}
        out = self._call_external_teacher(images, masks=masks)
        pred_key = "logits" if "logits" in out else "prob"
        pred = out[pred_key]
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if pred.shape[-2:] != images.shape[-2:]:
            pred = F.interpolate(pred.float(), size=images.shape[-2:], mode="bilinear", align_corners=False)
        out[pred_key] = pred.float()
        if "uncertainty" in out and out["uncertainty"] is not None:
            unc = out["uncertainty"]
            if unc.ndim == 3:
                unc = unc.unsqueeze(1)
            if unc.shape[-2:] != images.shape[-2:]:
                unc = F.interpolate(unc.float(), size=images.shape[-2:], mode="bilinear", align_corners=False)
            out["uncertainty"] = unc.float().clamp(0, 1)
        return out
