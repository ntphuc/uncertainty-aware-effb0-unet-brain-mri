#!/usr/bin/env python3
"""Evaluate Medical-SAM3 / Medical Samba teacher on the configured NPY dataset.

Important:
- When teacher.prompt.use_box=true, boxes are generated from the ground-truth mask.
  Therefore this is a prompt-conditioned teacher upper-bound, not automatic inference.
- For automatic-style evaluation, use a text-only prompt config.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import NPYSliceDataset
from kd import MedicalSambaTeacher
from utils.metrics import (
    AverageMeter,
    batch_surface_metrics,
    dice_score_from_logits,
    iou_score_from_logits,
    precision_recall_fbeta_from_logits,
)
from utils.misc import count_parameters, load_config, save_json
from utils.visualization import save_prediction_grid


def _as_logits(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "logits" in output:
        return output["logits"].float()
    if "prob" in output:
        prob = output["prob"].float().clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(prob)
    raise KeyError(f"Teacher output does not contain logits or prob: {list(output.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Medical-SAM3/Medical Samba teacher")
    parser.add_argument("--config", required=True, help="KD YAML containing paths/training/model/teacher")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only; omit for full evaluation")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    paths = cfg["paths"]
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    teacher_cfg = cfg.get("teacher", {})

    if args.split == "test":
        x_path, y_path = paths["x_test"], paths["y_test"]
    else:
        x_path, y_path = paths["x_train"], paths["y_train"]

    dataset = NPYSliceDataset(
        x_path,
        y_path,
        image_size=tcfg.get("image_size", 256),
        in_channels=mcfg.get("in_channels", 1),
        normalize=tcfg.get("normalize", "zscore"),
        augment=False,
    )
    batch_size = int(args.batch_size or min(int(tcfg.get("batch_size", 1)), 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else tcfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    teacher = MedicalSambaTeacher(teacher_cfg, image_size=tcfg.get("image_size", 256)).to(device)
    teacher.eval()

    threshold = float(args.threshold if args.threshold is not None else tcfg.get("threshold", 0.5))
    meters = {k: AverageMeter() for k in ["dice", "iou", "precision", "recall", "f2", "hd95", "assd", "boundary_f1"]}
    first_batch = None
    measured_images = 0
    measured_seconds = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (images, masks) in enumerate(tqdm(loader, desc=f"teacher-eval-{args.split}")):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = teacher(images, masks=masks)
            logits = _as_logits(output)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        if batch_index >= int(args.warmup_batches):
            measured_images += images.size(0)
            measured_seconds += elapsed

        bs = images.size(0)
        meters["dice"].update(dice_score_from_logits(logits, masks, threshold), bs)
        meters["iou"].update(iou_score_from_logits(logits, masks, threshold), bs)
        pr = precision_recall_fbeta_from_logits(logits, masks, threshold=threshold, beta=2.0)
        meters["precision"].update(pr["precision"], bs)
        meters["recall"].update(pr["recall"], bs)
        meters["f2"].update(pr["f_beta"], bs)
        surf = batch_surface_metrics(logits, masks, threshold=threshold)
        for key in ["hd95", "assd", "boundary_f1"]:
            meters[key].update(surf[key], bs)

        if first_batch is None:
            first_batch = (
                images.detach().cpu(),
                masks.detach().cpu(),
                {"seg": logits.detach().cpu(), "boundary": None, "deep_outputs": []},
            )

    prompt_cfg = teacher_cfg.get("prompt", {})
    prompt_name = "box" if prompt_cfg.get("use_box", False) else "no_box"
    text_prompt = prompt_cfg.get("text", None)
    if text_prompt not in [None, "", False]:
        prompt_name = f"{str(text_prompt).replace(' ', '_')}_{prompt_name}"
    else:
        prompt_name = f"only_{prompt_name}"

    out_dir = Path(args.output_dir or f"outputs/teacher_eval/{prompt_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = None
    if getattr(teacher, "teacher", None) is not None:
        params = count_parameters(teacher.teacher)

    checkpoint = teacher_cfg.get("checkpoint")
    checkpoint_size_mb = None
    if checkpoint and Path(checkpoint).exists():
        checkpoint_size_mb = Path(checkpoint).stat().st_size / (1024 ** 2)

    results = {key: meter.avg for key, meter in meters.items()}
    results.update({
        "model": "Medical-SAM3/Medical-Samba teacher",
        "config": args.config,
        "split": args.split,
        "threshold": threshold,
        "prompt_text": text_prompt,
        "prompt_use_box": bool(prompt_cfg.get("use_box", False)),
        "box_source": "ground_truth_mask" if prompt_cfg.get("use_box", False) else None,
        "prompt_conditioned_upper_bound": bool(prompt_cfg.get("use_box", False)),
        "params": params,
        "checkpoint_size_mb": checkpoint_size_mb,
        "evaluated_images": len(dataset) if args.max_batches is None else min(len(dataset), args.max_batches * batch_size),
        "timed_images": measured_images,
        "latency_ms_per_image": (1000.0 * measured_seconds / measured_images) if measured_images else None,
        "fps": (measured_images / measured_seconds) if measured_seconds > 0 else None,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None,
    })

    save_json(results, out_dir / f"{args.split}_results.json")
    if first_batch is not None:
        save_prediction_grid(*first_batch, out_dir / f"{args.split}_predictions.png", threshold=threshold)

    with (out_dir / f"{args.split}_results.txt").open("w", encoding="utf-8") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if results["prompt_conditioned_upper_bound"]:
        print("[NOTE] Box prompt came from ground-truth mask: treat this teacher result as a prompt-conditioned upper bound.")
    print(f"Saved teacher evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
