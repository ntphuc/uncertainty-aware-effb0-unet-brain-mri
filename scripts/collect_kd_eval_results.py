#!/usr/bin/env python3
"""Collect student KD/ablation evaluation JSON files into one CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

CONFIGS = [
    "configs/kd/student_no_kd.yaml",
    "configs/kd/ablation_vanilla_kd.yaml",
    "configs/kd/ablation_edge_kd.yaml",
    "configs/kd/ablation_selective_uncertainty_kd.yaml",
    "configs/kd/kd_medical_samba_only_box.yaml",
    "configs/kd/kd_medical_samba_only_tumor.yaml",
    "configs/kd/kd_medical_samba_only_polyp.yaml",
    "configs/kd/student_kd_medical_samba_text_tumor_box.yaml",
    "configs/kd/student_kd_medical_samba_text_polyp_box.yaml",
]


def main() -> None:
    rows: List[Dict[str, object]] = []
    for cfg_path in CONFIGS:
        path = Path(cfg_path)
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        out_dir = Path(cfg["paths"]["output_dir"])
        result_path = out_dir / "eval" / "test_results.json"
        if not result_path.exists():
            print(f"[SKIP] Missing result: {result_path}")
            continue
        metrics = json.loads(result_path.read_text(encoding="utf-8"))
        prompt = cfg.get("teacher", {}).get("prompt", {})
        kd = cfg.get("kd", {})
        rows.append({
            "project_name": cfg.get("project_name", path.stem),
            "config": cfg_path,
            "kd_enabled": bool(kd.get("enabled", False)),
            "prompt_text": prompt.get("text"),
            "prompt_use_box": prompt.get("use_box"),
            "dice": metrics.get("dice"),
            "iou": metrics.get("iou"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f2": metrics.get("f2"),
            "hd95": metrics.get("hd95"),
            "assd": metrics.get("assd"),
            "boundary_f1": metrics.get("boundary_f1"),
            "params": metrics.get("params"),
            "gflops": metrics.get("gflops"),
            "checkpoint_epoch": metrics.get("checkpoint_epoch"),
            "checkpoint_best_dice": metrics.get("checkpoint_best_dice"),
            "result_json": str(result_path),
        })

    output = Path("outputs/kd_ablation_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No student evaluation results found. Train/evaluate the models first.")
        return

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {output}")
    for row in rows:
        print(f"{row['project_name']}: Dice={row['dice']} IoU={row['iou']} HD95={row['hd95']}")


if __name__ == "__main__":
    main()
