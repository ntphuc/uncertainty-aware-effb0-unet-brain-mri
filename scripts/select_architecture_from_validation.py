#!/usr/bin/env python3
"""Select and lock the architecture using validation data only.

This script MUST be run before any factorial test-set evaluation.  It reads the
best validation record from every seed, averages metrics by architecture, and
applies the pre-specified rule:

1. keep variants within ``dice_tolerance`` of the best mean validation Dice;
2. choose the lowest mean validation HD95;
3. then the lowest ASSD;
4. then the highest Boundary-F1;
5. then the lexicographically first variant name.

It writes a cryptographic lock containing config/checkpoint hashes.  The test
evaluation script refuses to proceed without this lock or if files changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List

import yaml


METRICS = ["dice", "iou", "missing_prediction_rate", "hd95", "assd", "boundary_f1", "loss"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_std(values: List[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/factorial/manifest.yaml")
    parser.add_argument("--output-dir", default="outputs/factorial/selection")
    parser.add_argument("--dice-tolerance", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    lock_path = output_dir / "SELECTION_LOCK.json"
    if lock_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Selection lock already exists: {lock_path}. "
            "Use --overwrite only before any test-set evaluation."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    per_run: List[Dict[str, Any]] = []
    split_hashes = set()
    for run in manifest["runs"]:
        config_path = Path(run["config"])
        output_dir_run = Path(run["output_dir"])
        validation_path = output_dir_run / "validation_best.json"
        checkpoint_path = output_dir_run / "checkpoints" / "best_validation.pth"
        split_path = output_dir_run / "data_split.json"
        missing = [
            str(path)
            for path in (config_path, validation_path, checkpoint_path, split_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "All validation runs must finish before architecture selection. Missing:\n"
                + "\n".join(missing)
            )

        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        split_info = json.loads(split_path.read_text(encoding="utf-8"))
        split_hash = split_info["split_sha256"]
        split_hashes.add(split_hash)
        metrics = validation["best_validation_metrics"]
        row: Dict[str, Any] = {
            "variant": run["variant"],
            "seed": int(run["seed"]),
            "best_epoch": int(validation["best_epoch"]),
            "split_sha256": split_hash,
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        }
        for factor in ("multiscale", "skip_se", "boundary", "deep_supervision"):
            row[factor] = int(run[factor])
        for metric in METRICS:
            row[metric] = float(metrics[metric])
        per_run.append(row)

    if len(split_hashes) != 1:
        raise RuntimeError(
            "Factorial runs do not share one validation split. "
            f"Observed split hashes: {sorted(split_hashes)}"
        )

    variants = sorted({row["variant"] for row in per_run})
    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in per_run if row["variant"] == variant]
        seeds = sorted(row["seed"] for row in rows)
        summary: Dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(rows),
            "seeds": ",".join(map(str, seeds)),
            "multiscale": rows[0]["multiscale"],
            "skip_se": rows[0]["skip_se"],
            "boundary": rows[0]["boundary"],
            "deep_supervision": rows[0]["deep_supervision"],
            "mean_best_epoch": mean(row["best_epoch"] for row in rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = sample_std(values)
        summary_rows.append(summary)

    best_dice = max(row["dice_mean"] for row in summary_rows)
    candidates = [
        row
        for row in summary_rows
        if row["dice_mean"] >= best_dice - float(args.dice_tolerance)
    ]
    candidates.sort(
        key=lambda row: (
            row["missing_prediction_rate_mean"],
            row["hd95_mean"] if math.isfinite(row["hd95_mean"]) else float("inf"),
            row["assd_mean"] if math.isfinite(row["assd_mean"]) else float("inf"),
            -row["boundary_f1_mean"],
            row["variant"],
        )
    )
    selected = candidates[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_run, output_dir / "validation_per_run.csv")
    write_csv(summary_rows, output_dir / "validation_architecture_summary.csv")

    selected_runs = [row for row in per_run if row["variant"] == selected["variant"]]
    file_records = []
    for row in per_run:
        config_path = Path(row["config"])
        checkpoint_path = Path(row["checkpoint"])
        file_records.append(
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "config": str(config_path),
                "config_sha256": sha256_file(config_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "split_sha256": row["split_sha256"],
            }
        )

    lock = {
        "status": "architecture_locked_before_test_evaluation",
        "selection_basis": "validation only; no test metrics were read",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "validation_split_sha256": next(iter(split_hashes)),
        "policy": {
            "primary": "maximum mean validation Dice across seeds",
            "dice_tolerance": float(args.dice_tolerance),
            "candidate_rule": "within tolerance of best mean validation Dice",
            "tie_breakers": [
                "minimum mean validation missing-prediction rate",
                "minimum mean validation HD95",
                "minimum mean validation ASSD",
                "maximum mean validation Boundary-F1",
                "lexicographic variant name",
            ],
        },
        "selected_variant": selected["variant"],
        "selected_validation_summary": selected,
        "selected_runs": selected_runs,
        "all_locked_files": file_records,
    }
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Selected architecture: {selected['variant']}")
    print(
        f"Validation Dice={selected['dice_mean']:.6f}, "
        f"missing-rate={selected['missing_prediction_rate_mean']:.6f}, "
        f"HD95={selected['hd95_mean']:.6f}, "
        f"ASSD={selected['assd_mean']:.6f}, "
        f"BF1={selected['boundary_f1_mean']:.6f}"
    )
    print(f"Selection lock: {lock_path}")


if __name__ == "__main__":
    main()
