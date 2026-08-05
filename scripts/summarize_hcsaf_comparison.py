#!/usr/bin/env python3
"""Summarize post-lock Concat/RGAMF/HCSAF-BR comparison results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import yaml


METRICS = ["dice", "iou", "hd95", "assd", "boundary_f1", "params", "gflops"]


def sample_std(values: List[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/hcsaf_comparison/manifest.yaml"
    )
    parser.add_argument(
        "--lock", default="outputs/hcsaf_comparison/selection/SELECTION_LOCK.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/hcsaf_comparison/summary"
    )
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Selection lock is required before summarizing test results: {lock_path}"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    per_run: List[Dict[str, Any]] = []
    missing: List[str] = []
    for run in manifest["runs"]:
        result_path = Path(run["output_dir"]) / "eval" / "test_results.json"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        row: Dict[str, Any] = {
            "variant": run["variant"],
            "seed": int(run["seed"]),
            "fusion_design": run.get("fusion_design", ""),
            "selected_by_validation": run["variant"] == lock["selected_variant"],
        }
        for metric in METRICS:
            value = result.get(metric)
            row[metric] = None if value is None else float(value)
        per_run.append(row)

    if args.require_all and missing:
        raise FileNotFoundError("Missing test results:\n" + "\n".join(missing))
    if not per_run:
        raise RuntimeError("No test results were found")

    summaries: List[Dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in per_run}):
        rows = [row for row in per_run if row["variant"] == variant]
        summary: Dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda r: r["seed"])),
            "fusion_design": rows[0]["fusion_design"],
            "selected_by_validation": rows[0]["selected_by_validation"],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = sample_std(values)
        summaries.append(summary)

    write_csv(per_run, output_dir / "hcsaf_comparison_per_run.csv")
    write_csv(summaries, output_dir / "hcsaf_comparison_summary.csv")

    lines = [
        "# Validation-locked HCSAF comparison",
        "",
        f"Selected variant: **{lock['selected_variant']}**",
        "",
        "Test metrics below are post-lock and were not used for architecture selection.",
        "",
        "| Variant | Dice | IoU | HD95 | ASSD | BF1 | Params (M) | GFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def pm(metric: str, digits: int = 4) -> str:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key not in row:
                return "n/a"
            return f"{row[mean_key]:.{digits}f} ± {row.get(std_key, 0.0):.{digits}f}"

        params = row.get("params_mean")
        gflops = row.get("gflops_mean")
        lines.append(
            "| {variant} | {dice} | {iou} | {hd95} | {assd} | {bf1} | {params} | {gflops} |".format(
                variant=row["variant"],
                dice=pm("dice"),
                iou=pm("iou"),
                hd95=pm("hd95"),
                assd=pm("assd"),
                bf1=pm("boundary_f1"),
                params="n/a" if params is None else f"{params / 1e6:.3f}",
                gflops="n/a" if gflops is None else f"{gflops:.3f}",
            )
        )

    (output_dir / "hcsaf_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved summary to: {output_dir}")
    if missing:
        print(f"Warning: {len(missing)} test result files were missing")


if __name__ == "__main__":
    main()
