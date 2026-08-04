#!/usr/bin/env python3
"""Summarize validation-locked factorial experiments.

Outputs
-------
- validation_per_run.csv / validation_architecture_summary.csv (copied from selection)
- test_per_run.csv
- test_architecture_summary.csv (mean ± sample SD across training seeds)
- factorial_effects.csv (paired-by-seed main effects and interactions)
- factorial_report.md (paper-ready tables and interpretation notes)

Test results are never used for selecting the architecture; the script requires
SELECTION_LOCK.json and labels the full table as post-lock ablation evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


METRICS = ["dice", "iou", "hd95", "assd", "boundary_f1", "params", "gflops"]
ACCURACY_METRICS = ["dice", "iou", "hd95", "assd", "boundary_f1"]
FACTOR_NAMES = ["skip_se", "boundary", "deep_supervision"]


def sample_std(values: Sequence[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
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


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/factorial/manifest.yaml")
    parser.add_argument("--lock", default="outputs/factorial/selection/SELECTION_LOCK.json")
    parser.add_argument("--output-dir", default="outputs/factorial/summary")
    parser.add_argument(
        "--require-all-test-results",
        action="store_true",
        help="Fail unless every factorial run has test_results.json.",
    )
    return parser.parse_args()


def variant_summary(per_run: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in per_run}):
        rows = [row for row in per_run if row["variant"] == variant]
        summary: Dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(rows),
            "seeds": ",".join(str(v) for v in sorted(row["seed"] for row in rows)),
            "multiscale": rows[0]["multiscale"],
            "skip_se": rows[0]["skip_se"],
            "boundary": rows[0]["boundary"],
            "deep_supervision": rows[0]["deep_supervision"],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = sample_std(values)
        summaries.append(summary)
    return summaries


def cell_map(per_run: List[Dict[str, Any]]) -> Dict[Tuple[int, int, int, int], Dict[str, Any]]:
    return {
        (
            int(row["seed"]),
            int(row["skip_se"]),
            int(row["boundary"]),
            int(row["deep_supervision"]),
        ): row
        for row in per_run
        if int(row["multiscale"]) == 1
    }


def paired_factorial_effects(per_run: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells = cell_map(per_run)
    seeds = sorted({int(row["seed"]) for row in per_run})
    effects: List[Dict[str, Any]] = []

    # Ensure the complete 2x2x2 design exists for every included seed.
    complete_seeds = []
    for seed in seeds:
        required = [(seed, s, b, d) for s in (0, 1) for b in (0, 1) for d in (0, 1)]
        if all(key in cells for key in required):
            complete_seeds.append(seed)

    for metric in ACCURACY_METRICS:
        per_effect: Dict[str, List[float]] = {
            "M": [],
            "S": [],
            "B": [],
            "D": [],
            "S:B": [],
            "S:D": [],
            "B:D": [],
            "S:B:D": [],
        }

        # MultiScale anchor effect: M-only minus Base for the same seed.
        by_variant_seed = {(row["variant"], int(row["seed"])): row for row in per_run}
        for seed in seeds:
            base = by_variant_seed.get(("f0_base", seed))
            m_only = by_variant_seed.get(("f1_m_only", seed))
            if base is not None and m_only is not None:
                per_effect["M"].append(float(m_only[metric]) - float(base[metric]))

        for seed in complete_seeds:
            y = lambda s, b, d: float(cells[(seed, s, b, d)][metric])
            per_effect["S"].append(
                mean(y(1, b, d) for b in (0, 1) for d in (0, 1))
                - mean(y(0, b, d) for b in (0, 1) for d in (0, 1))
            )
            per_effect["B"].append(
                mean(y(s, 1, d) for s in (0, 1) for d in (0, 1))
                - mean(y(s, 0, d) for s in (0, 1) for d in (0, 1))
            )
            per_effect["D"].append(
                mean(y(s, b, 1) for s in (0, 1) for b in (0, 1))
                - mean(y(s, b, 0) for s in (0, 1) for b in (0, 1))
            )
            per_effect["S:B"].append(
                mean(y(1, 1, d) - y(1, 0, d) - y(0, 1, d) + y(0, 0, d) for d in (0, 1))
            )
            per_effect["S:D"].append(
                mean(y(1, b, 1) - y(1, b, 0) - y(0, b, 1) + y(0, b, 0) for b in (0, 1))
            )
            per_effect["B:D"].append(
                mean(y(s, 1, 1) - y(s, 1, 0) - y(s, 0, 1) + y(s, 0, 0) for s in (0, 1))
            )
            per_effect["S:B:D"].append(
                y(1, 1, 1)
                - y(1, 1, 0)
                - y(1, 0, 1)
                - y(0, 1, 1)
                + y(1, 0, 0)
                + y(0, 1, 0)
                + y(0, 0, 1)
                - y(0, 0, 0)
            )

        for effect_name, values in per_effect.items():
            if not values:
                continue
            mean_delta = mean(values)
            better_when = "positive" if metric in {"dice", "iou", "boundary_f1"} else "negative"
            effects.append(
                {
                    "metric": metric,
                    "effect": effect_name,
                    "n_paired_seeds": len(values),
                    "mean_raw_delta": mean_delta,
                    "std_raw_delta": sample_std(values),
                    "better_direction": better_when,
                    "descriptive_interpretation": (
                        "improves metric on average"
                        if (mean_delta > 0 and better_when == "positive")
                        or (mean_delta < 0 and better_when == "negative")
                        else "worsens metric on average"
                    ),
                }
            )
    return effects


def markdown_table(summaries: List[Dict[str, Any]], selected: str) -> str:
    lines = [
        "| Variant | M | S | B | D | Dice | IoU | HD95 | ASSD | BF1 | Params (M) | GFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        label = f"**{row['variant']}**" if row["variant"] == selected else row["variant"]
        cells = [label, row["multiscale"], row["skip_se"], row["boundary"], row["deep_supervision"]]
        for metric in ["dice", "iou", "hd95", "assd", "boundary_f1"]:
            cells.append(
                f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
            )
        cells.append(f"{row.get('params_mean', float('nan')) / 1e6:.3f}")
        cells.append(f"{row.get('gflops_mean', float('nan')):.3f}")
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    lock_path = Path(args.lock)
    output_dir = Path(args.output_dir)
    if not lock_path.exists():
        raise FileNotFoundError("Architecture selection lock is required before summarizing test results.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    per_run: List[Dict[str, Any]] = []
    missing: List[str] = []
    for run in manifest["runs"]:
        result_path = Path(run["output_dir"]) / "eval" / "test_results.json"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        results = json.loads(result_path.read_text(encoding="utf-8"))
        row: Dict[str, Any] = {
            "variant": run["variant"],
            "seed": int(run["seed"]),
            "multiscale": int(run["multiscale"]),
            "skip_se": int(run["skip_se"]),
            "boundary": int(run["boundary"]),
            "deep_supervision": int(run["deep_supervision"]),
            "result_path": str(result_path),
        }
        for metric in METRICS:
            row[metric] = results.get(metric)
        per_run.append(row)

    if args.require_all_test_results and missing:
        raise FileNotFoundError("Missing test results:\n" + "\n".join(missing))
    if not per_run:
        raise RuntimeError("No test results were found. Evaluate the locked selection first.")

    summaries = variant_summary(per_run)
    effects = paired_factorial_effects(per_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_run, output_dir / "test_per_run.csv")
    write_csv(summaries, output_dir / "test_architecture_summary.csv")
    write_csv(effects, output_dir / "factorial_effects.csv")

    selected = lock["selected_variant"]
    report_lines = [
        "# Validation-locked factorial ablation report",
        "",
        f"Selected architecture before test evaluation: **{selected}**.",
        "",
        "The architecture was selected from validation metrics only. The table below is post-lock test evidence and was not used to choose the model.",
        "",
        markdown_table(summaries, selected),
        "",
        "## Factorial contrasts",
        "",
        "Raw deltas are paired by training seed. Positive is favorable for Dice/IoU/BF1; negative is favorable for HD95/ASSD. With three seeds, these contrasts are descriptive and should not be presented as definitive significance tests.",
        "",
        "| Metric | Effect | Mean raw delta | SD across seeds | Interpretation |",
        "|---|---|---:|---:|---|",
    ]
    for row in effects:
        report_lines.append(
            f"| {row['metric']} | {row['effect']} | {row['mean_raw_delta']:.6f} | "
            f"{row['std_raw_delta']:.6f} | {row['descriptive_interpretation']} |"
        )
    if missing:
        report_lines.extend(
            [
                "",
                "## Missing runs",
                "",
                *[f"- `{path}`" for path in missing],
            ]
        )
    (output_dir / "factorial_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved summaries to: {output_dir}")


if __name__ == "__main__":
    main()
