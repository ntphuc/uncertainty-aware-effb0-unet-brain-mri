#!/usr/bin/env python3
"""Collect EfficientNet-B0 U-Net ablation results and export Table 4.

The evaluator writes one metric value per test case and stores the per-case
mean/std in ``test_results.json``. This script keeps the numerical columns for
machine processing and also creates Markdown/LaTeX tables in ``mean ± std``
form.

Variability scope:
    - one checkpoint/seed: standard deviation across test cases;
    - multiple independent seeds should be aggregated separately across seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


METRICS: Tuple[str, ...] = ("dice", "iou", "hd95", "assd", "boundary_f1")
METRIC_HEADERS: Mapping[str, str] = {
    "dice": "Dice↑",
    "iou": "IoU↑",
    "hd95": "HD95↓",
    "assd": "ASSD↓",
    "boundary_f1": "B-F1↑",
}

# Fixed order and paper-facing names for Table 4.
VARIANT_SPECS: Tuple[Tuple[str, str], ...] = (
    ("m1_effb0_unet", "EffB0 U-Net"),
    ("m2_multiscale", "EffB0 U-Net MultiScale (selected)"),
    ("m3_multiscale_se", "EffB0 MultiScale + SE"),
    ("m4_boundary", "EffB0 MultiScale + Boundary"),
    ("m5_full", "EffB0 MultiScale + Boundary + SE"),
)
VARIANT_LABELS = dict(VARIANT_SPECS)
VARIANT_ORDER = {key: index for index, (key, _) in enumerate(VARIANT_SPECS)}


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_deviation(values: Sequence[float], ddof: int) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    return statistics.stdev(finite) if ddof == 1 else statistics.pstdev(finite)


def _read_per_case_metric(csv_path: Path, metric: str) -> List[float]:
    if not csv_path.exists():
        return []
    values: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if metric not in (reader.fieldnames or []):
            return []
        for row in reader:
            value = _finite_float(row.get(metric))
            if value is not None:
                values.append(value)
    return values


def _metric_mean(metrics: Mapping[str, object], metric: str) -> Optional[float]:
    direct = _finite_float(metrics.get(metric))
    if direct is not None:
        return direct
    nested = metrics.get("metric_statistics")
    if isinstance(nested, Mapping):
        metric_stats = nested.get(metric)
        if isinstance(metric_stats, Mapping):
            return _finite_float(metric_stats.get("mean"))
    return None


def _metric_std(
    metrics: Mapping[str, object],
    metric: str,
    per_case_csv: Path,
    ddof: int,
) -> Tuple[Optional[float], str]:
    """Return std and its source label."""
    direct = _finite_float(metrics.get(f"{metric}_std"))
    if direct is not None:
        return direct, "test_results.json"

    nested = metrics.get("metric_statistics")
    if isinstance(nested, Mapping):
        metric_stats = nested.get(metric)
        if isinstance(metric_stats, Mapping):
            nested_std = _finite_float(metric_stats.get("std"))
            if nested_std is not None:
                return nested_std, "metric_statistics"

    values = _read_per_case_metric(per_case_csv, metric)
    fallback = _sample_deviation(values, ddof=ddof)
    if fallback is not None:
        return fallback, "test_per_case_metrics.csv"
    return None, "missing"


def _params_in_millions(value: object) -> Optional[float]:
    number = _finite_float(value)
    if number is None:
        return None
    # count_parameters() returns a raw count; older result files may already
    # contain a value in millions. Preserve both formats safely.
    return number / 1_000_000.0 if number >= 100_000 else number


def _result_files(model_dir: Path, split: str) -> Tuple[Path, Path]:
    eval_dir = model_dir / "eval"
    return (
        eval_dir / f"{split}_results.json",
        eval_dir / f"{split}_per_case_metrics.csv",
    )


def collect_rows(
    ablations_dir: Path,
    split: str = "test",
    ddof: int = 1,
    strict_std: bool = False,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model_dirs = sorted(
        (p for p in ablations_dir.iterdir() if p.is_dir()),
        key=lambda p: (VARIANT_ORDER.get(p.name, 10_000), p.name),
    )

    for model_dir in model_dirs:
        result_path, per_case_csv = _result_files(model_dir, split=split)
        if not result_path.exists():
            continue

        with result_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not isinstance(metrics, MutableMapping):
            raise ValueError(f"Expected a JSON object in {result_path}")

        row: Dict[str, object] = {
            "model": model_dir.name,
            "variant": VARIANT_LABELS.get(model_dir.name, model_dir.name),
            "selected": model_dir.name == "m2_multiscale",
            "params": metrics.get("params"),
            "params_m": _params_in_millions(metrics.get("params")),
            "gflops": _finite_float(metrics.get("gflops")),
            "n_cases": metrics.get("n_cases"),
            "statistics_scope": metrics.get("statistics_scope", "per-case"),
            "std_ddof": metrics.get("std_ddof", ddof),
            "checkpoint_epoch": metrics.get("checkpoint_epoch"),
            "checkpoint_best_dice": metrics.get("checkpoint_best_dice"),
            "json_path": str(result_path),
            "per_case_csv": str(per_case_csv),
        }

        missing_std: List[str] = []
        for metric in METRICS:
            mean = _metric_mean(metrics, metric)
            std, std_source = _metric_std(metrics, metric, per_case_csv, ddof=ddof)
            row[metric] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_std_source"] = std_source
            row[f"{metric}_mean_pm_std"] = format_mean_std(mean, std, decimals=4)
            if std is None:
                missing_std.append(metric)

        if strict_std and missing_std:
            joined = ", ".join(missing_std)
            raise RuntimeError(
                f"Missing standard deviation for {model_dir.name}: {joined}. "
                f"Rerun evaluate.py so {per_case_csv} is generated."
            )
        rows.append(row)

    return rows


def format_number(value: object, decimals: int) -> str:
    number = _finite_float(value)
    return "NA" if number is None else f"{number:.{decimals}f}"


def format_mean_std(mean: object, std: object, decimals: int = 4) -> str:
    mean_value = _finite_float(mean)
    std_value = _finite_float(std)
    if mean_value is None:
        return "NA"
    if std_value is None:
        return f"{mean_value:.{decimals}f} ± NA"
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_fields: List[str] = []
    for metric in METRICS:
        metric_fields.extend(
            [metric, f"{metric}_std", f"{metric}_mean_pm_std", f"{metric}_std_source"]
        )
    fieldnames = [
        "model",
        "variant",
        "selected",
        *metric_fields,
        "params",
        "params_m",
        "gflops",
        "n_cases",
        "statistics_scope",
        "std_ddof",
        "checkpoint_epoch",
        "checkpoint_best_dice",
        "json_path",
        "per_case_csv",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Iterable[Mapping[str, object]]) -> str:
    headers = [
        "Variant",
        *(METRIC_HEADERS[metric] for metric in METRICS),
        "Params(M)↓",
        "GFLOPs↓",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|:--|" + "|".join(["--:" for _ in headers[1:]]) + "|",
    ]
    for row in rows:
        selected = bool(row.get("selected"))
        variant = str(row.get("variant", row.get("model", "")))
        if selected:
            variant = f"**{variant}**"
        values = [variant]
        for metric in METRICS:
            display = str(row.get(f"{metric}_mean_pm_std", "NA"))
            values.append(f"**{display}**" if selected else display)
        values.extend(
            [
                format_number(row.get("params_m"), 2),
                format_number(row.get("gflops"), 2),
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_mean_std(mean: object, std: object, decimals: int = 4, bold: bool = False) -> str:
    mean_value = _finite_float(mean)
    std_value = _finite_float(std)
    if mean_value is None:
        content = r"\mathrm{NA}"
    elif std_value is None:
        content = f"{mean_value:.{decimals}f} \\pm \\mathrm{{NA}}"
    else:
        content = f"{mean_value:.{decimals}f} \\pm {std_value:.{decimals}f}"
    return f"$\\mathbf{{{content}}}$" if bold else f"${content}$"


def latex_table(rows: Iterable[Mapping[str, object]], caption: str, label: str) -> str:
    rows = list(rows)
    scopes = {str(row.get("statistics_scope", "per-case")) for row in rows}
    if scopes == {"across-seeds"}:
        variability_note = (
            r"\footnotesize Values are mean $\pm$ standard deviation across independent seed-level test means."
        )
    elif scopes == {"per-case"}:
        variability_note = (
            r"\footnotesize Values are mean $\pm$ standard deviation across test cases for a single checkpoint/seed."
        )
    else:
        variability_note = (
            r"\footnotesize Values are mean $\pm$ standard deviation; see the CSV statistics_scope column for the aggregation scope."
        )

    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Variant & Dice$\uparrow$ & IoU$\uparrow$ & HD95$\downarrow$ & ASSD$\downarrow$ & B-F1$\uparrow$ & Params(M)$\downarrow$ & GFLOPs$\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        selected = bool(row.get("selected"))
        variant = _latex_escape(str(row.get("variant", row.get("model", ""))))
        if selected:
            variant = rf"\textbf{{{variant}}}"
        cells = [variant]
        for metric in METRICS:
            cells.append(
                _latex_mean_std(
                    row.get(metric),
                    row.get(f"{metric}_std"),
                    decimals=4,
                    bold=selected,
                )
            )
        cells.extend(
            [
                format_number(row.get("params_m"), 2),
                format_number(row.get("gflops"), 2),
            ]
        )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            variability_note,
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect outputs/ablations/*/eval/test_results.json and export "
            "Table 4 with mean ± standard deviation."
        )
    )
    parser.add_argument(
        "--ablations_dir",
        type=Path,
        default=Path("outputs/ablations"),
        help="Directory containing ablation model folders.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train"],
        default="test",
        help="Evaluation split whose result files should be summarized.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/ablations/table4_ablation_with_std.csv"),
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("outputs/ablations/table4_ablation_with_std.md"),
    )
    parser.add_argument(
        "--output_tex",
        type=Path,
        default=Path("outputs/ablations/table4_ablation_with_std.tex"),
    )
    parser.add_argument(
        "--std-ddof",
        type=int,
        choices=[0, 1],
        default=1,
        help="Fallback convention when std must be recomputed from per-case CSV.",
    )
    parser.add_argument(
        "--strict-std",
        action="store_true",
        help="Fail if any metric lacks a standard deviation.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Separate ablation within the EfficientNet-B0 U-Net family. "
            "The MultiScale variant is selected for the subsequent uncertainty experiments."
        ),
    )
    parser.add_argument("--label", default="tab:effb0_ablation_std")
    args = parser.parse_args()

    if not args.ablations_dir.exists():
        raise FileNotFoundError(f"Ablations directory not found: {args.ablations_dir}")

    rows = collect_rows(
        args.ablations_dir,
        split=args.split,
        ddof=args.std_ddof,
        strict_std=args.strict_std,
    )
    if not rows:
        raise RuntimeError(f"No {args.split}_results.json found under: {args.ablations_dir}")

    write_csv(rows, args.output_csv)
    md = markdown_table(rows)
    tex = latex_table(rows, caption=args.caption, label=args.label)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md, encoding="utf-8")
    args.output_tex.write_text(tex, encoding="utf-8")

    print(md)
    print(f"Saved CSV:   {args.output_csv}")
    print(f"Saved MD:    {args.output_md}")
    print(f"Saved LaTeX: {args.output_tex}")


if __name__ == "__main__":
    main()
