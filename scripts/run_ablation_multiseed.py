#!/usr/bin/env python3
"""Run Table 4 ablations for multiple seeds and report mean ± std across runs.

This is the recommended mode for paper-level reproducibility. Each seed gets an
independent training output directory, while the validation split can be kept
fixed through ``--split-seed`` so the measured deviation mainly reflects model
training randomness rather than a changing validation partition.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def path_arg(path: Path) -> str:
    """Return a project-relative CLI path when possible, otherwise absolute."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


DEFAULT_CONFIGS = [
    "configs/ablations/m1_effb0_unet.yaml",
    "configs/ablations/m2_multiscale.yaml",
    "configs/ablations/m3_multiscale_se.yaml",
    "configs/ablations/m4_boundary.yaml",
    "configs/ablations/m5_full.yaml",
]
METRICS = ("dice", "iou", "hd95", "assd", "boundary_f1")


def finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def deviation(values: Sequence[float], ddof: int) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values) if ddof == 1 else statistics.pstdev(values))


def run(command: Sequence[str]) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def read_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def write_yaml(data: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False, allow_unicode=True)


def write_json(data: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_result(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def seed_config(
    source_cfg: Mapping[str, object],
    seed: int,
    split_seed: int,
    output_dir: Path,
) -> Dict[str, object]:
    cfg = copy.deepcopy(dict(source_cfg))
    cfg["seed"] = int(seed)

    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("config.training must be a mapping")
    training["split_seed"] = int(split_seed)

    paths = cfg.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    paths["output_dir"] = path_arg(output_dir)
    return cfg


def aggregate_model(
    model_key: str,
    seed_results: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    ddof: int,
    aggregate_root: Path,
    split: str,
) -> Dict[str, object]:
    aggregate: Dict[str, object] = {
        "statistics_scope": "across-seeds",
        "statistics_notation": (
            "mean ± sample standard deviation across seed-level test means"
            if ddof == 1
            else "mean ± population standard deviation across seed-level test means"
        ),
        "std_ddof": int(ddof),
        "n_seeds": len(seed_results),
        "seeds": list(seeds),
    }

    metric_statistics: Dict[str, object] = {}
    for metric in METRICS:
        values = [
            value
            for result in seed_results
            for value in [finite_float(result.get(metric))]
            if value is not None
        ]
        metric_mean = mean(values)
        metric_std = deviation(values, ddof=ddof)
        aggregate[metric] = metric_mean
        aggregate[f"{metric}_std"] = metric_std
        aggregate[f"{metric}_mean_pm_std"] = f"{metric_mean:.6f} ± {metric_std:.6f}"
        metric_statistics[metric] = {
            "mean": metric_mean,
            "std": metric_std,
            "n": len(values),
            "values": values,
            "scope": "across-seeds",
        }

    params_values = [
        value
        for result in seed_results
        for value in [finite_float(result.get("params"))]
        if value is not None
    ]
    gflops_values = [
        value
        for result in seed_results
        for value in [finite_float(result.get("gflops"))]
        if value is not None
    ]
    aggregate["params"] = mean(params_values)
    aggregate["gflops"] = mean(gflops_values)
    aggregate["params_std_across_seeds"] = deviation(params_values, ddof=ddof)
    aggregate["gflops_std_across_seeds"] = deviation(gflops_values, ddof=ddof)
    aggregate["metric_statistics"] = metric_statistics

    result_path = aggregate_root / model_key / "eval" / f"{split}_results.json"
    write_json(aggregate, result_path)
    return aggregate


def write_raw_csv(rows: Iterable[Mapping[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "seed", *METRICS, "params", "gflops", "result_path"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate Table 4 ablations for multiple seeds and aggregate mean ± std."
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed train/validation split seed used for all independent runs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ablations_multiseed"),
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip a train/eval stage when its expected output already exists.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Aggregate available seeds instead of failing on a missing seed result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    generated_configs = output_root / "generated_configs"
    aggregate_root = output_root / "aggregate"

    raw_rows: List[Dict[str, object]] = []
    grouped_results: Dict[str, List[Dict[str, object]]] = {}
    grouped_seeds: Dict[str, List[int]] = {}

    for config_text in args.configs:
        source_path = Path(config_text)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        source_cfg = read_yaml(source_path)
        model_key = source_path.stem
        grouped_results[model_key] = []
        grouped_seeds[model_key] = []

        for seed in args.seeds:
            run_dir = output_root / "runs" / model_key / f"seed_{seed}"
            generated_cfg_path = generated_configs / model_key / f"seed_{seed}.yaml"
            cfg = seed_config(
                source_cfg=source_cfg,
                seed=seed,
                split_seed=args.split_seed,
                output_dir=run_dir,
            )
            write_yaml(cfg, generated_cfg_path)

            checkpoint = run_dir / "checkpoints" / "best.pth"
            result_path = run_dir / "eval" / f"{args.split}_results.json"

            if not args.skip_train:
                if args.reuse_existing and checkpoint.exists():
                    print(f"[REUSE] {checkpoint}")
                else:
                    run(
                        [
                            sys.executable,
                            "train.py",
                            "--config",
                            path_arg(generated_cfg_path),
                            "--device",
                            args.device,
                        ]
                    )

            if not args.skip_eval:
                if not checkpoint.exists():
                    message = f"Checkpoint missing for {model_key}, seed={seed}: {checkpoint}"
                    if args.allow_missing:
                        print(f"[SKIP] {message}")
                        continue
                    raise FileNotFoundError(message)
                if args.reuse_existing and result_path.exists():
                    print(f"[REUSE] {result_path}")
                else:
                    run(
                        [
                            sys.executable,
                            "evaluate.py",
                            "--config",
                            path_arg(generated_cfg_path),
                            "--checkpoint",
                            path_arg(checkpoint),
                            "--split",
                            args.split,
                            "--device",
                            args.device,
                            "--std-ddof",
                            str(args.std_ddof),
                        ]
                    )

            if not result_path.exists():
                message = f"Result missing for {model_key}, seed={seed}: {result_path}"
                if args.allow_missing:
                    print(f"[SKIP] {message}")
                    continue
                raise FileNotFoundError(message)

            result = load_result(result_path)
            grouped_results[model_key].append(result)
            grouped_seeds[model_key].append(seed)
            row: Dict[str, object] = {
                "model": model_key,
                "seed": seed,
                "params": result.get("params"),
                "gflops": result.get("gflops"),
                "result_path": str(result_path),
            }
            for metric in METRICS:
                row[metric] = result.get(metric)
            raw_rows.append(row)

    raw_csv = output_root / "multiseed_raw_results.csv"
    write_raw_csv(raw_rows, raw_csv)

    for model_key, results in grouped_results.items():
        if not results:
            if args.allow_missing:
                continue
            raise RuntimeError(f"No seed results available for {model_key}")
        aggregate_model(
            model_key=model_key,
            seed_results=results,
            seeds=grouped_seeds[model_key],
            ddof=args.std_ddof,
            aggregate_root=aggregate_root,
            split=args.split,
        )

    summary_command = [
        sys.executable,
        "scripts/summarize_ablations.py",
        "--ablations_dir",
        path_arg(aggregate_root),
        "--split",
        args.split,
        "--std-ddof",
        str(args.std_ddof),
        "--strict-std",
        "--output_csv",
        path_arg(output_root / "table4_multiseed.csv"),
        "--output_md",
        path_arg(output_root / "table4_multiseed.md"),
        "--output_tex",
        path_arg(output_root / "table4_multiseed.tex"),
    ]
    run(summary_command)
    print(f"\nSaved raw seed results: {raw_csv}")


if __name__ == "__main__":
    main()
