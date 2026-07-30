#!/usr/bin/env python3
"""Train/evaluate all non-proxy paper baselines and export publication tables.

Examples
--------
Run all five models with the YAML defaults:
    python scripts/run_paper_baselines.py --stage train_eval

Run three seeds and aggregate mean ± std:
    python scripts/run_paper_baselines.py --stage train_eval --seeds 42 123 2025

Evaluate existing checkpoints only:
    python scripts/run_paper_baselines.py --stage eval --seeds 42

Quick smoke experiment:
    python scripts/run_paper_baselines.py --stage train_eval --epochs 2 --models unet attention_unet
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS: Dict[str, Path] = {
    "unet": Path("configs/paper_baselines/unet_original.yaml"),
    "unetpp": Path("configs/paper_baselines/unetpp_original.yaml"),
    "attention_unet": Path("configs/paper_baselines/attention_unet_original.yaml"),
    "deeplabv3plus": Path("configs/paper_baselines/deeplabv3plus_resnet50.yaml"),
    "umamba_bot": Path("configs/paper_baselines/umamba_bot_official.yaml"),
    "umamba_enc": Path("configs/paper_baselines/umamba_enc_official.yaml"),
    "vmunet": Path("configs/paper_baselines/vmunet_official.yaml"),
}
METRICS = ["dice", "iou", "precision", "recall", "f2", "hd95", "assd", "boundary_f1"]
HIGHER_IS_BETTER = {"dice", "iou", "precision", "recall", "f2", "boundary_f1"}


@dataclass
class RunRecord:
    model_key: str
    model: str
    seed: int
    config: str
    output_dir: str
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--models", nargs="+", choices=list(DEFAULT_CONFIGS), default=list(DEFAULT_CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override every config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override physical batch size")
    parser.add_argument("--accumulation-steps", type=int, default=None, help="Override gradient accumulation")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-root", default="outputs/paper_baselines")
    parser.add_argument("--skip-trained", action="store_true", help="Skip training when best.pth already exists")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with other models after a failure")
    parser.add_argument("--no-postprocess", action="store_true", help="Force raw-model evaluation")
    parser.add_argument("--tta-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=1, help="Per-case std convention used by evaluate.py")
    return parser.parse_args()


def run_command(command: Sequence[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def _command_text(command: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            list(command), cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def collect_environment() -> dict:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = None
    try:
        import thop
        thop_version = getattr(thop, "__version__", None)
    except Exception:
        thop_version = None
    try:
        import mamba_ssm
        mamba_ssm_version = getattr(mamba_ssm, "__version__", None)
    except Exception:
        mamba_ssm_version = None

    vm_repo = PROJECT_ROOT / "external" / "VM-UNet"
    vm_commit = None
    if (vm_repo / ".git").exists():
        vm_commit = _command_text(["git", "-C", str(vm_repo), "rev-parse", "HEAD"])

    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None

    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "thop": thop_version,
        "mamba_ssm": mamba_ssm_version,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "nvidia_smi": _command_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "official_vmunet_commit": vm_commit,
    }


def load_runtime_config(model_key: str, seed: int, args: argparse.Namespace) -> Tuple[dict, Path]:
    source_path = PROJECT_ROOT / DEFAULT_CONFIGS[model_key]
    cfg = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    cfg["seed"] = int(seed)
    tcfg = cfg.setdefault("training", {})
    if args.epochs is not None:
        tcfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        tcfg["batch_size"] = int(args.batch_size)
    if args.accumulation_steps is not None:
        tcfg["accumulation_steps"] = int(args.accumulation_steps)
    if args.num_workers is not None:
        tcfg["num_workers"] = int(args.num_workers)
    if args.image_size is not None:
        tcfg["image_size"] = int(args.image_size)
    if args.threshold is not None:
        tcfg["threshold"] = float(args.threshold)

    output_dir = Path(args.output_root) / model_key / f"seed_{seed}"
    cfg.setdefault("paths", {})["output_dir"] = str(output_dir)
    runtime_dir = Path(args.output_root) / "runtime_configs"
    runtime_path = runtime_dir / f"{model_key}_seed_{seed}.yaml"
    absolute_runtime_path = PROJECT_ROOT / runtime_path
    absolute_runtime_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_runtime_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, runtime_path


def validate_dataset_paths(cfg: Mapping) -> None:
    missing = []
    for key in ("x_train", "y_train", "x_test", "y_test"):
        path = PROJECT_ROOT / cfg["paths"][key]
        if not path.exists():
            missing.append(str(path))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "BRISC2025 preprocessed .npy files are missing:\n"
            f"  - {joined}\n"
            "Run the repository's data download/preprocessing scripts first."
        )


def execute_one(model_key: str, seed: int, args: argparse.Namespace) -> RunRecord:
    cfg, runtime_path = load_runtime_config(model_key, seed, args)
    validate_dataset_paths(cfg)
    model_name = cfg.get("paper_name", cfg.get("project_name", model_key))
    output_dir = Path(cfg["paths"]["output_dir"])
    checkpoint = output_dir / "checkpoints" / "best.pth"

    record = RunRecord(
        model_key=model_key,
        model=str(model_name),
        seed=seed,
        config=str(runtime_path),
        output_dir=str(output_dir),
        status="pending",
    )

    if args.stage in ("train", "train_eval"):
        if args.skip_trained and (PROJECT_ROOT / checkpoint).exists():
            print(f"Skip training existing checkpoint: {checkpoint}")
        else:
            run_command([
                sys.executable,
                "train.py",
                "--config",
                str(runtime_path),
                "--device",
                args.device,
            ])

    if args.stage in ("eval", "train_eval"):
        if not (PROJECT_ROOT / checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        command = [
            sys.executable,
            "evaluate.py",
            "--config",
            str(runtime_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--split",
            "test",
            "--tta-scales",
            *[str(x) for x in args.tta_scales],
            "--std-ddof",
            str(args.std_ddof),
        ]
        if args.no_postprocess:
            command.append("--no-postprocess")
        run_command(command)

    record.status = "ok"
    return record


def read_result(record: RunRecord) -> Optional[dict]:
    result_path = PROJECT_ROOT / record.output_dir / "eval" / "test_results.json"
    if not result_path.exists():
        return None
    data = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load((PROJECT_ROOT / record.config).read_text(encoding="utf-8"))
    row = {
        "model_key": record.model_key,
        "model": record.model,
        "seed": record.seed,
        "config": record.config,
        "output_dir": record.output_dir,
        "implementation": cfg.get("implementation", ""),
        "citation_key": cfg.get("citation_key", ""),
        "image_size": cfg.get("training", {}).get("image_size"),
        "split_seed": cfg.get("training", {}).get("split_seed", cfg.get("seed", 42)),
        "physical_batch_size": cfg.get("training", {}).get("batch_size"),
        "accumulation_steps": cfg.get("training", {}).get("accumulation_steps", 1),
        "effective_batch_size": int(cfg.get("training", {}).get("batch_size", 1))
        * int(cfg.get("training", {}).get("accumulation_steps", 1)),
        "epochs": cfg.get("training", {}).get("epochs"),
        "checkpoint_epoch": data.get("checkpoint_epoch"),
        "params": data.get("params"),
        "params_m": None if data.get("params") is None else data.get("params") / 1e6,
        "gflops": data.get("gflops"),
        "threshold": data.get("threshold"),
    }
    for metric in METRICS:
        row[metric] = data.get(metric)
        row[f"{metric}_case_std"] = data.get(f"{metric}_std")
        row[f"{metric}_case_sem"] = data.get(f"{metric}_sem")
        row[f"{metric}_case_ci95_half_width"] = data.get(f"{metric}_ci95_half_width")
    return row


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def aggregate_results(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate seed means and keep the ± scope explicit.

    * Multiple seeds: mean ± sample std across seed-level test means.
    * One seed: mean ± per-case sample std produced by evaluate.py.

    The fallback makes a one-seed run immediately useful while avoiding the
    common mistake of presenting case variability as run-to-run variability.
    """
    numeric_rows = []
    display_rows = []
    for (model_key, model), group in raw.groupby(["model_key", "model"], sort=False):
        n_seeds = int(len(group))
        scope = "across seeds" if n_seeds > 1 else "across test cases"
        numeric = {
            "model_key": model_key,
            "model": model,
            "n_seeds": n_seeds,
            "variability_scope": scope,
        }
        display = {"Model": model, "Seeds": n_seeds, "± scope": scope}
        for metric in METRICS + ["params_m", "gflops"]:
            vals = finite_values(group[metric]) if metric in group else np.array([])
            mean = float(vals.mean()) if len(vals) else math.nan
            if len(vals) > 1:
                std = float(vals.std(ddof=1))
            elif len(vals) == 1 and metric in METRICS:
                case_std_col = f"{metric}_case_std"
                case_std_values = finite_values(group[case_std_col]) if case_std_col in group else np.array([])
                std = float(case_std_values[0]) if len(case_std_values) else math.nan
            elif len(vals) == 1:
                std = 0.0
            else:
                std = math.nan

            numeric[f"{metric}_mean"] = mean
            numeric[f"{metric}_std"] = std
            numeric[f"{metric}_std_scope"] = scope if metric in METRICS else "across seeds"

            if not np.isfinite(mean):
                display_value = "NA"
            elif metric in {"params_m", "gflops"}:
                display_value = f"{mean:.3f}" if len(vals) == 1 else f"{mean:.3f} ± {std:.3f}"
            elif metric in {"hd95", "assd"}:
                display_value = f"{mean:.3f} ± {std:.3f}" if np.isfinite(std) else f"{mean:.3f}"
            else:
                display_value = f"{mean:.4f} ± {std:.4f}" if np.isfinite(std) else f"{mean:.4f}"
            label = {
                "dice": "Dice ↑",
                "iou": "IoU ↑",
                "precision": "Precision ↑",
                "recall": "Recall ↑",
                "f2": "F2 ↑",
                "hd95": "HD95 ↓",
                "assd": "ASSD ↓",
                "boundary_f1": "Boundary F1 ↑",
                "params_m": "Params (M) ↓",
                "gflops": "GFLOPs ↓",
            }[metric]
            display[label] = display_value
        numeric_rows.append(numeric)
        display_rows.append(display)
    return pd.DataFrame(numeric_rows), pd.DataFrame(display_rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No results.\n"
    columns = list(df.columns)
    rows = [[str(v) for v in df.iloc[i].tolist()] for i in range(len(df))]
    widths = [len(str(c)) for c in columns]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("±", r"$\pm$").replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")


def latex_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "% No results.\n"
    columns = list(df.columns)
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{BRISC2025 test-set segmentation results. The variability scope is stated explicitly: across seeds for repeated runs, or across test cases for a single-seed run.}",
        r"\label{tab:brisc2025_baselines}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(str(c)) for c in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(str(v)) for v in row.tolist()) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_outputs(records: Sequence[RunRecord], args: argparse.Namespace) -> None:
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps([record.__dict__ for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    environment_path = output_root / "environment.json"
    environment_path.write_text(
        json.dumps(collect_environment(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result_rows = [row for record in records if record.status == "ok" for row in [read_result(record)] if row is not None]
    if not result_rows:
        print(f"No evaluation JSON files found. Manifest: {manifest_path}")
        return

    raw = pd.DataFrame(result_rows)
    numeric_summary, display_summary = aggregate_results(raw)

    raw_path = output_root / "paper_results_raw.csv"
    summary_path = output_root / "paper_results_summary.csv"
    markdown_path = output_root / "paper_table.md"
    latex_path = output_root / "paper_table.tex"
    raw.to_csv(raw_path, index=False)
    numeric_summary.to_csv(summary_path, index=False)
    markdown_path.write_text(markdown_table(display_summary), encoding="utf-8")
    latex_path.write_text(latex_table(display_summary), encoding="utf-8")

    print("\nBRISC2025 PAPER BASELINES")
    print(markdown_table(display_summary))
    print("Saved:")
    for path in (raw_path, summary_path, markdown_path, latex_path, manifest_path, environment_path):
        print(f"  {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    args = parse_args()
    records: List[RunRecord] = []
    for model_key in args.models:
        for seed in args.seeds:
            try:
                record = execute_one(model_key, seed, args)
            except Exception as exc:
                cfg_path = str(DEFAULT_CONFIGS[model_key])
                record = RunRecord(
                    model_key=model_key,
                    model=model_key,
                    seed=seed,
                    config=cfg_path,
                    output_dir=str(Path(args.output_root) / model_key / f"seed_{seed}"),
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                records.append(record)
                print(f"\nFAILED [{model_key}, seed={seed}]: {record.error}", file=sys.stderr)
                if not args.continue_on_error:
                    write_outputs(records, args)
                    raise
            else:
                records.append(record)
    write_outputs(records, args)


if __name__ == "__main__":
    main()
