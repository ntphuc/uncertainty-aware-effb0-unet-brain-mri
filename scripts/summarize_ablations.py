#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def collect_rows(ablations_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for model_dir in sorted([p for p in ablations_dir.iterdir() if p.is_dir()]):
        result_path = model_dir / "eval" / "test_results.json"
        if not result_path.exists():
            continue

        with result_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        row: Dict[str, object] = {
            "model": model_dir.name,
            "dice": metrics.get("dice"),
            "iou": metrics.get("iou"),
            "hd95": metrics.get("hd95"),
            "assd": metrics.get("assd"),
            "boundary_f1": metrics.get("boundary_f1"),
            "params": metrics.get("params"),
            "gflops": metrics.get("gflops"),
            "checkpoint_epoch": metrics.get("checkpoint_epoch"),
            "checkpoint_best_dice": metrics.get("checkpoint_best_dice"),
            "json_path": str(result_path),
        }
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "dice",
        "iou",
        "hd95",
        "assd",
        "boundary_f1",
        "params",
        "gflops",
        "checkpoint_epoch",
        "checkpoint_best_dice",
        "json_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect outputs/ablations/*/eval/test_results.json into one CSV summary."
    )
    parser.add_argument(
        "--ablations_dir",
        type=Path,
        default=Path("outputs/ablations"),
        help="Directory containing ablation model folders.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/ablations/all_summary.csv"),
        help="Path to output summary CSV.",
    )

    args = parser.parse_args()

    if not args.ablations_dir.exists():
        raise FileNotFoundError(f"Ablations directory not found: {args.ablations_dir}")

    rows = collect_rows(args.ablations_dir)
    if not rows:
        raise RuntimeError(
            f"No test_results.json found under: {args.ablations_dir}"
        )

    write_csv(rows, args.output_csv)
    print(f"Saved {len(rows)} rows to: {args.output_csv}")


if __name__ == "__main__":
    main()
