#!/usr/bin/env python3
"""Train factorial cells and create validation-selected checkpoints only.

This phase never invokes evaluate.py and never reads X_test/Y_test.  It is safe
to resume: runs with validation_best.json are skipped unless --force is used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/factorial/manifest.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)

    selected_variants = set(args.variants or [])
    selected_seeds = set(args.seeds or [])
    runs = []
    for run in manifest["runs"]:
        if selected_variants and run["variant"] not in selected_variants:
            continue
        if selected_seeds and int(run["seed"]) not in selected_seeds:
            continue
        runs.append(run)

    if not runs:
        raise RuntimeError("No runs matched the requested variants/seeds.")

    for index, run in enumerate(runs, start=1):
        config = Path(run["config"])
        output_dir = Path(run["output_dir"])
        done_marker = output_dir / "validation_best.json"
        label = f"{run['variant']} seed={run['seed']}"
        if done_marker.exists() and not args.force:
            print(f"[{index}/{len(runs)}] SKIP {label}: {done_marker} exists")
            continue

        command = [sys.executable, "train.py", "--config", str(config), "--device", args.device]
        print(f"[{index}/{len(runs)}] TRAIN {label}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
            if not done_marker.exists():
                raise RuntimeError(f"Training finished but marker is missing: {done_marker}")

    print("Validation-only training phase complete.")
    print("Next: python scripts/select_architecture_from_validation.py")


if __name__ == "__main__":
    main()
