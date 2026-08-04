#!/usr/bin/env python3
"""Evaluate locked factorial checkpoints on the untouched test set.

The script verifies SHA-256 hashes from SELECTION_LOCK.json.  It refuses to
run if any configuration/checkpoint changed after validation-only selection.
Use --scope selected for the confirmatory final model.  Use --scope all only
after the architecture has been locked, to produce a transparently labelled
post-lock ablation table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="outputs/factorial/selection/SELECTION_LOCK.json")
    parser.add_argument("--scope", choices=["selected", "all"], default="selected")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def verify_lock(lock: Dict) -> None:
    for record in lock["all_locked_files"]:
        config = Path(record["config"])
        checkpoint = Path(record["checkpoint"])
        if not config.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"Locked file missing: {config} or {checkpoint}")
        if sha256_file(config) != record["config_sha256"]:
            raise RuntimeError(f"Config changed after selection lock: {config}")
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed after selection lock: {checkpoint}")


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    if not lock_path.exists():
        raise FileNotFoundError(
            f"Selection lock not found: {lock_path}. "
            "Run validation training and architecture selection first."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock)

    records: List[Dict] = lock["all_locked_files"]
    if args.scope == "selected":
        selected = lock["selected_variant"]
        records = [record for record in records if record["variant"] == selected]

    for index, record in enumerate(records, start=1):
        config_path = Path(record["config"])
        checkpoint_path = Path(record["checkpoint"])
        # Read output_dir without importing project code in this orchestration script.
        import yaml
        with config_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        result_path = Path(cfg["paths"]["output_dir"]) / "eval" / "test_results.json"
        label = f"{record['variant']} seed={record['seed']}"
        if result_path.exists() and not args.force:
            print(f"[{index}/{len(records)}] SKIP {label}: {result_path} exists")
            continue

        command = [
            sys.executable,
            "evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            args.device,
            "--split",
            "test",
        ]
        print(f"[{index}/{len(records)}] TEST {label}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
            if not result_path.exists():
                raise RuntimeError(f"Evaluation finished but result is missing: {result_path}")

    audit_path = lock_path.parent / f"TEST_EVALUATION_{args.scope.upper()}.json"
    if not args.dry_run:
        audit_path.write_text(
            json.dumps(
                {
                    "selection_lock": str(lock_path),
                    "scope": args.scope,
                    "selected_variant": lock["selected_variant"],
                    "evaluated_runs": [
                        {"variant": record["variant"], "seed": record["seed"]}
                        for record in records
                    ],
                    "interpretation": (
                        "confirmatory selected-model test evaluation"
                        if args.scope == "selected"
                        else "post-lock ablation test evaluation; not used for architecture selection"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Audit record: {audit_path}")


if __name__ == "__main__":
    main()
