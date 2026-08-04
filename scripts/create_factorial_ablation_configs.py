#!/usr/bin/env python3
"""Generate reviewer-grade factorial ablation configurations.

The design contains one base anchor and a complete 2x2x2 factorial over
Skip-SE (S), boundary supervision (B), and deep supervision (D), with
MultiScale fixed ON in the factorial cells.  This yields:

- one Base anchor (M=0,S=0,B=0,D=0), and
- eight MultiScale cells (M=1, S/B/D each in {0,1}).

All runs share the same validation split seed.  Training seeds alter only model
initialisation and stochastic training, so cell comparisons are paired by seed.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


FACTORIAL_CELLS = [
    # name, M, S, B, D
    ("f0_base", False, False, False, False),
    ("f1_m_only", True, False, False, False),
    ("f2_m_s", True, True, False, False),
    ("f3_m_b", True, False, True, False),
    ("f4_m_d", True, False, False, True),
    ("f5_m_s_b", True, True, True, False),
    ("f6_m_s_d", True, True, False, True),
    ("f7_m_b_d", True, False, True, True),
    ("f8_full_m_s_b_d", True, True, True, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/efficient_b0_boundary.yaml")
    parser.add_argument("--output-root", default="configs/factorial")
    parser.add_argument("--runs-root", default="outputs/factorial")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--boundary-weight", type=float, default=0.15)
    parser.add_argument("--deep-supervision-weight", type=float, default=0.25)
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic algorithms. The default is deterministic=True.",
    )
    return parser.parse_args()


def write_yaml(config: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_config)
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    output_root = Path(args.output_root)
    manifest: List[Dict] = []

    for variant, use_m, use_s, use_b, use_d in FACTORIAL_CELLS:
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["seed"] = int(seed)
            cfg.setdefault("experiment", {})
            cfg["experiment"].update(
                {
                    "family": "validation_locked_factorial_ablation",
                    "variant": variant,
                    "factors": {
                        "multiscale": int(use_m),
                        "skip_se": int(use_s),
                        "boundary": int(use_b),
                        "deep_supervision": int(use_d),
                    },
                    "training_seed": int(seed),
                    "split_seed": int(args.split_seed),
                }
            )

            model = cfg.setdefault("model", {})
            model["use_multiscale"] = bool(use_m)
            model["use_se"] = bool(use_s)
            model["use_boundary_head"] = bool(use_b)
            model["use_deep_supervision"] = bool(use_d)

            loss = cfg.setdefault("loss", {})
            loss["lambda_boundary"] = float(args.boundary_weight) if use_b else 0.0
            loss["beta_deep_supervision"] = (
                float(args.deep_supervision_weight) if use_d else 0.0
            )

            training = cfg.setdefault("training", {})
            training["split_seed"] = int(args.split_seed)
            training["deterministic"] = not args.non_deterministic
            training.setdefault(
                "checkpoint_selection",
                {
                    "primary": {"metric": "dice", "mode": "max", "tolerance": 0.001},
                    "tie_breakers": [
                        {"metric": "missing_prediction_rate", "mode": "min", "tolerance": 0.0},
                        {"metric": "hd95", "mode": "min", "tolerance": 0.0},
                        {"metric": "assd", "mode": "min", "tolerance": 0.0},
                        {"metric": "boundary_f1", "mode": "max", "tolerance": 0.0},
                    ],
                    "min_epoch": 1,
                },
            )

            run_dir = Path(args.runs_root) / variant / f"seed_{seed}"
            cfg.setdefault("paths", {})["output_dir"] = str(run_dir)
            cfg.setdefault("evaluation", {})
            cfg["evaluation"].update(
                {
                    "require_selection_lock_for_test": True,
                    "selection_lock": str(Path(args.runs_root) / "selection" / "SELECTION_LOCK.json"),
                }
            )
            config_path = output_root / variant / f"seed_{seed}.yaml"
            write_yaml(cfg, config_path)
            manifest.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "config": str(config_path),
                    "output_dir": str(run_dir),
                    **cfg["experiment"]["factors"],
                }
            )
            print(config_path)

    manifest_path = output_root / "manifest.yaml"
    write_yaml(
        {
            "base_config": str(base_path),
            "split_seed": int(args.split_seed),
            "seeds": [int(seed) for seed in args.seeds],
            "design": "Base anchor plus complete 2x2x2 factorial over S/B/D with M=1",
            "runs": manifest,
        },
        manifest_path,
    )
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
