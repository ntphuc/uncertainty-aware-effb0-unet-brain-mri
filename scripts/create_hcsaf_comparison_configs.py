#!/usr/bin/env python3
"""Create validation-locked Concat/RGAMF/HCSAF-BR comparison configs.

The generated manifest is compatible with:
- scripts/run_factorial_train_validation.py
- scripts/select_architecture_from_validation.py
- scripts/evaluate_factorial_after_lock.py

All variants keep Skip-SE, boundary supervision, and deep supervision enabled;
only the multi-scale fusion/upsampling/refinement design changes.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/efficient_b0_hcsaf_br_boundary.yaml",
    )
    parser.add_argument("--output-root", default="configs/hcsaf_comparison")
    parser.add_argument("--runs-root", default="outputs/hcsaf_comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def variant_table():
    return [
        (
            "a0_concat",
            {
                "name": "efficientnet_b0_unet_boundary",
                "fusion_type": "concat",
            },
        ),
        (
            "a1_rgamf",
            {
                "name": "efficientnet_b0_unet_rgamf",
                "fusion_type": "rgamf",
            },
        ),
        (
            "a2_hcsaf_channel",
            {
                "name": "efficientnet_b0_unet_hcsaf_br",
                "fusion_type": "hcsaf_br",
                "hcsaf_spatial_stage_indices": [],
                "hcsaf_learned_upsample_stage_indices": [],
                "hcsaf_use_learned_final_upsample": False,
                "hcsaf_use_boundary_refinement": False,
            },
        ),
        (
            "a3_hcsaf_spatial",
            {
                "name": "efficientnet_b0_unet_hcsaf_br",
                "fusion_type": "hcsaf_br",
                "hcsaf_spatial_stage_indices": [2, 3],
                "hcsaf_learned_upsample_stage_indices": [],
                "hcsaf_use_learned_final_upsample": False,
                "hcsaf_use_boundary_refinement": False,
            },
        ),
        (
            "a4_hcsaf_spatial_up",
            {
                "name": "efficientnet_b0_unet_hcsaf_br",
                "fusion_type": "hcsaf_br",
                "hcsaf_spatial_stage_indices": [2, 3],
                "hcsaf_learned_upsample_stage_indices": [3],
                "hcsaf_use_learned_final_upsample": True,
                "hcsaf_use_boundary_refinement": False,
            },
        ),
        (
            "a5_hcsaf_br_no_guide_loss",
            {
                "name": "efficientnet_b0_unet_hcsaf_br",
                "fusion_type": "hcsaf_br",
                "hcsaf_spatial_stage_indices": [2, 3],
                "hcsaf_learned_upsample_stage_indices": [3],
                "hcsaf_use_learned_final_upsample": True,
                "hcsaf_use_boundary_refinement": True,
                "_lambda_boundary_guide": 0.0,
            },
        ),
        (
            "a6_hcsaf_br_full",
            {
                "name": "efficientnet_b0_unet_hcsaf_br",
                "fusion_type": "hcsaf_br",
                "hcsaf_spatial_stage_indices": [2, 3],
                "hcsaf_learned_upsample_stage_indices": [3],
                "hcsaf_use_learned_final_upsample": True,
                "hcsaf_use_boundary_refinement": True,
                "_lambda_boundary_guide": 0.05,
            },
        ),
    ]


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_config)
    output_root = Path(args.output_root)
    runs_root = Path(args.runs_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)

    runs = []
    for variant, overrides in variant_table():
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["project_name"] = f"HCSAF_comparison_{variant}_seed_{seed}"
            cfg["seed"] = int(seed)
            cfg["training"]["split_seed"] = int(args.split_seed)
            cfg["paths"]["output_dir"] = str(runs_root / variant / f"seed_{seed}")
            model_overrides = {
                key: value for key, value in overrides.items() if not key.startswith("_")
            }
            cfg["model"].update(model_overrides)
            if "_lambda_boundary_guide" in overrides:
                cfg["loss"]["lambda_boundary_guide"] = float(
                    overrides["_lambda_boundary_guide"]
                )

            variant_dir = output_root / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            config_path = variant_dir / f"seed_{seed}.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)

            runs.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "config": str(config_path),
                    "output_dir": str(runs_root / variant / f"seed_{seed}"),
                    # Compatibility fields expected by the existing selection script.
                    "multiscale": 1,
                    "skip_se": 1,
                    "boundary": 1,
                    "deep_supervision": 1,
                    "fusion_design": overrides["fusion_type"],
                }
            )

    manifest = {
        "base_config": str(base_path),
        "split_seed": int(args.split_seed),
        "seeds": [int(seed) for seed in args.seeds],
        "design": "Concat -> RGAMF -> channel HCSAF -> spatial HCSAF -> learned upsampling -> HCSAF-BR",
        "runs": runs,
    }
    manifest_path = output_root / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    print(f"Created {len(runs)} configs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
