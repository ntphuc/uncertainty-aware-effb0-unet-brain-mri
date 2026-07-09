"""Generate ablation YAML configs for paper experiments."""
from pathlib import Path
import copy
import yaml

BASE = "configs/efficient_b0_boundary.yaml"
OUT_DIR = Path("configs/ablations")

EXPERIMENTS = {
    "m1_effb0_unet": {
        "model": {"use_multiscale": False, "use_se": False, "use_boundary_head": False, "use_deep_supervision": False},
        "loss": {"lambda_boundary": 0.0, "beta_deep_supervision": 0.0},
    },
    "m2_multiscale": {
        "model": {"use_multiscale": True, "use_se": False, "use_boundary_head": False, "use_deep_supervision": False},
        "loss": {"lambda_boundary": 0.0, "beta_deep_supervision": 0.0},
    },
    "m3_multiscale_se": {
        "model": {"use_multiscale": True, "use_se": True, "use_boundary_head": False, "use_deep_supervision": False},
        "loss": {"lambda_boundary": 0.0, "beta_deep_supervision": 0.0},
    },
    "m4_boundary": {
        "model": {"use_multiscale": True, "use_se": True, "use_boundary_head": True, "use_deep_supervision": False},
        "loss": {"lambda_boundary": 0.15, "beta_deep_supervision": 0.0},
    },
    "m5_full": {
        "model": {"use_multiscale": True, "use_se": True, "use_boundary_head": True, "use_deep_supervision": True},
        "loss": {"lambda_boundary": 0.15, "beta_deep_supervision": 0.25},
    },
}


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            dst.setdefault(k, {})
            deep_update(dst[k], v)
        else:
            dst[k] = v


def main():
    with open(BASE, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, patch in EXPERIMENTS.items():
        cfg = copy.deepcopy(base)
        deep_update(cfg, patch)
        cfg["paths"]["output_dir"] = f"outputs/ablations/{name}"
        out_path = OUT_DIR / f"{name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(out_path)


if __name__ == "__main__":
    main()
