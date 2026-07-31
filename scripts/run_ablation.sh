#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Usage examples:
#   Full retraining + evaluation + Table 4:
#     DEVICE=cuda TRAIN=1 EVAL=1 bash scripts/run_ablation.sh
#
#   Re-evaluate existing checkpoints and rebuild Table 4:
#     DEVICE=cuda TRAIN=0 EVAL=1 bash scripts/run_ablation.sh
#
#   Only rebuild Table 4 from existing JSON/CSV files:
#     TRAIN=0 EVAL=0 bash scripts/run_ablation.sh

DEVICE="${DEVICE:-cuda}"
TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
SPLIT="${SPLIT:-test}"
STD_DDOF="${STD_DDOF:-1}"
STRICT="${STRICT:-1}"

CONFIGS=(
  "configs/ablations/m1_effb0_unet.yaml"
  "configs/ablations/m2_multiscale.yaml"
  "configs/ablations/m3_multiscale_se.yaml"
  "configs/ablations/m4_boundary.yaml"
  "configs/ablations/m5_full.yaml"
)

python scripts/create_ablation_configs.py

for config in "${CONFIGS[@]}"; do
  output_dir="$(python - "$config" <<'PY'
import sys
from pathlib import Path
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(Path(cfg["paths"]["output_dir"]))
PY
)"
  checkpoint="$output_dir/checkpoints/best.pth"

  echo
  echo "============================================================"
  echo "Ablation config: $config"
  echo "Output:          $output_dir"
  echo "============================================================"

  if [[ "$TRAIN" == "1" ]]; then
    python train.py --config "$config" --device "$DEVICE"
  fi

  if [[ "$EVAL" == "1" ]]; then
    if [[ ! -f "$checkpoint" ]]; then
      echo "[MISSING] Checkpoint not found: $checkpoint"
      if [[ "$STRICT" == "1" ]]; then
        exit 3
      fi
      continue
    fi
    python evaluate.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split "$SPLIT" \
      --device "$DEVICE" \
      --std-ddof "$STD_DDOF"
  fi
done

SUMMARY_ARGS=(
  --ablations_dir outputs/ablations
  --split "$SPLIT"
  --std-ddof "$STD_DDOF"
  --output_csv outputs/ablations/table4_ablation_with_std.csv
  --output_md outputs/ablations/table4_ablation_with_std.md
  --output_tex outputs/ablations/table4_ablation_with_std.tex
)
if [[ "$STRICT" == "1" ]]; then
  SUMMARY_ARGS+=(--strict-std)
fi

python scripts/summarize_ablations.py "${SUMMARY_ARGS[@]}"
