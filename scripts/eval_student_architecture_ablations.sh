#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
STRICT="${STRICT:-0}"
STD_DDOF="${STD_DDOF:-1}"

CONFIGS=(
  "configs/ablations/m1_effb0_unet.yaml"
  "configs/ablations/m2_multiscale.yaml"
  "configs/ablations/m3_multiscale_se.yaml"
  "configs/ablations/m4_boundary.yaml"
  "configs/ablations/m5_full.yaml"
)

for config in "${CONFIGS[@]}"; do
  checkpoint="$(python - "$config" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(Path(cfg['paths']['output_dir']) / 'checkpoints' / 'best.pth')
PY
)"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] Missing architecture-ablation checkpoint: $checkpoint"
    if [[ "$STRICT" == "1" ]]; then exit 3; fi
    continue
  fi

  echo
  echo "============================================================"
  echo "Architecture ablation evaluation"
  echo "Config:     $config"
  echo "Checkpoint: $checkpoint"
  echo "============================================================"
  python evaluate.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --std-ddof "$STD_DDOF"
done

if find outputs/ablations -path '*/eval/test_results.json' -print -quit 2>/dev/null | grep -q .; then
  python scripts/summarize_ablations.py \
    --ablations_dir outputs/ablations \
    --std-ddof "$STD_DDOF" \
    --output_csv outputs/ablations/table4_ablation_with_std.csv \
    --output_md outputs/ablations/table4_ablation_with_std.md \
    --output_tex outputs/ablations/table4_ablation_with_std.tex
else
  echo "No architecture-ablation evaluation results found."
fi
