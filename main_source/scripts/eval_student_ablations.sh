#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
STRICT="${STRICT:-0}"

CONFIGS=(
  "configs/kd/student_no_kd.yaml"
  "configs/kd/kd_medical_samba_only_box.yaml"
)

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "[SKIP] Missing config: $config"
    if [[ "$STRICT" == "1" ]]; then exit 2; fi
    continue
  fi

  checkpoint="$(python - "$config" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(Path(cfg['paths']['output_dir']) / 'checkpoints' / 'best.pth')
PY
)"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[SKIP] Checkpoint not found: $checkpoint"
    if [[ "$STRICT" == "1" ]]; then exit 3; fi
    continue
  fi

  echo
  echo "============================================================"
  echo "Student ablation evaluation"
  echo "Config:     $config"
  echo "Checkpoint: $checkpoint"
  echo "============================================================"

  python evaluate.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split "$SPLIT" \
    --device "$DEVICE"
done

python scripts/collect_kd_eval_results.py
