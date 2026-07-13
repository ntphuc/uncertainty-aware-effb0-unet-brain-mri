#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/kd/student_no_kd.yaml}"
CHECKPOINT="${2:-}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(python - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(Path(cfg['paths']['output_dir']) / 'checkpoints' / 'best.pth')
PY
)"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Train first, for example:" >&2
  echo "  python train_student_kd.py --config $CONFIG" >&2
  exit 3
fi

python evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --device "$DEVICE"
