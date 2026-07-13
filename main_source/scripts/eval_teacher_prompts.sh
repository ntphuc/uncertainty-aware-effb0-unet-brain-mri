#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STRICT="${STRICT:-0}"

CONFIGS=(
  "configs/kd/kd_medical_sam3_checkpoint3d_tumor_box.yaml"
  "configs/kd/kd_medical_samba_only_box.yaml"
  "configs/kd/kd_medical_samba_only_tumor.yaml"
  "configs/kd/kd_medical_samba_only_polyp.yaml"
  "configs/kd/student_kd_medical_samba_text_polyp_box.yaml"
)

python scripts/check_medical_sam3_setup.py

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "[SKIP] Missing config: $config"
    if [[ "$STRICT" == "1" ]]; then exit 3; fi
    continue
  fi
  echo
  echo "============================================================"
  echo "Teacher evaluation: $config"
  echo "============================================================"
  python evaluate_teacher.py \
    --config "$config" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
done
