#!/usr/bin/env bash
set -euo pipefail

# Uncertainty-aware failure explanation pipeline.
# Override these env vars when needed:
#   CONFIG=...
#   CHECKPOINT=...
#   DEVICE=cuda or cpu
#   SPLIT=test
#   THRESHOLD=0.50
#   OUT_DIR=...
#   RUN_MC=1

CONFIG=${CONFIG:-configs/efficient_b0_multiscale_no_se.yaml}
CHECKPOINT=${CHECKPOINT:-outputs/effb0_multiscale_no_se/checkpoints/best.pth}
DEVICE=${DEVICE:-cpu}
SPLIT=${SPLIT:-test}
THRESHOLD=${THRESHOLD:-0.50}
OUT_DIR=${OUT_DIR:-outputs/uncertainty_analysis}
SAVE_TOP_K=${SAVE_TOP_K:-30}
TRAIN_REF_MAX=${TRAIN_REF_MAX:-2000}

mkdir -p "${OUT_DIR}"

echo "[1/3] Running main uncertainty analysis: entropy + TTA variance + train-coverage distance"
python scripts/uncertainty_failure_analysis.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --split "${SPLIT}" \
  --threshold "${THRESHOLD}" \
  --tta-scales 1.0 1.25 1.5 \
  --train-coverage \
  --train-ref-max-samples "${TRAIN_REF_MAX}" \
  --save-top-k "${SAVE_TOP_K}" \
  --out-dir "${OUT_DIR}/tta_train_distance" \
  --language en

if [[ "${RUN_MC:-0}" == "1" ]]; then
  echo "[2/3] Running optional MC Dropout + TTA uncertainty analysis"
  python scripts/uncertainty_failure_analysis.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --device "${DEVICE}" \
    --split "${SPLIT}" \
    --threshold "${THRESHOLD}" \
    --tta-scales 1.0 1.25 1.5 \
    --mc-runs 8 \
    --train-coverage \
    --train-ref-max-samples "${TRAIN_REF_MAX}" \
    --save-top-k "${SAVE_TOP_K}" \
    --out-dir "${OUT_DIR}/mc_dropout_tta_train_distance" \
    --language en
else
  echo "[2/3] Skipping MC Dropout. Set RUN_MC=1 to enable it."
fi

echo "[3/3] Running raw probability/entropy only baseline uncertainty"
python scripts/uncertainty_failure_analysis.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --split "${SPLIT}" \
  --threshold "${THRESHOLD}" \
  --tta-scales 1.0 \
  --no-hflip \
  --save-top-k "${SAVE_TOP_K}" \
  --out-dir "${OUT_DIR}/entropy_only_baseline" \
  --language en

echo "[DONE] All uncertainty analyses saved under: ${OUT_DIR}"
echo "Open: ${OUT_DIR}/tta_train_distance/uncertainty_report.md"
echo "Open: ${OUT_DIR}/tta_train_distance/uncertainty_cases.csv"
echo "Open: ${OUT_DIR}/tta_train_distance/visualizations/"
