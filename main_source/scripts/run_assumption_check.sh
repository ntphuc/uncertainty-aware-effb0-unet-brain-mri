#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Run dataset-centric diagnostics to check the hypothesis:
# "The failed test cases are unlike what the model saw during training."
#
# Usage examples:
#   bash scripts/run_assumption_check.sh
#
#   CONFIG=configs/efficient_b0_hard_negative_v3.yaml \
#   CHECKPOINT=outputs/effb0_hard_negative_v3/checkpoints/best.pth \
#   FAILED_CSV=outputs/effb0_hard_negative_v3/failed_cases.csv \
#   bash scripts/run_assumption_check.sh
#
# Optional:
#   WITH_CROP_NN=1 bash scripts/run_assumption_check.sh
#   THRESHOLD=0.60 bash scripts/run_assumption_check.sh
#   OUT_DIR=outputs/my_assumption_check bash scripts/run_assumption_check.sh
# ================================================================

CONFIG="${CONFIG:-configs/efficient_b0_hard_negative_v3.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/effb0_hard_negative_v3/checkpoints/best.pth}"
FAILED_CSV="${FAILED_CSV:-outputs/effb0_hard_negative_v3/failed_cases.csv}"
THRESHOLD="${THRESHOLD:-0.60}"
OUT_DIR="${OUT_DIR:-outputs/assumption_check}"
TOPK="${TOPK:-5}"
MAX_VIS="${MAX_VIS:-40}"
WITH_CROP_NN="${WITH_CROP_NN:-1}"
DEVICE="${DEVICE:-cpu}"

mkdir -p "${OUT_DIR}"

echo "[1/3] Dataset distribution diagnostic"
CROP_NN_ARG=""
if [[ "${WITH_CROP_NN}" == "1" ]]; then
  CROP_NN_ARG="--with-crop-nn"
fi

# python scripts/dataset_failure_diagnostic.py \
#   --config "${CONFIG}" \
#   --failed-csv "${FAILED_CSV}" \
#   --out-dir "${OUT_DIR}/dataset_diagnostic" \
#   --topk "${TOPK}" \
#   --max-visual-cases 12 \
#   ${CROP_NN_ARG}

echo "[2/3] Prediction probe WITHOUT postprocess"
python scripts/prediction_failure_probe.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --failed-csv "${FAILED_CSV}" \
  --threshold "${THRESHOLD}" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}/prediction_probe_no_postprocess" \
  --no-postprocess \
  --max-vis "${MAX_VIS}"

echo "[3/3] Prediction probe WITH config postprocess"
python scripts/prediction_failure_probe.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --failed-csv "${FAILED_CSV}" \
  --threshold "${THRESHOLD}" \
  --device "${DEVICE}" \
  --out-dir "${OUT_DIR}/prediction_probe_with_postprocess" \
  --max-vis "${MAX_VIS}"

echo "Done. Main files:"
echo "  ${OUT_DIR}/dataset_diagnostic/diagnostic_report.md"
echo "  ${OUT_DIR}/dataset_diagnostic/failed_cases_enriched.csv"
echo "  ${OUT_DIR}/dataset_diagnostic/nearest_train_by_stats.csv"
echo "  ${OUT_DIR}/prediction_probe_no_postprocess/prediction_probe_cases.csv"
echo "  ${OUT_DIR}/prediction_probe_with_postprocess/prediction_probe_cases.csv"
