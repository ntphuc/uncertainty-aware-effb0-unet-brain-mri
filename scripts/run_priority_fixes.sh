#!/usr/bin/env bash
set -euo pipefail

# Priority script after component-crop did not improve enough.
# Order:
# 0) Re-evaluate old trained model at better threshold 0.60 (no retrain)
# 1) Train balanced component crop v2
# 2) Train balanced component crop + component-weighted loss v2
# 3) Optional high-resolution run for tiny lesions
# 4) Sweep thresholds and evaluate optional multi-scale inference

DEVICE=${DEVICE:-cuda}
RUN_HIGHRES=${RUN_HIGHRES:-0}   # set RUN_HIGHRES=1 to run the high-res experiment

OLD_CONFIG="configs/efficient_b0_recall_ms_soft_boundary_component_ftv.yaml"
OLD_CKPT="outputs/effb0_recall_ms_soft_boundary_component_ftv/checkpoints/best.pth"

BALANCED_CONFIG="configs/efficient_b0_balanced_component_v2.yaml"
WEIGHTED_CONFIG="configs/efficient_b0_balanced_component_weighted_v2.yaml"
HIGHRES_CONFIG="configs/efficient_b0_balanced_component_highres_v2.yaml"

run_eval_and_sweep () {
  local cfg=$1
  local out_dir
  out_dir=$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$cfg"))
print(cfg["paths"]["output_dir"])
PY
)
  local ckpt="${out_dir}/checkpoints/best.pth"
  echo "\n[EVAL] ${cfg}"
  python evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --device "${DEVICE}" --threshold 0.60

  echo "\n[THRESHOLD SWEEP] ${cfg}"
  python scripts/sweep_threshold.py \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --split test \
    --device "${DEVICE}" \
    --thresholds 0.45 0.50 0.55 0.60 0.65 0.70

  echo "\n[MULTI-SCALE INFERENCE CHECK] ${cfg}"
  python evaluate.py \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --split test \
    --device "${DEVICE}" \
    --threshold 0.60 \
    --tta-scales 1.0 1.25 1.5
}

echo "========== PRIORITY 0: no-retrain threshold re-check on old model =========="
if [[ -f "${OLD_CKPT}" ]]; then
  python evaluate.py --config "${OLD_CONFIG}" --checkpoint "${OLD_CKPT}" --split test --device "${DEVICE}" --threshold 0.60
  python scripts/sweep_threshold.py \
    --config "${OLD_CONFIG}" \
    --checkpoint "${OLD_CKPT}" \
    --split test \
    --device "${DEVICE}" \
    --thresholds 0.45 0.50 0.55 0.60 0.65 0.70
else
  echo "Skip old-model re-check because checkpoint not found: ${OLD_CKPT}"
fi

echo "\n========== PRIORITY 1: train balanced component crop v2 =========="
python train.py --config "${BALANCED_CONFIG}" --device "${DEVICE}"
run_eval_and_sweep "${BALANCED_CONFIG}"

echo "\n========== PRIORITY 2: train component-weighted loss v2 =========="
python train.py --config "${WEIGHTED_CONFIG}" --device "${DEVICE}"
run_eval_and_sweep "${WEIGHTED_CONFIG}"

if [[ "${RUN_HIGHRES}" == "1" ]]; then
  echo "\n========== PRIORITY 3: train high-resolution v2 =========="
  python train.py --config "${HIGHRES_CONFIG}" --device "${DEVICE}"
  run_eval_and_sweep "${HIGHRES_CONFIG}"
else
  echo "\nSkip high-resolution run. To enable: RUN_HIGHRES=1 bash scripts/run_priority_fixes.sh"
fi

python scripts/collect_priority_results.py || true
