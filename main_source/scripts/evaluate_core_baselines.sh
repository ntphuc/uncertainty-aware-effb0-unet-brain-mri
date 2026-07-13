#!/usr/bin/env bash
set -e

# Evaluate all models trained by run_core_baselines.sh.
# Usage:
#   bash scripts/evaluate_core_baselines.sh
#   DEVICE=cuda bash scripts/evaluate_core_baselines.sh

DEVICE=${DEVICE:-$(python - <<EOF
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
EOF
)}
CONFIGS=(
  configs/baseline_unet.yaml
  configs/baseline_unetpp.yaml
  configs/baseline_transunet.yaml
  configs/baseline_umamba.yaml
  configs/baseline_vmunet.yaml
  configs/efficient_b0_multiscale_no_se.yaml
  configs/efficient_b0_multiscale_boundary_no_se.yaml
)

for cfg in "${CONFIGS[@]}"; do
  out_dir=$(python - <<PY
import yaml
cfg = yaml.safe_load(open('$cfg'))
print(cfg['paths']['output_dir'])
PY
)
  ckpt="$out_dir/checkpoints/best.pth"
  echo "============================================"
  echo "Evaluating $cfg"
  echo "Checkpoint: $ckpt"
  echo "============================================"
  python evaluate.py --config "$cfg" --checkpoint "$ckpt" --device "$DEVICE" --split test
done

python scripts/collect_eval_results.py
