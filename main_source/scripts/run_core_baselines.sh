#!/usr/bin/env bash
set -e

# Train all core baselines and the proposed no-SE variants.
# Usage:
#   bash scripts/run_core_baselines.sh
#   DEVICE=cuda bash scripts/run_core_baselines.sh

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
  echo "============================================"
  echo "Training $cfg"
  echo "============================================"
  python train.py --config "$cfg" --device "$DEVICE"
done
