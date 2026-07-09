#!/usr/bin/env bash
set -euo pipefail
DEVICE=${DEVICE:-cuda}
CONFIGS=(
  "configs/efficient_b0_balanced_component_v2.yaml"
  "configs/efficient_b0_balanced_component_weighted_v2.yaml"
  "configs/efficient_b0_balanced_component_highres_v2.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  out_dir=$(python - <<PY
import yaml
cfg=yaml.safe_load(open("$cfg"))
print(cfg["paths"]["output_dir"])
PY
)
  ckpt="${out_dir}/checkpoints/best.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skip ${cfg}; checkpoint not found: ${ckpt}"
    continue
  fi
  echo "Evaluate ${cfg} at threshold 0.60"
  python evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --device "${DEVICE}" --threshold 0.60
  echo "Evaluate ${cfg} with multi-scale inference"
  python evaluate.py --config "${cfg}" --checkpoint "${ckpt}" --split test --device "${DEVICE}" --threshold 0.60 --tta-scales 1.0 1.25 1.5
  echo "Sweep thresholds for ${cfg}"
  python scripts/sweep_threshold.py --config "${cfg}" --checkpoint "${ckpt}" --split test --device "${DEVICE}" --thresholds 0.45 0.50 0.55 0.60 0.65 0.70
done
python scripts/collect_priority_results.py || true
