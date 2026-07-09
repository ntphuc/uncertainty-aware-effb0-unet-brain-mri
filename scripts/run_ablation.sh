#!/usr/bin/env bash
set -e
python scripts/create_ablation_configs.py
for cfg in configs/ablations/*.yaml; do
  echo "Running $cfg"
  python train.py --config "$cfg"
  ckpt=$(python - "$cfg" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
print(cfg['paths']['output_dir'] + '/checkpoints/best.pth')
PY
)
  python evaluate.py --config "$cfg" --checkpoint "$ckpt"
done
