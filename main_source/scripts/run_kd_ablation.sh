#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
#   bash scripts/run_kd_ablation.sh
# Edit the teacher repo/checkpoint fields in configs/kd/*.yaml before running KD configs.

python train_student_kd.py --config configs/kd/student_no_kd.yaml
python train_student_kd.py --config configs/kd/kd_medical_samba_only_box.yaml
# python train_student_kd.py --config configs/kd/ablation_vanilla_kd.yaml
# python train_student_kd.py --config configs/kd/ablation_edge_kd.yaml
# python train_student_kd.py --config configs/kd/ablation_selective_uncertainty_kd.yaml



