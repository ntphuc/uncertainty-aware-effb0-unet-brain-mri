#!/usr/bin/env bash
set -euo pipefail

# Prompt ablation for Medical Samba / Medical-SAM3 teacher.
# 1) Box prompt only: no text prompt.
python train_student_kd.py --config configs/kd/kd_medical_samba_only_box.yaml

# # 2) Text prompt only: tumor, no box prompt.
# python train_student_kd.py --config configs/kd/kd_medical_samba_only_tumor.yaml

# # 3) Text prompt only: polyp, no box prompt.
# python train_student_kd.py --config configs/kd/kd_medical_samba_only_polyp.yaml

# 4) Text + box main configs.
# python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_tumor_box.yaml
# python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_polyp_box.yaml
