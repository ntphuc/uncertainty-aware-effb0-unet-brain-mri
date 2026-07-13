#!/usr/bin/env bash
set -euo pipefail

# Tumor text prompt + box prompt. This is the main config for brain tumor MRI.
python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_tumor_box.yaml

# Polyp text prompt + box prompt. Use this only for polyp data or prompt-sensitivity ablation.
python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_polyp_box.yaml

# Box-only prompt. No text prompt is passed to the teacher.
python train_student_kd.py --config configs/kd/kd_medical_samba_only_box.yaml

# Text-only prompt: tumor. No box prompt.
python train_student_kd.py --config configs/kd/kd_medical_samba_only_tumor.yaml

# Text-only prompt: polyp. No box prompt.
python train_student_kd.py --config configs/kd/kd_medical_samba_only_polyp.yaml
