#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

printf '\n[1/4] Evaluating Medical-SAM3 teacher...\n'
./scripts/eval_teacher.sh "${TEACHER_CONFIG:-configs/kd/kd_medical_sam3_checkpoint3d_tumor_box.yaml}"

printf '\n[2/4] Evaluating main student...\n'
./scripts/eval_student.sh "${STUDENT_CONFIG:-configs/kd/student_kd_medical_samba_text_tumor_box.yaml}" "${STUDENT_CHECKPOINT:-}"

printf '\n[3/4] Evaluating KD/prompt student ablations...\n'
./scripts/eval_student_ablations.sh

printf '\n[4/4] Evaluating architecture ablations...\n'
./scripts/eval_student_architecture_ablations.sh

printf '\nAll available evaluations completed.\n'
printf 'KD summary: outputs/kd_ablation_summary.csv\n'
printf 'Teacher results: outputs/teacher_eval/*/test_results.json\n'
