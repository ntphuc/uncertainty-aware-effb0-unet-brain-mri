#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

printf '\n[1/5] Installing Python requirements...\n'
"$PYTHON_BIN" -m pip install -r requirements.txt

printf '\n[2/5] Enabling all scripts...\n'
chmod -R +x scripts

printf '\n[3/5] Downloading BRISC2025 preprocessed data...\n'
./scripts/download_data.sh

printf '\n[4/5] Downloading Medical-SAM3 checkpoint...\n'
./scripts/download_medical_sam3_checkpoint.sh "$@"

printf '\n[5/5] Cloning Medical-SAM3 repository...\n'
./scripts/clone_medical_sam3_repo.sh

./scripts/eval_teacher.sh

printf '\nSetup completed.\n'
printf 'Next check: python scripts/check_medical_sam3_setup.py\n'

# printf 'Training: ./scripts/run_kd_ablation.sh\n'

printf 'Teacher eval: ./scripts/eval_teacher.sh\n'
./scripts/eval_teacher.sh
# printf 'Student eval: ./scripts/eval_student.sh\n'
# printf 'Ablation eval: ./scripts/eval_student_ablations.sh\n'
