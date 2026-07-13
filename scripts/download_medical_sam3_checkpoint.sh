#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/checkpoints/medical_sam3"

python -m pip install -U "huggingface_hub>=0.24" "hf_xet>=1.1.1"
python "${ROOT_DIR}/scripts/download_medical_sam3_checkpoint.py" \
  --output-dir "${DEST_DIR}" "$@"
