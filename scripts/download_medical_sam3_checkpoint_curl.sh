#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/checkpoints/medical_sam3"
DEST_FILE="${DEST_DIR}/checkpoint_3D.pt"
URL="https://huggingface.co/ChongCong/Medical-SAM3/resolve/main/checkpoint_3D.pt?download=true"
EXPECTED_SHA256="6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6"

mkdir -p "${DEST_DIR}"
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 2; }

# -C - resumes a partially downloaded 10.3 GB file.
curl -L --fail --retry 5 --retry-delay 10 -C - \
  -o "${DEST_FILE}" "${URL}"

echo "[INFO] Verifying SHA256..."
echo "${EXPECTED_SHA256}  ${DEST_FILE}" | sha256sum -c -
echo "[READY] ${DEST_FILE}"
