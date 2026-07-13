#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/datasets/brisc2025"
ZIP_PATH="$ROOT_DIR/data_preprocessed.zip"

mkdir -p "$OUT_DIR"

python3 "$ROOT_DIR/datasets/preprocess/download_from_gdrive_gdown.py" \
	--url "https://drive.google.com/file/d/1TqyOeXxUy_uEv_Fb3sqmd8XUB9rWJKn9/view?usp=sharing" \
	--output "$ZIP_PATH"

unzip -o "$ZIP_PATH" -d "$OUT_DIR"