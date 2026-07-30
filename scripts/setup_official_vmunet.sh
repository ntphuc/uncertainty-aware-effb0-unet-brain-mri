#!/usr/bin/env bash
set -euo pipefail

# Install and fetch the authors' official VM-UNet implementation.
# VM-UNet uses CUDA selective-scan kernels; an NVIDIA CUDA environment is expected.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: official VM-UNet/mamba-ssm is normally CUDA-only and may not run on CPU.")
PY

python -m pip install --upgrade packaging ninja
# Modern mamba packages preserve the selective_scan_interface imported by the official code.
# --no-build-isolation lets the build see the already-installed PyTorch/CUDA toolchain.
# python -m pip install --no-build-isolation "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2"
python -m  pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
python -m  pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

mkdir -p external
if [[ ! -d external/VM-UNet/.git ]]; then
  git clone https://github.com/JCruan519/VM-UNet.git external/VM-UNet
else
  git -C external/VM-UNet pull --ff-only
fi

echo "Official VM-UNet is ready at external/VM-UNet"
python scripts/check_paper_baselines.py --models vmunet --device cuda
