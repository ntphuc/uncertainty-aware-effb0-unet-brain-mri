#!/usr/bin/env bash
set -euo pipefail

# Install the selective-scan dependencies required by U-Mamba_Bot/U-Mamba_Enc.
# Run this inside the same Python environment used for this repository.

python - <<'PY'
import platform
import torch
print(f"Python platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "A CUDA-capable PyTorch environment is required for the official "
        "mamba_ssm selective-scan backend."
    )
PY

python -m pip install --upgrade pip setuptools wheel packaging ninja
# python -m pip install "causal-conv1d>=1.2.0" --no-build-isolation
# python -m pip install mamba-ssm --no-build-isolation --no-cache-dir
python -m  pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
python -m  pip install https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
python - <<'PY'
import torch
from mamba_ssm import Mamba

layer = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda().eval()
x = torch.randn(1, 64, 32, device="cuda")
with torch.no_grad():
    y = layer(x)
assert y.shape == x.shape
print("U-Mamba dependency check passed:", tuple(y.shape))
PY
