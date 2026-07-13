#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
ckpt = root / 'checkpoints' / 'medical_sam3' / 'checkpoint_3D.pt'
repo = root / 'external' / 'Medical-SAM3'

ok = True
if ckpt.exists():
    print(f'[OK] checkpoint: {ckpt} ({ckpt.stat().st_size / 1024**3:.2f} GiB)')
else:
    print(f'[MISSING] checkpoint: {ckpt}')
    print('          Run: bash scripts/download_medical_sam3_checkpoint.sh')
    ok = False

if repo.exists():
    print(f'[OK] repository: {repo}')
else:
    print(f'[MISSING] repository code: {repo}')
    print('          Clone Medical-SAM3 there, or change teacher.repo_path in YAML.')
    ok = False

raise SystemExit(0 if ok else 1)
