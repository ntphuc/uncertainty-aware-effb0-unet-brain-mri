#!/usr/bin/env bash
set -euo pipefail

# One-command runner for the five non-proxy BRISC2025 baselines.
# Forward any CLI arguments to scripts/run_paper_baselines.py.
# Examples:
#   bash scripts/run_paper_baselines.sh --stage train_eval --device cuda
#   bash scripts/run_paper_baselines.sh --stage train_eval --seeds 42 123 2025
#   bash scripts/run_paper_baselines.sh --stage eval --continue-on-error

python scripts/run_paper_baselines.py "$@"
