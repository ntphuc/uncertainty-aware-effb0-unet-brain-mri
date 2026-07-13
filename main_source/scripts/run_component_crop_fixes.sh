#!/usr/bin/env bash
set -e

echo "[1/3] Recall + component-aware crop, no boundary"
python train.py --config configs/efficient_b0_recall_ms_no_boundary_component_ftv.yaml

echo "[2/3] Recall + component-aware crop + hard boundary"
python train.py --config configs/efficient_b0_recall_ms_boundary_component_ftv.yaml

echo "[3/3] Recall + component-aware crop + soft boundary"
python train.py --config configs/efficient_b0_recall_ms_soft_boundary_component_ftv.yaml
