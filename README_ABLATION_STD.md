# Re-run Table 4 with standard deviation

The EfficientNet-B0 U-Net ablation pipeline now reports the five segmentation
metrics as **mean ± standard deviation**:

- Dice
- IoU
- HD95
- ASSD
- Boundary-F1

`Params(M)` and `GFLOPs` remain single values because they are deterministic for
a fixed architecture and input size.

## Important correction to the ablation definitions

The generated configurations now match the row names in Table 4:

| Config | Actual variant |
|---|---|
| `m1_effb0_unet.yaml` | EffB0 U-Net |
| `m2_multiscale.yaml` | EffB0 U-Net MultiScale |
| `m3_multiscale_se.yaml` | MultiScale + SE |
| `m4_boundary.yaml` | MultiScale + Boundary, **without SE** |
| `m5_full.yaml` | MultiScale + Boundary + SE |

Previously, `m4_boundary` also enabled SE, so it did not isolate the boundary
component correctly.

## 1. Full retraining, evaluation, and table export

```bash
DEVICE=cuda TRAIN=1 EVAL=1 STD_DDOF=1 bash scripts/run_ablation.sh
```

Use `DEVICE=cpu` only for a small/debug run because full training will be slow.

## 2. Re-evaluate existing checkpoints only

```bash
DEVICE=cuda TRAIN=0 EVAL=1 STD_DDOF=1 bash scripts/run_ablation.sh
```

## 3. Rebuild the table from existing result files only

```bash
TRAIN=0 EVAL=0 STD_DDOF=1 bash scripts/run_ablation.sh
```

## Generated files

```text
outputs/ablations/table4_ablation_with_std.csv
outputs/ablations/table4_ablation_with_std.md
outputs/ablations/table4_ablation_with_std.tex
```

Each model evaluation also creates:

```text
outputs/ablations/<model>/eval/test_results.json
outputs/ablations/<model>/eval/test_per_case_metrics.csv
```

The JSON includes fields such as:

```text
dice
dice_std
dice_sem
dice_ci95_half_width
dice_mean_pm_std
```

The same structure is produced for IoU, HD95, ASSD, and Boundary-F1.

## Statistical interpretation

With one checkpoint per model, the displayed deviation is the sample standard
deviation across test cases (`ddof=1`). This describes case-to-case variability,
not run-to-run reproducibility. For a paper claiming training stability, run at
least three independent seeds and aggregate the seed-level test means.

The generated LaTeX table uses `booktabs`, so include:

```latex
\usepackage{booktabs}
```

## Recommended paper mode: three independent seeds

For standard deviation across independent training runs rather than across test
cases, run:

```bash
python scripts/run_ablation_multiseed.py \
  --seeds 42 43 44 \
  --split-seed 42 \
  --device cuda \
  --std-ddof 1
```

This trains 15 models in total: 5 ablation variants × 3 seeds. The validation
split stays fixed at seed 42, while training randomness changes with each model
seed.

To aggregate previously completed seed folders without retraining or
re-evaluating:

```bash
python scripts/run_ablation_multiseed.py \
  --seeds 42 43 44 \
  --skip-train \
  --skip-eval
```

Multi-seed outputs:

```text
outputs/ablations_multiseed/multiseed_raw_results.csv
outputs/ablations_multiseed/table4_multiseed.csv
outputs/ablations_multiseed/table4_multiseed.md
outputs/ablations_multiseed/table4_multiseed.tex
```
