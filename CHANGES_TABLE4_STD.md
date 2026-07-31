# Table 4 standard-deviation update

- `evaluate.py` and `scripts/evaluate.py` now use per-case metrics and save mean,
  standard deviation, SEM, and 95% CI half-width.
- `scripts/summarize_ablations.py` now exports CSV, Markdown, and LaTeX Table 4
  with `mean ± std` for Dice, IoU, HD95, ASSD, and Boundary-F1.
- The summary script can recover standard deviation from
  `test_per_case_metrics.csv` when an older JSON lacks `*_std` fields.
- `scripts/run_ablation.sh` supports full retraining, evaluation-only, or
  table-only modes through `TRAIN` and `EVAL` environment variables.
- Corrected `m4_boundary` so it disables SE and truly represents
  `MultiScale + Boundary`; `m5_full` remains `MultiScale + Boundary + SE`.
- Added `README_ABLATION_STD.md` with exact commands and statistical scope.
- Added `scripts/run_ablation_multiseed.py` for the recommended paper-level
  `mean ± std` across independent seeds, with a fixed validation split.
