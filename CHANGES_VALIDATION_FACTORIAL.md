# Changes: Validation-Locked Selection and Factorial Ablation

## Modified files

### `train.py`

- Replaced seed-dependent `random_split` behavior with a fixed split controlled by `training.split_seed`.
- Saves exact split indices and SHA-256 to `data_split.json`.
- Uses independent train and validation dataset objects.
- Added per-case validation metrics rather than batch-level surface averaging.
- Added missing-prediction count/rate.
- Added explicit finite penalty for undefined HD95/ASSD during validation selection.
- Added configurable, pre-specified checkpoint-selection policy.
- Saves `best_validation.pth`, `validation_best.json`, and `checkpoint_selection.csv`.
- Stores validation metrics, policy, split hash, and selection reason in checkpoints.
- Preserves legacy `best.pth` for compatibility.

### `evaluate.py`

- Stores validation-selection provenance in test result JSON.
- Added optional strict test guard for factorial configs.
- Verifies `SELECTION_LOCK.json` and SHA-256 hashes before test evaluation.
- Labels selected-model evaluation versus post-lock ablation evaluation.

### `configs/efficient_b0_boundary.yaml`

- Added fixed `split_seed`.
- Added deterministic setting.
- Added explicit empty-surface penalty policy.
- Added pre-specified checkpoint-selection policy.

## New files

### `utils/selection.py`

Reusable lexicographic validation-selection logic with finite-value handling.

### `scripts/create_factorial_ablation_configs.py`

Generates one Base anchor plus the complete 2×2×2 S/B/D factorial with MultiScale enabled, for all declared seeds.

### `scripts/run_factorial_train_validation.py`

Runs training/validation only; never invokes test evaluation.

### `scripts/select_architecture_from_validation.py`

Aggregates validation results across seeds, selects an architecture, and writes an immutable SHA-256 selection lock.

### `scripts/evaluate_factorial_after_lock.py`

Evaluates either the selected model or all post-lock ablation cells only after lock verification.

### `scripts/summarize_factorial_results.py`

Creates seed-level summaries, mean ± SD tables, and paired factorial main effects/interactions.

### `tests/test_validation_selection.py`

Unit tests for the pre-specified selection rule.

### `configs/factorial/`

Pre-generated 27 run configs and one experiment manifest.

### `README_VALIDATION_SELECTION_FACTORIAL.md`

Full Vietnamese execution and reporting guide.

## Methodological corrections achieved

1. Test data are no longer needed for checkpoint or architecture selection.
2. Every run uses the same validation cases.
3. The architecture decision is cryptographically frozen before test evaluation.
4. Boundary and deep supervision are independently enabled/disabled.
5. Skip-SE interactions with both supervision mechanisms can be estimated.
6. Results are summarized across all declared seeds rather than a favorable single seed.

## Additional code-audit findings

- Normalization in `NPYSliceDataset` is per-slice z-score, not dataset-level z-score.
- The legacy `m4_boundary.yaml` is boundary-only (`use_deep_supervision: false`), so prior manuscript labels must be checked.
- The legacy `run_ablation.sh` evaluates test immediately and must not be used for validation-locked model selection.
