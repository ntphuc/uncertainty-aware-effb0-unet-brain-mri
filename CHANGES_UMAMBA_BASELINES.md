# Change log: U-Mamba baselines and metric variability

## Added

- `models/official_umamba.py`
  - `UMambaBot2D`
  - `UMambaEnc2D`
  - residual CNN encoder/decoder
  - spatial-token and channel-token Mamba modes
  - fp32 selective-scan execution inside AMP training
  - standard repository output dictionary
- `configs/paper_baselines/umamba_bot_official.yaml`
- `configs/paper_baselines/umamba_enc_official.yaml`
- `requirements-umamba.txt`
- `scripts/setup_official_umamba.sh`
- `README_UMAMBA_BASELINES.md`

## Changed

- `models/__init__.py`
  - added model factory names `umamba_bot_official` and `umamba_enc_official`
- `utils/metrics.py`
  - added per-image Dice, IoU, Precision, Recall, F2, HD95, ASSD and Boundary-F1
  - added mean, standard deviation, SEM and 95% CI summary helper
- `evaluate.py`
  - writes per-case CSV
  - writes `mean ± std` and uncertainty statistics to JSON
  - reports non-finite HD95/ASSD case counts
- `scripts/run_paper_baselines.py`
  - includes U-Mamba_Bot and U-Mamba_Enc
  - aggregates repeated seeds as mean ± sample standard deviation
  - falls back to explicitly labeled per-case std for a one-seed table
  - records `mamba_ssm` version in `environment.json`
- `scripts/check_paper_baselines.py`
  - smoke-test entries for both U-Mamba variants
- `scripts/run_paper_baselines.sh`
  - updated baseline count
- `paper_references.bib`
  - added U-Mamba citation
