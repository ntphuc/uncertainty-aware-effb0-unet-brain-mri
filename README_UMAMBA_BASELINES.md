# U-Mamba_Bot, U-Mamba_Enc, and ± metric reporting

This extension adds two non-proxy U-Mamba architecture variants to the existing
standalone BRISC2025 training/evaluation pipeline:

- `U-Mamba_Bot`: residual CNN encoder/decoder, with Mamba only at the bottleneck.
- `U-Mamba_Enc`: residual CNN encoder/decoder, with Mamba on alternating encoder
  stages and always on the deepest stage.

The implementation is adapted to the repository's output contract and dataset
loader. It uses the official `mamba_ssm.Mamba` selective-scan module; it does not
silently fall back to the old convolutional proxy.

## Install

Install the common requirements first, then the CUDA Mamba extension:

```bash
python -m pip install -r requirements.txt
bash scripts/setup_official_umamba.sh
```

The Mamba extension normally requires Linux, a compatible NVIDIA CUDA toolkit,
and a CUDA-enabled PyTorch build.

## Smoke test

```bash
python scripts/check_paper_baselines.py \
  --models umamba_bot umamba_enc \
  --device cuda \
  --image-size 256
```

## Train and evaluate

One seed:

```bash
python scripts/run_paper_baselines.py \
  --stage train_eval \
  --models umamba_bot umamba_enc \
  --device cuda \
  --seeds 42
```

Recommended repeated experiment:

```bash
python scripts/run_paper_baselines.py \
  --stage train_eval \
  --models umamba_bot umamba_enc \
  --device cuda \
  --seeds 42 123 2025
```

The full runner now recognizes:

```text
unet, unetpp, attention_unet, deeplabv3plus,
umamba_bot, umamba_enc, vmunet
```

## Mean ± standard deviation

`evaluate.py` now saves one row per test image to:

```text
<output_dir>/eval/test_per_case_metrics.csv
```

The JSON result also includes, for each metric:

```text
<metric>                     # mean, backward compatible
<metric>_std                 # per-case standard deviation
<metric>_sem                 # standard error of the mean
<metric>_ci95_half_width     # 1.96 × SEM
<metric>_mean_pm_std         # formatted "mean ± std"
<metric>_nonfinite_count     # relevant to HD95/ASSD complete mismatches
```

`metric_statistics` contains the same values in a nested, machine-readable
form. By default, `--std-ddof 1` reports sample standard deviation.

The paper baseline runner keeps the variability scope explicit:

- with multiple seeds: `mean ± std` across seed-level test means;
- with one seed: `mean ± std` across test cases, taken from `evaluate.py`.

Do not describe single-seed case variability as run-to-run reproducibility.

## Important implementation note

The official U-Mamba repository is built on nnU-Net and self-configures the
number of stages, feature widths, and patch plan. This repository uses an
explicit standalone 2D plan for a 256×256 input:

```text
channels: [32, 64, 128, 256, 320, 320]
strides:  [1,  2,   2,   2,   2,   2]
```

This preserves the defining U-Mamba_Bot/U-Mamba_Enc block placement and official
Mamba operator, but it is not a byte-for-byte reproduction of the full nnU-Net
planning and trainer stack. State this clearly in a paper or thesis.
