# BRISC2025 — five non-proxy segmentation baselines

This extension adds one reproducible benchmark runner for:

1. **U-Net** — canonical contracting/expanding architecture.
2. **U-Net++** — nested dense skip pathways with deep supervision.
3. **Attention U-Net** — additive attention gates on U-Net skip connections.
4. **DeepLabV3+** — ResNet-50 encoder, output stride 16, ASPP and separable decoder.
5. **VM-UNet** — the authors' official `VSSM` implementation loaded directly from
   `JCruan519/VM-UNet`; it is not the convolutional proxy previously present in
   this repository.

All five models expose raw binary-segmentation logits through the repository's
standard output contract:

```python
{"seg": logits, "boundary": None, "deep_outputs": [...]}
```

The adapter does not replace any VM-UNet VSS/selective-scan block. It only:

- repeats one-channel MRI input to three channels, as the official wrapper does;
- returns logits instead of applying sigmoid, because this repository trains with
  a logits-based segmentation loss;
- resizes the output only if a version of the official implementation returns a
  different spatial shape.

## 1. Files added or changed

```text
models/original_baselines.py          # U-Net, U-Net++, Attention U-Net, DeepLabV3+
models/official_vmunet.py             # direct adapter for authors' VM-UNet VSSM
models/__init__.py                    # model factory names
configs/paper_baselines/*.yaml        # matched benchmark configurations
scripts/setup_official_vmunet.sh      # clone/install official VM-UNet dependencies
scripts/check_paper_baselines.py      # architecture forward-pass smoke test
scripts/run_paper_baselines.py        # train/evaluate/aggregate all models
scripts/run_paper_baselines.sh        # shell wrapper
paper_references.bib                  # model and dataset citations
```

The old proxy is retained only for backward compatibility under the explicit
name `vmunet_proxy`; it is excluded from the paper runner.

## 2. Expected BRISC2025 files

The runner uses the paths already expected by this project:

```text
datasets/brisc2025/X_train.npy
datasets/brisc2025/Y_train.npy
datasets/brisc2025/X_test.npy
datasets/brisc2025/Y_test.npy
```

Supported sample layouts are documented in `datasets/npy_dataset.py`. Masks are
converted to binary masks by the existing data pipeline.

## 3. Install the common environment

```bash
cd <project-root>
python -m pip install -r requirements.txt
```

Check the four CNN baselines first:

```bash
python scripts/check_paper_baselines.py \
  --models unet unetpp attention_unet deeplabv3plus \
  --device cuda \
  --image-size 256
```

## 4. Install official VM-UNet

VM-UNet requires the CUDA selective-scan implementation from `mamba-ssm`.
Run on a Linux machine with a compatible NVIDIA CUDA/PyTorch toolchain:

```bash
bash scripts/setup_official_vmunet.sh
```

Then verify it:

```bash
python scripts/check_paper_baselines.py --models vmunet --device cuda --image-size 256
```

The authors' repository documents an older reference environment based on
Python 3.8, PyTorch 1.13, CUDA 11.7, `causal_conv1d==1.0.0`, and
`mamba_ssm==1.0.1`. The setup script first targets modern compatible packages so
it can coexist with this project's PyTorch environment. If compilation fails,
create the authors' pinned environment and run this repository from that
environment instead.

## 5. Run all models and produce the paper table

Recommended paper run with three seeds:

```bash
bash scripts/run_paper_baselines.sh \
  --stage train_eval \
  --device cuda \
  --seeds 42 123 2025
```

The validation split is held fixed with `split_seed: 42` for every run. The
training seed changes across runs. All default configurations use:

- input size: 256 × 256;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- 50 epochs;
- identical augmentation;
- effective batch size 8 through gradient accumulation;
- no post-processing and no test-time augmentation beyond scale 1.0;
- training from scratch by default, including the ResNet-50 encoder.

Run only the four models that do not require selective scan:

```bash
bash scripts/run_paper_baselines.sh \
  --stage train_eval \
  --models unet unetpp attention_unet deeplabv3plus \
  --device cuda \
  --seeds 42 123 2025
```

Useful quick checks:

```bash
# One seed, two epochs
bash scripts/run_paper_baselines.sh \
  --stage train_eval --epochs 2 --seeds 42 \
  --models unet attention_unet

# Evaluate existing best checkpoints without retraining
bash scripts/run_paper_baselines.sh \
  --stage eval --device cuda --seeds 42 123 2025

# Continue and still export completed models if one model fails
bash scripts/run_paper_baselines.sh \
  --stage train_eval --continue-on-error --device cuda
```

## 6. Generated outputs

For every model and seed:

```text
outputs/paper_baselines/<model>/seed_<seed>/
├── checkpoints/best.pth
├── train_log.csv
└── eval/
    ├── test_results.json
    └── test_predictions.png
```

Aggregated outputs:

```text
outputs/paper_baselines/paper_results_raw.csv
outputs/paper_baselines/paper_results_summary.csv
outputs/paper_baselines/paper_table.md
outputs/paper_baselines/paper_table.tex
outputs/paper_baselines/run_manifest.json
outputs/paper_baselines/environment.json
```

The terminal and publication tables include:

- Dice, IoU, Precision, Recall and F2 — higher is better;
- HD95 and ASSD — lower is better;
- Boundary F1 — higher is better;
- trainable parameters in millions;
- GFLOPs at the configured input size when THOP supports every model operation.

For multiple seeds, values are reported as mean ± sample standard deviation.
VM-UNet GFLOPs can appear as `NA` when the installed THOP version has no handler
for the custom selective-scan CUDA operator; the script intentionally does not
invent an estimate.

## 7. Paper-use notes

- Do not report values from `configs/baseline_vmunet.yaml` as VM-UNet results;
  that legacy configuration now explicitly uses `vmunet_proxy`.
- Keep the same image size, preprocessing, loss, threshold, split and augmentation
  across all baselines when comparing architectures.
- The generated LaTeX table requires `\usepackage{booktabs}`.
- `environment.json` records the GPU, CUDA, PyTorch, torchvision, THOP and the
  exact cloned official VM-UNet commit for the experimental setup section.
- The archive contains code and smoke-test validation, not trained BRISC2025
  scores. Real values are generated only after the dataset and training runs are
  available on your machine.
