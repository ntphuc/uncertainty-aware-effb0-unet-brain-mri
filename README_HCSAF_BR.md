# HCSAF-BR U-Net

This repository extends the existing EfficientNet-B0 U-Net with **Hierarchical Channel-Spatial Adaptive Fusion with Boundary Refinement (HCSAF-BR)** while preserving the original Concat and RGAMF implementations for controlled comparison.

## 1. What was implemented

At decoder stage `s`, every encoder feature is projected to a shared 64-channel space and resized to the target resolution.

### Channel reliability

For every encoder scale:

```text
projected encoder feature: [B, 64, Hl, Wl]
decoder query:             [B, 64, Hs, Ws]
GAP both tensors
concatenate vectors:       [B, 128]
MLP: 128 -> 32 -> 64
channel logits:            [B, 64]
```

The five scale logits are normalized with softmax across the scale dimension:

```text
channel weights: [B, 5, 64]
sum over 5 scales = 1 for every channel
```

### Spatial reliability

Spatial weighting is enabled only at high-resolution decoder stages D2 and D1 by default. The scorer uses inexpensive channel statistics from each projected encoder feature and the decoder query:

```text
encoder mean, encoder max
decoder mean, decoder max
absolute mean difference
mean product
```

A small `3x3 Conv -> GroupNorm -> SiLU -> 1x1 Conv` produces:

```text
spatial logits:  [B, 5, Hs, Ws]
spatial weights: [B, 5, Hs, Ws]
sum over 5 scales = 1 at every pixel
```

### Memory-efficient factorized fusion

The implementation does **not** materialize a full `[B, 5, 64, H, W]` attention tensor. Channel and spatial factors are accumulated scale by scale:

```text
factor_l = channel_weight_l * spatial_weight_l
numerator   += factor_l * projected_feature_l
denominator += factor_l
adaptive_feature = numerator / denominator
```

This keeps peak memory practical at 128×128 resolution.

### Residual uniform branch

The adaptive result is blended with an unweighted mean of all five scales:

```text
mixed = lambda * adaptive + (1 - lambda) * uniform_mean
```

`lambda` is learned and initialized to `0.8`. This stabilizes early training and reduces scale-weight collapse.

### Low-cost fusion block

The final HCSAF context uses:

```text
Depthwise 3x3 -> GroupNorm -> SiLU
Pointwise 1x1 -> GroupNorm -> SiLU
```

### Hierarchical deployment

Decoder order and indices:

```text
index 0: D4, 16x16
index 1: D3, 32x32
index 2: D2, 64x64
index 3: D1, 128x128
```

Default policy:

```text
D4/D3: channel reliability only
D2/D1: channel + spatial reliability
D2 -> D1: learned residual upsampling
D1 -> output: learned residual upsampling
```

### Boundary refinement

At full 256×256 resolution, a lightweight guide head predicts a preliminary boundary map. This map gates a depthwise-separable residual refinement block. The original final boundary head remains unchanged and supplies the main boundary output.

The preliminary guide can receive an auxiliary boundary loss:

```yaml
loss:
  lambda_boundary: 0.15
  lambda_boundary_guide: 0.05
```

## 2. Important files

```text
models/hcsaf_br.py
    HCSAFFeatureProjection
    ChannelReliabilityEstimator
    SpatialReliabilityEstimator
    HCSAF
    LearnedResidualUpsample
    BoundaryRefinementBlock

models/efficient_b0_unet.py
    Integration into the existing decoder and complete forward pass

configs/efficient_b0_hcsaf_br_boundary.yaml
    Full HCSAF-BR configuration

configs/hcsaf_comparison/
    Pre-generated 3-seed configs for Concat, RGAMF, and HCSAF ablations

scripts/visualize_hcsaf_weights.py
    Export scale summaries, spatial maps, and boundary-guide overlays

scripts/profile_fusion_models.py
    Identical-input Params/MAC/FLOP comparison

scripts/create_hcsaf_comparison_configs.py
    Regenerate the comparison configs and manifest

scripts/summarize_hcsaf_comparison.py
    Produce post-lock mean ± SD result tables

tests/test_hcsaf_br.py
    Shape, normalization, gradient, integration, and loss tests
```

## 3. Installation

```bash
unzip Archive_HCSAF_BR_UNet_implemented.zip
cd BE_UNet_validation_locked_factorial_fixed

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Dataset paths

The default configuration expects:

```text
datasets/brisc2025/X_train.npy
datasets/brisc2025/Y_train.npy
datasets/brisc2025/X_test.npy
datasets/brisc2025/Y_test.npy
```

Update the `paths` section in the YAML config when your data is stored elsewhere.

## 5. Verify the implementation

Run the focused tests:

```bash
pytest -q tests/test_hcsaf_br.py tests/test_rgamf.py tests/test_validation_selection.py
```

Expected result for the supplied package:

```text
16 passed
```

## 6. Quick single-seed training

```bash
python train.py \
  --config configs/efficient_b0_hcsaf_br_boundary.yaml \
  --device cuda
```

The validation-selected checkpoint is written to:

```text
outputs/hcsaf_br/checkpoints/best_validation.pth
```

Evaluate it:

```bash
python evaluate.py \
  --config configs/efficient_b0_hcsaf_br_boundary.yaml \
  --checkpoint outputs/hcsaf_br/checkpoints/best_validation.pth \
  --device cuda \
  --split test
```

This quick route is useful for debugging. For a paper, use the validation-locked multi-seed procedure below.

## 7. Recommended validation-locked comparison

Seven designs are supplied:

```text
a0_concat               Original Multi-scale Concat
a1_rgamf                Scalar RGAMF
a2_hcsaf_channel        Channel reliability only
a3_hcsaf_spatial        Channel + spatial reliability
a4_hcsaf_spatial_up              Add learned upsampling
a5_hcsaf_br_no_guide_loss        Add boundary refinement without direct guide loss
a6_hcsaf_br_full                 Add direct boundary-guide supervision
```

Each design has seeds 42, 43, and 44, giving 21 runs.

### Step 1: regenerate configs when needed

The package already contains the generated configs. To recreate them:

```bash
python scripts/create_hcsaf_comparison_configs.py \
  --base-config configs/efficient_b0_hcsaf_br_boundary.yaml \
  --output-root configs/hcsaf_comparison \
  --runs-root outputs/hcsaf_comparison \
  --seeds 42 43 44 \
  --split-seed 42
```

### Step 2: train using validation data only

```bash
python scripts/run_factorial_train_validation.py \
  --manifest configs/hcsaf_comparison/manifest.yaml \
  --device cuda
```

The script does not evaluate the test set.

Resume behavior is automatic: completed runs containing `validation_best.json` are skipped.

Run one design or seed only:

```bash
python scripts/run_factorial_train_validation.py \
  --manifest configs/hcsaf_comparison/manifest.yaml \
  --variants a6_hcsaf_br_full \
  --seeds 42 \
  --device cuda
```

### Step 3: select and cryptographically lock the architecture

```bash
python scripts/select_architecture_from_validation.py \
  --manifest configs/hcsaf_comparison/manifest.yaml \
  --output-dir outputs/hcsaf_comparison/selection \
  --dice-tolerance 0.001
```

Important files:

```text
outputs/hcsaf_comparison/selection/validation_per_run.csv
outputs/hcsaf_comparison/selection/validation_architecture_summary.csv
outputs/hcsaf_comparison/selection/SELECTION_LOCK.json
```

### Step 4: confirmatory test evaluation of the selected design

```bash
python scripts/evaluate_factorial_after_lock.py \
  --lock outputs/hcsaf_comparison/selection/SELECTION_LOCK.json \
  --scope selected \
  --device cuda
```

### Step 5: post-lock evaluation of every ablation

Only after the selection lock exists:

```bash
python scripts/evaluate_factorial_after_lock.py \
  --lock outputs/hcsaf_comparison/selection/SELECTION_LOCK.json \
  --scope all \
  --device cuda
```

These results are post-lock ablation evidence and must not be used to change the selected architecture.

### Step 6: summarize results

```bash
python scripts/summarize_hcsaf_comparison.py \
  --manifest configs/hcsaf_comparison/manifest.yaml \
  --lock outputs/hcsaf_comparison/selection/SELECTION_LOCK.json \
  --output-dir outputs/hcsaf_comparison/summary \
  --require-all
```

Outputs:

```text
hcsaf_comparison_per_run.csv
hcsaf_comparison_summary.csv
hcsaf_comparison_report.md
```

## 8. Visualize learned reliability

```bash
python scripts/visualize_hcsaf_weights.py \
  --config configs/efficient_b0_hcsaf_br_boundary.yaml \
  --checkpoint outputs/hcsaf_br/checkpoints/best_validation.pth \
  --output-dir outputs/hcsaf_br/weight_analysis \
  --split test \
  --max-samples 860 \
  --sample-indices 6 143 236 279
```

Main outputs:

```text
hcsaf_scale_summary.npy          [N, 4, 5]
hcsaf_scale_summary_long.csv
hcsaf_mean_scale_summary.png
hcsaf_scale_summary_sample_*.png
hcsaf_spatial_sample_*_stage_*.png
hcsaf_boundary_guide_sample_*.png
```

Model output diagnostics:

```python
outputs["hcsaf_weights"]  # list of four stage dictionaries

stage["channel_weights"]  # [B, 5, 64]
stage["spatial_weights"]  # [B, 5, H, W] or None
stage["scale_summary"]    # [B, 5]
stage["adaptive_mix"]     # scalar in [0,1]

outputs["boundary_guide"] # [B,1,256,256]
```

## 9. Profile computation

```bash
python scripts/profile_fusion_models.py \
  --config configs/efficient_b0_hcsaf_br_boundary.yaml \
  --image-size 256 \
  --batch-size 1 \
  --device cpu
```

For a publication, use the same:

- PyTorch, timm, and profiler versions;
- input shape and precision;
- batch size;
- device;
- MAC-to-FLOP convention;

for Concat, RGAMF, and HCSAF-BR.

## 10. Memory and training recommendations

Default paper setting:

```yaml
training:
  batch_size: 8
  amp: true

model:
  hcsaf_channels: 64
  hcsaf_spatial_stage_indices: [2, 3]
```

If GPU memory is insufficient, reduce in this order:

1. `batch_size: 8 -> 4`;
2. use gradient accumulation if required;
3. `hcsaf_channels: 64 -> 48`;
4. enable spatial reliability only at D1: `[3]`;
5. `hcsaf_spatial_hidden_channels: 8 -> 4`.

Do not silently change one architecture's batch size, augmentation, epochs, optimizer, or checkpoint-selection policy while leaving the other models unchanged.

## 11. Interpretation rules for the paper

A higher learned weight is a model preference, not automatically a causal explanation. Recommended analyses:

- mean and SD of weights by decoder stage;
- entropy and dominant-scale frequency;
- lesion-size quartiles;
- successful versus failed cases;
- stability across seeds;
- remove-highest-weight versus remove-lowest-weight scale intervention.

The supplied code has been checked for tensor shapes, normalization, gradients, complete forward integration, and auxiliary boundary-guide supervision. It has **not** been trained on the full BRISC dataset in this environment, so no Dice, HD95, ASSD, or BF1 improvement is claimed by the package itself.
