# RGAMF-U-Net implementation

This package replaces the original concat-based multi-scale context with
**Reliability-Guided Adaptive Multi-scale Fusion (RGAMF)** while preserving the
existing EfficientNet-B0 encoder, Skip-SE modules, boundary head, deep
supervision, validation-locked checkpoint selection, and evaluation pipeline.

## 1. Mathematical definition

At decoder stage `s`, every encoder feature is projected and resized:

```text
E_l -> 1x1 projection -> E_tilde_l -> resize to (H_s, W_s)
```

The reliability estimator receives global descriptors from the projected
encoder feature and the current upsampled decoder context:

```text
r_l,s = MLP([GAP(E_tilde_l), GAP(D_s)])
alpha_l,s = softmax_l(r_l,s / temperature)
M_s = phi_s(sum_l alpha_l,s * resize(E_tilde_l))
```

`phi_s` is `3x3 Conv -> BatchNorm -> ReLU`.

The decoder update is:

```text
D_s = psi_s(Up(D_s+1) || SkipSE(E_s) || M_s)
```

## 2. Implemented files

- `models/rgamf.py`
  - `FeatureProjection`
  - `ReliabilityEstimator`
  - `RGAMF`
- `models/efficient_b0_unet.py`
  - supports `fusion_type: concat` and `fusion_type: rgamf`
  - returns `rgamf_weights`, a list of four `[B,5]` tensors
- `models/__init__.py`
  - adds model aliases `efficientnet_b0_unet_rgamf`, `effb0_rgamf`, and `rgamf_unet`
- `configs/efficient_b0_rgamf_boundary.yaml`
- `scripts/profile_rgamf.py`
- `scripts/visualize_rgamf_weights.py`
- `tests/test_rgamf.py`

## 3. Encoder-channel warning

The requested paper assumption is:

```text
E1=24, E2=40, E3=80, E4=112, E5=320
```

The implementation does **not silently hard-code these values**. It reads the
actual channels from:

```python
self.encoder.feature_info.channels()
```

This is necessary because `timm` versions and feature taps can expose a
different channel list. To enforce the paper assumption, add:

```yaml
expected_encoder_channels: [24, 40, 80, 112, 320]
```

The model will then fail early if the encoder taps do not match. Never report
channel dimensions in the paper without printing and archiving the actual
`feature_info.channels()` output.

## 4. Install and test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests/test_rgamf.py
```

Expected result:

```text
4 passed
```

## 5. Train

```bash
python train.py --config configs/efficient_b0_rgamf_boundary.yaml
```

The validation-locked pipeline saves the selected checkpoint as:

```text
outputs/rgamf/checkpoints/best_validation.pth
```

Use the real output path printed by `train.py` if the project creates an
additional run directory.

## 6. Evaluate

```bash
python evaluate.py \
  --config configs/efficient_b0_rgamf_boundary.yaml \
  --checkpoint outputs/rgamf/checkpoints/best_validation.pth
```

Use the same seeds, split seed, preprocessing, threshold policy, and training
budget as the concat baseline.

## 7. Visualize adaptive alpha weights

```bash
python scripts/visualize_rgamf_weights.py \
  --config configs/efficient_b0_rgamf_boundary.yaml \
  --checkpoint outputs/rgamf/checkpoints/best_validation.pth \
  --output-dir outputs/rgamf/alpha_analysis \
  --split test \
  --max-samples 860 \
  --sample-indices 6 143 236 279
```

Outputs:

- `rgamf_weights.npy`: `[N,4,5]`
- `sample_indices.npy`
- `rgamf_weights_long.csv`
- `rgamf_mean_weights.png`
- one heatmap for every requested sample index

Decoder-stage order is deepest to shallowest:

```text
D4 (16x16), D3 (32x32), D2 (64x64), D1 (128x128)
```

Encoder-scale order is:

```text
E1, E2, E3, E4, E5
```

For a paper, report mean, standard deviation, entropy, and lesion-size-stratified
alpha weights. Do not interpret alpha as causal feature importance without an
intervention or scale-removal experiment.

## 8. Parameters and computation

Run:

```bash
python scripts/profile_rgamf.py
```

Under the requested channels, target sizes, `fusion_channels=64`, and
`hidden_channels=32`, the analytical comparison of the four context modules
plus four decoder blocks is:

```text
Concat parameters : 2,019,584
RGAMF parameters  : 1,834,388
Delta parameters  : -185,196 (-9.17%)

Concat MACs       : 2.5990 GMACs
RGAMF MACs        : 3.0486 GMACs
Delta MACs        : +0.4497 GMACs (+17.30%)
Approx FLOP delta : +0.8993 GFLOPs if one MAC is counted as two FLOPs
```

Why parameters fall but compute rises:

- weighted summation avoids the large stage-dependent concat context tensors;
- the shared RGAMF output width is 64, reducing early decoder input widths;
- the final 128x128 RGAMF stage now uses a 64-channel 3x3 fusion instead of a
  32-channel context, increasing high-resolution computation.

These are analytical estimates. For publication, profile both complete models
with the same profiler version, input tensor, precision, batch size, and device.
Also report latency, peak VRAM, and throughput.

## 9. Recommended paper ablations

### Essential mechanism ablations

1. Original concat fusion.
2. Uniform weighted sum (`alpha=1/5`) with the same 64-channel projections.
3. Learned static per-stage weights, independent of the image.
4. Encoder-only reliability: MLP receives only `GAP(E_tilde_l)`.
5. Full RGAMF: encoder feature plus decoder context.
6. Shared reliability MLP across scales versus five scale-specific MLPs.

### Capacity and optimization controls

7. Fusion channels: 24, 32, 64, 96.
8. Reliability hidden width: 16, 32, 64.
9. Softmax temperature: 0.5, 1.0, 2.0.
10. Parameter-matched concat and wider-decoder controls.

### Interaction ablations

11. RGAMF without Skip-SE.
12. RGAMF with Skip-SE.
13. RGAMF with/without boundary supervision.
14. RGAMF with/without deep supervision.

### Reliability analyses

15. Alpha entropy versus lesion size and failure category.
16. Alpha stability across seeds.
17. Paired scale-removal intervention: remove the highest-weight scale and compare
    the metric drop with removal of the lowest-weight scale.
18. Cross-dataset alpha shift, for example BRISC versus BraTS.

Use validation data to select all hyperparameters and lock the architecture
before final test evaluation.
