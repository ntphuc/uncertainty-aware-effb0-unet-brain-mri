# Changes made for RGAMF-U-Net

## New modules

### `FeatureProjection`
- Bias-free `1x1` convolution.
- Converts each encoder feature to 64 channels.
- Includes shape and channel validation.

### `ReliabilityEstimator`
- GAP on projected encoder feature.
- GAP on current decoder context.
- Concatenation of both vectors.
- `Linear -> ReLU -> Linear -> scalar`.

### `RGAMF`
- Projects all encoder scales independently.
- Computes five decoder-conditioned reliability scores.
- Applies numerically stable softmax across scales.
- Performs weighted summation.
- Applies `3x3 Conv -> BN -> ReLU`.
- Returns fused context and `[B,5]` alpha weights.

## Decoder integration

- Added `fusion_type: concat | rgamf`.
- Original concat implementation remains available for a controlled ablation.
- RGAMF receives the upsampled decoder tensor before concatenation.
- Decoder input is `upsampled decoder + Skip-SE output + 64-channel RGAMF context`.
- Forward output now includes `rgamf_weights`.

## Configuration and factory

- Added model aliases for RGAMF.
- Added `configs/efficient_b0_rgamf_boundary.yaml`.
- Added optional strict encoder-channel validation.

## Analysis utilities

- Added analytical parameter/MAC profiling.
- Added dataset-level and sample-level alpha export and heatmaps.
- Added unit and complete-forward tests.
