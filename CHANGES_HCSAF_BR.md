# HCSAF-BR implementation changes

## Added

- `models/hcsaf_br.py`
  - normalized feature projection;
  - channel reliability estimator;
  - lightweight spatial reliability estimator;
  - memory-efficient factorized HCSAF;
  - adaptive/uniform residual mixing;
  - depthwise-separable fusion;
  - learned residual upsampling;
  - boundary-guided full-resolution refinement.

- `configs/efficient_b0_hcsaf_br_boundary.yaml`
  - full HCSAF-BR training configuration;
  - fixed validation split and validation-only checkpoint selection;
  - auxiliary boundary-guide weight.

- `configs/hcsaf_comparison/`
  - 21 ready-to-run configs: seven fusion designs × three seeds;
  - compatible manifest for the validation-lock workflow.

- `scripts/create_hcsaf_comparison_configs.py`
  - regenerates the comparison design.

- `scripts/visualize_hcsaf_weights.py`
  - exports scale summaries, spatial reliability maps, and boundary-guide overlays.

- `scripts/profile_fusion_models.py`
  - identical-input full-model parameter/MAC/FLOP comparison.

- `scripts/summarize_hcsaf_comparison.py`
  - post-lock multi-seed result summary.

- `tests/test_hcsaf_br.py`
  - projection, channel reliability, spatial reliability, normalized weights,
    gradients, upsampling, refinement, complete model forward, and guide-loss tests.

- `README_HCSAF_BR.md`
  - architecture explanation and end-to-end commands.

## Modified

- `models/efficient_b0_unet.py`
  - accepts `fusion_type: hcsaf_br`;
  - integrates hierarchical spatial stages;
  - integrates learned D2→D1 and final upsampling;
  - integrates full-resolution boundary refinement;
  - returns HCSAF diagnostics without changing existing segmentation/boundary outputs;
  - preserves Concat and RGAMF behavior for ablations.

- `models/__init__.py`
  - adds model names:
    - `efficientnet_b0_unet_hcsaf_br`;
    - `effb0_hcsaf_br`;
    - `hcsaf_br_unet`;
  - passes all HCSAF-BR YAML options to the model.

- `losses/combined.py`
  - adds optional `lambda_boundary_guide`;
  - supervises the preliminary boundary guide with the same boundary target;
  - defaults to zero for backward compatibility.

- `train.py`
  - reads `lambda_boundary_guide`;
  - logs `boundary_guide_loss` in train and validation metrics.

## Validation performed

- Python compile check passed.
- YAML comparison configs parse successfully.
- Comparison orchestration dry-run passed.
- Focused test suite result: `16 passed`.

Full BRISC training was not executed in this environment because the dataset,
`timm`, and a training GPU were unavailable. Scientific performance values
must therefore be produced on the user's training system.
