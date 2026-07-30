# Validation report

Validation performed on the modified archive:

1. Compiled all Python source files outside `__MACOSX` with `py_compile`.
2. Instantiated `UMambaBot2D` and `UMambaEnc2D` through the model factory.
3. Ran forward passes at 64×64 and 256×256 using an API-compatible local test
   stub for `mamba_ssm.Mamba`; both variants returned logits with the original
   input resolution.
4. Ran `scripts/check_paper_baselines.py` for both U-Mamba variants with the test
   stub.
5. Ran `evaluate.py` end-to-end on a small synthetic `.npy` dataset and a random
   U-Net checkpoint; verified:
   - `test_results.json` contains mean, std, SEM, 95% CI half-width and formatted
     `mean ± std` fields;
   - `test_per_case_metrics.csv` is created;
   - non-finite HD95/ASSD cases are counted.
6. Tested one-seed result aggregation; the generated table labels the ± scope as
   `across test cases`.

## Environment limitation

The current build environment does not contain the compiled CUDA
`mamba_ssm` extension, so the real selective-scan CUDA kernel was not executed
here. The production configurations require the real package and fail with an
actionable error rather than silently using a proxy. Run
`scripts/setup_official_umamba.sh` and the CUDA smoke test on the target machine
before full BRISC2025 training.
