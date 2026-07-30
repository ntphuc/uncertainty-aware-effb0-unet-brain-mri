# Uncertainty-Aware EfficientNet-B0 U-Net for Brain Tumor MRI Segmentation

Official implementation for the paper:

> **A Lightweight Multi-Scale EfficientNet-B0 U-Net and Uncertainty Analysis for Brain Tumor Segmentation in MRI**

This repository implements a lightweight brain tumor MRI segmentation pipeline and an uncertainty-aware failure explanation module. The project follows the experimental logic of the paper:

1. Run an ablation study inside the EfficientNet-B0 U-Net family.
2. Select **EfficientNet-B0 U-Net MultiScale** as the lightweight backbone.
3. Compare the selected model against U-Net, U-Net++, TransUNet, U-Mamba proxy, and VM-UNet proxy.
4. Apply uncertainty estimation to detect unreliable predictions and generate textual reliability warnings.

Recommended GitHub repository name:

```text
uncertainty-aware-effb0-unet-brain-mri
```

Short alternatives:

```text
effb0-unet-mri-uncertainty
lightweight-brain-tumor-segmentation-uq
```

---

## Highlights

- Lightweight **EfficientNet-B0 U-Net MultiScale** backbone for 2D brain tumor MRI segmentation.
- BRISC2025-compatible `.npy` dataset loader.
- Baseline comparison with U-Net, U-Net++, TransUNet, U-Mamba proxy, and VM-UNet proxy.
- Deployment-aware metrics: Dice, IoU, HD95, ASSD, Boundary-F1, parameters, and GFLOPs.
- Uncertainty-aware failure analysis using:
  - probability entropy,
  - test-time augmentation variance,
  - train-distribution distance,
  - predicted-component distance,
  - component-level instability.
- Case-level risk score, risk-group analysis, risk-coverage analysis, and automatic textual reliability warnings.

> The generated text warnings are reliability explanations for model predictions. They are **not** clinical diagnoses and should not replace physician review.

---

## Repository Structure

```text
.
├── configs/                         # Training configs for ablation and baselines
├── datasets/                        # .npy dataset loader and preprocessing helpers
├── losses/                          # Segmentation, boundary, and component-aware losses
├── models/                          # U-Net, U-Net++, TransUNet, proxy SSM baselines, EffB0 U-Net
├── scripts/                         # Experiment, baseline, diagnostic, and uncertainty scripts
├── utils/                           # Metrics, TTA, uncertainty, visualization, text reports
├── train.py                         # Training entry point
├── evaluate.py                      # Evaluation entry point
├── infer.py                         # Single-slice inference
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/uncertainty-aware-effb0-unet-brain-mri.git
cd uncertainty-aware-effb0-unet-brain-mri

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Google Colab:

```python
!pip install timm thop scipy scikit-learn pyyaml tqdm matplotlib pandas gdown
```

---

## Dataset Preparation

The code expects preprocessed BRISC2025 arrays in `.npy` format:

```text
datasets/brisc2025/X_train.npy
datasets/brisc2025/Y_train.npy
datasets/brisc2025/X_test.npy
datasets/brisc2025/Y_test.npy
```

Supported input shapes:

```text
X: [N,H,W], [N,H,W,C], or [N,C,H,W]
Y: [N,H,W], [N,H,W,1], or [N,1,H,W]
```

The paper uses binary tumor segmentation: tumor pixels are foreground, regardless of tumor type.

To download real data datset:

```bash
python scripts/download_data.py
```

To check the pipeline without the real dataset:

```bash
python scripts/make_dummy_data.py
```

---

## Main Model

The selected backbone in the paper is:

```text
configs/efficient_b0_multiscale_no_se.yaml
```

This configuration uses:

- EfficientNet-B0 encoder,
- U-Net decoder,
- multi-scale feature fusion,
- no SE block,
- no boundary head,
- no deep supervision.

This model is selected from the ablation study because it provides the best overall accuracy-efficiency trade-off for the paper experiments.

---

## Training

Train the selected EfficientNet-B0 U-Net MultiScale model:

```bash
python train.py --config configs/efficient_b0_multiscale_no_se.yaml
```

Expected outputs:

```text
outputs/effb0_multiscale_no_se/
├── checkpoints/best.pth
├── train_log.csv
└── visualizations/
```

---

## Evaluation

Evaluate the selected model on the test set:

```bash
python evaluate.py \
  --config configs/efficient_b0_multiscale_no_se.yaml \
  --checkpoint outputs/effb0_multiscale_no_se/checkpoints/best.pth \
  --split test
```

Evaluation outputs:

```text
outputs/effb0_multiscale_no_se/eval/
├── test_results.json
└── test_predictions.png
```

Reported metrics:

```text
Dice, IoU, Precision, Recall, F2, HD95, ASSD, Boundary-F1, Params, GFLOPs
```

---

## Ablation Study

The paper separates the EfficientNet-B0 U-Net ablation from the external baseline comparison.

Main ablation configs:

| Config | Description |
|---|---|
| `configs/efficient_b0_multiscale_no_se.yaml` | EfficientNet-B0 U-Net MultiScale, selected model |
| `configs/efficient_b0_multiscale_boundary_no_se.yaml` | MultiScale + boundary head + deep supervision |
| `configs/efficient_b0_boundary.yaml` | Boundary/deep-supervision variant |

The paper reports the following ablation summary:

| Variant | Dice | IoU | HD95 | ASSD | B-F1 | Params(M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| EffB0 U-Net | 0.8752 | 0.8068 | 7.3076 | 2.6271 | 0.8581 | 6.10 | 3.20 |
| **EffB0 U-Net MultiScale** | **0.8772** | **0.8095** | **6.9258** | **2.4326** | **0.8639** | 6.91 | 4.26 |
| EffB0 MultiScale + SE | 0.8701 | 0.8017 | 7.2000 | 2.6037 | 0.8555 | 6.91 | 4.26 |
| EffB0 MultiScale + Boundary | 0.8736 | 0.8052 | 7.4557 | 2.6146 | 0.8607 | 6.92 | 4.57 |
| EffB0 MultiScale + Boundary + SE | 0.8753 | 0.8075 | 7.0762 | 2.5704 | 0.8624 | 6.92 | 4.57 |

---

## Baseline Comparison

Run all core baselines:

```bash
bash scripts/run_core_baselines.sh
```

Evaluate all core baselines:

```bash
bash scripts/evaluate_core_baselines.sh
```

The summary is saved to:

```text
outputs/core_baseline_summary.csv
```

Paper baseline summary:

| Model | Dice | IoU | HD95 | ASSD | B-F1 | Params(M) | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.8608 | 0.7899 | 9.1219 | 2.9200 | 0.8369 | 7.85 | 13.98 |
| U-Net++ | 0.8634 | 0.7926 | 9.5402 | 3.0462 | 0.8379 | 9.16 | 34.59 |
| TransUNet | 0.8604 | 0.7874 | 9.1760 | 2.9577 | 0.8326 | 6.48 | 13.04 |
| U-Mamba proxy | 0.8587 | 0.7862 | 8.7777 | 2.8665 | 0.8373 | 10.55 | 18.02 |
| VM-UNet proxy | 0.8664 | 0.7951 | 8.0131 | 2.6434 | 0.8445 | 10.50 | 17.74 |
| **EffB0 U-Net MultiScale** | **0.8772** | **0.8095** | **6.9258** | **2.4326** | **0.8639** | 6.91 | **4.26** |

### Note on U-Mamba and VM-UNet

`models/umamba.py` and `models/vmunet.py` remain runnable proxy baselines. The repository now also includes `models/official_umamba.py` for U-Mamba_Bot and U-Mamba_Enc using the real `mamba_ssm` operator, plus `models/official_vmunet.py` for the official VM-UNet adapter. Use the explicit paper-baseline configs and clearly distinguish proxies from official-architecture adapters.

---

## Uncertainty-Aware Failure Explanation

After training the selected model, run:

```bash
CONFIG=configs/efficient_b0_multiscale_no_se.yaml \
CHECKPOINT=outputs/effb0_multiscale_no_se/checkpoints/best.pth \
DEVICE=cuda \
SPLIT=test \
THRESHOLD=0.50 \
OUT_DIR=outputs/uncertainty_analysis \
bash scripts/run_uncertainty_analysis.sh
```

For CPU:

```bash
DEVICE=cpu bash scripts/run_uncertainty_analysis.sh
```

Optional MC Dropout:

```bash
RUN_MC=1 bash scripts/run_uncertainty_analysis.sh
```

The uncertainty script produces:

```text
outputs/uncertainty_analysis/
├── tta_train_distance/
│   ├── uncertainty_cases.csv
│   ├── summary.json
│   ├── uncertainty_report.md
│   ├── risk_group_summary.csv
│   ├── risk_coverage.csv
│   ├── visualizations/
│   └── text_reports/
└── entropy_only_baseline/
    ├── uncertainty_cases.csv
    ├── summary.json
    ├── uncertainty_report.md
    ├── visualizations/
    └── text_reports/
```

Important columns in `uncertainty_cases.csv`:

```text
case_idx
dice / iou / hd95 / assd / boundary_f1
gt_area / pred_area
gt_component_count / pred_component_count
missed_component_count / fp_component_count
entropy_p95
tta_var_p95
train_image_distance
pred_component_train_distance
component_instability
case_uncertainty_score
risk_level
failure_subtype
text_warning
```

---

## Uncertainty Metrics Reported in the Paper

Failure detection:

| Score | AUROC | AUPRC |
|---|---:|---:|
| Entropy-only | 0.6197 | 0.2168 |
| **Proposed uncertainty score** | **0.8041** | **0.4487** |

Risk-group analysis:

| Risk group | Number of cases | Mean Dice | Failure rate |
|---|---:|---:|---:|
| Low-risk | 745 | 0.8946 | 8.72% |
| High-risk | 41 | 0.6388 | 53.66% |

Risk-coverage analysis:

| Coverage | Mean Dice | Failure rate |
|---|---:|---:|
| 100% test cases | 0.8711 | 13.84% |
| 80% most reliable cases | 0.9025 | 7.27% |

Text explanation validity:

```text
Text-claim macro-F1 = 0.9333
```

This metric measures agreement with automatically derived failure labels and uncertainty conditions. It is not a substitute for physician validation.

---

## Qualitative Outputs

Each saved qualitative figure contains:

```text
MRI
GT mask
Prediction
Probability map
Uncertainty map
Error map: FP red / TP green / FN blue
Uncertainty overlay
Automatic textual warning
```

These figures correspond to the qualitative analysis section of the paper and can be used directly in a conference submission, provided patient/data-license requirements are satisfied.

---

## Single-Slice Inference

```bash
python infer.py \
  --config configs/efficient_b0_multiscale_no_se.yaml \
  --checkpoint outputs/effb0_multiscale_no_se/checkpoints/best.pth \
  --input_npy datasets/brisc2025/X_test.npy \
  --index 0 \
  --output outputs/inference_result.png
```

---

## Reproducibility Notes

- Random seed is set to `42` in the provided configs.
- Input resolution is `256 x 256`.
- MRI images are normalized with z-score normalization inside `NPYSliceDataset`.
- Dataset arrays and model checkpoints are not included in this repository.
- The code is designed for binary tumor segmentation from BRISC2025-style `.npy` arrays.
- Legacy `umamba` and `vmunet_proxy` remain proxies. Use `umamba_bot_official`, `umamba_enc_official`, and `vmunet_official` for the non-proxy paper baseline runner.

---

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@inproceedings{nguyen2026effb0uncertainty,
  title     = {A Lightweight Multi-Scale EfficientNet-B0 U-Net and Uncertainty Analysis for Brain Tumor Segmentation in MRI},
  author    = {Nguyen, Thien Phuc},
  booktitle = {Conference Proceedings},
  year      = {2026}
}
```

Update the `booktitle` field after the paper is accepted.

---

## Disclaimer

This repository is intended for research use. The model outputs, uncertainty maps, and textual warnings are decision-support artifacts and must not be used as standalone clinical diagnoses.

## Medical-SAM3 checkpoint

```bash
bash scripts/download_medical_sam3_checkpoint.sh
python train_student_kd.py --config configs/kd/kd_medical_sam3_checkpoint3d_tumor_box.yaml
```

The checkpoint is saved to `checkpoints/medical_sam3/checkpoint_3D.pt`. The Medical-SAM3 repository code must be available at `external/Medical-SAM3` or the YAML `teacher.repo_path` must be changed.
