# Multi-scale EfficientNet Student + Medical-SAM3/Medical Samba KD

This package extends the original multi-scale EfficientNet student project with a teacher-student training pipeline.

## What is added

- `train_student_kd.py`: train the multi-scale EfficientNet student with or without KD.
- `kd/medical_samba_teacher.py`: thin adapter for your 10GB Medical Samba / Medical-SAM3 checkpoint.
- `kd/losses.py`: KD losses: logit KD, soft Dice KD, boundary KD, selective KD, uncertainty-weighted KD.
- `kd/prompting.py`: text prompt and box prompt helpers.
- `configs/kd/student_no_kd.yaml`: baseline student without KD.
- `configs/kd/student_kd_medical_samba_text_tumor_box.yaml`: Medical Samba teacher with text=`tumor` and GT box prompt.
- `configs/kd/student_kd_medical_samba_text_polyp_box.yaml`: text=`polyp` + box prompt variant.
- `configs/kd/kd_medical_samba_only_box.yaml`: box-only prompt variant, no text prompt.
- `configs/kd/kd_medical_samba_only_tumor.yaml`: text-only `tumor` prompt variant, no box prompt.
- `configs/kd/kd_medical_samba_only_polyp.yaml`: text-only `polyp` prompt variant, no box prompt.
- `configs/kd/ablation_*.yaml`: No KD / Vanilla KD / Edge KD / Selective uncertainty KD ablations.

## Student model

The student is the existing multi-scale EfficientNet model:

```yaml
model:
  name: efficientnet_b0_unet_boundary
  encoder_name: efficientnet_b0
  use_multiscale: true
  use_se: true
  use_boundary_head: true
  use_deep_supervision: true
```

## Teacher model

The teacher is expected to be Medical Samba / Medical-SAM3. Because different Medical-SAM3 repositories expose different APIs, the code uses a thin adapter:

```text
kd/medical_samba_teacher.py
```

You only need to edit two methods if your repo API is different:

```python
_build_external_teacher()
_call_external_teacher()
```

The adapter expects the teacher to return either:

```python
{"logits": Tensor[B,1,H,W]}
```

or:

```python
{"prob": Tensor[B,1,H,W]}
```

If the teacher also returns uncertainty:

```python
{"logits": ..., "uncertainty": Tensor[B,1,H,W]}
```

then the KD loss will use it. If not, the code computes entropy from the teacher probability.

## How to point to your 10GB checkpoint

Edit the KD config:

```yaml
teacher:
  mode: external_callable
  repo_path: /workspace/Medical-SAM3
  builder: sam3.model_builder:build_sam3_image_model
  checkpoint: /workspace/checkpoints/medical_samba_10gb.pt
  output_type: logit
  output_key: logits
  prompt:
    text: tumor
    use_box: true
    box_pad: 6
```

`builder` means:

```text
python.module:function_name
```

For example, if your repo has:

```python
# /workspace/Medical-SAM3/medical_sam3/build_model.py

def build_model(checkpoint):
    ...
```

then use:

```yaml
builder: sam3.model_builder:build_sam3_image_model
```

## How prompts are used

The prompt config supports text + box, text-only, and box-only teacher calls:

```yaml
teacher:
  prompt:
    text: tumor   # tumor | polyp | null
    use_box: true # true for GT-box prompt, false for text-only
```

For box-only prompting, set:

```yaml
teacher:
  prompt:
    text: null
    use_box: true
```

For text-only prompting, set `use_box: false`.

During training only:

```text
MRI + text prompt + box prompt -> Medical-SAM3 teacher -> soft mask
MRI -> student -> student mask
student learns from ground truth + teacher soft mask
```

At inference:

```text
MRI -> student -> mask
```

The student receives no text prompt and no box prompt during inference.

## Main commands

### 1. Student without KD

```bash
python train_student_kd.py --config configs/kd/student_no_kd.yaml
```

### 2. Student with Medical Samba teacher using text `tumor` + box

```bash
python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_tumor_box.yaml
```

### 3. Student with Medical Samba teacher using text `polyp` + box

```bash
python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_polyp_box.yaml
```

### 4. Box-only Medical Samba KD

```bash
python train_student_kd.py --config configs/kd/kd_medical_samba_only_box.yaml
```

### 5. Text-only Medical Samba KD: `tumor`

```bash
python train_student_kd.py --config configs/kd/kd_medical_samba_only_tumor.yaml
```

### 6. Text-only Medical Samba KD: `polyp`

```bash
python train_student_kd.py --config configs/kd/kd_medical_samba_only_polyp.yaml
```

### 7. Full KD ablation

```bash
bash scripts/run_kd_ablation.sh
```

### 8. Full prompt ablation

```bash
bash scripts/run_kd_prompt_ablation.sh
```

The KD ablation script runs:

1. No KD
2. Vanilla KD
3. Edge/boundary KD
4. Selective + uncertainty-weighted + boundary KD

The prompt ablation script runs:

1. Box only
2. Tumor text only
3. Polyp text only
4. Tumor text + box
5. Polyp text + box

## Recommended ablation table

| Setting | Description |
|---|---|
| No KD | Student trained only with GT mask |
| Vanilla KD | BCE/Dice from teacher soft mask |
| Edge KD | Vanilla KD + teacher boundary supervision |
| Selective KD | Learn only confident teacher pixels |
| Selective + uncertainty KD | Downweight uncertain teacher pixels |
| Proposed | Selective + uncertainty-weighted + boundary-aware KD |

## Important note about lesion crop augmentation

For online teacher training with box prompts from ground truth, keep:

```yaml
augmentation:
  lesion_crop_prob: 0.0
```

Otherwise, the teacher box prompt is generated after augmentation/crop and can become inconsistent with the original Medical-SAM3 input expectation. You can enable lesion crop after you implement teacher precomputation on cropped images.

## Smoke test without real Medical Samba

To test the training loop only:

```bash
python train_student_kd.py --config configs/kd/debug_dummy_teacher_kd.yaml
```

This uses a dummy teacher from the GT mask. Do not report it in a paper.

## Download official Medical-SAM3 checkpoint_3D.pt

The official checkpoint is about **10.3 GB** and is stored inside this source tree at:

```text
checkpoints/medical_sam3/checkpoint_3D.pt
```

Recommended resumable download:

```bash
bash scripts/download_medical_sam3_checkpoint.sh
```

Alternative direct download using `curl`:

```bash
bash scripts/download_medical_sam3_checkpoint_curl.sh
```

The Python downloader verifies the official SHA256 by default. To skip the long hash pass:

```bash
python scripts/download_medical_sam3_checkpoint.py --skip-sha256
```

The configs now use this relative path:

```yaml
teacher:
  repo_path: external/Medical-SAM3
  checkpoint: checkpoints/medical_sam3/checkpoint_3D.pt
```

The model code repository is still required separately because a checkpoint only stores weights. Put/clone it at `external/Medical-SAM3`, then update `teacher.builder` if the repository exposes a different builder function.

Primary run config:

```bash
python train_student_kd.py \
  --config configs/kd/kd_medical_sam3_checkpoint3d_tumor_box.yaml
```

Prompt-specific configs using the same downloaded checkpoint:

```bash
python train_student_kd.py --config configs/kd/kd_medical_samba_only_box.yaml
python train_student_kd.py --config configs/kd/kd_medical_samba_only_tumor.yaml
python train_student_kd.py --config configs/kd/kd_medical_samba_only_polyp.yaml
```
