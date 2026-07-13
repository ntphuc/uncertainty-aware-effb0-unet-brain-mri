# Full setup, checkpoint download, training, and evaluation pipeline

Run every command from the project root.

## 1. Install requirements and download data

The exact manual commands are:

```bash
python -m pip install -r requirements.txt
chmod -R +x scripts
./scripts/download_data.sh
```

Or run the combined setup script, which also downloads the Medical-SAM3 checkpoint:

```bash
./scripts/00_setup_and_download.sh
```

## 2. Download Medical-SAM3 checkpoint

```bash
./scripts/01_download_checkpoint.sh
```

The checkpoint is stored at:

```text
checkpoints/medical_sam3/checkpoint_3D.pt
```

Before using the teacher, place or clone the Medical-SAM3 source repository at:

```text
external/Medical-SAM3
```

Then verify:

```bash
python scripts/check_medical_sam3_setup.py
```

## 3. Train students

### Student without KD

```bash
python train_student_kd.py --config configs/kd/student_no_kd.yaml
```

### Main student with Medical-SAM3, text `tumor` + GT box

```bash
python train_student_kd.py --config configs/kd/student_kd_medical_samba_text_tumor_box.yaml
```

### KD loss ablations

```bash
./scripts/run_kd_ablation.sh
```

### Prompt ablations

```bash
./scripts/run_kd_prompt_ablation.sh
```

## 4. Evaluate Medical-SAM3 teacher

Main teacher evaluation using text `tumor` + GT-derived box:

```bash
./scripts/eval_teacher.sh
```

Choose another teacher prompt config:

```bash
./scripts/eval_teacher.sh configs/kd/kd_medical_samba_only_box.yaml
./scripts/eval_teacher.sh configs/kd/kd_medical_samba_only_tumor.yaml
./scripts/eval_teacher.sh configs/kd/kd_medical_samba_only_polyp.yaml
```

Evaluate all teacher prompt variants:

```bash
./scripts/eval_teacher_prompts.sh
```

**Important:** when `use_box: true`, the box is generated from the ground-truth mask. The resulting teacher score is a prompt-conditioned upper bound, not automatic inference performance.

## 5. Evaluate one student

The script automatically reads `paths.output_dir` from the YAML and uses:

```text
<output_dir>/checkpoints/best.pth
```

Example:

```bash
./scripts/eval_student.sh configs/kd/student_no_kd.yaml
```

Or provide the checkpoint explicitly:

```bash
./scripts/eval_student.sh \
  configs/kd/student_kd_medical_samba_text_tumor_box.yaml \
  outputs/kd_ablation/student_kd_tumor_box/checkpoints/best.pth
```

## 6. Evaluate all available student ablations

```bash
./scripts/eval_student_ablations.sh
```

Missing checkpoints are skipped by default. To fail on the first missing checkpoint:

```bash
STRICT=1 ./scripts/eval_student_ablations.sh
```

The combined CSV is written to:

```text
outputs/kd_ablation_summary.csv
```

## 7. Train and evaluate the four KD-loss ablations

```bash
./scripts/train_and_eval_kd_ablations.sh
```

This runs:

1. No KD
2. Vanilla KD
3. Edge KD
4. Selective + uncertainty-weighted KD

and then evaluates all checkpoints that exist.

## 8. Evaluate teacher, main student, and all student ablations

```bash
./scripts/eval_all_models.sh
```

Optional environment variables:

```bash
DEVICE=cuda SPLIT=test BATCH_SIZE=1 ./scripts/eval_teacher.sh
DEVICE=cuda SPLIT=test ./scripts/eval_student_ablations.sh
```

## Output locations

```text
outputs/teacher_eval/<prompt>/test_results.json
outputs/teacher_eval/<prompt>/test_predictions.png
<student_output_dir>/eval/test_results.json
<student_output_dir>/eval/test_predictions.png
outputs/kd_ablation_summary.csv
```

## 9. Evaluate student architecture ablations

The source also contains the architecture ablation sequence:

1. EfficientNet-B0 U-Net
2. + multi-scale fusion
3. + SE attention
4. + boundary head/loss
5. full student

Evaluate all available checkpoints:

```bash
./scripts/eval_student_architecture_ablations.sh
```

Summary output:

```text
outputs/ablations/all_summary.csv
```
