# Validation-Locked Model Selection and Factorial Ablation

Tài liệu này hướng dẫn quy trình mới để sửa hai vấn đề phương pháp luận:

1. **Không chọn kiến trúc bằng test set.** Mọi checkpoint và kiến trúc được chọn bằng validation set theo quy tắc đã khai báo trước.
2. **Tách riêng đóng góp của Skip-SE, boundary supervision và deep supervision.** Quy trình mới chạy thiết kế factorial đầy đủ trên ba yếu tố này, đồng thời giữ một cặp anchor Base → MultiScale để đo đóng góp ban đầu của MultiScale.

## 1. Thiết kế thực nghiệm mới

Có 9 kiến trúc duy nhất. Mỗi kiến trúc chạy với 3 seed huấn luyện `42, 43, 44`, tổng cộng 27 lần train.

| ID | MultiScale | Skip-SE | Boundary | Deep supervision | Mục đích |
|---|---:|---:|---:|---:|---|
| `f0_base` | 0 | 0 | 0 | 0 | Anchor mô hình cơ sở |
| `f1_m_only` | 1 | 0 | 0 | 0 | Main effect ban đầu của MultiScale |
| `f2_m_s` | 1 | 1 | 0 | 0 | Skip-SE riêng |
| `f3_m_b` | 1 | 0 | 1 | 0 | Boundary riêng |
| `f4_m_d` | 1 | 0 | 0 | 1 | Deep supervision riêng |
| `f5_m_s_b` | 1 | 1 | 1 | 0 | Interaction Skip-SE × Boundary |
| `f6_m_s_d` | 1 | 1 | 0 | 1 | Interaction Skip-SE × Deep supervision |
| `f7_m_b_d` | 1 | 0 | 1 | 1 | Interaction Boundary × Deep supervision |
| `f8_full_m_s_b_d` | 1 | 1 | 1 | 1 | Mô hình Full |

Thiết kế trên cho phép tách:

- tác động của MultiScale: `f1_m_only - f0_base`;
- main effect của Skip-SE, Boundary và Deep supervision;
- các interaction hai chiều `S:B`, `S:D`, `B:D`;
- interaction ba chiều `S:B:D`.

## 2. Quy tắc chọn checkpoint đã khóa trước

Mỗi lần train chỉ dùng validation set để chọn checkpoint. Quy tắc mặc định:

1. Tối đa hóa validation Dice.
2. Nếu Dice nằm trong khoảng tuyệt đối `0.001` so với checkpoint tốt hơn, ưu tiên:
   - missing-prediction rate thấp hơn;
   - HD95 thấp hơn;
   - ASSD thấp hơn;
   - Boundary-F1 cao hơn.
3. Nếu vẫn hòa, giữ checkpoint xuất hiện sớm hơn để tránh thay đổi hậu nghiệm.

Quy tắc nằm trong YAML tại:

```yaml
training:
  split_seed: 42
  validation_empty_surface_penalty: image_diagonal
  checkpoint_selection:
    primary:
      metric: dice
      mode: max
      tolerance: 0.001
    tie_breakers:
      - metric: missing_prediction_rate
        mode: min
        tolerance: 0.0
      - metric: hd95
        mode: min
        tolerance: 0.0
      - metric: assd
        mode: min
        tolerance: 0.0
      - metric: boundary_f1
        mode: max
        tolerance: 0.0
```

`split_seed=42` được giữ giống nhau cho tất cả kiến trúc và seed huấn luyện. Seed `42, 43, 44` chỉ thay đổi khởi tạo mô hình và ngẫu nhiên trong huấn luyện.

## 3. Xử lý ca prediction rỗng trong validation

HD95 và ASSD không xác định khi một mask rỗng còn mask kia không rỗng. Code mới không loại bỏ im lặng các ca này.

Trong validation selection:

- prediction rỗng trên ground truth có tổn thương được tính vào `missing_prediction_rate`;
- HD95 và ASSD không hữu hạn được thay bằng đường chéo của ảnh đã resize;
- với ảnh `256 × 256`, penalty là `sqrt(255² + 255²)` pixel;
- quy tắc này được lưu trong checkpoint và `validation_best.json`.

Điều này ngăn một checkpoint có nhiều complete misses nhưng HD95 hữu hạn thấp giả tạo được chọn.

## 4. Chuẩn bị môi trường

Từ thư mục gốc của project:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
```
Tải dữ liệu: 
```bash
chmod -R +x scripts
./scripts/download_data.sh
```
Kiểm tra dữ liệu:

```bash
ls datasets/brisc2025/X_train.npy
ls datasets/brisc2025/Y_train.npy
ls datasets/brisc2025/X_test.npy
ls datasets/brisc2025/Y_test.npy
```

## 5. Bước 1 — Sinh toàn bộ config factorial

Các config đã được tạo sẵn trong `configs/factorial/`, nhưng có thể tái tạo bằng:

```bash
python scripts/create_factorial_ablation_configs.py \
  --base-config configs/efficient_b0_boundary.yaml \
  --output-root configs/factorial \
  --runs-root outputs/factorial \
  --seeds 42 43 44 \
  --split-seed 42
```

Kết quả:

```text
configs/factorial/manifest.yaml
configs/factorial/f0_base/seed_42.yaml
...
configs/factorial/f8_full_m_s_b_d/seed_44.yaml
```

## 6. Bước 2 — Train toàn bộ mô hình, chỉ dùng train/validation

```bash
python scripts/run_factorial_train_validation.py \
  --manifest configs/factorial/manifest.yaml \
  --device cuda
```

Script này **không gọi `evaluate.py` và không đọc test set**.

Chạy lại sẽ tự bỏ qua các run đã có `validation_best.json`. Muốn train lại một run:

```bash
python scripts/run_factorial_train_validation.py \
  --manifest configs/factorial/manifest.yaml \
  --device cuda \
  --variants f4_m_d \
  --seeds 42 \
  --force
```

Mỗi run tạo:

```text
outputs/factorial/<variant>/seed_<seed>/
├── data_split.json
├── train_log.csv
├── checkpoint_selection.csv
├── validation_best.json
└── checkpoints/
    ├── best_validation.pth
    └── epoch_XXX.pth
```

### Ý nghĩa các file

- `data_split.json`: danh sách train/validation index và SHA-256 của split.
- `train_log.csv`: metric từng epoch.
- `checkpoint_selection.csv`: epoch nào được chọn, lý do chọn hoặc không chọn.
- `validation_best.json`: metric validation tốt nhất theo quy tắc đã khóa.
- `best_validation.pth`: checkpoint được chọn hoàn toàn từ validation.

## 7. Chạy song song trên nhiều GPU

Ví dụ chia 9 biến thể trên 3 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_factorial_train_validation.py \
  --device cuda --variants f0_base f1_m_only f2_m_s &

CUDA_VISIBLE_DEVICES=1 python scripts/run_factorial_train_validation.py \
  --device cuda --variants f3_m_b f4_m_d f5_m_s_b &

CUDA_VISIBLE_DEVICES=2 python scripts/run_factorial_train_validation.py \
  --device cuda --variants f6_m_s_d f7_m_b_d f8_full_m_s_b_d &

wait
```

Không chạy hai process trên cùng một config/output directory.

## 8. Bước 3 — Chọn kiến trúc bằng validation và khóa quyết định

Sau khi đủ 27 run:

```bash
python scripts/select_architecture_from_validation.py \
  --manifest configs/factorial/manifest.yaml \
  --output-dir outputs/factorial/selection \
  --dice-tolerance 0.001
```

Script sẽ:

1. kiểm tra mọi run có checkpoint validation;
2. xác minh mọi run dùng cùng validation split;
3. tính mean ± SD validation qua ba seed cho từng kiến trúc;
4. chọn kiến trúc theo quy tắc đã định;
5. ghi SHA-256 của toàn bộ config và checkpoint;
6. tạo file khóa trước khi test.

Kết quả:

```text
outputs/factorial/selection/
├── validation_per_run.csv
├── validation_architecture_summary.csv
└── SELECTION_LOCK.json
```

**Không chỉnh YAML, checkpoint hoặc selection rule sau khi tạo lock.** Nếu phải thay đổi, xóa toàn bộ kết quả test cũ, giải thích thay đổi, train/select lại từ đầu rồi mới test.

## 9. Bước 4 — Đánh giá confirmatory chỉ kiến trúc đã chọn

```bash
python scripts/evaluate_factorial_after_lock.py \
  --lock outputs/factorial/selection/SELECTION_LOCK.json \
  --scope selected \
  --device cuda
```

Script xác minh hash của config và checkpoint trước khi đọc test set.

Kết quả của ba seed nằm tại:

```text
outputs/factorial/<selected_variant>/seed_42/eval/test_results.json
outputs/factorial/<selected_variant>/seed_43/eval/test_results.json
outputs/factorial/<selected_variant>/seed_44/eval/test_results.json
```

File audit:

```text
outputs/factorial/selection/TEST_EVALUATION_SELECTED.json
```

Đây là kết quả confirmatory chính để báo cáo cho mô hình cuối.

## 10. Bước 5 — Đánh giá toàn bộ ablation sau khi đã khóa kiến trúc

Chỉ thực hiện sau Bước 4 và sau khi lock đã tồn tại:

```bash
python scripts/evaluate_factorial_after_lock.py \
  --lock outputs/factorial/selection/SELECTION_LOCK.json \
  --scope all \
  --device cuda
```

Các kết quả này được mô tả là:

> Post-lock ablation test results; they were not used for architecture selection.

Không dùng bảng test này để thay đổi selected architecture.

## 11. Bước 6 — Tổng hợp bảng và factorial effects

```bash
python scripts/summarize_factorial_results.py \
  --manifest configs/factorial/manifest.yaml \
  --lock outputs/factorial/selection/SELECTION_LOCK.json \
  --output-dir outputs/factorial/summary \
  --require-all-test-results
```

Kết quả:

```text
outputs/factorial/summary/
├── test_per_run.csv
├── test_architecture_summary.csv
├── factorial_effects.csv
└── factorial_report.md
```

### File dùng để cập nhật paper

- `test_architecture_summary.csv`: mean ± sample SD qua ba seed cho từng kiến trúc.
- `factorial_effects.csv`: main effects và interaction, tính paired theo seed.
- `factorial_report.md`: bảng đọc nhanh, có thể chuyển sang LaTeX.
- `validation_architecture_summary.csv`: bằng chứng selected architecture được chọn bằng validation.
- `SELECTION_LOCK.json`: audit trail cho reviewer.

## 12. Cách đọc factorial effects

Trong `factorial_effects.csv`:

- `M`: MultiScale-only minus Base.
- `S`: mean cell có Skip-SE minus mean cell không Skip-SE.
- `B`: mean cell có Boundary minus mean cell không Boundary.
- `D`: mean cell có Deep supervision minus mean cell không Deep supervision.
- `S:B`, `S:D`, `B:D`: difference-of-differences.
- `S:B:D`: interaction ba chiều.

Dấu tốt:

- Dice, IoU, BF1: delta dương tốt hơn.
- HD95, ASSD: delta âm tốt hơn.

Với chỉ ba seed, các effect này nên được báo cáo là **descriptive paired contrasts**, chưa nên gọi là kiểm định thống kê mạnh.

## 13. Wording đề xuất cho Methods

Có thể dùng đoạn sau:

> All checkpoint and architecture decisions were made exclusively on a fixed validation partition shared by every factorial cell. Checkpoints were selected using a pre-specified lexicographic rule: validation Dice was maximized, and candidates within 0.001 absolute Dice were ranked by missing-prediction rate, HD95, ASSD, and Boundary-F1. The final architecture was locked from mean validation metrics across three training seeds before any test-set evaluation. Configuration and checkpoint hashes were stored in an immutable selection manifest. A base anchor and a complete 2×2×2 factorial over Skip-SE, boundary supervision, and deep supervision were evaluated with MultiScale enabled, allowing separate estimation of main effects and interactions.

## 14. Wording đề xuất cho Results

> The selected architecture was determined from validation data before test evaluation. Test-set ablations were computed only after the selection lock had been created and were not used to revise the architecture. Boundary-only, deep-supervision-only, combined, and Skip-SE interaction cells were evaluated independently, removing the confounding present when boundary and deep-supervision terms are introduced together.

## 15. Những điều không được làm

- Không chạy `scripts/run_ablation.sh` cũ để chọn mô hình; script cũ train rồi test ngay từng config.
- Không xem `test_results.json` trước khi tạo `SELECTION_LOCK.json`.
- Không đổi tolerance hoặc thứ tự tie-breaker sau khi đã xem test.
- Không chọn seed tốt nhất để báo cáo; phải báo cáo mean ± SD qua tất cả seed đã khai báo.
- Không gọi test ablation là confirmatory nếu bảng đó được xem trước khi khóa kiến trúc.
- Không gộp Boundary và Deep supervision thành một thành phần trong phần diễn giải mới.

## 16. Kiểm thử nhanh code

Chạy unit test cho selection policy:

```bash
python -m unittest tests/test_validation_selection.py -v
```

Kiểm tra syntax toàn bộ file mới:

```bash
python -m py_compile \
  train.py evaluate.py \
  utils/selection.py \
  scripts/create_factorial_ablation_configs.py \
  scripts/run_factorial_train_validation.py \
  scripts/select_architecture_from_validation.py \
  scripts/evaluate_factorial_after_lock.py \
  scripts/summarize_factorial_results.py
```

## 17. Lưu ý còn lại

Các sửa đổi này giải quyết trực tiếp:

- chọn checkpoint/kiến trúc bằng validation thay vì test;
- cùng một validation split cho mọi run;
- audit trail chống chỉnh config/checkpoint sau lựa chọn;
- tách Boundary và Deep supervision;
- đo interaction giữa Skip-SE, Boundary và Deep supervision.

Chúng chưa tự động giải quyết patient-level independence, external validation, physical-spacing metrics hoặc paired case-level significance testing. Các vấn đề đó cần dữ liệu hoặc thực nghiệm bổ sung riêng.

## 18. Các phát hiện thêm khi audit code gốc

Ba điểm cần đồng bộ lại trong paper:

1. `NPYSliceDataset._normalize()` đang tính mean và standard deviation **riêng trên từng ảnh/lát cắt**, không phải dataset-level z-score. Phần Methods nên ghi `per-slice z-score normalization` trừ khi bạn thay preprocessing.
2. Config cũ `configs/ablations/m4_boundary.yaml` thực tế bật boundary head nhưng tắt deep supervision. Nếu bảng paper trước đây gọi hàng này là `Boundary + Deep Supervision`, tên hàng đó không khớp code. Thiết kế factorial mới tách rõ Boundary-only, DS-only và Boundary+DS.
3. `scripts/run_ablation.sh` cũ train xong rồi chạy test cho từng kiến trúc ngay lập tức. Không dùng script này cho bản paper sửa đổi vì nó không tạo bước khóa kiến trúc trước test.
