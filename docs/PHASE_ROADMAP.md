# Lộ trình triển khai và bằng chứng thực nghiệm

## Phase 9 — Biểu diễn tri thức logic

### Nội dung

- Biểu diễn các dấu hiệu rủi ro bằng fuzzy predicates.
- Fit quantile threshold và category-risk mapping chỉ trên train.
- Kết hợp predicates thành luật có tên và mô tả nghiệp vụ.
- Kiểm tra tensor logic khả vi bằng PyTorch.

### File chính

- `src/logic/predicates.py`
- `src/logic/fraud_rules.py`
- `src/logic/knowledge_base.py`
- `src/logic/tensor_logic.py`
- `tests/test_rules.py`
- `tests/test_tensor_logic.py`
- `notebooks/04_IEEE_CIS_LTN_Rule_Analysis.ipynb`

### Kết quả và phạm vi chứng minh

- Bảng threshold đã fit, coverage, fraud precision, lift và rule AUC.
- So sánh chất lượng luật giữa validation và locked test.
- Chứng minh tri thức miền có thể được biểu diễn thành truth value trong `[0, 1]` và đánh giá định lượng; không chứng minh quan hệ nhân quả.

## Phase 10 — Giải thích cảnh báo dựa trên luật

### Nội dung

- Sinh top-k rule evidence cho từng giao dịch.
- Đánh giá riêng prediction quality và explanation quality.
- Phân tích TP, FP và FN.
- Bootstrap precision gain của nhóm cảnh báo được giải thích.

### File chính

- `src/explanation/rule_explainer.py`
- `src/explanation/explanation_metrics.py`
- `notebooks/05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb`

### Kết quả và phạm vi chứng minh

- Explanation coverage trên toàn bộ mẫu và trên model alerts.
- Sparsity, số rule trung bình, consistency và contradiction rate.
- Explained-alert precision, precision gain và khoảng tin cậy bootstrap 95%.
- Chứng minh rule evidence hỗ trợ kiểm tra một phần cảnh báo; không tuyên bố đây là causal explanation.

## Phase 11 — Ablation và external generalization

### Nội dung

- Only-one-rule và leave-one-rule-out ablation trên IEEE-CIS.
- Lặp prediction, rule quality và explanation evaluation trên BAF.
- Giữ nguyên split, predictor probability và threshold khi thực hiện ablation.

### File chính

- `notebooks/06_IEEE_CIS_Rule_Ablation.ipynb`
- `notebooks/07_BAF_LTN_Generalization.ipynb`

### Kết quả và phạm vi chứng minh

- Bảng thay đổi coverage và precision gain khi bỏ từng luật.
- Xác định luật hữu ích, dư thừa hoặc gây giảm chất lượng.
- Kiểm tra phương pháp có thực thi và tạo bằng chứng giải thích trên một phân phối dữ liệu khác.
- Không đồng nhất cross-dataset generalization với production generalization.

## Phase 12 — Tổng hợp, kiểm thử và đóng gói kết quả

### Nội dung

- Tổng hợp bảng prediction metrics từ các run.
- Xuất run metadata, predictions và bảng mean ± standard deviation.
- Kiểm tra top-to-bottom notebooks, unit tests, schema và leakage contract.
- Hoàn thiện hướng dẫn local, Kaggle và giới hạn claim.

### File chính

- `scripts/run_experiment.py`
- `scripts/export_thesis_results.py`
- `scripts/generate_notebooks.py`
- `notebooks/08_Cross_Dataset_Result_Synthesis.ipynb`
- `docs/EXPERIMENTAL_PROTOCOL.md`
- `docs/RESULTS_GUIDE.md`
- `docs/KAGGLE_GUIDE.md`

### Kết quả và phạm vi chứng minh

- Một pipeline tái lập từ dữ liệu đến prediction, rule evidence và report artifacts.
- Kết quả model chính thức trên ba seed `[42, 123, 2026]`, báo cáo mean ± standard deviation.
- Test chỉ được đánh giá bằng quyết định đã khóa từ train/validation.
- Frozen predictor artifacts đảm bảo explanation và ablation không thay đổi model outputs.

## Local và Kaggle

Tất cả notebook đều có thể chạy local khi máy đã cài dependencies và có dữ liệu. GPU không phải yêu cầu chức năng.

- Nên chạy local: Data Exploration, tree-only benchmark, rule analysis, explanation evaluation, ablation và smoke tests.
- Nên chạy Kaggle GPU: full-data MLP/TabularResNet benchmarks với ba seed.
- Có thể chạy neural full protocol local nếu máy có GPU CUDA tương thích và đủ thời gian; CPU vẫn chạy được nhưng chậm.
- Synthetic/quick mode chỉ xác nhận code thực thi, không hỗ trợ kết luận khóa luận.
