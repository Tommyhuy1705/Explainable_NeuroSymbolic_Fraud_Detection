# Explainable Neuro-Symbolic Learning for Financial Fraud Detection

Dự án nghiên cứu phương pháp phát hiện gian lận tài chính có khả năng giải thích bằng cách kết hợp mô hình học máy trên dữ liệu bảng với tri thức miền biểu diễn dưới dạng luật logic.

Hệ thống hướng đến hai mục tiêu bổ trợ: xây dựng mô hình dự đoán hiệu quả trên dữ liệu mất cân bằng và cung cấp bằng chứng dựa trên luật để hỗ trợ con người kiểm tra cảnh báo. Logic Tensor Network (LTN) được tiếp cận theo hướng fuzzy, differentiable logic để biểu diễn và đánh giá mức độ thỏa mãn của tri thức miền.

## Mục tiêu nghiên cứu

- Xây dựng pipeline xử lý dữ liệu giao dịch có mức mất cân bằng cao.
- So sánh MLP, TabularResNet và tree-based models dưới cùng giao thức đánh giá.
- Biểu diễn dấu hiệu gian lận bằng fuzzy predicates và luật logic.
- Sinh giải thích ngắn gọn từ các luật được kích hoạt.
- Đánh giá tách biệt prediction quality và explanation quality.
- Giới hạn data leakage bằng train/validation/test protocol rõ ràng.
- Cung cấp mã nguồn, cấu hình và notebook có thể chạy lại trên Kaggle.

## Câu hỏi nghiên cứu

1. Mô hình học máy nào phù hợp với dữ liệu gian lận mất cân bằng?
2. Tri thức miền dưới dạng luật có thể mô tả các vùng rủi ro như thế nào?
3. Bằng chứng logic hỗ trợ diễn giải cảnh báo của mô hình ở mức độ nào?
4. Có sự đánh đổi nào giữa hiệu quả dự đoán, độ bao phủ và độ ngắn gọn của giải thích?
5. Làm thế nào đánh giá hệ thống một cách leakage-safe và có thể tái lập?

## Kiến trúc tổng quát

```text
Dữ liệu giao dịch
       │
       ▼
Tiền xử lý và xây dựng đặc trưng
       │
       ▼
MLP / TabularResNet / Tree model
       │
       ├──────────────► Prediction metrics
       │
       ▼
Fuzzy predicates và fraud rules
       │
       ▼
Rule-based explanations
       │
       └──────────────► Explanation metrics và case studies
```

Prediction và explanation được đánh giá riêng. Calibration, threshold và các tham số lựa chọn được fit trên validation; test chỉ dùng để báo cáo cuối cùng.

## Datasets

### IEEE-CIS Fraud Detection

Bộ dữ liệu chính, gồm `train_transaction.csv` và tùy chọn `train_identity.csv`. Loader tự ghép hai bảng theo `TransactionID` và tạo các đặc trưng thời gian cơ bản.

### Bank Account Fraud Dataset Suite

Bộ dữ liệu mở rộng, sử dụng `Base.csv`. Cấu hình BAF mô tả các luật liên quan đến velocity, device-email linkage, foreign request, session length và credit risk.

Dataset không được lưu trong Git. Xem [data/README.md](data/README.md) để biết cách tải, gắn Kaggle dataset và bố trí file.

## Project structure

```text
configs/           Dataset, model, logic và evaluation settings
data/              Hướng dẫn dữ liệu; raw data bị gitignore
docs/              Phạm vi khóa luận và giao thức thực nghiệm
notebooks/         Sáu notebook độc lập, chạy được trên Kaggle
results/           Bảng và hình đã chọn cho báo cáo
scripts/           CLI chạy experiment và export kết quả
src/               Package triển khai pipeline nghiên cứu
tests/             Kiểm tra preprocessing, rules và protocol
```

Chi tiết package:

```text
src/
├── data/           Loading, temporal/stratified split, preprocessing
├── models/         MLP, TabularResNet, XGBoost/LightGBM factory
├── logic/          Fuzzy rules và differentiable tensor-logic knowledge base
├── explanation/    Rule explainer và explanation metrics
├── training/       PyTorch training và inference
├── evaluation/     Prediction metrics, calibration, threshold lock
└── experiment.py   Workflow dùng chung cho CLI và notebooks
```

## Cài đặt local

Khuyến nghị Python 3.11 hoặc 3.12:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Chạy smoke experiment không cần dữ liệu thật

```bash
python scripts/run_experiment.py \
  --config configs/ieee_cis.yaml \
  --models tree \
  --quick \
  --synthetic-fallback
```

Synthetic data chỉ dùng để kiểm tra khả năng thực thi. Không sử dụng kết quả synthetic để đưa ra kết luận nghiên cứu.

## Chạy với dữ liệu thật

```bash
python scripts/run_experiment.py \
  --config configs/ieee_cis.yaml \
  --data-root data/raw/ieee-cis
```

Để chạy nhanh trong giai đoạn phát triển:

```bash
python scripts/run_experiment.py \
  --config configs/ieee_cis.yaml \
  --data-root data/raw/ieee-cis \
  --quick \
  --max-rows 12000
```

Artifacts được ghi vào `results/runs/<dataset>/` và không được commit mặc định.

## Kaggle notebooks

| Notebook | Mục tiêu |
|---|---|
| `01_Data_Exploration.ipynb` | Data quality, imbalance và temporal distribution |
| `02_Predictive_Model_Benchmarks.ipynb` | So sánh ba nhóm predictor |
| `03_LTN_Rule_Analysis.ipynb` | Fit threshold train-only và phân tích từng luật |
| `04_Rule_Explanation_Evaluation.ipynb` | Đánh giá explanation coverage và consistency |
| `05_Rule_Ablation.ipynb` | Đánh giá vai trò của từng nhóm luật |
| `06_BAF_Generalization.ipynb` | Thí nghiệm mở rộng trên BAF |

Mỗi notebook có `QUICK_RUN`, auto-discovery cho `/kaggle/input`, synthetic fallback để smoke test và output riêng trong `/kaggle/working/thesis_outputs/`. Xem [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md).

## Evaluation contract

1. Split dữ liệu trước mọi thao tác học tham số.
2. Preprocessor chỉ fit trên train.
3. Model chỉ fit trên train.
4. Calibration và threshold chỉ fit/chọn trên validation.
5. Rule quantiles và category-risk mapping chỉ fit trên train.
6. Test được đánh giá một lần bằng toàn bộ quyết định đã khóa.
7. Không dùng test để chọn feature, rule, model hoặc hyperparameter.

Prediction metrics chính:

- PR-AUC.
- ROC-AUC.
- Precision, recall, F1 và F2.
- Brier score, ECE và NLL.

Explanation metrics chính:

- Rule coverage và fraud precision.
- Rule lift và rule AUC.
- Explanation coverage.
- Mean rule count và sparsity.
- Prediction-rule consistency.
- Contradiction rate.
- Case-study analysis.

## Tests

```bash
pytest -q
```

Tests tập trung vào các lỗi có thể làm sai kết luận:

- Temporal split không đảo thứ tự thời gian.
- Imputation/scaling chỉ fit trên train.
- Rule truth value nằm trong `[0, 1]`.
- Threshold selection chỉ nhận validation inputs.
- Explanation output và metrics có schema ổn định.

## Reproducibility

- Seed mặc định: `42`.
- Mọi tham số nằm trong YAML config.
- Mỗi run xuất metrics, prediction arrays và metadata.
- Notebook không chứa implementation chính; chúng gọi module trong `src/`.
- Kết quả synthetic phải được gắn nhãn rõ và không dùng trong báo cáo.
- Kết quả chính thức phải lưu config, data source, row count và feature count.

## Tài liệu

- [Phạm vi và nội dung khóa luận](docs/PHAM_VI_VA_NOI_DUNG_KHOA_LUAN.md)
- [Giao thức thực nghiệm](docs/EXPERIMENTAL_PROTOCOL.md)
- [Hướng dẫn Kaggle](docs/KAGGLE_GUIDE.md)
- [Hướng dẫn kết quả](docs/RESULTS_GUIDE.md)
- [Mô tả dữ liệu](data/README.md)

## Giới hạn

- Rule evidence thể hiện sự phù hợp với tri thức đã cấu hình, không tự động chứng minh quan hệ nhân quả.
- Kết quả trên một dataset không đại diện cho mọi hệ thống tài chính.
- Mô hình chưa được đánh giá bằng user study với fraud analysts.
- Prototype không phải hệ thống ra quyết định tài chính sẵn sàng triển khai.
- Temporal hoặc prior shift có thể làm giảm calibration và hiệu quả rule.

## License

Mã nguồn được phát hành theo MIT License. Dữ liệu phải được sử dụng theo giấy phép và điều khoản của nguồn tương ứng.
