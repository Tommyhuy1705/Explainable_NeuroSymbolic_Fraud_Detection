# An Explainable Neuro-Symbolic Audit Framework for Financial Fraud Detection

Dự án nghiên cứu một framework lai cho phát hiện gian lận tài chính: predictor học máy được huấn luyện và đánh giá trước, sau đó được khóa để một lớp fuzzy-rule/LTN-inspired cung cấp bằng chứng giải thích chọn lọc và hỗ trợ audit cảnh báo.

Hai thành phần prediction và explanation có vai trò tách biệt. Predictor tạo xác suất và quyết định cảnh báo; rule layer không huấn luyện lại hoặc thay đổi predictor mà biểu diễn tri thức miền bằng fuzzy predicates, đánh giá mức độ thỏa mãn luật và chỉ giải thích những trường hợp có đủ rule evidence. Đây là framework audit hậu kiểm lấy cảm hứng từ LTN, không phải mô hình LTN end-to-end hoặc một thuật toán LTN mới.

## Mục tiêu nghiên cứu

- Xây dựng pipeline xử lý dữ liệu giao dịch có mức mất cân bằng cao.
- So sánh MLP, TabularResNet và tree-based models dưới cùng giao thức đánh giá.
- Khóa reference predictor trước khi phân tích luật và giải thích.
- Biểu diễn dấu hiệu gian lận bằng fuzzy predicates và luật logic LTN-inspired.
- Sinh giải thích ngắn gọn từ các luật được kích hoạt.
- Đánh giá tách biệt prediction quality và explanation quality.
- Giới hạn data leakage bằng train/validation/test protocol rõ ràng.
- Cung cấp mã nguồn, cấu hình và notebook có thể chạy lại trên Kaggle.

## Câu hỏi nghiên cứu

1. Mô hình học máy nào phù hợp với dữ liệu gian lận mất cân bằng?
2. Tri thức miền dưới dạng luật có thể mô tả các vùng rủi ro như thế nào?
3. Bằng chứng logic hỗ trợ diễn giải cảnh báo của mô hình ở mức độ nào?
4. Bằng chứng giải thích chọn lọc có trade-off như thế nào giữa alert coverage và fraud precision, đồng thời giải thích ngắn gọn đến mức nào?
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
       ▼ khóa model, calibration và decision threshold
Frozen reference predictor
       │
       ├──────────────► Model alerts
       │
       ▼
Train-fitted fuzzy predicates và fraud rules
       │
       ▼
Selective rule evidence / audit explanations
       │
       └──────────────► Rule quality, explanation metrics và case studies
```

Prediction và explanation được đánh giá riêng. Calibration, decision threshold và các tham số lựa chọn được fit hoặc chọn trên validation; test chỉ dùng để báo cáo cuối cùng. Rule evidence thể hiện sự phù hợp với tập luật đã cấu hình, không mặc nhiên là nguyên nhân của dự đoán hoặc một mô tả faithful đầy đủ của predictor.

## Datasets

### IEEE-CIS Fraud Detection

Bộ dữ liệu chính, gồm `train_transaction.csv` và tùy chọn `train_identity.csv`. Loader tự ghép hai bảng theo `TransactionID` và tạo các đặc trưng thời gian cơ bản. `transaction_hour` chỉ là cyclic phase suy ra từ `TransactionDT`, không phải giờ địa phương đã biết vì clock origin/timezone không được xác định.

### Bank Account Fraud Dataset Suite

BAF là benchmark tổng hợp bảo toàn riêng tư (privacy-preserving synthetic benchmark), sử dụng `Base.csv`. Cấu hình BAF mô tả các luật liên quan đến velocity, device-email linkage, foreign request, session length và credit risk. Việc loader đọc file benchmark gốc và báo `data_source = real` hoặc `official_benchmark_file` chỉ phân biệt với synthetic fallback nội bộ của repository; nó không biến BAF thành dữ liệu giao dịch thực của một tổ chức tài chính.

BAF có thể mã hóa giá trị không có/không áp dụng bằng sentinel `-1` ở một số trường. Vì vậy, kết quả `0` native `NaN` không đồng nghĩa dữ liệu hoàn toàn không có missingness về mặt ngữ nghĩa. Protocol hiện tại giữ nguyên `-1` như giá trị mã hóa do benchmark cung cấp, không tự động đổi thành `NaN`; EDA kiểm đếm native `NaN` và sentinel `-1` riêng, và báo cáo phải nêu quyết định xử lý này như một giả định phương pháp.

### Stress-test extension: TransXion và AMLNet

Hai dataset bổ sung được dùng như **stress tests**, không phải dataset chính thứ ba và thứ tư:

- **TransXion** kiểm tra framework trong một AML benchmark khó, quy mô lớn và có nhãn rất hiếm. Tên notebook `TransXion_v2` chỉ revision v2 của paper arXiv; official repository không có một dataset release `v2` riêng. Full run khóa canonical `tx.csv` theo official repository commit và SHA-256.
- **AMLNet v1.0** kiểm tra trường hợp saturation/limited headroom, nơi score-only predictor có thể đã rất mạnh. Full run khóa Zenodo DOI, exact filename và MD5; dữ liệu mang giấy phép CC BY-NC 4.0.

Proposal còn đặt European Credit Card Fraud ở vai trò stress test phụ lục; phase hiện tại chỉ triển khai hai dataset được yêu cầu ở trên và không tự nhận là đã hoàn tất stress matrix ba dataset.

Hai stress tests dùng temporal split 60/20/20, ba predictor families TabularResNetV2/XGBoost/LightGBM, validation roles tách biệt, frozen predictor, rule audit và selective coverages cố định 5%/10%/25%/50%. Rule audit, TP/FP statistics, weighting và deduplication chỉ dùng vùng cảnh báo của predictor đã đóng băng; TN/FN ngoài vùng này không tham gia chọn luật. Chỉ alert có audited rule đạt activation threshold mới được tính là có explanation. Tier C CART được giữ làm baseline; primary policy chỉ dùng Tier A/B. Counterfactual interventions tái tính các đặc trưng dẫn xuất theo contract trong YAML để không tạo hàng nội bộ bất nhất. Contrastive meta-scorer dùng audited rule truth cùng frozen calibrated risk, được fit riêng trên TP/FP alerts của `rule_audit` rồi đóng băng. Guarded ensemble được khóa trước là trung bình của FP-penalized, attribution-gated và contrastive evidence; nó vẫn phải vượt score-only và stability guardrails trên `policy_select`. Baseline/ablation tách domain-only, exact counterfactual-only, unweighted, weighted-no-attribution, FP-penalized, attribution-gated, signed native-attribution top-k, CART và label/weight shuffle. Negative, inconclusive và AMLNet saturation đều được giữ nguyên theo các guardrail đã khóa.

Do quy mô của hai stress datasets và giới hạn một Kaggle session, stress config dùng một seed huấn luyện định trước (`42`) cho mỗi family; validation resampling và paired temporal-block bootstrap định lượng hai dạng bất định downstream nhưng không thay thế hoàn toàn training-seed variability. Protocol ba seed của IEEE-CIS/BAF không thay đổi.

Dataset không được lưu trong Git. Xem [data/README.md](data/README.md) để biết cách tải, gắn Kaggle dataset và bố trí file.

## Project structure

```text
configs/           Dataset, model, logic và evaluation settings
data/              Hướng dẫn dữ liệu; raw data bị gitignore
docs/              Phạm vi khóa luận và giao thức thực nghiệm
notebooks/         Notebook 01-08 cốt lõi và notebook 09-11 cho stress-test extension
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
├── stress_testing/ Pipeline stress test, frozen policy, audit và uncertainty
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

## Chạy smoke experiment không cần file benchmark

```bash
python scripts/run_experiment.py \
  --config configs/ieee_cis.yaml \
  --models tree \
  --quick \
  --synthetic-fallback
```

Synthetic data chỉ dùng để kiểm tra khả năng thực thi. Không sử dụng kết quả synthetic để đưa ra kết luận nghiên cứu.

## Chạy với file benchmark chính thức

```bash
python scripts/run_experiment.py \
  --config configs/ieee_cis.yaml \
  --data-root data/raw/ieee-cis \
  --repeated
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
| `01_Data_Exploration.ipynb` | Khảo sát IEEE-CIS, BAF và so sánh tổng quan giữa các dataset |
| `02_IEEE_CIS_Model_Benchmarks.ipynb` | Benchmark MLP, TabularResNet và tree model trên IEEE-CIS |
| `03_BAF_Model_Benchmarks.ipynb` | Benchmark MLP, TabularResNet và tree model trên BAF |
| `04_IEEE_CIS_LTN_Rule_Analysis.ipynb` | Fit train-only và phân tích luật trên IEEE-CIS |
| `05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb` | Đánh giá explanation coverage và consistency trên IEEE-CIS |
| `06_IEEE_CIS_Rule_Ablation.ipynb` | Đánh giá vai trò của từng nhóm luật IEEE-CIS |
| `07_BAF_LTN_Generalization.ipynb` | Cross-dataset replication/portability evaluation trên BAF |
| `08_Cross_Dataset_Result_Synthesis.ipynb` | Tổng hợp prediction, bootstrap, logic và explanation giữa hai dataset |
| `09_TransXion_v2_Stress_Test.ipynb` | Stress test AML khó trên canonical TransXion `tx.csv`; `v2` là paper revision |
| `10_AMLNet_v1_0_Stress_Test.ipynb` | Stress test saturation/limited headroom trên AMLNet v1.0 |
| `11_Stress_Test_Synthesis.ipynb` | Kiểm tra lineage và tổng hợp hai stress-test output packages |

Notebook exploration chạy nhiều dataset theo từng phần. BAF dùng các nhóm tháng không giao nhau: train `0-4`, validation `5`, test `6-7`. Mỗi notebook model/rule còn lại khóa vào một dataset cụ thể để tránh trộn cấu hình và kết quả. Full mode là mặc định, dùng toàn bộ dữ liệu, ngân sách tối đa 100 epoch cho MLP và 150 epoch cho TabularResNet, early stopping theo validation PR-AUC và ba seed độc lập. Quick/synthetic mode chỉ được bật tường minh để smoke test. Tên file Notebook 07 được giữ để không phá liên kết Kaggle cũ; nội dung được diễn giải là replication của framework trên benchmark thứ hai, không phải chuyển cùng model hoặc cùng rule base giữa hai dataset. Xem [docs/KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md).

Notebook 01 cần được rerun khi EDA/data-audit thay đổi, nhưng không phải input dependency của Notebook 08. Notebook 08 tổng hợp outputs 02-07 sau khi frozen-artifact và lineage checks bắt buộc đều qua.

Notebook 09 và 10 độc lập và có thể chạy song song. Notebook 11 chỉ chạy sau khi hai stress outputs đã sẵn sàng; notebook này không train lại predictor hoặc chọn lại policy. Full mode mới được dùng cho kết luận; fixture/quick mode chỉ là kiểm tra kỹ thuật. Xem [giao thức stress test](docs/STRESS_TEST_PROTOCOL.md).

## Evaluation contract

1. Split dữ liệu trước mọi thao tác học tham số.
2. Preprocessor chỉ fit trên train.
3. Model chỉ fit trên train.
4. Calibration fit trên nửa đầu validation; calibration method và threshold được chọn trên nửa sau validation.
5. Rule quantiles, category-risk mapping và median dùng điền numeric missing values chỉ fit trên train; validation/test tái sử dụng giá trị đã khóa.
6. Test được đánh giá một lần bằng toàn bộ quyết định đã khóa.
7. Không dùng test để chọn feature, rule, model hoặc hyperparameter.

Đối với stress-test extension, train/validation/test là 60/20/20 theo thời gian. Validation được chia thành các role không chồng lấn cho predictor/calibration lock, alert-conditional rule audit và policy lock. Candidate rules có Tier A/B/C và provenance rõ ràng; mọi baseline/ablation cũng phải qua cùng validation-resampling stability gate. Test chỉ được đánh giá sau khi predictor, calibration, threshold, audited rules, evidence method và coverage policies đã freeze.

Prediction metrics chính:

- Raw PR-AUC là metric chính để so sánh và chọn model vì phù hợp với class imbalance.
- ROC-AUC.
- Precision, recall, F1 và F2 trên calibrated probabilities với decision threshold đã khóa.
- Brier score, ECE và NLL.

Calibration được chọn theo Brier score trên validation partition riêng. Brier, ECE và NLL đánh giá chất lượng xác suất sau calibration; không dùng calibrated test PR-AUC để chọn lại model. Mục tiêu F2 đặt trọng số recall lớn hơn precision vì giả định bỏ sót fraud tốn kém hơn việc kiểm tra thêm false alerts; đây là giả định vận hành của khóa luận, chưa phải cost function đã được xác nhận bởi một tổ chức tài chính.

Explanation metrics chính:

- Rule coverage và fraud precision, kèm denominator `evaluated_rows` của split tương ứng.
- Rule lift và rule AUC.
- Explanation coverage.
- `mean_active_rule_count`, `mean_displayed_rule_count` và `rule_sparsity`, trong đó `rule_sparsity = 1 - mean(active_rule_count / available_rule_count)`.
- Prediction-rule consistency, balanced consistency và contradiction rate, luôn diễn giải cùng alert-conditioned coverage, unsupported-alert rate và rule-evidence-without-alert rate để tránh ảnh hưởng của class imbalance.
- Case-study analysis cho TP, FP, FN và TN.

## Tests

```bash
pytest -q
```

Sau khi thay đổi source cell, sinh lại notebook và kiểm tra source-sync bằng:

```bash
python scripts/generate_notebooks.py
pytest -q tests/test_notebook_generation.py
```

Stress notebooks 09-11 dùng generator/test riêng để không làm thay đổi lineage đã khóa của Notebook 01-08:

```bash
python scripts/generate_stress_test_notebooks.py
pytest -q tests/test_stress_notebook_generation.py
```

Tests tập trung vào các lỗi có thể làm sai kết luận:

- Temporal split không đảo thứ tự thời gian.
- Imputation/scaling chỉ fit trên train.
- Rule truth value nằm trong `[0, 1]`.
- Threshold selection chỉ nhận validation inputs.
- Explanation output và metrics có schema ổn định.

## Reproducibility

- Kết quả model chính thức dùng các seed `[42, 123, 2026]` và báo cáo mean ± standard deviation.
- Mọi tham số nằm trong YAML config.
- Mỗi run xuất raw/calibrated predictions, calibration comparison, metrics và environment metadata.
- Benchmark chọn reference predictor bằng mean validation raw PR-AUC và xuất frozen artifact có checksum.
- Reference artifact dùng seed đã khóa trước là `42`. Paired bootstrap resample các test rows tương ứng của seed 42; nó ước lượng độ bất định theo mẫu test, không thay thế biến thiên huấn luyện được báo cáo qua ba seed.
- Notebook 05-07 chỉ đọc frozen predictor; ablation không được train lại model. Ablation trên locked test là phân tích hậu nghiệm, không được dùng để tối ưu rule set rồi tuyên bố kết quả test mới là unbiased.
- Notebook không chứa implementation chính; chúng gọi module trong `src/` và được sinh từ `scripts/generate_notebooks.py`. Không sửa notebook thủ công: thay đổi generator, sinh lại notebook và chạy source-sync checks.
- Preflight của Notebook 05-07 phải tìm đúng một frozen artifact tương ứng trong các input roots, sau đó kiểm tra dataset, full/quick mode, config hash, checksum và label alignment trước khi tính kết quả.
- Output Notebook 04-07 lưu lineage với source/config fingerprint và checksum của các file đã khai báo; Notebook 08 kiểm tra lineage này trước khi tổng hợp.
- Mỗi full run phải ghi thủ công Kaggle dataset slug/reference và input version trong experiment log hoặc notebook version notes; tên file và row count không đủ để xác định dataset version.
- Kết quả synthetic phải được gắn nhãn rõ và không dùng trong báo cáo.
- Kết quả chính thức phải lưu config, data source, row count và feature count.

## Tài liệu

- [Phạm vi và nội dung khóa luận](docs/PHAM_VI_VA_NOI_DUNG_KHOA_LUAN.md)
- [Lộ trình phase và bằng chứng thực nghiệm](docs/PHASE_ROADMAP.md)
- [Giao thức thực nghiệm](docs/EXPERIMENTAL_PROTOCOL.md)
- [Hướng dẫn Kaggle](docs/KAGGLE_GUIDE.md)
- [Hướng dẫn kết quả](docs/RESULTS_GUIDE.md)
- [Giao thức stress test TransXion/AMLNet](docs/STRESS_TEST_PROTOCOL.md)
- [Mô tả dữ liệu](data/README.md)

## Giới hạn

- Rule evidence thể hiện sự phù hợp với tri thức đã cấu hình, không tự động chứng minh quan hệ nhân quả.
- Rule evidence không tự động chứng minh model faithfulness; lớp luật là lớp audit chọn lọc quanh frozen predictor.
- Kết quả trên một dataset không đại diện cho mọi hệ thống tài chính.
- IEEE-CIS và BAF dùng feature space, model và rule base riêng; kết quả BAF chỉ hỗ trợ tính khả chuyển của quy trình trên benchmark thứ hai, không chứng minh model/rule transfer hoặc external validation trên dữ liệu ngân hàng thực.
- Mô hình chưa được đánh giá bằng user study với fraud analysts.
- Prototype không phải hệ thống ra quyết định tài chính sẵn sàng triển khai.
- Temporal hoặc prior shift có thể làm giảm calibration và hiệu quả rule.
- TransXion và AMLNet chỉ mở rộng stress-test coverage; chúng không phải external validation trên dữ liệu của một tổ chức tài chính độc lập.
- AMLNet saturation, nếu quan sát thấy, chỉ có nghĩa score-only không để lại residual headroom có bằng chứng trong protocol này; nó không chứng minh production performance.

## License

Mã nguồn được phát hành theo MIT License. Dữ liệu phải được sử dụng theo giấy phép và điều khoản của nguồn tương ứng.
