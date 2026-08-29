# Lộ trình triển khai và bằng chứng thực nghiệm

## Phase 9 — Biểu diễn tri thức logic

### Nội dung

- Biểu diễn các dấu hiệu rủi ro bằng fuzzy predicates.
- Fit quantile threshold và category-risk mapping chỉ trên train.
- Fit median cho numeric-rule missing values chỉ trên train, tái sử dụng trên validation/test và xuất `fitted_missing_value`.
- Dùng absolute-domain softness cho các mốc `greater`/`less`/`outside_range` và kiểm thử boundary semantics.
- Kết hợp predicates thành luật có tên và mô tả nghiệp vụ.
- Kiểm tra tensor logic khả vi bằng PyTorch.
- Xuất lineage của Notebook 04 với audit-pipeline/config fingerprint và checksum của các output rule-analysis.

### File chính

- `src/logic/predicates.py`
- `src/logic/fraud_rules.py`
- `src/logic/knowledge_base.py`
- `src/logic/tensor_logic.py`
- `tests/test_rules.py`
- `tests/test_tensor_logic.py`
- `notebooks/04_IEEE_CIS_LTN_Rule_Analysis.ipynb`

### Kết quả và phạm vi chứng minh

- Bảng threshold và `fitted_missing_value` đã fit, `evaluated_rows`, coverage, fraud precision, lift và rule AUC cho từng split.
- So sánh chất lượng luật giữa validation và locked test.
- Chứng minh tri thức miền có thể được biểu diễn thành truth value trong `[0, 1]` và đánh giá định lượng; không chứng minh quan hệ nhân quả.
- Đây là lớp fuzzy-rule/LTN-inspired quanh predictor được khóa, không phải LTN end-to-end hoặc một thuật toán LTN mới.

## Phase 10 — Giải thích cảnh báo dựa trên luật

### Nội dung

- Sinh top-k rule evidence cho từng giao dịch.
- Đánh giá riêng prediction quality và explanation quality.
- Phân tích TP, FP, FN và TN.
- Bootstrap precision gain của nhóm cảnh báo được giải thích.

### File chính

- `src/explanation/rule_explainer.py`
- `src/explanation/explanation_metrics.py`
- `notebooks/05_IEEE_CIS_Rule_Explanation_Evaluation.ipynb`

### Kết quả và phạm vi chứng minh

- Explanation coverage trên toàn bộ mẫu và trên model alerts.
- `mean_active_rule_count`, `mean_displayed_rule_count`, `rule_sparsity`, unsupported-alert rate, overall/balanced consistency và contradiction rate. `rule_sparsity` dùng tỷ lệ active rule trên tổng rule khả dụng; `rule_count` chỉ là alias backward-compatible của active count.
- Explained-alert precision, precision gain và khoảng tin cậy bootstrap 95%.
- Chứng minh rule evidence hỗ trợ kiểm tra một phần cảnh báo; không tuyên bố đây là causal explanation.
- Các tỷ lệ có denominator bằng 0 được ghi `NaN`; consistency toàn tập không được diễn giải tách khỏi alert-conditioned coverage.

## Phase 11 — Ablation và cross-dataset replication

### Nội dung

- Only-one-rule và leave-one-rule-out ablation trên IEEE-CIS.
- Lặp prediction, rule quality và explanation evaluation trên privacy-preserving synthetic benchmark BAF với model, feature space và rule base riêng.
- Giữ nguyên split, predictor probability và threshold khi thực hiện ablation.

### File chính

- `notebooks/06_IEEE_CIS_Rule_Ablation.ipynb`
- `notebooks/07_BAF_LTN_Generalization.ipynb`

### Kết quả và phạm vi chứng minh

- Bảng thay đổi coverage và precision gain khi bỏ từng luật.
- Mô tả hậu nghiệm luật có liên hệ với trade-off coverage-precision; không dùng locked test để chọn policy mới.
- Kiểm tra phương pháp có thực thi và tạo bằng chứng giải thích trên một phân phối dữ liệu khác.
- Không đồng nhất cross-dataset replication với rule/model transfer, external validation trên dữ liệu ngân hàng thực hoặc production generalization.

Tên file `07_BAF_LTN_Generalization.ipynb` được giữ vì tương thích với Kaggle linkage và output slug cũ; title/narrative phải dùng thuật ngữ cross-dataset replication/portability.

## Phase 12 — Tổng hợp, kiểm thử và đóng gói kết quả

### Nội dung

- Tổng hợp bảng prediction metrics từ các run.
- Xác định reference model từ frozen manifest đã chọn bằng validation raw PR-AUC; không chọn lại model theo test trong Notebook 08.
- Xuất run metadata, predictions và bảng mean ± standard deviation.
- Kiểm tra top-to-bottom notebooks, unit tests, schema và leakage contract.
- Kiểm tra lineage Notebook 04-07, audit-pipeline/config fingerprints và SHA-256 của các output đã khai báo.
- Ghi thủ công `dataset_reference`/`dataset_version` trong experiment log hoặc Kaggle version notes.
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
- Output Notebook 04-07 truy vết được đến source/config fingerprint và checksum; Notebook 08 từ chối input không qua lineage audit.
- Paired row bootstrap của seed tham chiếu 42 và mean ± standard deviation qua ba training seed trả lời hai nguồn bất định khác nhau.
- Notebook 08 trình bày lift cùng coverage/active count, precision gain cùng bootstrap interval và đánh dấu test ablation là post-hoc.

## Local và Kaggle

Tất cả notebook đều có thể chạy local khi máy đã cài dependencies và có dữ liệu. GPU không phải yêu cầu chức năng.

- Nên chạy local: Data Exploration, tree-only benchmark, rule analysis, explanation evaluation, ablation và smoke tests.
- Nên chạy Kaggle GPU: full-data MLP/TabularResNet benchmarks với ba seed.
- Có thể chạy neural full protocol local nếu máy có GPU CUDA tương thích và đủ thời gian; CPU vẫn chạy được nhưng chậm.
- Synthetic/quick mode chỉ xác nhận code thực thi, không hỗ trợ kết luận khóa luận.

Notebook được sinh từ `scripts/generate_notebooks.py`; preflight của Notebook 05-07 phải được duy trì trong generator và kiểm tra frozen artifact trước khi chạy. Sau mọi thay đổi source cell, chạy:

```bash
python scripts/generate_notebooks.py
pytest -q tests/test_notebook_generation.py
```

Sau vòng sửa logic/metric hiện tại, giữ frozen outputs 02/03 nếu vẫn qua compatibility checks; rerun Notebook 01 để cập nhật EDA và rerun 04-07 để cập nhật bằng chứng rule/explanation. Notebook 01 có thể chạy song song và không phải dependency của 08; Notebook 08 chỉ chạy sau khi outputs 02-07 và lineage audits tương ứng đã sẵn sàng.

## Phase 13 — Stress-test foundation và provenance lock

### Nội dung

- Xây dựng pipeline stress test độc lập để không làm thay đổi lineage Notebook 01-08.
- Khóa TransXion canonical `tx.csv` theo official repository commit/SHA-256; ghi rõ `v2` là paper revision, không phải dataset release.
- Khóa AMLNet v1.0 theo Zenodo DOI/exact filename/MD5/CC BY-NC 4.0 và ghi source-version discrepancy.
- Implement checksum-first discovery, schema validation, safe AMLNet metadata timestamp parser và stable temporal split 60/20/20.
- Xây causal history features theo chiều quá khứ, feature denylist và split-integrity checks.
- Implement TabularResNetV2, XGBoost và LightGBM theo cùng protocol; calibration, threshold lock và frozen predictor artifact.

### File chính

- `configs/stress/transxion_v2.yaml`
- `configs/stress/amlnet_v1.yaml`
- `src/stress_testing/data.py`
- `src/stress_testing/predictors.py`
- `src/stress_testing/metrics.py`
- `src/stress_testing/artifacts.py`
- `tests/test_stress_data.py`
- `tests/test_stress_predictors.py`
- `tests/test_stress_metrics.py`

### Kết quả và phạm vi chứng minh

- Data/provenance manifests và split-integrity tables có thể audit.
- Ba predictor families được so sánh dưới cùng split/feature contract; không silent-fallback trong full mode.
- Frozen predictor boundary ngăn rule layer thay đổi prediction sau khi model/calibration/threshold đã khóa.
- Phase này chỉ xây nền tảng và protocol; không tự tạo ra bằng chứng rule-added value khi chưa chạy full experiments.

## Phase 14 — Dataset-specific stress experiments

### Nội dung

- Chạy một notebook riêng cho TransXion và một notebook riêng cho AMLNet.
- Sinh Tier A domain-prespecified, Tier B train-derived/counterfactual-assisted và Tier C empirical baseline candidates; counterfactual interventions tái tính derived semantic features theo contract.
- Tính attribution support quanh frozen predictor; audit, ghi failure reasons và deduplicate candidate rules chỉ trên frozen-predictor alerts của validation audit role.
- Fit và freeze contrastive TP-vs-FP meta-scorer từ audited rules cùng frozen risk trên `rule_audit`, có explicit fallback khi thiếu mẫu/lớp/variation.
- Kiểm tra guarded ensemble và toàn bộ proposal baselines bằng cùng validation stability gate.
- Khóa evidence method, rule weights, risk bins và selective policies trên validation policy role.
- Đánh giá fixed coverages 5%/10%/25%/50%, trong đó 10%/25% là primary; so sánh score-only guardrail tại cùng exact budget.
- Đánh giá abstention, matched-risk, residual TP-vs-FP, paired bootstrap, policy stability và negative controls.

### File chính

- `src/stress_testing/attribution.py`
- `src/stress_testing/rule_audit.py`
- `src/stress_testing/selective_policy.py`
- `src/stress_testing/matched_risk.py`
- `src/stress_testing/experiment.py`
- `notebooks/09_TransXion_v2_Stress_Test.ipynb`
- `notebooks/10_AMLNet_v1_0_Stress_Test.ipynb`
- `scripts/generate_stress_test_notebooks.py`
- `tests/test_stress_attribution.py`
- `tests/test_stress_vasre.py`
- `tests/test_stress_notebook_generation.py`

### Kết quả và phạm vi chứng minh

- Full output package cho từng dataset: data/split/predictor manifests, rule registry/audit/redundancy, locked policies, coverage results, matched-risk, residual evidence, bootstrap, stability, negative controls và lineage.
- TransXion kiểm tra framework trong difficult-AML setting; AMLNet kiểm tra saturation/limited headroom.
- Rule-added value chỉ được hỗ trợ nếu kết quả vượt score-only ổn định tại coverage đã khóa và qua uncertainty/negative-control checks.
- Negative hoặc inconclusive result là kết quả hợp lệ; không thay rule/coverage theo locked test.

## Phase 15 — Stress synthesis và thesis claim audit

### Nội dung

- Gắn completed output packages của Notebook 09 và 10, kiểm tra exact dataset identity, full/fixture mode, config/source fingerprints và output checksums.
- Tổng hợp predictor quality, rule audit, coverage trade-off, score-only guardrail, abstention, matched-risk, residual TP-vs-FP, uncertainty, stability và controls.
- Phân loại kết quả thành positive, negative, inconclusive hoặc blocked-by-data-quality/lineage.
- Viết claim matrix theo vai trò stress test thay vì gộp chúng như hai primary datasets.

### File chính

- `notebooks/11_Stress_Test_Synthesis.ipynb`
- `docs/STRESS_TEST_PROTOCOL.md`
- `docs/RESULTS_GUIDE.md`
- `docs/KAGGLE_GUIDE.md`

### Kết quả và phạm vi chứng minh

- Synthesis chỉ đọc artifacts; không train model, sinh rule hoặc chọn lại policy.
- Có thể nêu framework hoạt động/không hoạt động đến mức nào trong hai điều kiện stress đã định nghĩa.
- Không tuyên bố causal explanation, universal superiority, external validation trên dữ liệu ngân hàng thật hoặc production readiness.

## Thứ tự chạy stress extension

Notebook 09 và 10 độc lập và có thể chạy song song trên hai Kaggle T4 sessions. Chỉ chạy Notebook 11 sau khi cả hai completed full output versions đã được gắn. Fixture/quick mode chỉ dùng cho smoke test và không được đưa vào thesis claims.
