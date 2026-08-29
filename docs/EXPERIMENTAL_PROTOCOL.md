# Giao thức thực nghiệm

## 1. Mục đích

Tài liệu này khóa các quyết định đánh giá trước khi đọc kết quả test, nhằm giảm data leakage, test overfitting và cherry-picking.

## 2. Data split

IEEE-CIS sử dụng temporal row split:

- Train: 70% thời gian đầu.
- Validation: 15% tiếp theo.
- Test: 15% cuối.

BAF sử dụng month-disjoint split theo tám tháng:

- Train: tháng 0-4.
- Validation: tháng 5.
- Test: tháng 6-7.

Không được cắt cùng một giá trị `month` qua nhiều split. Raw `TransactionDT` và `month` chỉ dùng để chia tập; predictor không dùng raw split key.

BAF là privacy-preserving synthetic benchmark. Trạng thái `data_source = real` trong experiment metadata hoặc `official_benchmark_file` trong EDA chỉ có nghĩa loader đã đọc `Base.csv` chính thức thay vì synthetic fallback dùng cho smoke test. Một số trường BAF dùng `-1` như sentinel cho giá trị không có hoặc không áp dụng; đây là semantic missingness và phải được kiểm đếm riêng với native `NaN`. Protocol giữ nguyên `-1` như giá trị mã hóa của benchmark và không tự động chuyển thành `NaN`; báo cáo phải nêu quyết định xử lý này và không được suy luận rằng BAF không có missingness chỉ từ số lượng native `NaN` bằng 0.

Mỗi full run phải ghi thủ công `dataset_reference`, `dataset_version`, danh sách file đã dùng và thời điểm gắn/tải input trong experiment log hoặc Kaggle version notes. Nếu input provider không cung cấp version, ghi `dataset_version: unavailable`; filename và row count không được coi là dataset version.

Nếu dataset không có biến thời gian đáng tin cậy, sử dụng stratified split với seed cố định và ghi rõ lý do.

## 3. Train-only operations

Chỉ train được phép dùng để fit:

- Missing-value statistics.
- Scaling statistics.
- Categorical encodings.
- Feature availability filters.
- Neural/tree model parameters.
- Quantile threshold của fuzzy predicates.
- Category-risk mapping trong logic rules.
- Median điền numeric missing values trong từng rule condition.

## 4. Validation-only operations

Validation được dùng để:

- Theo dõi early stopping.
- Nửa đầu validation fit Platt/isotonic calibration.
- Nửa sau validation chọn calibration method theo Brier score và chọn decision threshold theo F1/F2.
- Chọn model/config trong phạm vi budget đã định trước.
- Chọn activation threshold nếu có thí nghiệm sensitivity được thiết kế trên validation.

Mọi quyết định phải được khóa trước khi chạy test report.

## 5. Test-only reporting

Test chỉ dùng để:

- Tính prediction metrics cuối cùng.
- Tính rule/explanation metrics với engine đã fit.
- Tạo case studies sau khi policy đã khóa.

Không điều chỉnh feature, threshold, calibration, rule hoặc model sau khi đọc test.

## 6. Predictive models

| Nhóm | Vai trò |
|---|---|
| MLP | Neural baseline |
| TabularResNet | Tabular deep-learning model |
| XGBoost/LightGBM | Strong tree baseline |

Các model dùng cùng split và preprocessing contract. Hyperparameter budget phải được mô tả trong báo cáo.

## 7. Prediction metrics

Primary metric: raw PR-AUC, phù hợp với class imbalance. Reference model được chọn theo mean validation raw PR-AUC qua các seed; calibrated test PR-AUC không được dùng để xếp hạng hoặc chọn lại model.

Secondary metrics:

- ROC-AUC.
- Precision.
- Recall.
- F1/F2.
- Brier score.
- Expected Calibration Error.
- Negative log-likelihood.

Brier score, Expected Calibration Error và negative log-likelihood đánh giá xác suất sau calibration. Các metric phụ thuộc threshold, gồm precision, recall, F1 và F2, được tính từ calibrated probabilities với threshold đã chọn trên validation và khóa trên test.

Protocol mặc định tối ưu F2 với `beta = 2`. Theo công thức F-beta, lựa chọn này đặt trọng số recall gấp bốn lần precision, phản ánh giả định rằng bỏ sót fraud tốn kém hơn việc kiểm tra thêm false alerts. Đây là giả định vận hành phục vụ thực nghiệm, không phải cost ratio đã được xác nhận bằng dữ liệu vận hành hoặc user study.

## 8. Logical rules

Rules được khai báo trong YAML. Threshold dạng quantile và category-risk mapping chỉ fit trên train. Rule không có đủ feature sẽ được bỏ qua và ghi lý do.

Với numeric rule condition, missing/coercion value được điền bằng median chỉ fit trên training split. Cùng median đó phải được tái sử dụng nguyên vẹn trên validation/test; tuyệt đối không recompute median trên split downstream. Bảng fitted conditions xuất giá trị này trong cột `fitted_missing_value` để audit, và Notebook 05 ghi nó trong case-level audit evidence.

`softness` phải có ngữ nghĩa nhất quán với operator: predicate có threshold theo quantile có thể dùng độ mềm tương đối, còn `greater`, `less` và `outside_range` với mốc nằm trong đơn vị gốc dùng độ mềm tuyệt đối theo miền của feature. Boundary semantics phải có unit test để tránh việc nhân độ mềm với độ lớn threshold làm thay đổi ngoài ý muốn vùng kích hoạt.

Rule metrics:

- Coverage.
- Fraud precision.
- Lift so với base fraud rate.
- Mean truth value.
- Rule AUC.
- Validation/test stability.

Khi `active_count = 0`, fraud precision và lift là không xác định và phải ghi `NaN`, không ghi `0`. Lift luôn được diễn giải cùng coverage và active count; lift lớn trên một tập rất nhỏ không đủ chứng minh rule có giá trị bao quát.

## 9. Explanation evaluation

Một explanation gồm tối đa `top_k_rules` có truth value vượt activation threshold.

System metrics:

- Coverage trên toàn bộ mẫu.
- Coverage trên model alerts.
- `mean_active_rule_count`: số rule vượt activation threshold trung bình trên mỗi mẫu.
- `mean_displayed_rule_count`: số rule trung bình thực sự được hiển thị sau giới hạn `top_k_rules`.
- `rule_sparsity = 1 - mean(active_rule_count / available_rule_count)`. Giá trị càng cao nghĩa là càng ít rule active so với số rule khả dụng.
- Prediction-rule consistency.
- Balanced prediction-rule consistency.
- Contradiction rate.
- Unsupported-alert rate và rule-evidence-without-alert rate.
- Explained-alert precision.

Mọi tỷ lệ phải kèm denominator tương ứng, đặc biệt là `predicted_alert_count`, `explained_count`, `explained_alert_count` và `explained_alert_fraud_count`. Khi không có explained alert, explained-alert precision và precision gain là `NaN` thay vì `0`.

`rule_count` chỉ là alias backward-compatible của active-rule count; báo cáo và bảng mới phải dùng các tên metric tường minh bên trên. Trong rule-quality tables, denominator chung có tên `evaluated_rows` vì cùng hàm được dùng cho train, validation và test; không gọi denominator này là `test_rows` khi đánh giá split khác.

Prediction-rule consistency trên toàn bộ test có thể bị chi phối bởi số lượng non-alert lớn. Vì vậy, metric này chỉ là chỉ báo mô tả và phải được đọc cùng explanation coverage trên alerts, unsupported-alert rate và rule-evidence-without-alert rate. Các metrics không chứng minh causal explanation hoặc model faithfulness. Case study cần trình bày TP, FP, FN và TN với model probability, decision threshold, raw feature values liên quan, fitted rule thresholds, `fitted_missing_value` của numeric conditions và rule strengths.

## 10. Ablation

Tối thiểu:

- Từng rule riêng lẻ.
- Bỏ từng rule khỏi full rule set.
- Full rule set.
- Sensitivity theo activation threshold.

Ablation chỉ thay đổi thành phần đang kiểm tra; data split và predictor probabilities giữ nguyên.

Ablation và activation-threshold sensitivity được tính trên locked test chỉ là post-hoc diagnostics để hiểu trade-off coverage-precision. Không chọn rule set hoặc activation threshold tốt nhất từ các bảng test rồi báo lại kết quả trên cùng test như một ước lượng unbiased. Nếu muốn thay đổi policy dựa trên ablation, phải chọn trên validation và đánh giá bằng một holdout mới; trong phạm vi khóa luận hiện tại, giữ full rule set đã khóa và trình bày ablation như phân tích hậu nghiệm là đủ.

## 11. Repeated runs

Smoke run dùng seed 42. Kết quả model chính thức bắt buộc dùng ba seed `[42, 123, 2026]`; báo cáo mean và standard deviation. Mỗi run lưu:

- Config.
- Seed.
- Data source và row count.
- Feature count.
- Metrics.
- Prediction arrays.
- Thời điểm chạy và môi trường.
- Raw và calibrated prediction arrays.
- Config hash, prediction checksum và split-group summary.
- `dataset_reference`/`dataset_version` trong experiment log hoặc Kaggle version notes.

Reference predictor được chọn theo mean validation raw PR-AUC. Seed tham chiếu được khóa trước là 42. Paired stratified bootstrap resample các test rows tương ứng để định lượng chênh lệch PR-AUC giữa reference model seed 42 và các baseline seed 42. Khoảng tin cậy bootstrap này phản ánh độ bất định do tập test, không phải độ bất định giữa các lần huấn luyện; mean ± standard deviation qua `[42, 123, 2026]` vẫn phải được báo cáo riêng.

Notebook 04-07 phải ghi lineage chứa git commit, audit-pipeline/config fingerprint, danh sách output và SHA-256 của từng output. Notebook 05-07 còn phải gắn frozen manifest/artifact checksum; Notebook 04 không dùng frozen predictor nhưng vẫn phải có source/config/output lineage. Notebook 08 chỉ tổng hợp khi toàn bộ lineage checks bắt buộc đều qua. Notebook 01 cần được chạy lại khi EDA thay đổi, nhưng không phải input dependency của Notebook 08.

Giới hạn của frozen artifacts 02/03 hiện tại: alignment downstream khóa checksum artifact và so sánh nguyên chuỗi label validation/test, nhưng legacy manifest chưa hash row identifiers hoặc feature rows. Vì vậy loader, dataset version và split order phải được giữ deterministic, không được reorder dữ liệu sau split; giới hạn này phải được nêu trong threats to validity thay vì diễn giải label alignment như bằng chứng nhận dạng đầy đủ từng row.

## 12. Claim boundaries

Kết quả chỉ hỗ trợ claim trong dataset và protocol đã chạy. Không tuyên bố:

- Universal superiority.
- Causal explanation.
- Rule evidence là faithful explanation đầy đủ của frozen predictor.
- Hệ thống là LTN end-to-end hoặc logic làm tăng predictive performance.
- Cùng model hoặc cùng rule base transfer giữa IEEE-CIS và BAF.
- Production readiness.
- Generalization đến tổ chức tài chính khác khi chưa có dữ liệu tương ứng.

Notebook 07 là cross-dataset replication/portability evaluation: cùng quy trình được cấu hình lại trên BAF với feature space, predictor và rule base riêng. Kết quả đó hỗ trợ khả năng tái áp dụng framework trên benchmark thứ hai, không phải rule/model transfer hoặc external validation trên dữ liệu ngân hàng thực.

## 13. Proposal-aligned stress-test extension

### 13.1. Vai trò và dataset identity

TransXion và AMLNet là stress tests sau protocol cốt lõi, không phải hai co-primary datasets mới. TransXion kiểm tra framework trong bối cảnh AML khó; AMLNet kiểm tra saturation/giới hạn khi score-only có thể đã rất mạnh.

Dataset identity bắt buộc:

- TransXion canonical `tx.csv`, official repository commit `53932595c37c23b9f55ea5ddf5984e4d57b88369`, SHA-256 `d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35`. `v2` chỉ paper revision `arXiv:2604.17420v2`, không phải dataset release.
- AMLNet v1.0 Zenodo DOI `10.5281/zenodo.16736515`, exact file `AMLNet_August 2025.csv`, MD5 `7668fc7d74c787e07546ce85c6f790b9`, CC BY-NC 4.0. Manifest phải ghi discrepancy giữa Zenodo v1/v1.0 và `VERSION 2.0` có thể xuất hiện trong README preview.

Full run dừng khi checksum/schema không khớp. Fixture/quick run chỉ kiểm tra kỹ thuật và không được dùng cho claim.

### 13.2. Safe time construction và split

Với AMLNet, row order và `step` không phải chronological key tin cậy. `event_time` phải được trích từ `metadata` bằng parser giới hạn/regular expression; nghiêm cấm `eval`/`exec`. Full run yêu cầu parse đầy đủ và stable-sort theo timestamp/original row order. Raw metadata, target, `isFraud`, `laundering_typology`, `fraud_probability`, `step` và raw IDs bị loại khỏi predictor.

Với cả hai stress datasets, split sau stable temporal sort là:

- Train: 60% đầu.
- Validation: 20% tiếp theo.
- Locked test: 20% cuối.

Validation được chia chronological, không chồng lấn thành bốn roles:

1. Calibration fit.
2. Calibration/model/threshold selection theo contract đã khai báo.
3. Rule audit và deduplication.
4. Selective-policy lock và stability.

Mọi preprocessor, causal history feature contract, model, calibration, threshold, rule candidate/fitted threshold, attribution method, audit criterion, evidence method, rule weight, risk bin và coverage policy phải được fit/chọn/khóa trước locked test.

### 13.3. Predictors và freeze boundary

Ba predictor families là TabularResNetV2, XGBoost và LightGBM. Chúng dùng cùng split, feature eligibility và evaluation contract. Full mode không silent-fallback sang model family khác khi dependency/GPU không khả dụng.

Predictor metrics tối thiểu:

- Raw PR-AUC là primary ranking metric.
- ROC-AUC, precision, recall, F2 và Recall@1%FPR.
- Brier score, Expected Calibration Error và negative log-likelihood.

Sau model/calibration/threshold selection trên validation role tương ứng, reference predictor được freeze cùng preprocessor, feature schema, calibration method, threshold, seed và fingerprints. Rule pipeline chỉ đọc frozen outputs.

### 13.4. Candidate tiers, attribution và audit

Candidate rules có provenance tier:

- Tier A: domain-prespecified rules, khai báo trước validation/test.
- Tier B: train-derived/counterfactual-assisted candidates quanh frozen predictor; không gọi là domain rules thuần túy. Counterfactual raw intervention phải tái tính deterministic derived features theo YAML contract; không perturb trực tiếp derived outputs.
- Tier C: shallow-path/attribution-derived empirical baselines, chỉ dùng train để sinh candidate.

Validation audit tính coverage, precision/lift, TP-vs-FP rule AUC, attribution support và active count chỉ trong alert region của frozen predictor. Population statistics là diagnostics tách biệt, không làm pass/fail. Rule không qua criterion phải có failure reason. Near-duplicate rules được deduplicate theo alert-conditional activation overlap/Jaccard đã khai báo. Rule weights, evidence aggregation và risk bins chỉ được khóa từ train/validation alerts.

Biến thể contrastive meta-scorer dùng các rule A/B đã qua audit cùng frozen calibrated risk để phân biệt true-positive và false-positive alerts. Nó chỉ fit trên `rule_audit`, có minimum sample/class/risk-variation guardrail và trả zero evidence kèm fallback reason khi không đủ dữ liệu; `policy_select` và locked test chỉ transform bằng schema/mean/std/coefficients đã khóa.

Guarded ensemble là arithmetic mean được pre-register của FP-penalized, attribution-gated và contrastive evidence. Full A/B, domain-only, exact counterfactual-only, all-Tier-B, từng method, guarded ensemble, CART, signed native-attribution top-k và shuffled controls đều phải qua cùng policy-select score/fidelity/support và resampling-stability guardrails trước locked test.

Attribution support chỉ kiểm tra liên hệ giữa rule features và frozen predictor; nó không tự chứng minh causal explanation hoặc full faithfulness.

### 13.5. Fixed-coverage selective evaluation

Coverage budgets được pre-register là 5%, 10%, 25% và 50%; 10%/25% là primary, 5%/50% là sensitivity. Selection phải deterministic và ghi riêng requested count/coverage với realized count/coverage. Support shift có thể làm rule policy underfill; unsupported alerts không được thêm vào để lấp budget.

Mỗi rule-evidence policy có score-only guardrail dùng cùng frozen probabilities. Core comparison luôn rank score-only ở đúng realized selected count của rule policy; kết quả score-only tại requested budget chỉ là diagnostic có tên riêng. So sánh equal-count không hữu hạn thì positive/saturation claim bị chặn. Report gồm selected precision, recall, gain/lift, requested và realized explanation coverage, abstention, sparsity, contradiction và fidelity/attribution support theo đúng denominator. Row không có đủ evidence phải mang nhãn abstained/unsupported thay vì được gán giải thích rỗng.

Rule-added value chỉ được tuyên bố khi chênh lệch với score-only ổn định và có uncertainty report phù hợp. Không chọn coverage tốt nhất từ locked test.

### 13.6. Residual evidence, uncertainty và controls

- **Matched-risk:** risk bins/matching rule fit trên validation, pair/select row trên test không dùng label; so sánh evidence status ở mức model risk tương đương.
- **Residual TP-vs-FP:** so sánh rule evidence giữa TP và FP alerts bên trong locked risk strata.
- **Paired bootstrap:** resample cùng rows hoặc temporal blocks cho rule policy và score-only; báo point estimate, 95% interval và valid replicate count.
- **Policy stability:** validation resampling/seeds, rule survival, selection overlap, coverage error và metric dispersion.
- **Mandatory shuffled controls:** validation-label shuffle và validation-weight shuffle phải có metric hữu hạn ở cùng realized count với locked primary và với score-only comparator của chính control. Thiếu control, count mismatch hoặc control abstain/non-finite làm final claim fail closed; control tương đương/vượt primary chặn positive claim, còn material positive gain của control chặn saturation claim.
- **Negative controls:** shuffled/permuted evidence hoặc control tương đương mà không thay frozen predictor.

Negative result là kết quả hợp lệ. AMLNet chỉ được gắn diễn giải saturation khi score-only đã mạnh, residual gain không ổn định và các data-quality/lineage/stability checks đều qua. Saturation benchmark không chứng minh production performance.

### 13.7. Notebook boundary

Notebook 09 và 10 độc lập, có thể chạy song song. Notebook 11 chỉ đọc completed full output packages, kiểm tra lineage và tổng hợp; nó không train, sinh rule hoặc chọn lại policy. Output contract và claim boundaries chi tiết nằm trong [STRESS_TEST_PROTOCOL.md](STRESS_TEST_PROTOCOL.md).
