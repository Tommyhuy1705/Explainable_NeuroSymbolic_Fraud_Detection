# Giao thức stress test trên TransXion và AMLNet

Tài liệu này định nghĩa protocol và result contract trước full experiments. Sự tồn tại của code, config hoặc notebook không đồng nghĩa stress tests đã chạy, và không có kết luận thực nghiệm nào được suy ra chỉ từ tài liệu này.

## 1. Vị trí trong phạm vi khóa luận

TransXion và AMLNet là hai **stress-test datasets**, không phải dataset chính thứ ba và thứ tư có vai trò ngang hàng với IEEE-CIS và BAF. Phần mở rộng này kiểm tra xem kết luận về lớp giải thích rule-audited chọn lọc có còn đứng vững trong hai điều kiện biên hay không:

- **TransXion**: stress test AML khó, có quy mô lớn, nhãn hiếm và cấu trúc giao dịch theo thời gian. Vai trò chính là kiểm tra khả năng tái áp dụng framework khi chuyển từ fraud benchmark sang bối cảnh AML phức tạp hơn.
- **AMLNet v1.0**: stress test bão hòa. Vai trò chính là kiểm tra giới hạn của rule evidence khi frozen predictor hoặc score-only ranking đã rất mạnh, khiến phần headroom còn lại có thể rất nhỏ.

Stress test không dùng để thay thế kết quả chính trên IEEE-CIS/BAF, không dùng để tuyên bố universal generalization và không biến benchmark tổng hợp thành bằng chứng production.

Proposal còn liệt kê European Credit Card Fraud như stress test bổ sung/phụ lục. Snapshot này triển khai đúng hai dataset được yêu cầu ở phase hiện tại là TransXion và AMLNet; vì vậy không được ghi rằng toàn bộ ma trận ba stress datasets trong proposal đã hoàn tất cho đến khi European có notebook/output riêng.

## 2. Dataset identity và provenance lock

### 2.1. TransXion

Trong repository này, cụm từ **TransXion v2** chỉ revision v2 của bài báo arXiv `2604.17420v2`. Official repository hiện cung cấp canonical dataset `tx.csv`; không có bằng chứng về một dataset release/tag riêng mang tên `v2`. Vì vậy, mọi manifest và báo cáo phải tách rõ:

```text
paper_revision: arXiv:2604.17420v2
dataset_name: TransXion
dataset_file: tx.csv
source_repository_commit: 53932595c37c23b9f55ea5ddf5984e4d57b88369
sha256: d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35
```

Nguồn chính thức:

- Paper revision v2: <https://arxiv.org/html/2604.17420v2>
- Source repository: <https://github.com/chaos-max/TransXion>

Giá trị tham chiếu từ nguồn là 3.029.170 giao dịch, 4.641 nhãn dương, 365 ngày và 47.526 tài khoản. Pipeline phải tính lại và ghi các con số thực tế sau khi load; các giá trị tham chiếu không được dùng thay cho data audit.

### 2.2. AMLNet v1.0

AMLNet được khóa theo Zenodo record v1.0 và exact file, không chỉ theo filename:

```text
dataset_version: AMLNet v1.0
zenodo_record: https://zenodo.org/records/16736515
doi: 10.5281/zenodo.16736515
dataset_file: AMLNet_August 2025.csv
md5: 7668fc7d74c787e07546ce85c6f790b9
license: CC BY-NC 4.0
```

Nguồn chính thức: <https://zenodo.org/records/16736515>.

Giá trị tham chiếu từ record/file là 1.090.173 giao dịch, 1.745 nhãn AML dương và khoảng 195 ngày. Cần ghi rõ một source-version discrepancy: Zenodo record được khóa là v1/v1.0, trong khi phần README preview của nguồn có thể hiển thị `VERSION 2.0`. Protocol không suy diễn version từ README preview; nó dùng DOI, exact filename và MD5 nêu trên làm identity contract, đồng thời ghi discrepancy trong manifest.

## 3. Data integrity và leakage controls

### 3.1. Checksum-first loading

Full run chỉ được tiếp tục khi:

1. Tìm đúng exact filename trong các input roots.
2. File là duy nhất hoặc người chạy đã chỉ định tường minh một file.
3. Checksum khớp identity contract.
4. Schema, target và timestamp parser qua preflight.
5. Manifest ghi source path, file size, checksum, thời điểm chạy, git commit, config hash và full/fixture mode.

Không commit raw data vào Git. Không phân phối lại TransXion khi chưa xác minh đầy đủ quyền phân phối. AMLNet chỉ được sử dụng theo CC BY-NC 4.0 và các điều khoản Zenodo tương ứng.

### 3.2. Timestamp AMLNet

File AMLNet không được coi là đã sắp thời gian theo row order. `step` không duy nhất và không đủ làm khóa thời gian. Timestamp nằm trong chuỗi Python-like của cột `metadata`, ví dụ `datetime.datetime(...)`. Loader phải:

- Trích timestamp bằng parser giới hạn/regular expression đã kiểm thử.
- Không dùng `eval`, `exec` hoặc thực thi chuỗi metadata.
- Báo tỷ lệ parse thành công và dừng full run nếu timestamp bắt buộc không parse được đầy đủ.
- Stable-sort theo timestamp và original row order trước khi split.

Sau khi trích `event_time`, raw `metadata` bị loại khỏi predictor features.

### 3.3. Feature eligibility

Chỉ dùng feature khả dụng tại thời điểm giao dịch. Mọi rolling/history feature phải tính theo chiều quá khứ và phải dùng `shift` hoặc cơ chế tương đương để giao dịch hiện tại không tự rò rỉ vào feature của chính nó.

Với AMLNet, denylist tối thiểu gồm:

- `isMoneyLaundering` và target alias.
- `isFraud` nếu nhiệm vụ là AML detection.
- `laundering_typology`.
- `fraud_probability`.
- Raw `metadata` và `step`.
- Raw account IDs, trừ khi chỉ được dùng để xây causal history feature rồi bị loại.

Profile hoặc entity fields của TransXion chỉ được join bằng composite key phù hợp và phải audit join coverage/cardinality. Không dùng nhãn, thông tin điều tra sau giao dịch hoặc biến được sinh từ target.

## 4. Temporal split 60/20/20

Mỗi stress dataset được stable-sort theo `event_time`, sau đó chia theo thời gian:

- Train: 60% đầu.
- Validation: 20% tiếp theo.
- Locked test: 20% cuối.

Pipeline phải xuất min/max timestamp, row count, positive count, prevalence và kiểm tra thứ tự cho từng split. Preprocessor, feature selection, category mapping, rule threshold và candidate generation chỉ fit trên train. Test không được dùng để chọn predictor, calibration, threshold, rule, rule weight, evidence method hoặc coverage budget.

History counts/sums/time-since chỉ được đọc các timestamp **nhỏ hơn nghiêm ngặt** timestamp
hiện tại. Source row order chỉ dùng để stable-sort; các giao dịch trùng timestamp không được xem
là history của nhau.

Validation phải được chia thành bốn chronological roles không chồng lấn, được ghi trong manifest:

1. **Calibration fit**: fit calibration candidates.
2. **Calibration select**: so sánh calibration method, khóa operating threshold và predictor-selection evidence theo contract đã khai báo.
3. **Rule audit**: đánh giá candidate rules, attribution support, loại rule không đủ bằng chứng và deduplicate.
4. **Policy select**: chọn evidence method, coverage policy và các tham số abstention; đánh giá stability mà không quay lại chỉnh rule audit.

Nếu cấu hình gộp hoặc chia khác các vai trò này, run không được gắn nhãn confirmatory và phải nêu rõ là sensitivity analysis.

## 5. Predictor protocol và freeze boundary

Ba predictor families phải được đánh giá dưới cùng split contract:

- TabularResNetV2.
- XGBoost.
- LightGBM.

Training, calibration, threshold selection và model-family selection chỉ dùng train/validation theo các role đã khóa. Predictor tham chiếu sau đó được freeze; downstream rule audit, selective policy và synthesis không được huấn luyện lại hoặc thay đổi probability/threshold.

Prediction report tối thiểu gồm:

- PR-AUC là primary ranking metric trong bối cảnh nhãn hiếm.
- ROC-AUC.
- Precision, recall và F2 tại locked threshold.
- Recall@1%FPR.
- Brier score, Expected Calibration Error và negative log-likelihood.

Freeze manifest phải gắn model family, seed, feature schema, preprocessor, calibration method, threshold, source/config fingerprints và checksums của prediction artifacts.

Vì đây là hai stress tests bổ sung có quy mô rất lớn, cấu hình full khóa **một seed huấn luyện định trước (`42`) cho mỗi family**, thay vì lặp ba seed như hai benchmark chính. Đây là quyết định về tính khả thi tính toán, không phải bằng chứng rằng training variance bằng không. Độ bất định được bổ sung bằng validation policy resampling và paired calendar-day bootstrap, nhưng hai phép này không thay thế hoàn toàn multi-seed training variance; giới hạn này phải được nêu trong khóa luận.

## 6. Candidate rules và audit

Candidate rules được tách theo provenance tier:

- **Tier A — domain-prespecified**: luật được khai báo trước khi xem validation/test, ví dụ high amount, rapid activity, new counterparty, currency inconsistency, cross-bank flow hoặc unusual hour khi feature cho phép.
- **Tier B — train-derived/counterfactual-assisted**: threshold hoặc candidate được sinh từ train-only perturbation/counterfactual analysis quanh frozen predictor. Provenance phải ghi rõ; không được gọi là domain rule thuần túy.
- **Tier C — empirical baseline candidates**: shallow decision-path hoặc attribution-derived candidates dùng để so sánh. Chúng là baseline thực nghiệm, không phải tri thức AML đã được chuyên gia xác nhận.

Tier C bị tách vật lý khỏi primary evidence pool. CART surrogate chỉ xuất hiện trong
`cart_tier_c_baseline`; nó không được audit/weight như Tier A/B và không thể được chọn làm
primary VASRE policy. Cùng primary A/B rule set được replay trên XGBoost, LightGBM và
TabularResNetV2 với attribution/policy riêng để tạo predictor-sensitivity evidence; đây là
sensitivity của rule layer đã khóa, không phải tuyên bố cùng model hoặc rule tự động transfer.

Tất cả candidate generation/fitting chỉ dùng train. Mỗi counterfactual intervention trên raw/causal field phải tái tính các semantic derived fields theo contract whitelist trong YAML trước khi predictor chấm điểm; derived output không được perturb trực tiếp. `sender_seconds_since_previous`, `receiver_seconds_since_previous` và `pair_prior_transaction_count` là causal history-derived fields, không phải raw transaction inputs. Vì single-row median intervention không thể tái tạo hợp lệ toàn bộ lịch sử giao dịch trước đó, ba field này được khai báo trong `non_intervenable_derived_features`, bị loại khỏi candidate interventions và chỉ được dùng như các feature lịch sử đã tính causal trong rule/predictor. Rule audit trên validation audit role chỉ tính coverage, positive precision/lift, TP-vs-FP rule AUC, attribution support và active count trong **frozen-predictor alert region**. Các chỉ số toàn population chỉ là diagnostics mang tên `*_population_*`, không được dùng để pass/fail hoặc weight. Rule bị loại phải có failure reason; rule gần trùng được deduplicate bằng activation overlap/Jaccard cũng chỉ trong alert region. Rule weights chỉ được khóa từ train/validation alerts, không fit trên test.

Proposal yêu cầu thêm **contrastive meta-scorer**. Trong stress extension, đây là logistic scorer cân bằng lớp, dùng soft-truth của các rule A/B đã qua audit **cùng frozen calibrated predictor risk đã standardize trên audit alerts** để phân biệt TP alerts với FP alerts. Nó chỉ fit trên frozen-predictor alerts thuộc `rule_audit`; `policy_select` và locked test chỉ được label-free transform bằng mean/std, feature schema và coefficients đã khóa. Cấu hình khóa trước gồm tối thiểu 50 alerts, 10 rows mỗi lớp, L2 với `C=1.0`, seed cố định và tối đa 1.000 iterations. Nếu không đủ rule, alerts, TP/FP, rule variation hoặc risk variation, scorer trả evidence bằng 0 cùng fallback reason; không chuyển sang fit trên partition khác.

`guarded_ensemble` là một candidate method được pre-register trong YAML: arithmetic mean của `fp_penalized`, `attribution_gated` và `contrastive_meta`. Ensemble không dùng test label, không tự thay component/weight sau khi xem kết quả, và chỉ có thể được khóa nếu vượt score-only, support/fidelity/TP-FP và resampling-stability guardrails trên `policy_select`.

Attribution support chỉ đo mức các feature của rule có liên quan đến frozen predictor theo attribution method đã khai báo. Nó không tự động chứng minh quan hệ nhân quả hoặc faithful explanation đầy đủ.

## 7. Selective explanation policy

Coverage budgets được khóa trước:

```text
5%, 10%, 25%, 50%
```

Trong đó 10% và 25% là primary coverages; 5% và 50% là secondary/sensitivity coverages. Mỗi budget phải có exact deterministic selection và denominator rõ ràng. Artifact phải ghi riêng `requested_selected_count`, `realized_selected_count`, requested coverage và realized coverage. Nếu active-rule support trên locked test thấp hơn requested budget, policy được phép underfill nhưng tuyệt đối không chèn unsupported alert để đủ số lượng.

Với mỗi rule-evidence policy, bắt buộc có **score-only guardrail** dùng chính frozen predictor score. Core precision delta và paired bootstrap luôn dùng score-only mask có đúng bằng `realized_selected_count` của rule policy. Score-only tại requested budget có thể được báo riêng như diagnostic, nhưng không được dùng trong core delta khi rule policy underfill. Nếu không tạo được so sánh equal-count hữu hạn, mọi positive hoặc saturation claim phải fail closed. Không được tuyên bố rule layer cải thiện triage nếu chênh lệch so với score-only không ổn định hoặc khoảng tin cậy bao gồm kết quả không cải thiện.

Những row không được policy chọn hoặc không có rule qua audit phải được ghi là **abstained/unsupported**, không được gán ngầm một giải thích rỗng. `abstention_rate` là tỷ lệ frozen-predictor alerts không được chọn, tức `1 - explanation_coverage`; nó khác với cờ policy-level `abstain` khi toàn bộ evidence method không qua validation guardrail. Report tối thiểu gồm selected precision, `selected_positive_count`, `test_positive_count`, `alert_positive_count`, recall với cả test-positive và alert-positive denominator, lift/gain so với population và score-only, explanation coverage, abstention rate, sparsity và contradiction.

Một alert chỉ đủ điều kiện vào fixed budget khi ít nhất một rule đã qua audit đạt
`activation_threshold=0.60`; soft-truth dương rất nhỏ không được tính là support. Validation
phải có đủ active-rule alerts để lấp đầy budget, vượt score-only ít nhất `0.005`, có TP-FP
evidence dương, positive-region fidelity tối thiểu `0.50` và policy stability tối thiểu `0.60`.
Không đạt bất kỳ guardrail nào thì khóa thành policy-level abstention trước khi mở test labels.

## 8. Bằng chứng bổ sung và uncertainty

### 8.1. Matched-risk analysis

Risk bins hoặc matching rule được fit/khóa trên validation. Trên locked test, so sánh các row có frozen predictor risk tương đương nhưng khác rule-evidence status. Matching không được dùng test label để chọn pair. Mục tiêu là kiểm tra rule evidence có cung cấp thông tin hậu kiểm bổ sung sau khi đã kiểm soát model risk hay không.

### 8.2. Residual TP-vs-FP evidence

Trong các risk strata đã khóa, so sánh rule evidence giữa true-positive và false-positive alerts. Phân tích này không khẳng định quan hệ nhân quả; nó chỉ kiểm tra khả năng phân biệt hậu kiểm còn lại sau model score.

### 8.3. Paired bootstrap

Rule policy và score-only guardrail được so sánh trên cùng test rows. Bootstrap phải paired; khi có temporal grouping phù hợp, ưu tiên resample theo time block thay vì coi mọi row độc lập. Báo point estimate, percentile 95% interval, valid replicate count và denominator. Interval này ước lượng sampling uncertainty của locked test, không thay thế training-seed variability.

### 8.4. Stability và negative controls

Policy stability được đánh giá bằng các validation resample/seed đã khai báo: rule survival, selection overlap, metric dispersion và coverage error. Cùng stability gate được áp dụng cho primary method và mọi baseline/ablation trước khi mở test. Negative controls gồm row-permuted/random evidence, validation-label shuffle và validation-weight shuffle; weight shuffle chỉ được chọn giữa các weighted methods, không được fallback sang unweighted. Các control không được đọc test label để tạo ranking. Row-permuted/random rankings giữ nguyên active-support count trong alert region và được so sánh ở cùng realized count của locked policy. Hai shuffled validation controls là bắt buộc trong final confirmation gate: thiếu/non-finite/abstain, khác realized count với primary, hoặc có comparator score-only khác count đều chặn positive và saturation claim. Positive claim còn yêu cầu cả hai control kém locked primary; saturation chỉ được xác nhận khi cả hai không tạo material positive gain. Không bao giờ bù count thiếu bằng unsupported alerts. Nếu negative control cho kết quả tương đương policy thật, bằng chứng rule-added value là không đủ.

Ablation bắt buộc tách: full A/B; domain Tier A only; **exact counterfactual-only** (`source=train_model_counterfactual`); all train-derived Tier B; unweighted; audit-weighted không attribution; FP-penalized không attribution; attribution-gated; contrastive meta; guarded ensemble; shallow CART Tier C; và signed native-attribution top-k. Ba candidate-pool baselines (Tier A only, exact counterfactual-only và all Tier B) phải bắt đầu từ toàn bộ rule audit table **trước** primary A/B deduplication. Mỗi pool tự chạy lại cùng audit-pass filter, alert-conditional Jaccard deduplication, `max_rules`, weight fitting và validation policy-stability gate; không được lọc từ tập primary weights đã chọn vì cross-tier redundancy/rule budget có thể làm mất một luật hợp lệ của baseline. Provenance phải lưu candidate count, audit-pass count, selected rules, threshold/budget và xác nhận không dùng test label. Top-k baseline chọn 5 local effects có absolute magnitude lớn nhất từ tree `pred_contribs` hoặc neural gradient×input, sau đó cộng chỉ các contribution dương theo raw fraud-logit direction. Nó không còn là một concentration ratio bỏ qua magnitude/sign.

Nhãn `positive_incremental_evidence` chỉ được ghi khi năm hướng bằng chứng cùng hội tụ: paired-bootstrap CI có lower bound dương, matched-risk CI có lower bound dương, residual TP-vs-FP evidence dương, locked rule evidence vượt các shuffled/random controls, và tối thiểu 80% validation resamples giữ cùng method, không abstain và có full-validation delta dương. Nếu paired CI dương nhưng một guardrail còn lại không đạt hoặc không xác định, kết quả phải là `inconclusive_incremental_evidence`.

## 9. Cách diễn giải negative result và saturation

Negative result là kết quả hợp lệ nếu protocol đã được khóa và data/lineage checks đều qua. `counterfactual_only` chỉ gồm candidate sinh bởi train-only model intervention; `train_derived_tier_b_only` là baseline rộng hơn, bao gồm toàn bộ Tier B đã fit trên train. Cả hai được deduplicate và reweight độc lập trong candidate pool riêng, không kế thừa subset survivors của full A/B. Trên AMLNet, **saturation** chỉ được ghi nhận khi:

- Score-only đã có hiệu quả rất cao tại primary coverages.
- Rule-audited policy không có residual gain ổn định sau matched-risk/paired-bootstrap.
- Kết quả không do pipeline failure, leakage, checksum mismatch, quá ít positive rows hoặc unstable policy.

Các materiality thresholds phải nằm trong YAML và được ghi lại trong outcome artifacts/manifest: score-only headroom tối đa `0.05`, precision no-gain margin `0.02`, evidence no-gain margin `0.05`, minimum positive stability fraction `0.80`, ít nhất `20` positive alerts, `20` matched pairs và `200` valid paired-bootstrap replicates. Saturation scope chỉ hợp lệ cho AMLNet full/checksummed run có exact model backends, đủ attribution và lineage; TransXion không được gắn nhãn saturation. Không được thay các giá trị này sau khi xem locked-test outcomes.

Saturation trên AMLNet không đồng nghĩa với hiệu năng production hoặc hiệu năng trên dữ liệu ngân hàng thật. Nếu rule layer chỉ tăng khả năng mô tả/audit mà không cải thiện triage so với score-only, phải báo cáo đúng giới hạn đó.

## 10. Notebook order và execution modes

Ba notebook mở rộng có dependency như sau:

```text
09_TransXion_v2_Stress_Test.ipynb ─┐
                                        ├─► 11_Stress_Test_Synthesis.ipynb
10_AMLNet_v1_0_Stress_Test.ipynb ──┘
```

- Notebook 09 và 10 độc lập, có thể chạy song song trên hai Kaggle sessions.
- Notebook 11 chỉ đọc hai output packages đã hoàn tất, kiểm tra lineage/checksum/mode rồi tổng hợp. Nó không train model, không sinh lại rule và không chọn lại policy trên test.

Full mode là chế độ duy nhất được dùng cho claim. Fixture/quick mode chỉ dùng để kiểm tra schema, dependency, artifact writing và flow control; mọi artifact của chế độ này phải mang cờ `fixture`/`quick_run` và bị loại khỏi synthesis chính thức.

## 11. Output contract

Mỗi dataset stress run tối thiểu phải xuất:

```text
data_manifest.json
split_integrity.csv
predictive_metrics.csv
predictor_explanation_sensitivity.csv
predictor_attribution_sensitivity.csv
predictor_contrastive_sensitivity.csv
predictor_manifest.json
rule_registry.csv
rule_audit.csv
rule_audit_stability.csv
rule_redundancy.csv
contrastive_meta_coefficients.csv
contrastive_meta_provenance.json
locked_policies.json
policy_candidates_validation.csv
ablation_validation_results.csv
ablation_validation_stability.csv
ablation_results.csv
coverage_results.csv
matched_risk_results.csv
residual_evidence.csv
paired_bootstrap.csv
validation_stability.csv
negative_controls.csv
stress_test_manifest.json
stress_lineage.json
```

Notebook 11 phải xuất `primary_stress_test_synthesis.csv`, `synthesis_manifest.json` và
`synthesis_lineage.json`; lineage cuối lưu checksum của cả hai upstream manifests/lineages,
source commit/fingerprint và checksum của các bảng tổng hợp.

Mọi bảng tỷ lệ phải có numerator/denominator hoặc row counts đủ để audit. `NaN` được dùng cho metric không xác định; không thay bằng `0`. Synthesis phải phân biệt confirmatory primary coverages 10%/25% với sensitivity 5%/50%, phân biệt positive, negative, inconclusive và blocked-by-data-quality results.

## 12. Claim boundaries

Stress-test evidence có thể hỗ trợ các tuyên bố hạn chế sau:

- Framework có thể được thiết lập và audit trên hai AML benchmarks bổ sung.
- Selective rule evidence có hoặc không có giá trị bổ sung so với frozen predictor score tại coverage đã khóa.
- Giá trị bổ sung, nếu có, ổn định đến mức nào qua validation resampling, matched-risk và paired bootstrap.
- Framework gặp giới hạn nào trong bối cảnh saturation.

Không tuyên bố causal explanation, universal superiority, model/rule transfer không thay đổi, external validation trên tổ chức tài chính thật, hay production readiness.
