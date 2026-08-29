# Hướng dẫn quản lý kết quả

## Generated outputs

Mỗi run ghi vào `results/runs/<dataset>/`:

- `predictive_metrics.csv`.
- `predictions.npz`.
- `run_metadata.json`.
- `calibration_comparison.csv`.
- `split_summary.csv`.
- `config_snapshot.json`.

Mỗi repeated benchmark còn xuất:

- `predictive_metrics_all_seeds.csv` và `predictive_metrics_summary.csv`.
- `calibration_comparison_all_seeds.csv`.
- `paired_bootstrap_model_differences.csv`.
- `frozen_reference_artifact.npz` và `frozen_reference_manifest.json`.
- `repeated_run_metadata.json`.

Thư mục `results/runs/` bị gitignore để tránh commit output thử nghiệm hoặc file lớn.

## Curated thesis outputs

Chỉ chuyển các artifact đã kiểm tra vào:

```text
results/
├── tables/
└── figures/
```

Mỗi bảng/hình dùng trong khóa luận phải truy vết được đến:

- Notebook/script tạo ra.
- Config.
- `dataset_reference`/`dataset_version` được ghi thủ công trong experiment log hoặc Kaggle notebook version notes; nếu provider không cung cấp version, ghi `unavailable`.
- Seed.
- Run metadata.
- Frozen artifact checksum và config hash khi kết quả phụ thuộc predictor.
- Source `git_commit`, notebook version và full/quick mode.

Notebook 05-07 phải qua frozen-artifact preflight trước khi xuất kết quả. Preflight xác nhận đúng dataset, đúng một manifest/artifact hợp lệ, reference seed, config hash, checksum, mode và label alignment.

Notebook 04-07 đều phải xuất lineage chứa git commit, audit-pipeline/config fingerprint, danh sách output và SHA-256 của từng output. Notebook 04 không phụ thuộc frozen predictor nhưng vẫn phải qua source/config/output lineage audit. Notebook 08 chỉ tổng hợp output 02-07 sau khi các frozen checks và lineage checks bắt buộc đều qua; nó không train model hoặc chọn lại model/rule policy bằng test. Notebook 01 phải được rerun khi EDA thay đổi, nhưng không phải input dependency của Notebook 08.

`label alignment` của frozen artifacts 02/03 nghĩa là toàn bộ label arrays validation/test khớp đúng thứ tự; legacy manifests chưa hash row IDs hoặc feature rows. Không đổi loader/split order khi tái sử dụng hai artifacts này, và ghi đây là một provenance limitation trong báo cáo.

Notebook là generated artifact từ `scripts/generate_notebooks.py`. Mọi thay đổi cell dùng cho kết quả chính thức phải có trong generator và vượt source-sync checks; không curate output từ notebook sửa tay lệch source.

```bash
python scripts/generate_notebooks.py
pytest -q tests/test_notebook_generation.py
```

## Quy ước diễn giải metric

- Dùng mean validation raw PR-AUC qua ba seed để chọn reference model; dùng raw test PR-AUC để báo cáo ranking đã khóa.
- Dùng Brier, ECE và NLL để đánh giá calibrated probabilities; precision/recall/F1/F2 dùng calibrated threshold đã chọn trên validation.
- F2 với `beta = 2` phản ánh giả định ưu tiên recall, không phải business cost ratio đã được kiểm chứng.
- Reference artifact dùng seed 42. Paired bootstrap resample cùng test rows của seed 42 và không thay thế mean ± standard deviation qua ba training seed.
- Rule lift phải đi cùng coverage và active count. Precision/lift của rule không có active rows và precision gain khi không có explained alerts được ghi `NaN`, không ghi `0`.
- Rule-quality denominator dùng tên `evaluated_rows` vì bảng có thể thuộc train, validation hoặc test split.
- Numeric-rule missing values trên validation/test phải dùng training median đã khóa. Bảng fitted conditions và case evidence phải mang theo `fitted_missing_value`; không curate kết quả nếu median bị recompute theo split.
- Explanation table phân biệt `mean_active_rule_count` với `mean_displayed_rule_count`. `rule_sparsity = 1 - mean(active_rule_count / available_rule_count)`; `rule_count` chỉ là alias backward-compatible của active count và không nên dùng làm nhãn báo cáo mới.
- Test ablation là post-hoc diagnostic. Không đặt tên output theo kiểu `best_rule_set`; dùng cách gọi `diagnostic_condition` hoặc `coverage_precision_tradeoff`.
- BAF là privacy-preserving synthetic benchmark. Kết quả Notebook 07 là cross-dataset replication/portability, không phải transfer cùng model/rule base hoặc external validation trên dữ liệu ngân hàng thực.

## Naming convention

```text
table_<dataset>_<experiment>_<metric>.csv
figure_<dataset>_<experiment>_<description>.png
```

Ví dụ:

```text
table_ieee_predictive_test_metrics.csv
table_ieee_rule_quality.csv
figure_ieee_pr_curves.png
figure_ieee_rule_coverage.png
```

Với hình rule quality, ưu tiên tên phản ánh cả coverage và lift, ví dụ:

```text
figure_ieee_rule_coverage_lift.png
figure_cross_dataset_explanation_precision_gain.png
```

## Synthetic results

Không chuyển synthetic smoke outputs vào `results/tables` hoặc `results/figures`. Synthetic results chỉ xác nhận code chạy được và không hỗ trợ kết luận nghiên cứu.

`data_source = real` hoặc `official_benchmark_file` nghĩa loader đã đọc file benchmark thay vì fallback; riêng BAF vẫn là dữ liệu tổng hợp bảo toàn riêng tư. Khi curate EDA BAF, lưu cả bảng native `NaN` và sentinel `-1` để tránh tuyên bố missingness không chính xác. Protocol hiện tại giữ `-1` như giá trị mã hóa của benchmark, không tự động chuyển thành `NaN`; quyết định này phải đi kèm result package.

## Stress-test outputs

TransXion và AMLNet outputs thuộc một stress-test extension riêng. Chúng không được gộp vào bảng IEEE-CIS/BAF như hai co-primary datasets, và không được dùng để thay thế kết quả cốt lõi.

Mỗi full stress run dự kiến xuất tối thiểu:

- `data_manifest.json`: exact dataset identity, checksum, source/version discrepancy, schema, row/label/time audit và mode.
- `split_integrity.csv`: train/validation/test và validation-role boundaries, row/positive counts, prevalence, min/max time.
- `predictive_metrics.csv` và `predictor_manifest.json`: ba model families, calibration, threshold và frozen reference identity.
- `predictor_explanation_sensitivity.csv` và `predictor_attribution_sensitivity.csv`: replay cùng reference-fitted A/B rule set trên cả ba predictor; không được diễn giải là model/rule transfer tự động.
- `contrastive_meta_coefficients.csv`, `contrastive_meta_provenance.json` và `predictor_contrastive_sensitivity.csv`: contrastive TP-vs-FP scorer dùng audited rule truth cùng frozen calibrated risk trên `rule_audit`, hệ số/fallback và sensitivity theo predictor; scorer không được fit lại trên test.
- `rule_registry.csv`, `rule_audit.csv`, `rule_redundancy.csv`: Tier A/B/C provenance, alert-conditional pass/fail statistics, attribution support và alert-conditional deduplication; các cột `*_population_*` chỉ là diagnostics.
- `locked_policies.json`, `policy_candidates_validation.csv`: policy decisions đã khóa hoàn toàn từ validation.
- `ablation_validation_results.csv`, `ablation_validation_stability.csv`, `ablation_results.csv`: full A/B, domain Tier A only, exact counterfactual-only, all Tier B, từng evidence method, guarded ensemble, CART Tier C, signed native-attribution top-k và validation label/weight shuffle controls.
- `ablation_rule_selection.csv`, `ablation_rule_pool_provenance.json`: audit/pass/dedup/selection và provenance của ba candidate-pool baselines. Tier A only, exact counterfactual-only và all Tier B phải tự chọn từ complete pre-dedup audit pool, tự tính lại weights với cùng `max_jaccard`/`max_rules`, không filter primary survivors và không dùng test label.
- `coverage_results.csv`: requested và realized count/coverage của rule-audited policy, score-only comparator ở cùng realized count, và score-only requested-budget diagnostic tại 5%, 10%, 25%, 50%.
- `matched_risk_results.csv`, `residual_evidence.csv`, `paired_bootstrap.csv`: residual evidence và uncertainty.
- `rule_audit_stability.csv`, `validation_stability.csv`, `negative_controls.csv`: rule survival, policy/selection stability và falsification checks.
- `stress_test_manifest.json`, `stress_lineage.json`: source/config/input/output fingerprints và checksums.

`results/runs/` vẫn bị gitignore. Không commit full raw predictions hoặc dữ liệu gốc. Chỉ curate các bảng/hình nhỏ đã qua completeness gate.

## Stress-test completeness gate

Không curate hoặc diễn giải kết quả chính thức nếu thiếu một trong các điều kiện sau:

- TransXion exact SHA-256/source commit hoặc AMLNet exact MD5/DOI khớp identity contract.
- `fixture=false`, `quick_run=false` và không có synthetic substitution.
- Temporal 60/20/20 split và validation roles không chồng lấn.
- AMLNet timestamp parse rate đạt 100% và safe parser contract qua audit.
- Predictor/preprocessor/calibration/threshold đã freeze trước rule audit/policy evaluation.
- Feature denylist và causal-history checks qua.
- Rule candidates có Tier A/B/C provenance, audit decision và failure reason; audit/TP-FP/weight/dedup chỉ dùng frozen-predictor alerts; Tier C không được nằm trong primary rule audit/weight/policy.
- Counterfactual derived-feature contract và recomputation provenance qua kiểm tra; không perturb trực tiếp một derived output.
- Có cả rule-audited và score-only results ở cùng exact realized selected count; requested budget được báo tách riêng khi support shift làm policy underfill.
- Có primary coverages 10%/25%; 5%/50% được đánh dấu sensitivity.
- Có abstention, matched-risk, residual TP-vs-FP, paired bootstrap, stability và negative controls.
- Chỉ row có ít nhất một audited rule đạt activation threshold được tính là explanation; requested budget thiếu support phải fail/abstain thay vì điền bằng zero-evidence alerts.
- Predictor-sensitivity và ablation tables có pre-label selection invariance/provenance; mọi confirmatory baseline/ablation có validation stability gate.
- Ba candidate-pool ablations có independent selection provenance, candidate/audit-pass/selected counts và selected-rule list; `primary_selected_weights_filtered=false`.
- `stress_lineage.json` xác minh source/config/input/output checksums.
- Notebook 11 có `synthesis_manifest.json` và `synthesis_lineage.json` kiểm tra checksum upstream và output synthesis.

Notebook 11 chỉ được tổng hợp khi cả hai upstream packages qua gate. Nếu một dataset bị blocked, báo `blocked_by_data_quality` hoặc `blocked_by_lineage`; không impute một kết luận từ dataset còn lại.

## Quy ước diễn giải stress-test results

- TransXion là difficult-AML/generalization stress test; AMLNet là saturation/limit stress test. Không gọi chúng là hai external banking datasets.
- `v2` trong tên TransXion notebook là paper revision v2, không phải dataset version. Curated caption phải ghi canonical `tx.csv`, source commit và SHA-256.
- AMLNet caption phải ghi v1.0 Zenodo DOI, exact filename, MD5, CC BY-NC 4.0 và source-version discrepancy.
- Report predictive PR-AUC/ROC-AUC/F2/precision/recall/Recall@1%FPR cùng Brier/ECE/NLL trước khi diễn giải rule layer.
- Tại mỗi coverage, core rule-audited/score-only comparison phải có cùng realized selected count. Không lấp budget bằng unsupported alerts và không dùng score-only requested-budget diagnostic để tính core delta khi rule policy underfill.
- Selected precision/gain/lift phải đi cùng coverage, selected count, `selected_positive_count`, `test_positive_count`, `alert_positive_count`, hai recall denominator và row-level abstention (`1 - explanation_coverage`). Cờ policy-level `abstain` phải được báo riêng. Metric không xác định ghi `NaN`, không ghi `0`.
- Paired-bootstrap interval phải đi cùng valid replicate count và resampling unit. Khoảng tin cậy bao gồm zero không hỗ trợ tuyên bố improvement.
- Matched-risk và residual TP-vs-FP là bằng chứng hậu kiểm có kiểm soát model score; chúng không chứng minh causal effect.
- Negative control tương đương policy thật là bằng chứng chống lại rule-added-value claim.
- `positive_incremental_evidence` đòi hỏi paired bootstrap, matched-risk, residual TP-vs-FP, validation stability và negative controls cùng hỗ trợ; label-shuffle và weight-shuffle controls phải đầy đủ, hữu hạn và cùng realized count với primary. Nếu chỉ paired CI dương, control bị thiếu/abstain/count mismatch hoặc control tương đương/vượt primary thì ghi `inconclusive_incremental_evidence`.
- Negative/inconclusive result phải được giữ. Không thay rule, weight hoặc coverage sau khi xem test.
- AMLNet chỉ được gọi là saturation khi score-only đã mạnh, residual gain không ổn định, hai shuffled controls hợp lệ không tạo material positive gain, có tối thiểu 20 positive alerts, 20 matched pairs, 200 valid bootstrap replicates và data-quality/lineage/stability checks đều qua. TransXion không mang nhãn saturation. Saturation không chứng minh production performance.

Stress-result naming convention:

```text
table_stress_<dataset>_<analysis>_<coverage>.csv
figure_stress_<dataset>_<analysis>_<description>.png
```

Ví dụ:

```text
table_stress_transxion_coverage_primary.csv
table_stress_amlnet_matched_risk_25pct.csv
figure_stress_synthesis_rule_vs_score_only.png
```
