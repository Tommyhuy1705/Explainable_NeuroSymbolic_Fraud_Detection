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
