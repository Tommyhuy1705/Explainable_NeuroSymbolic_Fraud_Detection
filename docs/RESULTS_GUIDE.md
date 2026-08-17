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
- Dataset version.
- Seed.
- Run metadata.
- Frozen artifact checksum và config hash khi kết quả phụ thuộc predictor.

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

## Synthetic results

Không chuyển synthetic smoke outputs vào `results/tables` hoặc `results/figures`. Synthetic results chỉ xác nhận code chạy được và không hỗ trợ kết luận nghiên cứu.
