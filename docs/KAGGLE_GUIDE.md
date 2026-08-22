# Hướng dẫn chạy trên Kaggle

## 1. Chuẩn bị notebook

Upload notebook từ thư mục `notebooks/` hoặc tạo Kaggle Notebook rồi import file `.ipynb`.

Các file trong `notebooks/` được sinh từ `scripts/generate_notebooks.py`. Khi thay đổi source cell, sửa generator rồi sinh lại toàn bộ notebook; không duy trì một bản sửa tay chỉ tồn tại trong `.ipynb`. Trước khi upload, chạy notebook source-sync check để bảo đảm file đã sinh đúng generator.

Project code có thể được cung cấp theo một trong ba cách:

1. Clone public repository vào `/kaggle/working` khi Internet được bật; đây là cách mặc định của notebook hiện tại.
2. Upload repository thành Kaggle Dataset và gắn bằng **Add Input** khi cần chạy offline hoặc khóa một snapshot source.
3. Upload source cùng notebook.

Setup cell của Notebook 01 và 04-08 tìm một source tree có cả `configs/` và `src/` trong current directory, parent directories, `/kaggle/working` và `/kaggle/input` trước. Chỉ khi không tìm thấy source tree nào và đang ở Kaggle, cell mới clone public repository vào `/kaggle/working`. Vì vậy, source snapshot đã gắn bằng **Add Input** có thể chạy offline mà không bị buộc clone.

Notebook 02 và 03 là hai executed benchmark artifacts đang được giữ nguyên từng byte để cell source tiếp tục khớp đúng output Kaggle đã chạy. Hai file này vì thế vẫn giữ setup của lần chạy gốc: trên Kaggle, chúng clone public repository nếu `/kaggle/working/Explainable_NeuroSymbolic_Fraud_Detection` chưa tồn tại và luôn khóa full mode. Không sửa setup của 02/03 mà giữ lại output cũ. Nếu sau này cần chạy lại 02/03 bằng source snapshot offline, phải tạo một notebook version mới, chạy lại toàn bộ benchmark và xuất frozen artifacts mới tương ứng.

Clone qua HTTPS chỉ hoạt động không cần credentials khi repository là public. Nếu repository private, Kaggle cần GitHub token/secret hoặc một source dataset snapshot; không ghi token trực tiếp vào notebook. Khi clone source mới nhất, lưu `git_commit` và audit-pipeline fingerprint trong lineage để các artifact downstream truy vết được đúng phiên bản code.

Trước khi upload notebook, chạy đúng hai lệnh source-sync sau từ project root:

```bash
python scripts/generate_notebooks.py
pytest -q tests/test_notebook_generation.py
```

## 2. Gắn dữ liệu

`01_Data_Exploration.ipynb` đọc lần lượt IEEE-CIS và BAF trong cùng một lần chạy. Notebook model/rule được tách theo dataset: file có `IEEE_CIS` chỉ đọc IEEE-CIS, file có `BAF` chỉ đọc BAF.

IEEE-CIS cần Kaggle competition dataset IEEE-CIS Fraud Detection. BAF cần dataset chứa `Base.csv`.

Sau khi Add Input, chạy cell data discovery và kiểm tra trạng thái tương ứng:

```text
Notebook 01: data_source = official_benchmark_file
Notebook 02-07: data_source = real
```

Nếu hiển thị `synthetic_fallback` hoặc `synthetic`, notebook chỉ đang smoke test. Với BAF, `official_benchmark_file`/`real` nghĩa là đã đọc file benchmark thay vì fallback nội bộ; BAF vẫn là privacy-preserving synthetic benchmark.

EDA phải phân biệt hai loại missingness của BAF:

- Native `NaN` do parser nhận diện.
- Sentinel `-1` biểu diễn giá trị không có/không áp dụng ở một số feature.

Do đó, bảng native missingness rỗng không đủ để kết luận mọi feature BAF đều đầy đủ về mặt ngữ nghĩa.

Protocol giữ nguyên sentinel `-1` như giá trị mã hóa do BAF cung cấp; preprocessing không tự động đổi `-1` thành `NaN`. Trước full run, ghi `dataset_reference`, `dataset_version`, danh sách file và thời điểm gắn input trong Kaggle version notes hoặc experiment log. Nếu Kaggle không hiển thị version, ghi `dataset_version: unavailable`.

## 3. QUICK_RUN

Notebook mặc định chạy full protocol:

```python
QUICK_RUN = False
ALLOW_SYNTHETIC_FALLBACK = False
```

Setup của Notebook 01 và 04-08 chỉ gán giá trị mặc định nếu biến môi trường chưa tồn tại; nó không ghi đè giá trị đã được cấu hình trước khi cell đầu tiên chạy. Chỉ bật smoke mode bằng biến môi trường khi kiểm tra kỹ thuật:

```text
THESIS_QUICK_RUN=1
THESIS_SYNTHETIC_FALLBACK=1
```

Hai biến này phải được đặt trước setup cell; không sửa tay notebook chính thức chỉ để bật smoke mode. Trên PowerShell local có thể dùng `$env:THESIS_QUICK_RUN="1"` và `$env:THESIS_SYNTHETIC_FALLBACK="1"` trước khi mở kernel. Setup Kaggle đã khóa của Notebook 02/03 không nhận override quick mode; đây là chủ ý để bảo toàn source/output của hai full benchmark đã hoàn tất. Kết quả khóa luận luôn dùng full mode.

Smoke mode:

- Giới hạn số dòng.
- Giảm epoch.
- Giảm số tree estimators.
- Phù hợp để kiểm tra pipeline.

Ba epoch quan sát được trong smoke mode là giới hạn kiểm tra khả năng thực thi, không phải cấu hình huấn luyện chính thức. Full mode dùng tối đa 100 epoch cho MLP, 150 epoch cho TabularResNet, early stopping theo validation PR-AUC và ba seed độc lập.

## 4. Chọn nơi chạy

`01_Data_Exploration.ipynb` chỉ cần CPU và có thể chạy local. Các notebook tree-only và rule analysis cũng có thể chạy CPU. Dùng Kaggle GPU cho notebook model benchmarks khi huấn luyện MLP và TabularResNet trên dữ liệu đầy đủ.

## 5. Accelerator

Chọn GPU Tesla T4 cho notebook neural benchmarks. Không dùng P100 nếu PyTorch hiện tại không còn kernel `sm_60`. Tree-only và rule analysis có thể chạy CPU.

Code tự chọn CUDA khi `torch.cuda.is_available()`.

## 6. Output

Artifacts được ghi vào:

```text
/kaggle/working/thesis_outputs/<notebook_name>/
```

Sau khi run xong, tải về:

- CSV tables.
- Figures.
- `run_metadata.json`.
- `predictions.npz` nếu cần tái phân tích.
- `frozen_reference_artifact.npz` và `frozen_reference_manifest.json`.
- `calibration_comparison_all_seeds.csv` và `paired_bootstrap_model_differences.csv`.
- `upstream_lineage.json` của Notebook 04-07 và toàn bộ output files được lineage khai báo.

Không coi cell output là artifact duy nhất.

## 7. Artifact dependency giữa notebooks

Notebook 05 và 06 cần output của Notebook 02. Notebook 07 cần output của Notebook 03. Dùng **Add Input -> Notebook Output** để gắn benchmark output tương ứng; không upload lại source repo.

Notebook 08 cần outputs của Notebook 02-07. Notebook 01 là EDA/data-audit artifact cần được chạy lại sau thay đổi EDA, nhưng không phải input dependency của Notebook 08.

Notebook 04-07 ghi lineage với git commit, audit-pipeline/config fingerprint và SHA-256 của các output đã khai báo. Notebook 08 kiểm tra lineage, source/config fingerprint, frozen-artifact alignment khi áp dụng, sự tồn tại và checksum của output trước khi tổng hợp. Notebook 04 không phụ thuộc frozen predictor, nhưng output rule analysis vẫn phải qua source/config/output lineage checks.

Notebook 05, 06 và 07 phải có preflight cell được sinh từ generator. Preflight tìm manifest/artifact trong `INPUT_ROOTS`, yêu cầu đúng một artifact cho dataset tương ứng và xác nhận:

- Dataset name và reference seed.
- Full/quick mode khớp notebook hiện tại.
- `data_source` không phải synthetic cho kết quả khóa luận.
- Config hash và artifact checksum hợp lệ.
- Validation/test label alignment khớp dữ liệu đang load.

Nếu preflight không tìm thấy artifact, dùng **Add Input -> Notebook Output** để gắn output của Notebook 02 cho Notebook 05/06 và output của Notebook 03 cho Notebook 07. Không cần upload repository thành Kaggle Dataset khi setup cell đã clone public repository thành công.

Thứ tự chạy đầy đủ từ đầu:

1. Notebook 01.
2. Notebook 02, 03 và 04 có thể chạy song song.
3. Notebook 05, 06 và 07 có thể chạy song song sau khi frozen artifacts đã có.
4. Notebook 08 chạy cuối.

Thứ tự cho vòng sửa rule/metric hiện tại:

1. Giữ output Notebook 02 và 03; không train lại nếu frozen manifest vẫn vượt qua config hash, checksum, mode và label-alignment checks.
2. Sinh lại notebooks từ generator và chạy tests/source-sync checks local.
3. Rerun Notebook 01 để cập nhật EDA, BAF sentinel audit và identity-join summary. Notebook này có thể chạy song song với 04-07 và không chặn Notebook 08.
4. Rerun Notebook 04, 05, 06 và 07. Bốn notebook có thể chạy song song; Notebook 05/06 gắn output 02, Notebook 07 gắn output 03, còn Notebook 04 chỉ cần IEEE-CIS data.
5. Lưu các output version mới của Notebook 01 và 04-07.
6. Gắn outputs Notebook 02-07 vào Notebook 08 và chỉ chạy Notebook 08 sau khi tất cả input audits bắt buộc đều qua.

Tên file `07_BAF_LTN_Generalization.ipynb` được giữ để không phá Kaggle linkage hiện có. Nội dung học thuật của notebook là cross-dataset replication/portability evaluation trên BAF, không phải transfer cùng predictor hoặc cùng rule base từ IEEE-CIS.

## 8. Full-run checklist

- Internet không cần thiết sau khi code và data đã được gắn.
- `QUICK_RUN=False`.
- `ALLOW_SYNTHETIC_FALLBACK=False`.
- Không có synthetic fallback: Notebook 01 phải báo `official_benchmark_file`, Notebook 02-07 phải báo `real`.
- Accelerator đúng với model.
- Chạy **Run All** từ kernel sạch.
- Không chỉnh threshold sau khi xem test.
- Không chọn rule set hoặc activation threshold từ test ablation; Notebook 06 là post-hoc diagnostic.
- Lưu notebook version và output artifacts.
- Ghi lại seed, config và Kaggle environment.
- Ghi `dataset_reference`, `dataset_version`, files used và input attachment time.
- Xác nhận Notebook 04-07 đã xuất lineage và output checksums.
- Xác nhận fitted-condition/case outputs có `fitted_missing_value`; numeric rule median được fit trên train và tái sử dụng trên validation/test.
- Xác nhận bảng kết quả chính thức có `n_seeds = 3` và báo cáo mean ± standard deviation.
- Xác nhận model được chọn bằng validation raw PR-AUC; calibrated metrics và F2 chỉ dùng policy/calibration reporting đã khóa.

## 9. Troubleshooting

### Không tìm thấy project root

Đảm bảo input chứa cả `src/` và `configs/`. Có thể đặt thủ công:

```python
PROJECT_ROOT = Path("/kaggle/input/<project-dataset>/Explainable_NeuroSymbolic_Fraud_Detection")
```

Nếu dùng clone workflow, kiểm tra Internet đang bật và repository public, sau đó xác nhận thư mục sau tồn tại:

```python
Path("/kaggle/working/Explainable_NeuroSymbolic_Fraud_Detection/src").is_dir()
```

### Không tìm thấy dataset

Kiểm tra tên file bằng:

```python
list(Path("/kaggle/input").glob("**/train_transaction.csv"))
```

hoặc:

```python
list(Path("/kaggle/input").glob("**/Base.csv"))
```

### Không tìm thấy frozen artifact

Liệt kê artifact đã gắn:

```python
list(Path("/kaggle/input").glob("**/frozen_reference_manifest.json"))
list(Path("/kaggle/input").glob("**/frozen_reference_artifact.npz"))
```

Nếu có nhiều output cùng dataset, bỏ các input version cũ để preflight chỉ tìm đúng một artifact hợp lệ. Không bỏ qua checksum hoặc label-alignment check để ép notebook chạy.

### Out of memory

- Giảm `max_rows`.
- Giảm `max_features` trong config.
- Giảm batch size.
- Chạy từng model riêng.
- Không giữ nhiều DataFrame copy trong notebook.
