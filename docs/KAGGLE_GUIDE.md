# Hướng dẫn chạy trên Kaggle

## 1. Chuẩn bị notebook

Upload notebook từ thư mục `notebooks/` hoặc tạo Kaggle Notebook rồi import file `.ipynb`.

Project code có thể được cung cấp theo một trong hai cách:

1. Upload repository thành Kaggle Dataset và gắn bằng **Add Input**.
2. Clone repository vào `/kaggle/working` khi Internet được bật.
3. Upload source cùng notebook.

Setup cell tự tìm `configs/` và `src/` trong current directory, parent directories, `/kaggle/working` và `/kaggle/input`.

## 2. Gắn dữ liệu

`01_Data_Exploration.ipynb` đọc lần lượt IEEE-CIS và BAF trong cùng một lần chạy. Notebook model/rule được tách theo dataset: file có `IEEE_CIS` chỉ đọc IEEE-CIS, file có `BAF` chỉ đọc BAF.

IEEE-CIS cần Kaggle competition dataset IEEE-CIS Fraud Detection. BAF cần dataset chứa `Base.csv`.

Sau khi Add Input, chạy cell data discovery và kiểm tra:

```text
data_source = real
```

Nếu hiển thị `synthetic`, notebook chỉ đang smoke test.

## 3. QUICK_RUN

Notebook mặc định chạy full protocol:

```python
QUICK_RUN = False
ALLOW_SYNTHETIC_FALLBACK = False
```

Chỉ bật smoke mode bằng biến môi trường khi kiểm tra kỹ thuật:

```python
THESIS_QUICK_RUN=1
THESIS_SYNTHETIC_FALLBACK=1
```

Smoke mode:

- Giới hạn số dòng.
- Giảm epoch.
- Giảm số tree estimators.
- Phù hợp để kiểm tra pipeline.

Ba epoch quan sát được trong smoke mode là giới hạn kiểm tra khả năng thực thi, không phải cấu hình huấn luyện chính thức. Full mode dùng tối đa 100 epoch cho MLP, 150 epoch cho TabularResNet, early stopping theo validation PR-AUC và ba seed độc lập.

## 4. Chọn nơi chạy

`01_Data_Exploration.ipynb` chỉ cần CPU và có thể chạy local. Các notebook tree-only và rule analysis cũng có thể chạy CPU. Dùng Kaggle GPU cho notebook model benchmarks khi huấn luyện MLP và TabularResNet trên dữ liệu đầy đủ.

## 5. Accelerator

Chọn GPU T4/P100 cho notebook neural benchmarks. Tree-only và rule analysis có thể chạy CPU.

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

Không coi cell output là artifact duy nhất.

## 7. Full-run checklist

- Internet không cần thiết sau khi code và data đã được gắn.
- `QUICK_RUN=False`.
- `ALLOW_SYNTHETIC_FALLBACK=False`.
- Data source là `real`.
- Accelerator đúng với model.
- Chạy **Run All** từ kernel sạch.
- Không chỉnh threshold sau khi xem test.
- Lưu notebook version và output artifacts.
- Ghi lại seed, config và Kaggle environment.
- Xác nhận bảng kết quả chính thức có `n_seeds = 3` và báo cáo mean ± standard deviation.

## 8. Troubleshooting

### Không tìm thấy project root

Đảm bảo input chứa cả `src/` và `configs/`. Có thể đặt thủ công:

```python
PROJECT_ROOT = Path("/kaggle/input/<project-dataset>/Explainable_NeuroSymbolic_Fraud_Detection")
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

### Out of memory

- Giảm `max_rows`.
- Giảm `max_features` trong config.
- Giảm batch size.
- Chạy từng model riêng.
- Không giữ nhiều DataFrame copy trong notebook.
