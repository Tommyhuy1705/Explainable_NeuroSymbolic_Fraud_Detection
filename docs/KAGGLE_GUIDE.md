# Hướng dẫn chạy trên Kaggle

## 1. Chuẩn bị notebook

Upload notebook từ thư mục `notebooks/` hoặc tạo Kaggle Notebook rồi import file `.ipynb`.

Project code có thể được cung cấp theo một trong hai cách:

1. Upload repository thành Kaggle Dataset và gắn bằng **Add Input**.
2. Clone repository vào `/kaggle/working` khi Internet được bật.
3. Upload source cùng notebook.

Setup cell tự tìm `configs/` và `src/` trong current directory, parent directories, `/kaggle/working` và `/kaggle/input`.

## 2. Gắn dữ liệu

IEEE notebook cần Kaggle competition dataset IEEE-CIS Fraud Detection. BAF notebook cần dataset chứa `Base.csv`.

Sau khi Add Input, chạy cell data discovery và kiểm tra:

```text
data_source = real
```

Nếu hiển thị `synthetic`, notebook chỉ đang smoke test.

## 3. QUICK_RUN

Notebook mặc định:

```python
QUICK_RUN = True
```

Chế độ này:

- Giới hạn số dòng.
- Giảm epoch.
- Giảm số tree estimators.
- Phù hợp để kiểm tra pipeline.

Để tạo kết quả chính thức:

```python
QUICK_RUN = False
ALLOW_SYNTHETIC_FALLBACK = False
```

## 4. Accelerator

Chọn GPU T4/P100 cho notebook neural benchmarks. Tree-only và rule analysis có thể chạy CPU.

Code tự chọn CUDA khi `torch.cuda.is_available()`.

## 5. Output

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

## 6. Full-run checklist

- Internet không cần thiết sau khi code và data đã được gắn.
- `QUICK_RUN=False`.
- `ALLOW_SYNTHETIC_FALLBACK=False`.
- Data source là `real`.
- Accelerator đúng với model.
- Chạy **Run All** từ kernel sạch.
- Không chỉnh threshold sau khi xem test.
- Lưu notebook version và output artifacts.
- Ghi lại seed, config và Kaggle environment.

## 7. Troubleshooting

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

