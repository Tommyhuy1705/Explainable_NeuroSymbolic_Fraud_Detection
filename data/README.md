# Hướng dẫn dữ liệu

Repository không lưu dữ liệu gốc, dữ liệu đã xử lý hoặc model artifacts. Các file này thường lớn và có điều khoản sử dụng riêng.

## 1. Cấu trúc local đề xuất

```text
data/
├── raw/
│   ├── ieee-cis/
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   └── baf/
│       └── Base.csv
└── processed/             # generated, không commit
```

`data/raw/` và `data/processed/` đã được khai báo trong `.gitignore`.

## 2. IEEE-CIS Fraud Detection

Nguồn: Kaggle competition **IEEE-CIS Fraud Detection**.

File cần thiết:

- `train_transaction.csv`: bắt buộc.
- `train_identity.csv`: tùy chọn nhưng được khuyến nghị.

Target: `isFraud`.

ID ghép bảng: `TransactionID`.

Biến thời gian: `TransactionDT`.

Loader tạo thêm:

- `transaction_hour`.
- `transaction_day`.
- `transaction_amount_log`.

`transaction_hour = floor(TransactionDT / 3600) mod 24` chỉ là cyclic phase suy ra từ mốc `TransactionDT`. Do clock origin/timezone gốc không được xác định trong pipeline, feature này không được diễn giải như giờ địa phương đã biết.

## 3. Bank Account Fraud Dataset Suite

Nguồn: Bank Account Fraud Dataset Suite, NeurIPS 2022. BAF là privacy-preserving synthetic benchmark, không phải dữ liệu giao dịch thực của một tổ chức tài chính.

File mặc định: `Base.csv`.

Target: `fraud_bool`.

Biến thời gian: `month`.

Protocol chính thức dùng train tháng `0-4`, validation tháng `5`, test tháng `6-7`. Các nhóm tháng phải hoàn toàn không giao nhau. `month` chỉ dùng làm split key và bị loại khỏi predictor features.

Một số feature BAF dùng `-1` để mã hóa giá trị không có hoặc không áp dụng. Protocol hiện tại:

- Kiểm đếm native `NaN` và sentinel `-1` thành hai loại missingness riêng trong EDA.
- Giữ nguyên `-1` như giá trị mã hóa của benchmark trong preprocessing; loader không tự động đổi `-1` thành `NaN`.
- Ghi rõ quyết định này trong báo cáo và không suy luận "không có missingness" chỉ vì native `NaN` bằng 0.

Trong metadata, `data_source = real` hoặc `official_benchmark_file` chỉ có nghĩa pipeline đã đọc file `Base.csv` chính thức thay vì synthetic fallback nội bộ; các nhãn này không thay đổi bản chất synthetic của BAF.

Nếu Kaggle dataset dùng tên thư mục khác, không cần đổi code. Loader tìm file theo tên bên dưới `/kaggle/input/`.

## 4. Kaggle setup

Gắn dataset tương ứng vào notebook bằng nút **Add Input**. Các path phổ biến:

```text
/kaggle/input/ieee-fraud-detection/train_transaction.csv
/kaggle/input/ieee-fraud-detection/train_identity.csv
```

và:

```text
/kaggle/input/<baf-dataset-name>/Base.csv
```

Notebook exploration đọc lần lượt IEEE-CIS và BAF theo hai phần riêng. Các notebook model/rule khóa vào dataset ghi trong tên file. Nếu không tìm thấy file benchmark, notebook chỉ được dùng synthetic fallback để smoke test. Nhãn nguồn dữ liệu là:

```text
Notebook 01, file benchmark: data_source = official_benchmark_file
Notebook 01, fallback:       data_source = synthetic_fallback
Notebook 02-07, file benchmark: data_source = real
Notebook 02-07, fallback:       data_source = synthetic
```

Mọi kết quả fallback đều không được dùng cho claim khóa luận.

## 5. Dataset version record

Tên file, số dòng và `data_source` không đủ để xác định duy nhất một dataset version. Trước mỗi full run, ghi thủ công các field sau trong experiment log hoặc Kaggle notebook version notes:

```text
dataset_reference: <Kaggle owner/slug hoặc source identifier>
dataset_version: <Kaggle input version hoặc release identifier>
files_used: <train_transaction.csv, train_identity.csv hoặc Base.csv>
retrieved_or_attached_at: <UTC date/time>
```

Nếu giao diện không cung cấp version, ghi rõ `dataset_version: unavailable` thay vì ngầm coi filename là version.

## 6. Data integrity checklist

Trước khi chạy kết quả chính thức, kiểm tra:

- Target chỉ chứa `0/1`.
- ID không bị trùng ngoài dự kiến.
- Tỷ lệ gian lận theo từng split.
- Thứ tự thời gian của train, validation và test.
- Với BAF, `overlap_groups` phải rỗng cho cả ba split.
- Số dòng trước và sau merge.
- Tỷ lệ IEEE-CIS identity join coverage khi có `train_identity.csv`.
- Native `NaN` và sentinel `-1` của BAF được báo riêng.
- Feature không chứa target proxy hoặc thông tin phát sinh sau quyết định.
- Không fit imputer, scaler, category mapping hoặc rule threshold trên validation/test.
- Numeric rule missing values trên validation/test dùng lại training median đã xuất trong `fitted_missing_value`; không recompute median theo split.
- `dataset_reference` và `dataset_version` đã được ghi trong experiment log/version notes.

## 7. Quyền sử dụng

Người dùng chịu trách nhiệm chấp nhận điều khoản của từng dataset. Không đưa dữ liệu gốc vào GitHub, file nộp khóa luận hoặc artifact công khai nếu giấy phép không cho phép.
