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

## 3. Bank Account Fraud Dataset Suite

Nguồn: Bank Account Fraud Dataset Suite, NeurIPS 2022.

File mặc định: `Base.csv`.

Target: `fraud_bool`.

Biến thời gian: `month`.

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

Notebook exploration đọc lần lượt IEEE-CIS và BAF theo hai phần riêng. Các notebook model/rule khóa vào dataset ghi trong tên file. Nếu không tìm thấy dữ liệu thật, notebook có thể dùng synthetic fallback để kiểm tra code. Kết quả fallback sẽ hiển thị `data_source = synthetic` và không được dùng trong luận văn.

## 5. Data integrity checklist

Trước khi chạy kết quả chính thức, kiểm tra:

- Target chỉ chứa `0/1`.
- ID không bị trùng ngoài dự kiến.
- Tỷ lệ gian lận theo từng split.
- Thứ tự thời gian của train, validation và test.
- Số dòng trước và sau merge.
- Feature không chứa target proxy hoặc thông tin phát sinh sau quyết định.
- Không fit imputer, scaler, category mapping hoặc rule threshold trên validation/test.

## 6. Quyền sử dụng

Người dùng chịu trách nhiệm chấp nhận điều khoản của từng dataset. Không đưa dữ liệu gốc vào GitHub, file nộp khóa luận hoặc artifact công khai nếu giấy phép không cho phép.
