# Hướng dẫn dữ liệu

Repository không lưu dữ liệu gốc, dữ liệu đã xử lý hoặc model artifacts. Các file này thường lớn và có điều khoản sử dụng riêng.

## 1. Cấu trúc local đề xuất

```text
data/
├── raw/
│   ├── ieee-cis/
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   ├── baf/
│   │   └── Base.csv
│   ├── transxion/
│   │   ├── tx.csv
│   │   ├── person.csv       # optional profile enrichment
│   │   └── merchant.csv     # optional profile enrichment
│   └── amlnet/
│       └── AMLNet_August 2025.csv
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

## 4. TransXion stress dataset

Nguồn chính thức:

- Paper revision v2: <https://arxiv.org/html/2604.17420v2>
- Source repository: <https://github.com/chaos-max/TransXion>

`v2` trong tên notebook là revision v2 của paper arXiv, **không phải** một dataset release/tag `v2` riêng. Full run dùng canonical file sau:

```text
file: tx.csv
source_repository_commit: 53932595c37c23b9f55ea5ddf5984e4d57b88369
sha256: d6c345f07a8d8e26123dba5fe4f6572ef198e04fb94f9b53facb3fef6d197a35
```

Target: `Is Laundering`.

Biến thời gian: `Timestamp`.

Expected raw schema của `tx.csv`:

```text
Timestamp, From Bank, From Account, To Bank, To Account,
Amount Received, Receiving Currency, Amount Paid, Payment Currency,
Payment Format, Is Laundering
```

Một số tài liệu/snapshot của official repository dùng trùng header `Account` cho phía gửi
và nhận. Khi `pandas.read_csv` chuẩn hóa dạng đó thành `Account`/`Account.1`, loader vẫn ánh
xạ đúng sang `sender_account_id`/`receiver_account_id`; dạng explicit
`From Account`/`To Account` của file checksum-pinned cũng được hỗ trợ. Raw schema thực tế
được ghi vào manifest và identity của full run luôn do checksum quyết định.

`person.csv` và `merchant.csv` là profile tables tùy chọn. Join phải dùng composite bank-account identity thay vì chỉ dùng account number; pipeline phải báo duplicate keys, join cardinality và join coverage. Raw entity IDs không được đưa trực tiếp vào predictor. Chúng chỉ có thể được dùng để xây causal history features, trong đó giao dịch hiện tại bị loại bằng `shift` hoặc cơ chế tương đương.

Giá trị tham chiếu từ nguồn là 3.029.170 rows, 4.641 positive rows, 365 ngày và 47.526 tài khoản. Full run phải tính lại các giá trị này và dừng nếu checksum/schema không khớp. Hiện không có official Kaggle mirror được project xác minh; nếu upload file làm private Kaggle Dataset, vẫn phải giữ exact checksum và ghi official source commit.

Counterfactual candidate generation chỉ perturb raw `amount_paid` hoặc `amount_received`; sau mỗi intervention, pipeline bắt buộc tính lại `amount_paid_log`, `amount_received_log` và `amount_relative_difference` theo `derived_feature_contract`. Các history feature không phụ thuộc trực tiếp vào amount hiện tại được giữ nguyên. Contract/formula/output được ghi trong candidate provenance; không thực thi code từ cấu hình.

## 5. AMLNet v1.0 stress dataset

Nguồn chính thức: Zenodo record <https://zenodo.org/records/16736515>.

Identity contract:

```text
version: AMLNet v1.0
doi: 10.5281/zenodo.16736515
file: AMLNet_August 2025.csv
md5: 7668fc7d74c787e07546ce85c6f790b9
license: CC BY-NC 4.0
```

Target: `isMoneyLaundering`.

Expected raw schema:

```text
step, type, amount, category, nameOrig, nameDest,
oldbalanceOrg, newbalanceOrig, isFraud, isMoneyLaundering,
laundering_typology, metadata, fraud_probability,
hour, day_of_week, day_of_month, month
```

Source-version note: Zenodo record/DOI được khóa là v1/v1.0, trong khi README preview của nguồn có thể hiển thị `VERSION 2.0`. Repository không suy diễn dataset version từ preview; exact DOI, filename và MD5 nêu trên là identity contract, và discrepancy phải được lưu trong `data_manifest.json`.

Row order và `step` không được coi là chronological key. Timestamp được trích an toàn từ chuỗi `metadata`, thường có dạng `datetime.datetime(...)`:

- Chỉ dùng parser giới hạn/regular expression đã kiểm thử.
- Không dùng `eval`, `exec` hoặc thực thi metadata.
- Full run yêu cầu timestamp parsing đầy đủ, stable-sort theo timestamp và original row order, sau đó mới split 60/20/20.

Leakage denylist tối thiểu gồm target, `isFraud`, `laundering_typology`, `fraud_probability`, raw `metadata`, `step` và raw account IDs. `isFraud` là auxiliary outcome, không phải causal AML feature. Feature thời gian được tính lại từ parsed timestamp thay vì tin mù quáng các reported time columns.

`newbalanceOrig`/`sender_balance_after` là trạng thái hậu giao dịch và bị loại. Pipeline cũng cấm proxy `balance_depletion_fraction` tính từ before/after. Rule theo tỷ lệ số tiền chỉ được dùng `amount_to_sender_balance_fraction = amount / oldbalanceOrg`, tức số tiền giao dịch và số dư trước giao dịch; quyết định này phải được audit như một giả định “available at scoring time”.

Trong counterfactual generation, chỉ `amount` và `sender_balance_before` được perturb; `amount_log` và `amount_to_sender_balance_fraction` luôn được tính lại theo YAML contract trước khi chấm điểm. Derived ratio không được perturb như một raw field độc lập.

Giá trị tham chiếu là 1.090.173 rows, 1.745 AML positives và khoảng 195 ngày. Full run phải audit lại. AMLNet mang giấy phép CC BY-NC 4.0; không được dùng artifacts/raw data vượt quá phạm vi phi thương mại mà giấy phép cho phép.

## 6. Kaggle setup

Gắn dataset tương ứng vào notebook bằng nút **Add Input**. Các path phổ biến:

```text
/kaggle/input/ieee-fraud-detection/train_transaction.csv
/kaggle/input/ieee-fraud-detection/train_identity.csv
```

và:

```text
/kaggle/input/<baf-dataset-name>/Base.csv
```

Stress notebooks cần gắn input chứa exact file tương ứng:

```text
/kaggle/input/<transxion-input>/tx.csv
/kaggle/input/<amlnet-input>/AMLNet_August 2025.csv
```

Notebook 09 chỉ đọc TransXion; Notebook 10 chỉ đọc AMLNet. Notebook 11 không cần raw datasets, nhưng cần gắn output package của cả Notebook 09 và 10. Tên thư mục Kaggle có thể thay đổi; loader tìm exact filename đệ quy dưới `/kaggle/input`, sau đó checksum preflight mới quyết định input có hợp lệ hay không. Các history features chỉ dùng timestamp nhỏ hơn nghiêm ngặt; hai rows trùng timestamp không được dùng làm lịch sử của nhau dù source order khác nhau.

Notebook exploration đọc lần lượt IEEE-CIS và BAF theo hai phần riêng. Các notebook model/rule khóa vào dataset ghi trong tên file. Nếu không tìm thấy file benchmark, notebook chỉ được dùng synthetic fallback để smoke test. Nhãn nguồn dữ liệu là:

```text
Notebook 01, file benchmark: data_source = official_benchmark_file
Notebook 01, fallback:       data_source = synthetic_fallback
Notebook 02-07, file benchmark: data_source = real
Notebook 02-07, fallback:       data_source = synthetic
```

Mọi kết quả fallback đều không được dùng cho claim khóa luận.

## 7. Dataset version record

Tên file, số dòng và `data_source` không đủ để xác định duy nhất một dataset version. Trước mỗi full run, ghi thủ công các field sau trong experiment log hoặc Kaggle notebook version notes:

```text
dataset_reference: <Kaggle owner/slug hoặc source identifier>
dataset_version: <Kaggle input version hoặc release identifier>
files_used: <train_transaction.csv, train_identity.csv hoặc Base.csv>
retrieved_or_attached_at: <UTC date/time>
checksum_algorithm: <sha256 hoặc md5>
checksum: <giá trị thực tế>
```

Nếu giao diện không cung cấp version, ghi rõ `dataset_version: unavailable` thay vì ngầm coi filename là version.

## 8. Data integrity checklist

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
- Với TransXion, exact `tx.csv` SHA-256 và source repository commit khớp identity contract; không gọi file là dataset release `v2`.
- Với AMLNet, exact file MD5 và Zenodo DOI khớp; source-version discrepancy được ghi trong manifest.
- AMLNet timestamp được parse an toàn, tỷ lệ parse đạt 100% trong full run và row order được stable-sort trước split.
- Stress split là chronological 60/20/20 và validation roles không chồng lấn.
- Raw IDs/target proxies/post-outcome fields không xuất hiện trong predictor feature schema.
- Mọi history feature chỉ dùng giao dịch quá khứ và không tính cả row hiện tại.

## 9. Quyền sử dụng

Người dùng chịu trách nhiệm chấp nhận điều khoản của từng dataset. Không đưa dữ liệu gốc vào GitHub, file nộp khóa luận hoặc artifact công khai nếu giấy phép không cho phép.

Riêng AMLNet v1.0 được công bố theo CC BY-NC 4.0. Với TransXion, không giả định quyền tái phân phối chỉ vì file có thể tải từ official repository; người chạy phải kiểm tra điều khoản nguồn và giữ raw input private khi cần.
