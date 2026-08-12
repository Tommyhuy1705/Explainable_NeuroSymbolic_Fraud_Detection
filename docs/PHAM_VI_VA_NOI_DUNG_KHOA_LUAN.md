# Phạm vi và nội dung khóa luận tốt nghiệp

## Tên đề tài

**Nghiên cứu phương pháp học máy có khả năng giải thích kết hợp tri thức logic cho phát hiện gian lận tài chính**

**An Explainable Neuro-Symbolic Learning Framework for Financial Fraud Detection**

## Bài toán trung tâm

Khóa luận trả lời câu hỏi:

> Làm thế nào xây dựng và đánh giá một hệ thống phát hiện gian lận tài chính vừa có hiệu quả dự đoán, vừa cung cấp được bằng chứng giải thích dựa trên tri thức miền?

## Nội dung dự kiến

### Chương 1 — Giới thiệu

- Bối cảnh phát hiện gian lận.
- Dữ liệu mất cân bằng và yêu cầu giải thích.
- Mục tiêu, câu hỏi nghiên cứu và phạm vi.
- Đóng góp và cấu trúc báo cáo.

### Chương 2 — Cơ sở lý thuyết

- Fraud detection và imbalanced learning.
- Machine learning cho dữ liệu bảng.
- Explainable AI.
- Fuzzy logic, neuro-symbolic AI và LTN.
- Nghiên cứu liên quan.

### Chương 3 — Phương pháp

- Pipeline dữ liệu.
- MLP, TabularResNet và tree models.
- Fuzzy predicates và fraud rules.
- Rule-based explanation.
- Leakage-safe evaluation protocol.

### Chương 4 — Thực nghiệm

- Data exploration và quality checks.
- Predictive model benchmarks.
- Rule quality analysis.
- Explanation evaluation.
- Rule ablation.
- BAF generalization nếu đủ điều kiện.

### Chương 5 — Thảo luận

- So sánh model performance.
- Vai trò và hạn chế của từng rule.
- Trade-off giữa prediction và explanation.
- Phân tích TP, FP, FN.
- Threats to validity.

### Chương 6 — Kết luận

- Trả lời câu hỏi nghiên cứu.
- Tóm tắt bằng chứng.
- Giới hạn claim.
- Hướng phát triển.

## Kết quả cần có

1. Pipeline có thể tái lập từ dữ liệu đến dự đoán và giải thích.
2. Bảng so sánh ba nhóm predictor.
3. Tập fuzzy rules có ý nghĩa nghiệp vụ.
4. Bảng coverage, fraud precision, lift và rule AUC.
5. Bảng explanation coverage, sparsity, consistency và contradiction.
6. Ablation theo rule group.
7. Case studies cho TP, FP và FN.
8. Mã nguồn, config, notebook và run metadata.

## Bằng chứng khóa luận có thể cung cấp

Trong phạm vi dataset và protocol đã chạy, khóa luận có thể chứng minh bằng thực nghiệm rằng:

- Các model tabular phù hợp xử lý fraud imbalance tốt hơn baseline đơn giản.
- Tri thức miền có thể biểu diễn thành fuzzy predicates và luật tính toán được.
- Một số luật xác định vùng dữ liệu có fraud rate cao hơn base rate.
- Rule evidence hỗ trợ kiểm tra một phần model alerts.
- Không phải mọi cảnh báo đều có đủ logical evidence.
- Prediction quality và explanation quality cần được đánh giá riêng.
- Validation-only protocol giảm nguy cơ data leakage và test overfitting.

Claim trung tâm:

> Trong phạm vi dữ liệu và giao thức thực nghiệm được sử dụng, việc kết hợp mô hình dự đoán dựa trên dữ liệu với tri thức miền biểu diễn bằng logic có thể tạo ra một hệ thống phát hiện gian lận vừa duy trì hiệu quả dự đoán, vừa cung cấp bằng chứng giải thích có thể kiểm tra cho một phần cảnh báo.

## Những điều không tuyên bố

- LTN luôn tốt hơn tree models.
- Logic luôn cải thiện prediction performance.
- Rule explanation là causal explanation.
- Hệ thống sẵn sàng triển khai trong tổ chức tài chính.
- Kết quả khái quát cho mọi dataset và mọi môi trường.

## Tiêu chí hoàn thành

- Pipeline chạy top-to-bottom.
- Ít nhất ba model được so sánh công bằng.
- Rules có ý nghĩa và được fit train-only.
- Có metrics cho prediction và explanation.
- Có ablation và case studies.
- Kết luận trả lời trực tiếp research questions.
- Mọi claim được giới hạn đúng theo bằng chứng.
