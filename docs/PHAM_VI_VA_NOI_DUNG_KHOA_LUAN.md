# Phạm vi và nội dung khóa luận tốt nghiệp

## Tên đề tài

Tên tiếng Việt:

**Nghiên cứu phương pháp học máy có khả năng giải thích kết hợp tri thức logic cho phát hiện gian lận tài chính**

Tên tiếng Anh và project title thống nhất trong repository:

**An Explainable Neuro-Symbolic Audit Framework for Financial Fraud Detection**

## Bài toán trung tâm

Khóa luận trả lời câu hỏi:

> Làm thế nào xây dựng và đánh giá một framework lai trong đó predictor học máy được khóa sau huấn luyện, còn lớp fuzzy-rule/LTN-inspired cung cấp bằng chứng giải thích chọn lọc để hỗ trợ audit cảnh báo gian lận?

Khóa luận không phát triển LTN end-to-end và không dùng logic để huấn luyện lại predictor. Phạm vi trung tâm là đánh giá tách biệt chất lượng dự đoán, chất lượng rule evidence và trade-off coverage-precision trong cùng một protocol hạn chế leakage.

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
- Quy trình chọn và khóa reference predictor, calibration và decision threshold.
- Fuzzy predicates, fraud rules, train-fitted numeric missing-value medians và tensor-logic semantics LTN-inspired.
- Selective rule-based explanation/audit layer.
- Leakage-safe evaluation protocol.

### Chương 4 — Thực nghiệm

- Data exploration và quality checks.
- Predictive model benchmarks.
- Rule quality analysis.
- Explanation evaluation.
- Post-hoc rule ablation và activation-threshold sensitivity.
- Cross-dataset replication/portability evaluation trên BAF.

### Chương 5 — Thảo luận

- So sánh model performance.
- Vai trò và hạn chế của từng rule.
- Trade-off coverage-precision của explanation chọn lọc, độ ngắn gọn của explanation và mối quan hệ với prediction quality được báo cáo tách biệt.
- Phân tích TP, FP, FN và TN.
- Phân biệt association/enrichment với causal explanation và model faithfulness.
- Phân biệt replication trên benchmark thứ hai với rule/model transfer.
- Threats to validity.

### Chương 6 — Kết luận

- Trả lời câu hỏi nghiên cứu.
- Tóm tắt bằng chứng.
- Giới hạn claim.
- Hướng phát triển.

## Kết quả cần có

1. Pipeline có thể tái lập từ dữ liệu đến frozen prediction, rule evidence và giải thích.
2. Bảng so sánh ba nhóm predictor.
3. Tập fuzzy rules có domain rationale được ghi rõ, threshold và `fitted_missing_value` truy vết được, và hành vi được đánh giá định lượng; chưa tuyên bố đã được fraud analyst xác nhận.
4. Bảng `evaluated_rows`, coverage, fraud precision, lift và rule AUC theo split.
5. Bảng explanation coverage, unsupported-alert rate, `mean_active_rule_count`, `mean_displayed_rule_count`, `rule_sparsity`, overall/balanced consistency và contradiction; `rule_sparsity` được định nghĩa theo tỷ lệ rule active trên tổng rule khả dụng.
6. Ablation theo rule group, được ghi rõ là post-hoc khi dùng locked test.
7. Case studies cho TP, FP, FN và TN, gồm feature values, fitted thresholds, numeric `fitted_missing_value`, rule strengths và model probability.
8. Mã nguồn, config, notebook và run metadata, bao gồm source/config fingerprint, output checksums và bản ghi thủ công `dataset_reference`/`dataset_version`.

## Bằng chứng khóa luận có thể cung cấp

Trong phạm vi dataset và protocol đã chạy, khóa luận có thể chứng minh bằng thực nghiệm rằng:

- Trong model set và protocol đã chạy, reference tree models đạt raw PR-AUC cao hơn hai neural baselines.
- Tri thức miền có thể biểu diễn thành fuzzy predicates và luật tính toán được.
- Một số luật xác định vùng dữ liệu có fraud rate cao hơn base rate.
- Nhóm model alerts có rule evidence có thể có fraud precision cao hơn toàn bộ alerts, với độ bất định được định lượng bằng row bootstrap.
- Không phải mọi cảnh báo đều có đủ logical evidence.
- Prediction quality và explanation quality cần được đánh giá riêng.
- Validation-only protocol giảm nguy cơ data leakage và test overfitting.
- Cùng quy trình có thể được cấu hình và chạy lại trên IEEE-CIS và privacy-preserving synthetic benchmark BAF, dù hai dataset dùng predictor và rule base riêng.

Claim trung tâm:

> Trong phạm vi dữ liệu và giao thức thực nghiệm được sử dụng, một frozen predictor kết hợp với lớp fuzzy-rule/LTN-inspired có thể giữ nguyên pipeline dự đoán đã đánh giá và cung cấp bằng chứng audit có thể kiểm tra cho một phần cảnh báo; lợi ích và giới hạn được thể hiện qua coverage, precision gain và case studies.

## Những điều không tuyên bố

- LTN luôn tốt hơn tree models.
- Logic luôn cải thiện prediction performance.
- Rule explanation là causal explanation.
- Rule evidence là faithful explanation đầy đủ của predictor.
- Đây là một LTN end-to-end hoặc thuật toán LTN mới.
- Kết quả BAF chứng minh cùng model/rule base transfer từ IEEE-CIS.
- Post-hoc test ablation cung cấp một rule policy mới đã được đánh giá unbiased.
- Hệ thống sẵn sàng triển khai trong tổ chức tài chính.
- Kết quả khái quát cho mọi dataset và mọi môi trường.

## Tiêu chí hoàn thành

- Pipeline chạy top-to-bottom.
- Ít nhất ba model được so sánh công bằng.
- Rules có domain rationale được tài liệu hóa; threshold, category-risk mapping và numeric missing-value median data-informed được fit train-only; giới hạn chưa có analyst validation được nêu rõ.
- Reference predictor, calibration và decision threshold được khóa bằng train/validation trước explanation.
- Có metrics cho prediction và explanation.
- Có ablation và case studies.
- Raw PR-AUC được dùng cho model comparison; calibrated probability metrics và F2 được báo cáo đúng vai trò.
- Kết luận trả lời trực tiếp research questions.
- Mọi claim được giới hạn đúng theo bằng chứng.
