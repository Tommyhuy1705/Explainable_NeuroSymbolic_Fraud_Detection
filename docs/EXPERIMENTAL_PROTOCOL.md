# Giao thức thực nghiệm

## 1. Mục đích

Tài liệu này khóa các quyết định đánh giá trước khi đọc kết quả test, nhằm giảm data leakage, test overfitting và cherry-picking.

## 2. Data split

Mặc định sử dụng temporal split:

- Train: 70% thời gian đầu.
- Validation: 15% tiếp theo.
- Test: 15% cuối.

Nếu dataset không có biến thời gian đáng tin cậy, sử dụng stratified split với seed cố định và ghi rõ lý do.

## 3. Train-only operations

Chỉ train được phép dùng để fit:

- Missing-value statistics.
- Scaling statistics.
- Categorical encodings.
- Feature availability filters.
- Neural/tree model parameters.
- Quantile threshold của fuzzy predicates.
- Category-risk mapping trong logic rules.

## 4. Validation-only operations

Validation được dùng để:

- Theo dõi early stopping.
- Fit Platt hoặc isotonic calibration khi bật.
- Chọn decision threshold theo F1/F2.
- Chọn model/config trong phạm vi budget đã định trước.
- Chọn activation threshold nếu có thí nghiệm sensitivity riêng.

Mọi quyết định phải được khóa trước khi chạy test report.

## 5. Test-only reporting

Test chỉ dùng để:

- Tính prediction metrics cuối cùng.
- Tính rule/explanation metrics với engine đã fit.
- Tạo case studies sau khi policy đã khóa.

Không điều chỉnh feature, threshold, calibration, rule hoặc model sau khi đọc test.

## 6. Predictive models

| Nhóm | Vai trò |
|---|---|
| MLP | Neural baseline |
| TabularResNet | Tabular deep-learning model |
| XGBoost/LightGBM | Strong tree baseline |

Các model dùng cùng split và preprocessing contract. Hyperparameter budget phải được mô tả trong báo cáo.

## 7. Prediction metrics

Primary metric: PR-AUC, phù hợp với class imbalance.

Secondary metrics:

- ROC-AUC.
- Precision.
- Recall.
- F1/F2.
- Brier score.
- Expected Calibration Error.
- Negative log-likelihood.

Threshold được chọn trên validation và khóa trên test.

## 8. Logical rules

Rules được khai báo trong YAML. Threshold dạng quantile và category-risk mapping chỉ fit trên train. Rule không có đủ feature sẽ được bỏ qua và ghi lý do.

Rule metrics:

- Coverage.
- Fraud precision.
- Lift so với base fraud rate.
- Mean truth value.
- Rule AUC.
- Validation/test stability.

## 9. Explanation evaluation

Một explanation gồm tối đa `top_k_rules` có truth value vượt activation threshold.

System metrics:

- Coverage trên toàn bộ mẫu.
- Coverage trên model alerts.
- Mean rule count.
- Sparsity.
- Prediction-rule consistency.
- Contradiction rate.
- Explained-alert precision.

Các metrics này không chứng minh causal faithfulness. Case study cần trình bày cả TP, FP và FN.

## 10. Ablation

Tối thiểu:

- Từng rule riêng lẻ.
- Bỏ từng rule khỏi full rule set.
- Full rule set.
- Sensitivity theo activation threshold.

Ablation chỉ thay đổi thành phần đang kiểm tra; data split và predictor probabilities giữ nguyên.

## 11. Repeated runs

Smoke run dùng seed 42. Kết quả báo cáo nên dùng tối thiểu 3 seeds nếu compute cho phép. Mỗi run lưu:

- Config.
- Seed.
- Data source và row count.
- Feature count.
- Metrics.
- Prediction arrays.
- Thời điểm chạy và môi trường.

## 12. Claim boundaries

Kết quả chỉ hỗ trợ claim trong dataset và protocol đã chạy. Không tuyên bố:

- Universal superiority.
- Causal explanation.
- Production readiness.
- Generalization đến tổ chức tài chính khác khi chưa có dữ liệu tương ứng.
