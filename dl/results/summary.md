| Dataset | Model | Trials | CV RMSE | CV R² | Test MAE | Test RMSE [95% CI] | Test R² [95% CI] | NRMSE | Fit (s) | Predict (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| housing | FT-Transformer | 15 + 35 pruned | 1,010,087 ± 61,494 | 0.6607 ± 0.0271 | 987,572 | 1,351,028 [1,078,338, 1,633,617] | 0.6389 [0.534, 0.726] | 0.270 | 15.5 | 0.01 |
| housing | TabNet | 35 + 15 pruned | 1,085,143 ± 112,323 | 0.6074 ± 0.0654 | 1,100,857 | 1,510,134 [1,207,546, 1,799,102] | 0.5488 [0.410, 0.667] | 0.302 | 9.9 | 0.03 |
| housing | MLP | 5 cấu hình cố định | 988,391 ± 60,358 | 0.6712 ± 0.0593 | 1,003,632 | 1,368,702 [1,085,341, 1,668,441] | 0.6294 [0.537, 0.712] | 0.273 | 0.3 | 0.00 |
| kc | TabNet | 1 (mặc định, chưa tune) | 119,408 ± 3,584 | 0.8888 ± 0.0177 | 67,330 | 129,377 [114,206, 147,870] | 0.8893 [0.858, 0.913] | 0.236 | 275.7 | 0.21 |

**Phân tích sai số trên test** (chỉ để phân tích — metric chính là bảng trên; chi tiết 10 căn sai nhiều nhất trong `largest_errors_test.csv`)

| Dataset | Model | % SSE từ căn sai nhất | % SSE từ 5 căn sai nhất | max ŷ/y | RMSE nếu bỏ căn sai nhất | R² nếu bỏ căn sai nhất |
|---|---|---|---|---|---|---|
| housing | FT-Transformer | 14.4% | 41.6% | 1.82 | 1,255,691 | 0.6581 |
| housing | TabNet | 15.4% | 39.7% | 2.33 | 1,395,696 | 0.5776 |
| housing | MLP | 15.1% | 41.8% | 1.76 | 1,267,251 | 0.6398 |
| kc | TabNet | 9.0% | 24.3% | 3.52 | 123,434 | 0.8984 |
